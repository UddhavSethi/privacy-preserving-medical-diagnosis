"""ADR-1 fine-tuning x ADR-2 Differential Privacy -- the real experiment scoped by
CLAUDE.md's pending decision 4, single-seed, centralized only.

`scripts/smoke_test_dp_finetune.py` confirmed this combination fits comfortably in
this machine's 4GB VRAM (441MB peak at batch_size=16, no BatchMemoryManager needed)
and fixed a real bug along the way (torchvision's `denseblock4` hardcodes
`inplace=True` ReLUs, which crash Opacus's per-sample-gradient hooks -- fixed in
`DenseNet121Head.__init__`). This script is the actual measurement that smoke test
unblocked: does DP-SGD's utility cost get materially worse once fine-tuning raises
the trainable parameter count from ~263K (head-only, `src/privacy/dp.py`'s
production-proven path) to ~2.4M -- exactly the axis ADR-1's own DP-utility-collapse
rationale warns about?

Single seed (42, matching every other baseline's first seed), epsilon=4/delta=1e-5
(project default, DG-7), centralized/pooled natural partition only -- DP/SecAgg rows
and the balanced regime are out of scope here, same scoping the fine-tuning campaign
itself used (CLAUDE.md pending decision 3). This is a first measurement, not a
3-seed campaign -- CLAUDE.md section 11.2's evidentiary bar (mean +/- std over >= 3
seeds) is intentionally not claimed by this run alone.

Design, extending `src/privacy/dp.py`'s own established pattern (only ever wrapping
`model.classifier` with Opacus, since the frozen backbone's `denseblock1-3` still
contain real BatchNorm that Opacus's ModuleValidator rejects outright) to the larger
fine-tuned tail:
  - The frozen prefix (`DenseNet121Head._FROZEN_PREFIX_NAMES`) runs ONCE, outside
    Opacus, under `torch.no_grad()`, on the deterministic eval-style view (not the
    K-augmented-view training transform `train_centralized_finetune.py`'s non-DP
    path uses) -- matching `src/privacy/dp.py`'s own stated reason: Opacus's
    Poisson-sampling DataLoader doesn't compose with a different random view per
    epoch. This means DP fine-tuning trains on non-augmented images; that is a real,
    reportable difference from the non-DP fine-tuning baseline (row 2), not an
    oversight.
  - Only `denseblock4 + norm5 + classifier` (already GroupNorm/Linear -- DP-safe per
    ADR-1's resolved decision 11, and now Opacus-safe per the smoke test's ReLU fix)
    is wrapped by `PrivacyEngine.make_private()`, taking the frozen prefix's
    precomputed output as its input tensor.
  - Same differential-learning-rate protocol as the non-DP fine-tuning baseline
    (`train_centralized_finetune.py`): backbone tail at a smaller LR than the
    from-scratch classifier head.

Usage: uv run python scripts/train_centralized_finetune_dp.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from src.data.transforms import build_eval_transform
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporting import save_results
from src.models.densenet_head import _FROZEN_PREFIX_NAMES, DenseNet121Head
from src.models.freezing import count_trainable_parameters
from src.privacy.accounting import compute_noise_multiplier, compute_total_steps
from src.privacy.dp import _strip_opacus_prefix
from src.training.trainer import compute_class_weights
from src.utils.seeding import set_global_seed

# Sibling-module import, not `scripts.train_centralized_finetune` -- see that
# script's own note (or `run_federated_finetune_pilot.py`'s) on why: `scripts/` is
# not an installed package, only `src.*` resolves from an arbitrary cwd, and
# running this file directly puts its own directory on sys.path[0].
from train_centralized_finetune import (
    BACKBONE_LR,
    BATCH_SIZE,
    HEAD_LR,
    IMAGE_SIZE,
    LABEL_TO_INDEX,
    NUM_EPOCHS,
    PARTITION_PATH,
    PATIENCE,
    RawImageDataset,
    _records_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HOSPITALS = ["A", "B", "C"]
CHECKPOINT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "finetuned" / "centralized_dp_natural_seed42.pt"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "centralized_finetune_dp_singleseed.json"

SEED = 42
TARGET_EPSILON = 4.0  # project default, DG-7
TARGET_DELTA = 1e-5  # project default, DG-7
MAX_GRAD_NORM = 1.0  # DPConfig's own default, src/privacy/dp.py


class FineTuneTail(nn.Module):
    """denseblock4 + norm5 + pool + classifier -- the exact subset of
    `DenseNet121Head.forward()` that runs on the frozen prefix's *output* rather
    than raw pixels, and the only part of the fine-tuned model Opacus ever sees.
    Mirrors `scripts/smoke_test_dp_finetune.py`'s own class of the same name."""

    def __init__(self, head: DenseNet121Head) -> None:
        super().__init__()
        self.denseblock4 = head.features.denseblock4
        self.norm5 = head.features.norm5
        self.pool = head.pool
        self.classifier = head.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.denseblock4(x)
        x = self.norm5(x)
        x = torch.relu(x)
        x = self.pool(x)
        return self.classifier(x)


@torch.no_grad()
def _frozen_prefix_features(head: DenseNet121Head, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Everything up to (not including) denseblock4, computed once outside Opacus
    on the deterministic eval-style view -- see module docstring."""
    head.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = x
        for name in _FROZEN_PREFIX_NAMES:
            out = getattr(head.features, name)(out)
        feats.append(out.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def _evaluate_tail(tail: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device, batch_size: int = 256) -> dict:
    tail.eval()
    all_probs = []
    for i in range(0, len(features), batch_size):
        x = features[i : i + batch_size].to(device)
        probs = F.softmax(tail(x), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
    metrics = compute_metrics(labels.numpy(), np.array(all_probs)).to_dict()
    tn, fp, fn, tp = np.array(metrics["confusion_matrix"]).ravel()
    metrics["accuracy"] = float((tn + tp) / (tn + fp + fn + tp))
    return metrics


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = OmegaConf.load(REPO_ROOT / "conf" / "config.yaml")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("centralized_finetune_dp")

    set_global_seed(seed=SEED, data_partition_seed=SEED, client_sampling_seed=SEED)

    partition = json.loads(PARTITION_PATH.read_text())
    train_records = _records_for(partition, HOSPITALS, "train")
    val_records = _records_for(partition, HOSPITALS, "val")
    test_records = _records_for(partition, HOSPITALS, "test")
    print(f"pooled centralized DP fine-tune (natural): train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    # Deterministic eval-style view for ALL splits, including train -- see module
    # docstring for why DP training can't use the augmented view.
    eval_transform = build_eval_transform(image_size=IMAGE_SIZE)
    train_ds = RawImageDataset(train_records, eval_transform)
    val_ds = RawImageDataset(val_records, eval_transform)
    test_ds = RawImageDataset(test_records, eval_transform)

    raw_loader_kwargs = dict(batch_size=64, shuffle=False, num_workers=2)
    train_raw_loader = DataLoader(train_ds, **raw_loader_kwargs)
    val_raw_loader = DataLoader(val_ds, **raw_loader_kwargs)
    test_raw_loader = DataLoader(test_ds, **raw_loader_kwargs)

    head = DenseNet121Head(fine_tune_last_block=True).to(device)

    print("computing frozen-prefix features for train/val/test (outside Opacus, no_grad, eval-style view)...")
    t0 = time.monotonic()
    train_features, train_labels = _frozen_prefix_features(head, train_raw_loader, device)
    val_features, val_labels = _frozen_prefix_features(head, val_raw_loader, device)
    test_features, test_labels = _frozen_prefix_features(head, test_raw_loader, device)
    print(f"  done in {time.monotonic() - t0:.1f}s -- train feature shape {tuple(train_features.shape)}")

    tail = FineTuneTail(head)
    n_trainable = count_trainable_parameters(tail)
    print(f"trainable params in the DP-wrapped tail: {n_trainable:,}")

    class_weights = compute_class_weights(train_labels).to(device)

    opt = torch.optim.Adam(
        [
            {"params": list(tail.denseblock4.parameters()) + list(tail.norm5.parameters()), "lr": BACKBONE_LR},
            {"params": tail.classifier.parameters(), "lr": HEAD_LR},
        ]
    )

    train_dataset = TensorDataset(train_features, train_labels)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, generator=generator)

    sample_rate = BATCH_SIZE / len(train_dataset)
    total_steps = compute_total_steps(dataset_size=len(train_dataset), batch_size=BATCH_SIZE, local_epochs=1, num_rounds=NUM_EPOCHS)
    noise_multiplier = compute_noise_multiplier(
        target_epsilon=TARGET_EPSILON, target_delta=TARGET_DELTA, sample_rate=sample_rate, total_steps=total_steps
    )
    print(f"noise_multiplier for eps={TARGET_EPSILON}, delta={TARGET_DELTA}, total_steps={total_steps}: {noise_multiplier:.4f}")

    privacy_engine = PrivacyEngine(accountant="rdp")
    dp_tail, dp_opt, dp_loader = privacy_engine.make_private(
        module=tail, optimizer=opt, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=MAX_GRAD_NORM,
    )

    with mlflow.start_run(run_name=f"centralized_finetune_dp_natural_seed{SEED}"):
        mlflow.log_params({
            "seed": SEED,
            "fine_tune_last_block": True,
            "dp_enabled": True,
            "target_epsilon": TARGET_EPSILON,
            "target_delta": TARGET_DELTA,
            "max_grad_norm": MAX_GRAD_NORM,
            "noise_multiplier": noise_multiplier,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "head_lr": HEAD_LR,
            "backbone_lr": BACKBONE_LR,
            "n_train": len(train_records),
            "n_val": len(val_records),
            "n_test": len(test_records),
            "n_trainable_params": n_trainable,
        })

        best_val_auroc = -1.0
        best_state = None
        epochs_without_improvement = 0
        history = []

        for epoch in range(NUM_EPOCHS):
            dp_tail.train()
            t0 = time.time()
            epoch_loss, n_seen = 0.0, 0
            for x, y in dp_loader:
                x, y = x.to(device), y.to(device)
                out = dp_tail(x)
                loss = F.cross_entropy(out, y, weight=class_weights)
                dp_opt.zero_grad()
                loss.backward()
                dp_opt.step()
                epoch_loss += loss.item() * len(y)
                n_seen += len(y)
            epoch_loss /= max(n_seen, 1)
            epoch_time = time.time() - t0

            val_metrics = _evaluate_tail(dp_tail, val_features, val_labels, device)
            try:
                epsilon_so_far = privacy_engine.get_epsilon(delta=TARGET_DELTA)
            except OverflowError:
                epsilon_so_far = float("inf")
            history.append({
                "epoch": epoch, "train_loss": epoch_loss, "val_auroc": val_metrics["auroc"],
                "epoch_seconds": epoch_time, "epsilon_so_far": epsilon_so_far,
            })
            print(
                f"    epoch {epoch}: train_loss={epoch_loss:.4f} val_auroc={val_metrics['auroc']:.4f} "
                f"epsilon_so_far={epsilon_so_far:.3f} ({epoch_time:.1f}s)"
            )
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("val_auroc", val_metrics["auroc"], step=epoch)
            mlflow.log_metric("epsilon_so_far", epsilon_so_far, step=epoch)

            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                best_state = _strip_opacus_prefix(dp_tail.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"    early stopping at epoch {epoch}")
                    break

        try:
            epsilon_spent = privacy_engine.get_epsilon(delta=TARGET_DELTA)
        except OverflowError:
            epsilon_spent = float("inf")

        # Re-evaluate the BEST checkpoint (not necessarily the last epoch) on a
        # fresh, unwrapped tail -- dp_tail is a GradSampleModule and best_state was
        # already stripped of its "_module." prefix by _strip_opacus_prefix.
        eval_head = DenseNet121Head(fine_tune_last_block=True)
        eval_tail = FineTuneTail(eval_head)
        eval_tail.load_state_dict(best_state)
        eval_tail.to(device)
        test_metrics = _evaluate_tail(eval_tail, test_features, test_labels, device)

        print(
            f"\nbest val_auroc={best_val_auroc:.4f}  pooled test_auroc={test_metrics['auroc']:.4f}  "
            f"epsilon_spent={epsilon_spent:.4f}"
        )
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"pooled_test_{k}", v)
        mlflow.log_metric("best_val_auroc", best_val_auroc)
        mlflow.log_metric("epsilon_spent", epsilon_spent)

        # eval_tail shares its submodules BY REFERENCE with eval_head (FineTuneTail
        # doesn't copy) -- loading best_state into eval_tail above already updated
        # eval_head's own features.denseblock4/norm5/classifier in place.
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(eval_head.trainable_state_dict(), CHECKPOINT_PATH)
        print(f"checkpoint saved: {CHECKPOINT_PATH}")

        save_results({
            "history": history,
            "best_val_auroc": best_val_auroc,
            "epsilon_spent": epsilon_spent,
            "target_epsilon": TARGET_EPSILON,
            "target_delta": TARGET_DELTA,
            "noise_multiplier": noise_multiplier,
            "n_trainable_params": n_trainable,
            "pooled_test": test_metrics,
        }, RESULTS_PATH)
        print(f"Results written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
