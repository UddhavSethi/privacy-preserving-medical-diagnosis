"""ADR-1 GroupNorm fallback -- centralized fine-tuned baseline (ablation row 2).

Debugging session, 2026-08-31: a real "the model looks at the wrong part of the
X-ray" complaint traced to a structural cause, not a bug -- with the backbone fully
frozen, the classifier only ever sees a globally-average-pooled 1024-number summary
(`DenseNet121Head.pooled_features`), so no spatial layout survives to reach it, and
Grad-CAM's heatmap is at best a reconstruction, not what the classifier used.
Measured: quantitative Grad-CAM localization (`docs/gradcam_localization.md`'s
pointing-game metric) collapses from an already-weak 18.6% to 0.0% specifically on
images the frozen-backbone model gets wrong (150 real RSNA boxed test images).

This script trains ADR-1's own documented approved fallback (owner-approved before
implementation -- see docs/adr1_groupnorm_fallback.md): `DenseNet121Head
(fine_tune_last_block=True)` (`src/models/densenet_head.py`) unfreezes denseblock4 +
norm5 and swaps their BatchNorm for GroupNorm via Opacus's `ModuleValidator.fix()`,
so the model can adapt spatial features to chest X-rays instead of relying entirely
on frozen generic ImageNet channels behind a fixed pooling op.

**2026-09-21: scaled from the original single-seed pilot to a proper 3-seed
campaign** (owner-approved scope: rows 1-3, natural regime, fine-tuning -- see
CLAUDE.md's pending decision 3), matching CLAUDE.md section 11.2's own evidentiary
bar ("mean +/- std over at least 3 seeds -- single-run numbers are not credible").
The original pilot (seed 42 only, `outputs/results/centralized_finetune_pilot.json`)
is left untouched as its own historical record; this script writes to a separate
`centralized_finetune_multiseed.json` and re-runs seed 42 fresh under the unified
multi-seed protocol below (adds MLflow logging + a shared results format with rows
1/3) rather than splicing the old pilot's artifact into a new aggregate, for a
single consistent source per CLAUDE.md section 12's "every reported number traces
to a config hash + seed set + MLflow run."

Differs from `scripts/train_centralized.py` (Stage 12, the frozen-backbone version)
in the same three ways the original pilot did:
  - Trains on raw CLAHE-cached images through the (partially) unfrozen backbone,
    not Stage 9's cached pooled features -- those are frozen-backbone-only and
    invalid once denseblock4/norm5 become trainable.
  - Differential learning rates: the pretrained-but-now-trainable backbone tail
    (denseblock4+norm5) uses a smaller LR than the from-scratch classifier head,
    standard fine-tuning practice to avoid destroying pretrained features.
  - Checkpoint format changes: saves the WHOLE model's state_dict (backbone tail +
    classifier), not classifier-only, since the backbone tail is no longer a fixed,
    reproducible-from-`pretrained=True` function.

Scope stays natural-partition-only, matching rows 2/3's owner-approved scope for
this campaign (balanced regime and DP/SecAgg rows are explicitly out of scope).

Usage: uv run python scripts/train_centralized_finetune.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.data.transforms import build_eval_transform, build_train_transform
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporting import aggregate_metrics_over_seeds, save_results
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import compute_class_weights
from src.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
CLAHE_CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "finetuned"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "centralized_finetune_multiseed.json"

HOSPITALS = ["A", "B", "C"]
LABEL_TO_INDEX = {"Normal": 0, "Pneumonia": 1}
SEEDS = [42, 123, 2024]
IMAGE_SIZE = 224

# Same bounded protocol the pilot used (session-realistic on a 4GB laptop GPU
# running full backbone forward/backward passes, no feature cache) -- the pilot's
# own 8-epoch run already showed val AUROC plateauing (0.9186->0.9199->0.919->
# 0.9205 across epochs 4-7), so this is not under-budgeted.
NUM_EPOCHS = 8
PATIENCE = 3
BATCH_SIZE = 16
HEAD_LR = 1e-3  # matches Stage 12's classifier LR exactly
BACKBONE_LR = 1e-4  # smaller: denseblock4/norm5 are pretrained, not from scratch


class RawImageDataset(Dataset):
    """Reads CLAHE-cached images directly for a list of `hospitals_natural.json`
    records (mixed sources allowed -- each record carries its own `source`, unlike
    `src/data/datasets.py::ChestXrayDataset` which fixes one source per instance)."""

    def __init__(self, records: list[dict], transform: Callable) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        cache_path = cache_path_for(CLAHE_CACHE_DIR, record["source"], record["relative_path"], ClaheParams())
        image = load_from_cache(cache_path)
        tensor = self.transform(image)
        label = LABEL_TO_INDEX[record["label"]]
        return tensor, label


def _records_for(partition: dict, hospitals: list[str], split: str) -> list[dict]:
    out = []
    for h in hospitals:
        out += [r for r in partition["hospitals"][h] if r["frozen_split"] == split]
    return out


@torch.no_grad()
def _evaluate(model: DenseNet121Head, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs)).to_dict()
    tn, fp, fn, tp = np.array(metrics["confusion_matrix"]).ravel()
    metrics["accuracy"] = float((tn + tp) / (tn + fp + fn + tp))
    return metrics


def _train_one_seed(
    seed: int,
    train_records: list,
    val_records: list,
    test_records: list,
    device: torch.device,
    checkpoint_name: str | None = None,
) -> dict:
    """`checkpoint_name` defaults to `centralized_natural_seed{seed}.pt` (row 2's
    own naming). Callers training more than one model under the same seed value
    (e.g. `train_local_finetune.py`, one model per hospital) MUST pass a distinct
    name -- 2026-09-21 post-mortem: the original per-hospital caller relied on
    training into this fixed default name and renaming the file afterward, which
    let row 1's hospital-A run silently overwrite row 2's still-needed seed-42/123/
    2024 checkpoints before they were renamed away, permanently losing those three
    files (the JSON metrics were unaffected, saved independently beforehand)."""
    set_global_seed(seed=seed, data_partition_seed=seed, client_sampling_seed=seed)

    train_ds = RawImageDataset(train_records, build_train_transform(image_size=IMAGE_SIZE))
    val_ds = RawImageDataset(val_records, build_eval_transform(image_size=IMAGE_SIZE))
    test_ds = RawImageDataset(test_records, build_eval_transform(image_size=IMAGE_SIZE))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    train_labels_t = torch.tensor([LABEL_TO_INDEX[r["label"]] for r in train_records])
    class_weights = compute_class_weights(train_labels_t).to(device)

    model = DenseNet121Head(fine_tune_last_block=True).to(device)
    backbone_params = list(model.features.denseblock4.parameters()) + list(model.features.norm5.parameters())
    opt = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": HEAD_LR},
        ]
    )

    best_val_auroc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        t0 = time.time()
        epoch_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y, weight=class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(y)
            n_seen += len(y)
        epoch_loss /= n_seen
        epoch_time = time.time() - t0

        val_metrics = _evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_auroc": val_metrics["auroc"], "epoch_seconds": epoch_time})
        print(f"    epoch {epoch}: train_loss={epoch_loss:.4f} val_auroc={val_metrics['auroc']:.4f} ({epoch_time:.1f}s)")
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("val_auroc", val_metrics["auroc"], step=epoch)

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"    early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate(model, test_loader, device)
    print(f"    best val_auroc={best_val_auroc:.4f}  pooled test_auroc={test_metrics['auroc']:.4f}")

    for k, v in test_metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mlflow.log_metric(f"pooled_test_{k}", v)
    mlflow.log_metric("best_val_auroc", best_val_auroc)

    ckpt_path = CHECKPOINT_DIR / (checkpoint_name or f"centralized_natural_seed{seed}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)
    print(f"    checkpoint saved: {ckpt_path}")

    return {"history": history, "best_val_auroc": best_val_auroc, "pooled_test": test_metrics}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = OmegaConf.load(REPO_ROOT / "conf" / "config.yaml")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("centralized_finetune")

    partition = json.loads(PARTITION_PATH.read_text())
    train_records = _records_for(partition, HOSPITALS, "train")
    val_records = _records_for(partition, HOSPITALS, "val")
    test_records = _records_for(partition, HOSPITALS, "test")
    print(f"pooled centralized fine-tune (natural): train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    per_seed_results = []
    per_seed_test_metrics = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        with mlflow.start_run(run_name=f"centralized_finetune_natural_seed{seed}"):
            mlflow.log_params(
                {
                    "seed": seed,
                    "fine_tune_last_block": True,
                    "num_epochs": NUM_EPOCHS,
                    "patience": PATIENCE,
                    "batch_size": BATCH_SIZE,
                    "head_lr": HEAD_LR,
                    "backbone_lr": BACKBONE_LR,
                    "n_train": len(train_records),
                    "n_val": len(val_records),
                    "n_test": len(test_records),
                }
            )
            result = _train_one_seed(seed, train_records, val_records, test_records, device)
            per_seed_results.append(result)
            per_seed_test_metrics.append(result["pooled_test"])

    aggregated = aggregate_metrics_over_seeds(per_seed_test_metrics)
    print(
        f"\ncentralized fine-tune (natural), pooled test: AUROC = "
        f"{aggregated['auroc']['mean']:.4f} +/- {aggregated['auroc']['std']:.4f} "
        f"(n={aggregated['auroc']['n_seeds']} seeds)"
    )
    for k in ("accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy"):
        if k in aggregated:
            print(f"  {k}: {aggregated[k]['mean']:.4f} +/- {aggregated[k]['std']:.4f}")

    save_results(
        {
            "per_seed": per_seed_results,
            "pooled_test": aggregated,
        },
        RESULTS_PATH,
    )
    print(f"\nResults written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
