"""ADR-1 GroupNorm fallback -- pilot fine-tuned centralized baseline.

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

**Deliberately scoped as a bounded PILOT, not the full Stage 21 campaign**: one
config (centralized, natural partition -- the privacy-free ceiling, and the most
informative single comparison point since it isolates "does fine-tuning help at
all" from FL/DP complexity), one seed (42, matching the existing baseline this is
compared against), reduced epoch budget. This is the same protocol
`scripts/train_centralized.py` (Stage 12) used, EXCEPT:
  - Trains on raw CLAHE-cached images through the (partially) unfrozen backbone,
    not Stage 9's cached pooled features -- those are frozen-backbone-only and
    invalid once denseblock4/norm5 become trainable.
  - Differential learning rates: the pretrained-but-now-trainable backbone tail
    (denseblock4+norm5) uses a smaller LR than the from-scratch classifier head,
    standard fine-tuning practice to avoid destroying pretrained features.
  - Checkpoint format changes: saves the WHOLE model's state_dict (backbone tail +
    classifier), not classifier-only, since the backbone tail is no longer a fixed,
    reproducible-from-`pretrained=True` function.

Usage: uv run python scripts/train_centralized_finetune.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.data.transforms import build_eval_transform, build_train_transform
from src.evaluation.metrics import compute_metrics
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import compute_class_weights
from src.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
CLAHE_CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "finetuned"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "centralized_finetune_pilot.json"

HOSPITALS = ["A", "B", "C"]
LABEL_TO_INDEX = {"Normal": 0, "Pneumonia": 1}
SEED = 42
IMAGE_SIZE = 224

# Pilot-scoped, not Stage 12's full protocol (30 epochs/patience 5) -- see module
# docstring. Bounded so a single pilot run finishes in a session-realistic time on
# a 4GB laptop GPU running full backbone forward/backward passes (no feature cache).
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
    import numpy as np

    return compute_metrics(np.array(all_labels), np.array(all_probs)).to_dict()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    partition = json.loads(PARTITION_PATH.read_text())
    train_records = _records_for(partition, HOSPITALS, "train")
    val_records = _records_for(partition, HOSPITALS, "val")
    test_records = _records_for(partition, HOSPITALS, "test")
    print(f"pooled centralized (natural): train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    set_global_seed(seed=SEED, data_partition_seed=SEED, client_sampling_seed=SEED)

    train_ds = RawImageDataset(train_records, build_train_transform(image_size=IMAGE_SIZE))
    val_ds = RawImageDataset(val_records, build_eval_transform(image_size=IMAGE_SIZE))
    test_ds = RawImageDataset(test_records, build_eval_transform(image_size=IMAGE_SIZE))

    generator = torch.Generator().manual_seed(SEED)
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
        print(f"epoch {epoch}: train_loss={epoch_loss:.4f} val_auroc={val_metrics['auroc']:.4f} ({epoch_time:.1f}s)")

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate(model, test_loader, device)
    print(f"\nbest val_auroc={best_val_auroc:.4f}  pooled test_auroc={test_metrics['auroc']:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "centralized_natural_seed42.pt"
    torch.save(best_state, ckpt_path)
    print(f"checkpoint saved: {ckpt_path}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"history": history, "best_val_auroc": best_val_auroc, "pooled_test": test_metrics}, indent=2))
    print(f"results saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
