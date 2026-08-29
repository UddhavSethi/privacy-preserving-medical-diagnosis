"""Frozen-backbone feature cache (Stage 9, `REC`, Decision Gate DG-5).

Because the backbone is frozen (Stage 8, ADR-1) and never changes during training,
its pooled 1024-dim output for a given (image, transform) pair is a pure function of
that pair — it can be precomputed once and reused across every head-training epoch,
every epsilon value, every seed in the ablation campaign, instead of re-running a
~7M-parameter CNN forward pass every single step. Only the ~263K-parameter classifier
then needs to run live during head training, which is what makes the full ablation
campaign (6 configs x multiple epsilons x 3+ seeds) realistically finish on a 4GB
laptop GPU rather than taking days.

DG-5 (resolved 2026-08-29, owner-approved): cache K=5 augmented views per training
image, not just one deterministic view, so head training retains real
augmentation-driven regularization instead of seeing an identical feature vector every
epoch. Each training image also gets one deterministic ("eval-style", no augmentation)
feature cached alongside its K augmented ones — used for the augmentation-disabled
comparison test (Stage 9's own acceptance criterion) and available for fast
validation-time evaluation during head training. Val/test images only ever get the
single deterministic feature (no augmentation is ever applied to them).

This cache is invalidated by ANY change to: the frozen backbone's weights, the CLAHE
cache's contents/parameters, or the transform pipeline (image size, augmentation
parameters) — `FeatureCacheKey.hash_suffix()` folds in the transform parameters so a
change produces a new cache path rather than silently reusing stale features (the
CLAHE cache's own parameters are a separate concern: rebuilding this cache after a
CLAHE parameter change is a manual step, since the feature cache reads from
`data/clahe_cache/` and has no way to detect an upstream change on its own).

Full-image forward passes (not this cache) remain in use for final test-set
evaluation, inference, and Grad-CAM (Stage 18), which need the real image, not a
stored feature vector, once the model is producing a result someone will actually
read. If ADR-1's GroupNorm fallback is ever adopted, this entire stage becomes invalid
(the backbone is no longer frozen) and the cache must be rebuilt or retired.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

FEATURE_DIM = 1024  # DenseNet121's backbone output dimension


@dataclass(frozen=True)
class FeatureCacheKey:
    """Everything that must match for a cached feature bank to still be valid."""

    image_size: int
    num_augmented_views: int
    rotation_degrees: float
    brightness: float
    contrast: float

    def hash_suffix(self) -> str:
        raw = (
            f"size{self.image_size}_views{self.num_augmented_views}"
            f"_rot{self.rotation_degrees}_b{self.brightness}_c{self.contrast}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def cache_file_path(cache_dir: Path, source: str, split: str, key: FeatureCacheKey) -> Path:
    return cache_dir / key.hash_suffix() / f"{source}_{split}.pt"


@torch.no_grad()
def compute_pooled_features(model, image_rgb: np.ndarray, transform: Callable) -> torch.Tensor:
    """Transform one CLAHE-cached RGB image and run it through the frozen backbone's
    `pooled_features` (Stage 8) to get its 1024-dim feature vector. `model` must be a
    `DenseNet121Head` in eval mode.
    """
    tensor = transform(image_rgb).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    pooled = model.pooled_features(tensor)
    return pooled.squeeze(0).cpu()  # (1024,)


def save_feature_bank(
    path: Path,
    features: torch.Tensor,  # (N, V, FEATURE_DIM) — V = num_augmented_views+1 (train) or 1 (val/test)
    record_ids: list[str],
    labels: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "record_ids": record_ids, "labels": labels}, path)


def load_feature_bank(path: Path) -> dict:
    return torch.load(path, weights_only=True)
