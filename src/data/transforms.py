"""torchvision transform pipelines (Stage 7). Everything except CLAHE lives here
(ADR-6) — this module never touches OpenCV; it consumes the already-CLAHE'd RGB
uint8 arrays that `src/data/preprocessing.py::load_from_cache` returns.

Two pipelines, deliberately separate objects so training augmentation can never leak
into evaluation (a flagged Stage 7 risk):
  - `build_train_transform`: resize, mild augmentation, scale-to-[0,1], normalize.
  - `build_eval_transform`: resize, scale-to-[0,1], normalize — no randomness.

Scaling to [0,1] and normalizing are two separate, explicit steps (`ToDtype(...,
scale=True)` then `Normalize`) rather than a single fused call, so it is visible in
the pipeline definition that normalization is applied exactly once (the "double
normalization" risk Stage 7 flags).
"""
from __future__ import annotations

import torch
import torchvision.transforms.v2 as transforms_v2

# ImageNet statistics — required because the model (Stage 8) is an ImageNet-pretrained
# DenseNet121 (ADR-1); using dataset-specific statistics instead would be inconsistent
# with the pretrained backbone's expected input distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(
    image_size: int = 224,
    rotation_degrees: float = 10.0,
    brightness: float = 0.1,
    contrast: float = 0.1,
) -> transforms_v2.Compose:
    """Mild augmentation only. No horizontal flip: chest X-ray laterality
    (heart/mediastinum position) is clinically meaningful, so flipping is avoided
    here, consistent with common practice in chest X-ray classification work."""
    return transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.Resize((image_size, image_size)),
            transforms_v2.RandomRotation(degrees=rotation_degrees),
            transforms_v2.ColorJitter(brightness=brightness, contrast=contrast),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms_v2.Compose:
    """Deterministic: resize + normalize only, zero randomness."""
    return transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.Resize((image_size, image_size)),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
