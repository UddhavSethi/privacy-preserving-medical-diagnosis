"""Stage 18 — Grad-CAM explainability (CLAUDE.md section 9, objective 6's
explanation half): class-discriminative heatmaps over a chest X-ray showing
which lung regions drove the Pneumonia/Normal prediction.

Real, verified interaction with ADR-1's frozen backbone, not assumed (this
stage's own flagged risk) — and the verification surfaced a real bug, not
just confirmed a theory: Grad-CAM needs gradients to flow BACKWARD through
the target layer's *activations*, not its weights, and ADR-1's freezing
(`src/models/freezing.py`) sets `requires_grad=False` on the backbone's
*parameters* only, which by itself does not stop PyTorch from building the
autograd graph through the backbone's forward computation. But
`pytorch-grad-cam`'s `BaseCAM.forward()` only wraps the input tensor as
`torch.autograd.Variable(input_tensor, requires_grad=True)` when its
`compute_input_gradient` flag is `True` — and plain `GradCAM` does **not**
set that flag, defaulting to `False`. With every backbone parameter frozen
AND the input not requiring grad, *nothing* in the forward pass needs a
gradient, so PyTorch never builds a graph at all: the target layer's forward
hook fires (captures real activations), but its backward hook never does
(`grads=None`), producing a hard crash, not a silently-wrong heatmap — caught
immediately by `tests/test_gradcam.py`'s first real run, not by reading the
library's source and assuming the default handled it. Fixed here by
explicitly marking the input tensor `requires_grad_(True)` before calling
into the library, which is unnecessary for an ordinary fully-trainable model
(the plain-`GradCAM` default silently "works" there because *some* parameter
upstream already requires grad) but load-bearing for this project's
frozen-backbone architecture.

Target layer: `model.features.norm5` — DenseNet121's final backbone
BatchNorm2d (pre-ReLU, pre-pool; see `DenseNet121Head.pooled_features`).
Matches CLAUDE.md section 9. If the ADR-1 GroupNorm fallback is ever adopted,
this target layer name must be revisited (both here and in CLAUDE.md §9).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.transforms import build_eval_transform
from src.models.densenet_head import DenseNet121Head

NORMAL_CLASS_INDEX = 0
PNEUMONIA_CLASS_INDEX = 1


def get_target_layer(model: DenseNet121Head) -> torch.nn.Module:
    """CLAUDE.md section 9's chosen Grad-CAM target layer."""
    return model.features.norm5


def compute_gradcam_heatmap(
    model: DenseNet121Head,
    image_tensor: torch.Tensor,  # (1, 3, H, W), already normalized (build_eval_transform)
    target_class: int,
) -> np.ndarray:
    """Class-discriminative Grad-CAM heatmap, resized to the input's spatial
    resolution, values in [0, 1]. `model` must not be run inside
    `torch.no_grad()` by the caller (see module docstring)."""
    model.eval()
    # Load-bearing for this project's frozen backbone (ADR-1) — see module
    # docstring. Without this, no gradient graph gets built at all, since
    # every backbone parameter is also frozen; plain GradCAM does not set
    # this on its own (its `compute_input_gradient` flag defaults to False).
    image_tensor = image_tensor.clone().requires_grad_(True)
    with GradCAM(model=model, target_layers=[get_target_layer(model)]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(target_class)])
    return grayscale_cam[0]


@dataclass
class GradCAMOverlay:
    heatmap: np.ndarray  # (H, W) in [0, 1]
    overlay_rgb: np.ndarray  # (H, W, 3) uint8 — heatmap blended over the input image


def generate_overlay(
    model: DenseNet121Head,
    image_rgb_uint8: np.ndarray,  # already CLAHE'd (ADR-6), RGB, arbitrary size
    target_class: int,
    image_size: int = 224,
) -> GradCAMOverlay:
    """Raw CLAHE'd RGB image -> resized/normalized tensor (Stage 7's exact eval
    transform, so what the model sees here matches what it saw during
    evaluation) -> Grad-CAM heatmap -> RGB overlay for visual inspection."""
    tensor = build_eval_transform(image_size=image_size)(image_rgb_uint8).unsqueeze(0)
    heatmap = compute_gradcam_heatmap(model, tensor, target_class)

    resized = np.array(Image.fromarray(image_rgb_uint8).resize((image_size, image_size)))
    image_float01 = resized.astype(np.float32) / 255.0
    overlay_rgb = show_cam_on_image(image_float01, heatmap, use_rgb=True)
    return GradCAMOverlay(heatmap=heatmap, overlay_rgb=overlay_rgb)
