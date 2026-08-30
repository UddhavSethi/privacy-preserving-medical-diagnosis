"""OPT-3 — quantitative Grad-CAM evaluation (owner-approved 2026-08-30, Phase 6
priority 3). CLAUDE.md section 15, item 6: "Grad-CAM is evaluated qualitatively"
— this module converts heatmaps into a measured localization result against
RSNA's real bounding-box annotations, the standard weakly-supervised-localization
metrics: the **pointing game** (does the heatmap's single most-activated pixel fall
inside a ground-truth box?) and **Intersection-over-Union** against a thresholded
heatmap mask.

RSNA-only, by construction (CLAUDE.md's own known limitation): Kermany carries no
bounding-box annotations, so this analysis cannot include hospital A.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def rescale_boxes(
    boxes: list[tuple[float, float, float, float]],
    original_size: tuple[int, int],  # (width, height)
    target_size: tuple[int, int],  # (width, height)
) -> list[tuple[float, float, float, float]]:
    """Rescales (x, y, width, height) boxes from the original image's pixel space
    (RSNA DICOMs, typically 1024x1024) to the Grad-CAM heatmap's spatial resolution
    (224x224, matching `build_eval_transform`'s resize)."""
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    sx, sy = target_w / orig_w, target_h / orig_h
    return [(x * sx, y * sy, w * sx, h * sy) for x, y, w, h in boxes]


def _box_mask(boxes: list[tuple[float, float, float, float]], shape: tuple[int, int]) -> np.ndarray:
    """Union of all boxes as a boolean mask over `shape` (H, W)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def pointing_game_hit(heatmap: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> bool:
    """True if the heatmap's single most-activated pixel falls inside at least one
    ground-truth box — the standard pointing-game metric (Zhang et al. 2018)."""
    if not boxes:
        raise ValueError("pointing_game_hit requires at least one ground-truth box")
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    mask = _box_mask(boxes, heatmap.shape)
    return bool(mask[peak_y, peak_x])


def iou_against_boxes(
    heatmap: np.ndarray, boxes: list[tuple[float, float, float, float]], threshold_fraction: float = 0.5
) -> float:
    """Binarizes the heatmap at `threshold_fraction` of its own max value (a
    standard, simple choice in weakly-supervised localization evaluation — not
    tuned to this dataset) and computes IoU against the union of ground-truth
    boxes. Returns 0.0 if the thresholded heatmap and the box union share no
    pixels and neither is degenerate-empty in a way that would make IoU undefined
    (both empty is treated as IoU=1.0, matching the standard convention, though it
    should not occur here since boxes are always non-empty)."""
    if not boxes:
        raise ValueError("iou_against_boxes requires at least one ground-truth box")
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError(f"threshold_fraction must be in (0, 1], got {threshold_fraction}")

    peak = heatmap.max()
    pred_mask = heatmap >= (threshold_fraction * peak) if peak > 0 else np.zeros_like(heatmap, dtype=bool)
    gt_mask = _box_mask(boxes, heatmap.shape)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


@dataclass(frozen=True)
class LocalizationSummary:
    pointing_game_accuracy: float
    mean_iou: float
    n_images: int

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_localization(hits: list[bool], ious: list[float]) -> LocalizationSummary:
    if len(hits) != len(ious):
        raise ValueError(f"hits has {len(hits)} entries but ious has {len(ious)}")
    if not hits:
        raise ValueError("hits/ious must be non-empty")
    return LocalizationSummary(
        pointing_game_accuracy=float(np.mean(hits)),
        mean_iou=float(np.mean(ious)),
        n_images=len(hits),
    )
