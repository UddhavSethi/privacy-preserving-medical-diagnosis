"""OPT-3 — real tests for src/evaluation/gradcam_eval.py against hand-constructed
heatmaps and boxes with known answers, matching this project's convention."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.gradcam_eval import (
    iou_against_boxes,
    pointing_game_hit,
    rescale_boxes,
    summarize_localization,
)


def test_rescale_boxes_halves_coordinates_for_half_size_target():
    boxes = [(100.0, 200.0, 50.0, 60.0)]
    scaled = rescale_boxes(boxes, original_size=(1024, 1024), target_size=(512, 512))
    x, y, w, h = scaled[0]
    assert (x, y, w, h) == pytest.approx((50.0, 100.0, 25.0, 30.0))


def test_pointing_game_hit_when_peak_inside_box():
    heatmap = np.zeros((10, 10))
    heatmap[5, 5] = 1.0  # peak at (row=5, col=5)
    boxes = [(3, 3, 4, 4)]  # x=3,y=3,w=4,h=4 -> covers cols 3-6, rows 3-6
    assert pointing_game_hit(heatmap, boxes) is True


def test_pointing_game_miss_when_peak_outside_box():
    heatmap = np.zeros((10, 10))
    heatmap[0, 0] = 1.0
    boxes = [(5, 5, 2, 2)]
    assert pointing_game_hit(heatmap, boxes) is False


def test_pointing_game_requires_at_least_one_box():
    with pytest.raises(ValueError):
        pointing_game_hit(np.zeros((5, 5)), [])


def test_iou_perfect_overlap_is_one():
    heatmap = np.zeros((10, 10))
    heatmap[2:6, 2:6] = 1.0  # a 4x4 block, all at max value
    boxes = [(2, 2, 4, 4)]  # identical 4x4 region
    iou = iou_against_boxes(heatmap, boxes, threshold_fraction=0.5)
    assert iou == pytest.approx(1.0)


def test_iou_no_overlap_is_zero():
    heatmap = np.zeros((10, 10))
    heatmap[0:2, 0:2] = 1.0
    boxes = [(8, 8, 2, 2)]
    iou = iou_against_boxes(heatmap, boxes, threshold_fraction=0.5)
    assert iou == pytest.approx(0.0)


def test_iou_partial_overlap_matches_hand_computation():
    heatmap = np.zeros((10, 10))
    heatmap[0:4, 0:4] = 1.0  # predicted mask: rows/cols 0-3 (16 pixels)
    boxes = [(2, 2, 4, 4)]  # gt mask: rows/cols 2-5 (16 pixels)
    # intersection: rows/cols 2-3 -> 2x2 = 4 pixels. union = 16+16-4 = 28.
    iou = iou_against_boxes(heatmap, boxes, threshold_fraction=0.5)
    assert iou == pytest.approx(4 / 28)


def test_iou_invalid_threshold_raises():
    with pytest.raises(ValueError):
        iou_against_boxes(np.ones((5, 5)), [(0, 0, 1, 1)], threshold_fraction=0.0)
    with pytest.raises(ValueError):
        iou_against_boxes(np.ones((5, 5)), [(0, 0, 1, 1)], threshold_fraction=1.5)


def test_iou_requires_at_least_one_box():
    with pytest.raises(ValueError):
        iou_against_boxes(np.zeros((5, 5)), [])


def test_summarize_localization_averages_correctly():
    hits = [True, True, False, True]
    ious = [0.8, 0.6, 0.0, 0.4]
    summary = summarize_localization(hits, ious)
    assert summary.pointing_game_accuracy == pytest.approx(0.75)
    assert summary.mean_iou == pytest.approx(0.45)
    assert summary.n_images == 4


def test_summarize_localization_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        summarize_localization([True, False], [0.5])


def test_summarize_localization_empty_raises():
    with pytest.raises(ValueError):
        summarize_localization([], [])
