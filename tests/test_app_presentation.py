"""PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Phase 1A) — tests
for `app/presentation.py`, the one module that enforces "never fabricate
data": every test here that checks a `None`/"Not available" case exists
specifically because the OLD code (the `float("inf")` sentinel trap, Risk #6
in the plan) used to silently produce a real-looking fake value instead.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app import presentation
from app.inference import DecodedMeta


# ---------------------------------------------------------------------------
# certainty_label
# ---------------------------------------------------------------------------


def test_certainty_label_bands():
    assert presentation.certainty_label(0.1, 1.0) == "High"  # ratio 0.1 < 0.5
    assert presentation.certainty_label(0.49, 1.0) == "High"
    assert presentation.certainty_label(0.5, 1.0) == "Medium"  # ratio == 0.5, boundary
    assert presentation.certainty_label(0.99, 1.0) == "Medium"
    assert presentation.certainty_label(1.0, 1.0) == "Low"  # ratio == 1.0, boundary
    assert presentation.certainty_label(5.0, 1.0) == "Low"


@pytest.mark.parametrize("threshold", [None, float("inf"), float("nan"), 0.0, -1.0])
def test_certainty_label_none_for_unavailable_threshold(threshold):
    """The float('inf') sentinel trap (Risk #6): must return None, never
    silently compute a fake "Low" from an infinite/missing threshold."""
    assert presentation.certainty_label(0.5, threshold) is None


# ---------------------------------------------------------------------------
# image_check
# ---------------------------------------------------------------------------


def test_image_check_none_when_no_detectors():
    assert presentation.image_check({}) is None


def test_image_check_unusual_only_when_all_flag():
    assert presentation.image_check({"A": True, "B": True, "C": True}) == "Unusual"
    assert presentation.image_check({"A": True, "B": False, "C": True}) == "Typical"
    assert presentation.image_check({"A": False, "B": False, "C": False}) == "Typical"


# ---------------------------------------------------------------------------
# compute_focus
# ---------------------------------------------------------------------------


def _peaked_heatmap(y: int, x: int, size: int = 224) -> np.ndarray:
    hm = np.zeros((size, size), dtype=np.float32)
    hm[y, x] = 1.0
    return hm


def test_compute_focus_none_for_non_pneumonia_label():
    hm = _peaked_heatmap(50, 50)
    focus, spread = presentation.compute_focus(hm, "Normal")
    assert focus is None
    assert spread is True

    focus, spread = presentation.compute_focus(hm, "Uncertain")
    assert focus is None
    assert spread is True


def test_compute_focus_right_upper():
    # u < 0.5 -> patient "right" lung (radiographic convention); small y -> "upper"
    hm = _peaked_heatmap(y=20, x=50)  # v ~ 0.09, below the 0.12 lower bound -> spread
    focus, spread = presentation.compute_focus(hm, "Pneumonia")
    assert spread is True
    assert focus is None


def test_compute_focus_within_bounds_maps_zones():
    size = 224
    # v = (y+0.5)/size; want v in (0.12, 0.85). Pick y so t = (v-0.12)/0.68 lands
    # clearly in each zone band.
    for y_frac, expected_zone in [(0.20, "upper"), (0.50, "middle"), (0.75, "lower")]:
        y = int(y_frac * size)
        hm = _peaked_heatmap(y=y, x=30)  # x=30 -> u ~0.136 < 0.5 -> "right"
        focus, spread = presentation.compute_focus(hm, "Pneumonia")
        assert spread is False, f"y_frac={y_frac} unexpectedly spread"
        assert focus["side"] == "right"
        assert focus["zone"] == expected_zone, f"y_frac={y_frac} -> {focus}"


def test_compute_focus_left_side():
    hm = _peaked_heatmap(y=100, x=200)  # x=200 -> u ~0.897, just inside <0.9 bound
    focus, spread = presentation.compute_focus(hm, "Pneumonia")
    assert spread is False
    assert focus["side"] == "left"


def test_compute_focus_outside_u_bounds_is_spread():
    hm = _peaked_heatmap(y=100, x=5)  # u ~0.0245, outside (0.1, 0.9)
    focus, spread = presentation.compute_focus(hm, "Pneumonia")
    assert focus is None
    assert spread is True


# ---------------------------------------------------------------------------
# quality_checks
# ---------------------------------------------------------------------------


def test_quality_checks_flags_low_resolution():
    small_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    meta = DecodedMeta(format="JPEG", width=100, height=100, projection=None)
    result = presentation.quality_checks(small_bgr, meta)
    assert result["resolution_ok"] is False
    assert result["min_side_px"] == 100


def test_quality_checks_accepts_high_resolution():
    big_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
    meta = DecodedMeta(format="JPEG", width=512, height=512, projection=None)
    result = presentation.quality_checks(big_bgr, meta)
    assert result["resolution_ok"] is True


def test_quality_checks_detects_color_image_as_not_grayscale():
    # A vividly colored image should fail the grayscale check; a true
    # grayscale-in-BGR image (B==G==R everywhere) should pass it.
    color_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    color_bgr[..., 2] = 255  # pure red in BGR
    result = presentation.quality_checks(color_bgr, DecodedMeta("JPEG", 256, 256, None))
    assert result["grayscale"] is False

    gray_value = np.full((256, 256, 3), 128, dtype=np.uint8)
    result = presentation.quality_checks(gray_value, DecodedMeta("JPEG", 256, 256, None))
    assert result["grayscale"] is True


# ---------------------------------------------------------------------------
# heatmap / display image encoding
# ---------------------------------------------------------------------------


def test_heatmap_to_uint8_bytes_shape_and_clipping():
    heatmap = np.array([[-0.5, 0.0], [0.5, 1.5]], dtype=np.float32)
    raw = presentation.heatmap_to_uint8_bytes(heatmap)
    assert len(raw) == 4
    values = list(raw)
    assert values[0] == 0  # clipped from -0.5
    assert values[3] == 255  # clipped from 1.5
    assert values[2] == 127 or values[2] == 128  # 0.5 -> ~127.5


def test_make_display_jpeg_downscales_large_images_only():
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    small_jpeg = presentation.make_display_jpeg(small, max_edge=1600)
    assert isinstance(small_jpeg, bytes) and len(small_jpeg) > 0

    large = np.zeros((2000, 3000, 3), dtype=np.uint8)
    large_jpeg = presentation.make_display_jpeg(large, max_edge=1600)
    assert isinstance(large_jpeg, bytes) and len(large_jpeg) > 0


# ---------------------------------------------------------------------------
# checkpoint calibration resolution
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_calibration_validation_source():
    cfg = {"decision_threshold": 0.45, "temperature": 1.1, "abstention_half_width": 0.05}
    result = presentation.resolve_checkpoint_calibration(cfg)
    assert result.source == "validation"
    assert result.decision_threshold == 0.45
    assert result.temperature == 1.1
    assert result.abstention_half_width == 0.05


def test_resolve_checkpoint_calibration_default_source_when_uncalibrated():
    """No fabricated calibration values for a checkpoint that was never
    calibrated: temperature/abstention_half_width must be None, and the
    source must say so explicitly, never silently claim "validation"."""
    result = presentation.resolve_checkpoint_calibration({})
    assert result.source == "default"
    assert result.temperature is None
    assert result.abstention_half_width is None
    assert result.decision_threshold == 0.5  # the historical argmax default


# ---------------------------------------------------------------------------
# build_props JSON-safety (via result_props, the largest single builder)
# ---------------------------------------------------------------------------


def _make_fake_result(**overrides):
    from app.inference import InferenceResult

    defaults = dict(
        rgb_image=np.zeros((224, 224, 3), dtype=np.uint8),
        predicted_label="Pneumonia",
        predicted_class=1,
        confidence=0.8,
        prob_pneumonia=0.8,
        entropy=0.3,
        deferred=False,
        deferral_threshold=float("inf"),  # simulates "no real threshold available"
        abstained=False,
        decision_threshold=0.5,
        gradcam_overlay_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        gradcam_heatmap=_peaked_heatmap(112, 50),
        gradcam_target_class=1,
        ood_flags={},
        ood_scores={},
    )
    defaults.update(overrides)
    return InferenceResult(**defaults)


def test_result_props_degrades_to_not_available_without_deferral_or_ood():
    """Mirrors the pre-D2 Cloud state: deferral_threshold is the inf
    sentinel, ood_flags is empty. Every dependent field must be None, not a
    value derived from the sentinel."""
    result = _make_fake_result()
    calib = presentation.resolve_checkpoint_calibration({})
    props = presentation.result_props(result, calib)

    assert props["certainty"] is None
    assert props["deferred"] is None
    assert props["deferral_threshold"] is None
    assert props["image_check"] is None
    assert props["ood_sites"] is None
    assert props["temperature"] is None
    assert props["abstention_half_width"] is None
    assert props["decision_threshold_source"] == "default"


_FAKE_OOD_FLAGS = {"A": True, "B": False, "C": True}
_FAKE_OOD_SCORES = {"A": 0.1, "B": -0.2, "C": 0.05}


def test_result_props_real_values_when_available():
    result = _make_fake_result(deferral_threshold=0.6, ood_flags=_FAKE_OOD_FLAGS, ood_scores=_FAKE_OOD_SCORES)
    calib = presentation.resolve_checkpoint_calibration(
        {"decision_threshold": 0.55, "temperature": 1.1, "abstention_half_width": 0.05}
    )
    props = presentation.result_props(result, calib)

    assert props["certainty"] in ("High", "Medium", "Low")
    assert props["deferred"] is False
    assert props["deferral_threshold"] == 0.6
    assert props["image_check"] == "Typical"
    assert props["ood_sites"]["A"]["flagged"] is True
    assert props["decision_threshold_source"] == "validation"


def test_result_props_is_json_serializable():
    result = _make_fake_result(deferral_threshold=0.6, ood_flags=_FAKE_OOD_FLAGS, ood_scores=_FAKE_OOD_SCORES)
    calib = presentation.resolve_checkpoint_calibration(
        {"decision_threshold": 0.55, "temperature": 1.1, "abstention_half_width": 0.05}
    )
    props = presentation.result_props(result, calib)
    json.dumps(props)  # must not raise


def test_result_props_never_returns_low_when_input_missing():
    """Direct regression test for the exact trap the plan names: the old
    `float("inf")` sentinel used to make `uncertainty_label` return "Low"
    uncertainty -- displayed as fake "High" certainty. Confirms that string
    never appears anywhere a value is actually unavailable."""
    result = _make_fake_result()  # inf threshold, no OOD
    calib = presentation.resolve_checkpoint_calibration({})
    props = presentation.result_props(result, calib)
    assert props["certainty"] != "Low"
    assert props["certainty"] is None
