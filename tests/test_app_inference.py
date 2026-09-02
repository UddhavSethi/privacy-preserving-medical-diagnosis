"""OPT-6 — smoke tests for app/inference.py (CLAUDE.md section 11.3 / the OPT-6
plan's own testing criterion: "a basic smoke test that the app imports cleanly
and its inference-calling function produces the expected shape/types" — not a
full research-grade suite, since UI/demo code isn't privacy-critical).

Deliberately imports `app.inference` directly, never `app.streamlit_app` —
confirms the inference layer works standalone, with no Streamlit process
required, which is the whole point of keeping Streamlit out of that module.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app import inference

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
FEATURE_CACHE_DIR = REPO_ROOT / "data" / "feature_cache"
CENTRALIZED_CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "centralized_baseline" / "natural_seed42.pt"

requires_real_artifacts = pytest.mark.skipif(
    not PARTITION_PATH.exists() or not FEATURE_CACHE_DIR.exists() or not CENTRALIZED_CHECKPOINT.exists(),
    reason="requires the real frozen partition + feature cache + a trained checkpoint (Stages 4-9, 12)",
)


def _encode_jpeg_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr)
    assert ok
    return buf.tobytes()


def test_decode_uploaded_image_from_real_jpeg_bytes():
    synthetic = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    jpeg_bytes = _encode_jpeg_bytes(synthetic)
    decoded = inference.decode_uploaded_image(jpeg_bytes, "upload.jpg")
    assert decoded.shape == (64, 64, 3)
    assert decoded.dtype == np.uint8


def test_decode_uploaded_image_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        inference.decode_uploaded_image(b"not an image", "upload.jpg")


def test_uncertainty_label_bands():
    threshold = 1.0
    assert inference.uncertainty_label(0.1, threshold) == "Low"
    assert inference.uncertainty_label(0.6, threshold) == "Medium"
    assert inference.uncertainty_label(1.5, threshold) == "High"


def test_preprocess_image_produces_model_ready_tensor():
    synthetic_bgr = np.random.default_rng(1).integers(0, 256, size=(200, 180, 3), dtype=np.uint8)
    rgb_image, tensor = inference.preprocess_image(synthetic_bgr, image_size=224)
    assert rgb_image.shape == (200, 180, 3)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


@requires_real_artifacts
def test_load_classifier_produces_a_working_model():
    model = inference.load_classifier(CENTRALIZED_CHECKPOINT)
    assert isinstance(model, inference.DenseNet121Head)


requires_xray_gate = pytest.mark.skipif(
    not inference.XRAY_GATE_WEIGHTS_PATH.exists(),
    reason="requires committed xray_gate_weights.json (scripts/build_xray_gate.py)",
)


@requires_xray_gate
def test_xray_gate_accepts_a_real_chest_xray():
    if not PARTITION_PATH.exists():
        pytest.skip("requires the real partition for a genuine test X-ray")
    partition = json.loads(PARTITION_PATH.read_text())
    from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache

    record = next(r for r in partition["hospitals"]["A"] if r["frozen_split"] == "test")
    clahe_cache_dir = REPO_ROOT / "data" / "clahe_cache"
    cache_path = cache_path_for(clahe_cache_dir, record["source"], record["relative_path"], ClaheParams())
    if not cache_path.exists():
        pytest.skip("CLAHE cache entry for this record not present")
    rgb_image = load_from_cache(cache_path)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    gate = inference.load_xray_gate()
    result = inference.check_is_xray(bgr_image, gate)
    assert result.is_xray is True


@requires_xray_gate
def test_xray_gate_rejects_random_noise():
    # Not the exact bootstrap negatives (those are local, uncommitted real
    # photos — see scripts/build_xray_gate.py) but the same synthetic-noise
    # generation the gate was trained to reject alongside them, and portable
    # to any environment (CI included) without needing local image files.
    rng = np.random.default_rng(123)
    noise_bgr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    gate = inference.load_xray_gate()
    result = inference.check_is_xray(noise_bgr, gate)
    assert result.is_xray is False


@requires_real_artifacts
def test_run_full_inference_end_to_end_shape_and_types():
    model = inference.load_classifier(CENTRALIZED_CHECKPOINT)

    # A real cached CLAHE'd chest X-ray, decoded straight back to bytes so this
    # exercises the exact same decode path a real upload would.
    from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache

    import json

    partition = json.loads(PARTITION_PATH.read_text())
    record = next(r for r in partition["hospitals"]["A"] if r["frozen_split"] == "test")
    cache_path = cache_path_for(FEATURE_CACHE_DIR.parent / "clahe_cache", record["source"], record["relative_path"], ClaheParams())
    if not cache_path.exists():
        pytest.skip("CLAHE cache entry for this record not present")
    rgb_image = load_from_cache(cache_path)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    threshold = inference.calibrate_deferral_threshold(
        model, PARTITION_PATH, FEATURE_CACHE_DIR, target_defer_fraction=0.10, num_mc_passes=5
    )
    assert np.isfinite(threshold)

    detectors, thresholds = inference.build_ood_detectors(
        PARTITION_PATH, FEATURE_CACHE_DIR, ["A", "B", "C"], seed=42, target_flag_fraction=0.05
    )
    assert set(detectors.keys()) == {"A", "B", "C"}

    result = inference.run_full_inference(
        model, bgr_image, deferral_threshold=threshold, ood_detectors=detectors, ood_thresholds=thresholds, num_mc_passes=5
    )

    assert result.predicted_label in ("Normal", "Pneumonia")
    assert result.predicted_class in (0, 1)
    assert result.abstained is False
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.prob_pneumonia <= 1.0
    assert result.entropy >= 0.0
    assert isinstance(result.deferred, bool)
    assert result.gradcam_overlay_rgb.shape[-1] == 3
    assert set(result.ood_flags.keys()) == {"A", "B", "C"}
    assert all(isinstance(v, bool) for v in result.ood_flags.values())
    assert all(np.isfinite(v) for v in result.ood_scores.values())

    # Same image, but a wide abstention band around 0.5 should force "Uncertain"
    # for anything not extremely confident -- proves the new abstention path
    # actually engages, not just that its default (off) leaves old behavior
    # unchanged (the assertions above already cover that).
    abstained_result = inference.run_full_inference(
        model, bgr_image, deferral_threshold=threshold, ood_detectors=detectors, ood_thresholds=thresholds,
        num_mc_passes=5, decision_threshold=0.5, abstention_half_width=0.49,
    )
    assert abstained_result.abstained is True
    assert abstained_result.predicted_label == inference.UNCERTAIN_LABEL
    assert abstained_result.predicted_class is None
