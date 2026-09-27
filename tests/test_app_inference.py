"""OPT-6 — smoke tests for app/inference.py (CLAUDE.md section 11.3 / the OPT-6
plan's own testing criterion: "a basic smoke test that the app imports cleanly
and its inference-calling function produces the expected shape/types" — not a
full research-grade suite, since UI/demo code isn't privacy-critical).

Deliberately imports `app.inference` directly, never `app.streamlit_app` —
confirms the inference layer works standalone, with no Streamlit process
required, which is the whole point of keeping Streamlit out of that module.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

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

    # PneumoScan redesign (docs/pneumoscan_redesign_plan.md Phase 1A): the raw
    # Grad-CAM heatmap must actually be exposed now, not silently discarded
    # (the exact bug this field was added to fix).
    assert result.gradcam_heatmap.shape == (224, 224)
    assert result.gradcam_heatmap.dtype == np.float32
    assert 0.0 <= float(result.gradcam_heatmap.min())
    assert float(result.gradcam_heatmap.max()) <= 1.0
    assert result.gradcam_target_class in (inference.NORMAL_CLASS_INDEX, inference.PNEUMONIA_CLASS_INDEX)


# ---------------------------------------------------------------------------
# PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Phase 1A) additions
# ---------------------------------------------------------------------------


def test_supported_upload_extensions_includes_dicom_and_dot_dicom():
    """The pre-existing bug this constant was added to fix: the old file
    uploader's own extension list had ".dcm" but not ".dicom", despite the
    decoder always supporting both."""
    assert ".dcm" in inference.SUPPORTED_UPLOAD_EXTENSIONS
    assert ".dicom" in inference.SUPPORTED_UPLOAD_EXTENSIONS
    assert ".jpg" in inference.SUPPORTED_UPLOAD_EXTENSIONS
    assert ".jpeg" in inference.SUPPORTED_UPLOAD_EXTENSIONS
    assert ".png" in inference.SUPPORTED_UPLOAD_EXTENSIONS


def _make_synthetic_dicom_bytes(rows: int = 64, cols: int = 64, view_position: str | None = None) -> bytes:
    """A minimal, valid, in-memory DICOM file — no real patient data, no
    fixture file on disk (portable to CI, same reasoning as this file's
    existing synthetic-noise X-ray-gate test)."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if view_position is not None:
        ds.ViewPosition = view_position
    pixel_array = np.random.default_rng(0).integers(0, 4096, size=(rows, cols), dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_decode_uploaded_image_with_meta_dicom_extracts_projection():
    dicom_bytes = _make_synthetic_dicom_bytes(rows=48, cols=32, view_position="PA")
    bgr, meta = inference.decode_uploaded_image_with_meta(dicom_bytes, "upload.dcm")
    assert bgr.shape == (48, 32, 3)
    assert meta.format == "DICOM"
    assert meta.width == 32
    assert meta.height == 48
    assert meta.projection == "PA"


def test_decode_uploaded_image_with_meta_dicom_dotdicom_extension():
    """The `.dicom` extension specifically (not just `.dcm`) must decode too."""
    dicom_bytes = _make_synthetic_dicom_bytes(rows=32, cols=32)
    bgr, meta = inference.decode_uploaded_image_with_meta(dicom_bytes, "upload.dicom")
    assert bgr.shape == (32, 32, 3)
    assert meta.format == "DICOM"


def test_decode_uploaded_image_with_meta_dicom_no_projection_tag():
    """No ViewPosition tag present -> None, never a guessed/default value."""
    dicom_bytes = _make_synthetic_dicom_bytes(rows=32, cols=32, view_position=None)
    _, meta = inference.decode_uploaded_image_with_meta(dicom_bytes, "upload.dcm")
    assert meta.projection is None


def test_decode_uploaded_image_with_meta_jpeg_never_fabricates_projection():
    synthetic = np.random.default_rng(0).integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    jpeg_bytes = _encode_jpeg_bytes(synthetic)
    bgr, meta = inference.decode_uploaded_image_with_meta(jpeg_bytes, "upload.jpg")
    assert bgr.shape == (64, 96, 3)
    assert meta.format == "JPEG"
    assert meta.width == 96
    assert meta.height == 64
    assert meta.projection is None  # JPEG carries no such tag -- must never be guessed


@requires_real_artifacts
def test_run_full_inference_progress_callback_fires_all_stages_in_order():
    model = inference.load_classifier(CENTRALIZED_CHECKPOINT)
    synthetic_bgr = np.random.default_rng(2).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    events: list[tuple[str, str]] = []

    def callback(stage: str, event: str) -> None:
        events.append((stage, event))

    inference.run_full_inference(
        model,
        synthetic_bgr,
        deferral_threshold=float("inf"),
        ood_detectors={},
        ood_thresholds={},
        num_mc_passes=3,
        progress_callback=callback,
    )

    assert events == [
        ("preprocess", "start"),
        ("preprocess", "done"),
        ("mc_dropout", "start"),
        ("mc_dropout", "done"),
        ("ood", "start"),
        ("ood", "done"),
        ("gradcam", "start"),
        ("gradcam", "done"),
    ]
    assert tuple(s for s, _ in events[::2]) == inference.INFERENCE_STAGES


@requires_real_artifacts
def test_run_full_inference_identical_results_with_and_without_callback():
    """The progress callback must be purely observational -- reordering OOD
    before Grad-CAM (to match the design's step order) must not change any
    computed output, since the two steps are independent."""
    model = inference.load_classifier(CENTRALIZED_CHECKPOINT)
    synthetic_bgr = np.random.default_rng(3).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    kwargs = dict(
        deferral_threshold=0.5,
        ood_detectors={},
        ood_thresholds={},
        num_mc_passes=3,
    )
    result_no_callback = inference.run_full_inference(model, synthetic_bgr, **kwargs)
    result_with_callback = inference.run_full_inference(model, synthetic_bgr, progress_callback=lambda *_: None, **kwargs)

    assert result_no_callback.predicted_label == result_with_callback.predicted_label
    assert result_no_callback.prob_pneumonia == pytest.approx(result_with_callback.prob_pneumonia)
    assert result_no_callback.entropy == pytest.approx(result_with_callback.entropy)
    np.testing.assert_array_equal(result_no_callback.gradcam_heatmap, result_with_callback.gradcam_heatmap)


def test_check_is_xray_accepts_a_shared_frozen_model():
    """PneumoScan redesign Phase 1A: the new `frozen_model` param must be
    usable to skip rebuilding a fresh DenseNet121Head on every call, and
    produce the same result as the default (no param) path."""
    if not inference.XRAY_GATE_WEIGHTS_PATH.exists():
        pytest.skip("requires committed xray_gate_weights.json")
    from src.models.densenet_head import DenseNet121Head

    rng = np.random.default_rng(123)
    noise_bgr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    gate = inference.load_xray_gate()

    shared_model = DenseNet121Head()
    result_shared = inference.check_is_xray(noise_bgr, gate, frozen_model=shared_model)
    result_default = inference.check_is_xray(noise_bgr, gate)

    assert result_shared.is_xray == result_default.is_xray
    assert result_shared.p_xray == pytest.approx(result_default.p_xray)
