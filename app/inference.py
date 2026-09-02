"""OPT-6 — inference wrapper for the Streamlit demo (`app/streamlit_app.py`).

Deliberately contains NO Streamlit import and NO new inference logic — every
prediction, uncertainty, explanation, and anomaly-detection step is a direct call
into the same modules the rest of this project's tests and scripts already use
(`src/models`, `src/uncertainty`, `src/explain`, `src/data`). This module exists
only to (a) decode an uploaded file into the BGR array those modules expect and
(b) bundle their outputs into one result object for the UI layer to render.

Kept import-clean of Streamlit specifically so it stays testable and reusable
without a running Streamlit process (`tests/test_app_inference.py`) — "keep the
frontend separated from the research/training code" cuts both ways: the research
code must not import the frontend, and the frontend's logic layer should not
require the frontend framework to be tested.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

from src.data.preprocessing import ClaheParams, preprocess_to_rgb, window_pixel_array_to_uint8
from src.data.transforms import build_eval_transform
from src.explain.gradcam import NORMAL_CLASS_INDEX, PNEUMONIA_CLASS_INDEX, generate_overlay
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_hospital_features, load_pooled_features
from src.uncertainty.deferral import compute_deferral
from src.uncertainty.mc_dropout import MCDropoutResult, compute_mc_dropout_uncertainty
from src.uncertainty.ood_detector import build_and_calibrate, compute_anomaly_scores, flag_ood
from src.uncertainty.probability_calibration import apply_temperature
from src.uncertainty.xray_gate import XrayGateResult, load_gate_weights, predict_is_xray

XRAY_GATE_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "src" / "uncertainty" / "xray_gate_weights.json"

CLASS_NAMES = {NORMAL_CLASS_INDEX: "Normal", PNEUMONIA_CLASS_INDEX: "Pneumonia"}
UNCERTAIN_LABEL = "Uncertain"
SUPPORTED_DICOM_EXTENSIONS = (".dcm", ".dicom")
DEFAULT_DECISION_THRESHOLD = 0.5  # unchanged default for every checkpoint that
                                   # doesn't specify its own (conf/app.yaml)


def decode_uploaded_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Decodes an uploaded file into a BGR uint8 array — the same format
    `src/data/preprocessing.py`'s functions expect (`load_jpeg_bgr` /
    `load_dicom_as_uint8_bgr`, generalized here to work from in-memory bytes
    rather than a filesystem path, since a Streamlit upload never touches disk).
    """
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_DICOM_EXTENSIONS:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        uint8_img = window_pixel_array_to_uint8(ds.pixel_array)
        return cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)

    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {filename}")
    return bgr


def load_classifier(checkpoint_path: Path, fine_tune_last_block: bool = False) -> DenseNet121Head:
    """Loads a trained classifier head. Two checkpoint formats exist in this
    project (see `DenseNet121Head.trainable_state_dict`/`load_trainable_state_dict`,
    src/models/densenet_head.py): the default frozen-backbone path stores a bare
    classifier `state_dict` (unprefixed keys, `evaluate_classifier` and every
    OPT-1-5 script's format), while an ADR-1 GroupNorm-fallback checkpoint
    (`fine_tune_last_block=True`, docs/adr1_groupnorm_fallback.md) stores the
    prefixed classifier+denseblock4+norm5 subset instead — loading it with
    `model.classifier.load_state_dict()` would raise a key-mismatch error, not
    silently produce a wrong result, so this has to branch on the checkpoint's
    own architecture rather than assume the default."""
    model = DenseNet121Head(fine_tune_last_block=fine_tune_last_block)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if fine_tune_last_block:
        model.load_trainable_state_dict(state_dict)
    else:
        model.classifier.load_state_dict(state_dict)
    return model


def load_xray_gate():
    """Loads the chest-X-ray input gate (src/uncertainty/xray_gate.py) from its
    committed weights file. No checkpoint/training data needed at load time."""
    return load_gate_weights(XRAY_GATE_WEIGHTS_PATH)


def check_is_xray(bgr_image: np.ndarray, gate, image_size: int = 224) -> XrayGateResult:
    """Runs the gate against a FRESH, always-frozen backbone — deliberately
    NOT whichever checkpoint the user has selected for diagnosis. The gate was
    trained on Stage 9's cached pooled features, which are a frozen-backbone
    artifact (see docs/adr1_groupnorm_fallback.md's own repeated note on this);
    round 9's checkpoint partially unfreezes the backbone, so using ITS
    pooled_features here would feed the gate a different feature distribution
    than it was trained on — the exact mismatch already found and disabled for
    deferral/OOD. A plain, un-checkpointed DenseNet121Head's backbone is always
    the same frozen ImageNet weights regardless of which classifier head is
    loaded, so no checkpoint is needed here at all."""
    frozen_model = DenseNet121Head()
    _, tensor = preprocess_image(bgr_image, image_size=image_size)
    with torch.no_grad():
        pooled_features = frozen_model.pooled_features(tensor).numpy()[0]
    return predict_is_xray(gate, pooled_features)


@dataclass
class InferenceResult:
    rgb_image: np.ndarray  # CLAHE'd RGB, original resolution — what the model actually saw
    predicted_label: str  # "Normal" / "Pneumonia" / "Uncertain" (see `abstained`)
    predicted_class: int | None  # None when abstained — no class is asserted
    confidence: float  # calibrated max(mean_probs); see `abstained` for what this means when True
    prob_pneumonia: float  # calibrated P(pneumonia)
    entropy: float
    deferred: bool
    deferral_threshold: float
    abstained: bool  # True when prob_pneumonia fell inside the abstention band around the
                      # decision threshold — predicted_label is "Uncertain", not forced
    decision_threshold: float
    gradcam_overlay_rgb: np.ndarray
    ood_flags: dict[str, bool]  # per hospital: True = flagged out-of-distribution
    ood_scores: dict[str, float]


def preprocess_image(bgr_image: np.ndarray, image_size: int = 224) -> tuple[np.ndarray, torch.Tensor]:
    """Full pipeline used everywhere else in this project (ADR-6 CLAHE -> explicit
    RGB -> Stage 7's eval transform): returns the CLAHE'd RGB array (for display)
    and the normalized model-input tensor (for inference)."""
    rgb_image = preprocess_to_rgb(bgr_image, ClaheParams())
    tensor = build_eval_transform(image_size=image_size)(rgb_image).unsqueeze(0)
    return rgb_image, tensor


def run_full_inference(
    model: DenseNet121Head,
    bgr_image: np.ndarray,
    deferral_threshold: float,
    ood_detectors: dict[str, IsolationForest],
    ood_thresholds: dict[str, float],
    image_size: int = 224,
    num_mc_passes: int = 20,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    abstention_half_width: float = 0.0,
    temperature: float = 1.0,
) -> InferenceResult:
    """End-to-end single-image inference: preprocessing -> pooled features ->
    MC Dropout (prediction + confidence + uncertainty, Stage 19's own design,
    not a second point-estimate path) -> temperature calibration -> decision
    threshold + abstention band -> deferral -> Grad-CAM -> per-hospital OOD
    check. Every step calls directly into the existing, already-tested modules —
    this function only sequences them for one new image.

    `decision_threshold`/`abstention_half_width`/`temperature` default to 0.5 /
    0.0 / 1.0 (the prior behavior — argmax at 0.5, no abstention, no
    calibration) for every checkpoint that doesn't specify its own values in
    conf/app.yaml; see docs/adr1_groupnorm_fallback.md sec. 10 for how round
    9's own values were derived, entirely from its validation set."""
    rgb_image, tensor = preprocess_image(bgr_image, image_size=image_size)

    with torch.no_grad():
        pooled_features = model.pooled_features(tensor)  # (1, 1024)

    # MC Dropout's T stochastic passes are otherwise unseeded, so re-analyzing
    # the SAME uploaded image gave a different confidence (and occasionally a
    # different Uncertain/not verdict) every time — found live, 2026-09-02.
    # Seeding from the image's own bytes makes a given image's result exactly
    # reproducible on every re-upload, while still giving different images
    # different (not all-identical) dropout mask sequences.
    torch.manual_seed(int.from_bytes(hashlib.sha256(bgr_image.tobytes()).digest()[:8], "big") % (2**31))
    mc_result: MCDropoutResult = compute_mc_dropout_uncertainty(model, pooled_features, num_passes=num_mc_passes)
    calibrated_probs = apply_temperature(mc_result.mean_probs, temperature)
    prob_pneumonia = float(calibrated_probs[0, PNEUMONIA_CLASS_INDEX].item())
    entropy = float(mc_result.entropy.item())
    deferred = entropy >= deferral_threshold

    lo, hi = decision_threshold - abstention_half_width, decision_threshold + abstention_half_width
    abstained = lo <= prob_pneumonia <= hi
    if abstained:
        predicted_class = None
        predicted_label = UNCERTAIN_LABEL
        confidence = float(max(prob_pneumonia, 1.0 - prob_pneumonia))
        gradcam_target_class = PNEUMONIA_CLASS_INDEX if prob_pneumonia >= 0.5 else NORMAL_CLASS_INDEX
    else:
        predicted_class = PNEUMONIA_CLASS_INDEX if prob_pneumonia >= decision_threshold else NORMAL_CLASS_INDEX
        predicted_label = CLASS_NAMES[predicted_class]
        confidence = prob_pneumonia if predicted_class == PNEUMONIA_CLASS_INDEX else 1.0 - prob_pneumonia
        gradcam_target_class = predicted_class

    overlay = generate_overlay(model, rgb_image, target_class=gradcam_target_class, image_size=image_size)

    features_np = pooled_features.numpy()
    ood_flags, ood_scores = {}, {}
    for hospital, detector in ood_detectors.items():
        score = float(compute_anomaly_scores(detector, features_np)[0])
        ood_scores[hospital] = score
        ood_flags[hospital] = bool(flag_ood(np.array([score]), ood_thresholds[hospital])[0])

    return InferenceResult(
        rgb_image=rgb_image,
        predicted_label=predicted_label,
        predicted_class=predicted_class,
        confidence=confidence,
        prob_pneumonia=prob_pneumonia,
        entropy=entropy,
        deferred=deferred,
        deferral_threshold=deferral_threshold,
        abstained=abstained,
        decision_threshold=decision_threshold,
        gradcam_overlay_rgb=overlay.overlay_rgb,
        ood_flags=ood_flags,
        ood_scores=ood_scores,
    )


def calibrate_deferral_threshold(
    model: DenseNet121Head,
    partition_path: Path,
    feature_cache_dir: Path,
    target_defer_fraction: float,
    num_mc_passes: int = 20,
) -> float:
    """DG-10's deferral policy (`src.uncertainty.deferral.compute_deferral`) is
    defined relative to a BATCH's own uncertainty distribution ("defer the worst
    10% of this batch") — which has no meaning for a single newly-uploaded image
    in isolation. The standard way to deploy a batch-relative policy for
    real one-at-a-time inference is to calibrate it once against a reference
    population (here: the real pooled validation set, exactly as OPT-1/OPT-4
    already used it) and freeze the resulting entropy threshold as an absolute
    operating point. This calls `compute_deferral` directly on that reference
    batch — the exact same function DG-10 uses — rather than inventing a new
    single-example policy."""
    pooled = load_pooled_features(partition_path, ["A", "B", "C"], feature_cache_dir)
    mc_result = compute_mc_dropout_uncertainty(model, pooled.val_features, num_passes=num_mc_passes)
    return compute_deferral(mc_result.entropy, target_defer_fraction).threshold


def build_ood_detectors(
    partition_path: Path,
    feature_cache_dir: Path,
    hospitals: list[str],
    seed: int,
    target_flag_fraction: float,
) -> tuple[dict[str, IsolationForest], dict[str, float]]:
    """One Isolation Forest per hospital (`src.uncertainty.ood_detector`, OPT-5) —
    trained on that hospital's own real cached training features, calibrated on
    its own held-out val set. Returns (detectors, thresholds), both keyed by
    hospital, for `run_full_inference` to score a new image against all three."""
    detectors, thresholds = {}, {}
    for hospital in hospitals:
        features = load_hospital_features(partition_path, hospital, feature_cache_dir)
        train_features = features.train_features[:, -1, :].numpy()
        val_features = features.val_features.numpy()
        detector, calibration = build_and_calibrate(
            train_features, val_features, seed=seed, target_flag_fraction=target_flag_fraction
        )
        detectors[hospital] = detector
        thresholds[hospital] = calibration.threshold
    return detectors, thresholds


def uncertainty_label(entropy: float, deferral_threshold: float) -> str:
    """A coarse, human-readable band for the trust panel — NOT a new policy;
    "deferred" vs. "accepted" is still exactly DG-10's own binary decision
    (`deferred` field above). This just gives the "Low / Medium / High" framing
    the UI shows, relative to the same calibrated threshold."""
    if entropy < deferral_threshold * 0.5:
        return "Low"
    if entropy < deferral_threshold:
        return "Medium"
    return "High"
