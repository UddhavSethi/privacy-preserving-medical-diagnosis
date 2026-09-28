"""PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Phase 1A) —
translates `app/inference.py`'s real outputs into the exact JSON-safe props
contract the React component consumes, and is the one place that enforces
the redesign's "never fabricate data" rule: every function here returns
`None` (never a fake default like "Low" or "Typical") when the backend
genuinely didn't produce a value.

Deliberately contains NO Streamlit import, same reasoning as
`app/inference.py`'s own module docstring: this logic is real presentation
logic, not framework glue, and stays independently testable
(`tests/test_app_presentation.py`) without a running Streamlit process.

The `float("inf")` deferral-threshold trap (docs/pneumoscan_redesign_plan.md
Risk #6): the OLD app code passed `float("inf")` as a sentinel for "no
calibrated threshold available", which silently made
`inference.uncertainty_label` return "Low" uncertainty and `deferred`
return `False` — both of which read as real, confident answers rather than
"not available". Every function below checks availability explicitly
(`None` or `math.isfinite`) instead of trusting that sentinel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.inference import DecodedMeta, InferenceResult
from src.uncertainty.xray_gate import XrayGateResult

PROTOCOL_VERSION = 1

_MIN_RESOLUTION_PX = 256
_GRAYSCALE_THRESHOLD = 12.0  # matches the design prototype's own method (README Assets section)

# Peak -> lung-zone mapping constants, copied verbatim from the design
# README's own "Peak -> lung-zone mapping (image space, u = x/width, v =
# y/height)" section -- a presentation choice specified by the design, not a
# measured/derived value, so it belongs here rather than in inference.py.
_FOCUS_U_MIN, _FOCUS_U_MAX = 0.1, 0.9
_FOCUS_V_MIN, _FOCUS_V_MAX = 0.12, 0.85
_ZONE_V_OFFSET = 0.12
_ZONE_V_SCALE = 0.68
_ZONE_UPPER_MAX_T = 0.34
_ZONE_MIDDLE_MAX_T = 0.67


def certainty_label(entropy: float, deferral_threshold: float | None) -> str | None:
    """"How sure the AI is" -- High/Medium/Low, per the design's own formula
    (predictive entropy / deferral threshold: <0.5 High, <1 Medium, >=1 Low).
    Returns None (never a guessed band) when no real threshold exists."""
    if deferral_threshold is None or not math.isfinite(deferral_threshold) or deferral_threshold <= 0:
        return None
    ratio = entropy / deferral_threshold
    if ratio < 0.5:
        return "High"
    if ratio < 1.0:
        return "Medium"
    return "Low"


def image_check(ood_flags: dict[str, bool]) -> str | None:
    """"Image check" -- Typical/Unusual, matching the existing
    `all(ood_flags.values())` rule (app/streamlit_app.py's pre-redesign
    banner logic) exactly: Unusual only if every hospital's detector flags
    it. None (not "Typical") when there are no detectors at all."""
    if not ood_flags:
        return None
    return "Unusual" if all(ood_flags.values()) else "Typical"


def compute_focus(heatmap: np.ndarray, label: str) -> tuple[dict[str, Any] | None, bool]:
    """Returns `(focus, spread)`. `focus` is the Grad-CAM peak mapped to a
    lung side/zone (design README's own mapping, reproduced verbatim in the
    module-level constants above); `spread` is True whenever no single-region
    focus applies (non-Pneumonia label, or the peak falls outside the
    design's own bounding region) -- the UI then shows "The AI's focus is
    spread out rather than in one area" instead of a specific zone."""
    if label != "Pneumonia":
        return None, True

    h, w = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    y, x = divmod(flat_idx, w)
    u = (x + 0.5) / w
    v = (y + 0.5) / h

    if not (_FOCUS_U_MIN < u < _FOCUS_U_MAX and _FOCUS_V_MIN < v < _FOCUS_V_MAX):
        return None, True

    side = "right" if u < 0.5 else "left"  # radiographic convention: patient right on viewer left
    t = (v - _ZONE_V_OFFSET) / _ZONE_V_SCALE
    if t < _ZONE_UPPER_MAX_T:
        zone = "upper"
    elif t < _ZONE_MIDDLE_MAX_T:
        zone = "middle"
    else:
        zone = "lower"

    return {"side": side, "zone": zone, "u": u, "v": v}, False


def quality_checks(bgr: np.ndarray, meta: DecodedMeta) -> dict[str, Any]:
    """Real, server-side image-quality checks against the actual decoded
    pixels -- `resolution_ok` and `grayscale` are measured here, not asserted
    from the file format. `decoded`/`format_supported` are always True by
    the time this runs (decode already succeeded), included for the
    frontend's quality-list rendering, not because they could be False here."""
    min_side = min(meta.width, meta.height)
    resolution_ok = min_side >= _MIN_RESOLUTION_PX

    small = cv2.resize(bgr, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    grayscale_metric = float(np.mean(np.abs(b - g) + np.abs(g - r)))
    grayscale = grayscale_metric < _GRAYSCALE_THRESHOLD

    return {
        "decoded": True,
        "format_supported": True,
        "min_side_px": min_side,
        "resolution_ok": resolution_ok,
        "grayscale": grayscale,
    }


def xray_gate_props(gate_result: XrayGateResult | None) -> dict[str, Any]:
    """`quality.xray_gate` -- `status` is "unavailable" (not "passed") when
    the gate itself couldn't run (missing weights), never silently treated
    as a pass."""
    if gate_result is None:
        return {"status": "unavailable", "p_xray": None}
    return {
        "status": "passed" if gate_result.is_xray else "failed",
        "p_xray": float(gate_result.p_xray),
    }


def make_display_jpeg(bgr: np.ndarray, max_edge: int = 1600) -> bytes:
    """The image sent to the frontend for display -- downscaled so repeated
    polling reruns (Phase 1B) don't resend a full-resolution image every
    ~0.3s. Never upscales."""
    height, width = bgr.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode display JPEG")
    return buf.tobytes()


def heatmap_to_uint8_bytes(heatmap: np.ndarray) -> bytes:
    """Raw (224, 224) float heatmap -> row-major uint8 bytes (50,176 bytes
    at 224x224) the frontend colorizes client-side. Clips to [0,1] first --
    Grad-CAM's own output is already in that range, but this is the
    boundary to the untrusted-by-convention wire format, so it's asserted
    here rather than assumed."""
    clipped = np.clip(heatmap, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8).tobytes()


def study_props(
    meta: DecodedMeta, filename: str, upload_sha256_hex: str, uploaded_at_iso: str, is_sample: bool
) -> dict[str, Any]:
    """`study` -- `study_id` derived from the upload's own hash (not a
    counter or random id) so re-uploading the exact same file reproducibly
    gets the same id. `patient` is always "Not recorded": this app never
    collects patient identity, for any file."""
    return {
        "study_id": f"PS-{upload_sha256_hex[:6].upper()}",
        "filename": filename,
        "format": meta.format,
        "width": meta.width,
        "height": meta.height,
        "file_size_bytes": None,  # filled by the caller, which has the raw byte count
        "projection": meta.projection,
        "loaded_at": uploaded_at_iso,
        "patient": None,
        "is_sample": is_sample,
    }


@dataclass(frozen=True)
class CheckpointCalibration:
    """What `conf/app.yaml`'s active checkpoint entry actually specifies,
    resolved to explicit None where a key is absent -- never silently
    defaulted to a value that would misrepresent an uncalibrated checkpoint
    as calibrated. `source` records which case applied, for the frontend's
    own "Technical details" text."""

    decision_threshold: float
    temperature: float | None
    abstention_half_width: float | None
    source: str  # "validation" | "default"


def resolve_checkpoint_calibration(checkpoint_cfg: dict[str, Any]) -> CheckpointCalibration:
    """Reads calibration fields directly off the checkpoint's own
    `conf/app.yaml` entry -- NOT off `InferenceResult` (which always carries
    a value, defaulted or not, since `run_full_inference` needs a concrete
    number to run). Whether that number was actually derived from real
    validation-set calibration (D1) or is just the historical 0.5/1.0/0.0
    fallback is a fact about the config, only visible by checking which keys
    are actually present there."""
    has_calibration = "temperature" in checkpoint_cfg and "abstention_half_width" in checkpoint_cfg
    return CheckpointCalibration(
        decision_threshold=float(checkpoint_cfg.get("decision_threshold", 0.5)),
        temperature=float(checkpoint_cfg["temperature"]) if "temperature" in checkpoint_cfg else None,
        abstention_half_width=(
            float(checkpoint_cfg["abstention_half_width"]) if "abstention_half_width" in checkpoint_cfg else None
        ),
        source="validation" if has_calibration else "default",
    )


def quality_props(bgr: np.ndarray, meta: DecodedMeta, gate_result: XrayGateResult | None) -> dict[str, Any]:
    """`quality` -- merges the real pixel-level checks (`quality_checks`) with
    the X-ray gate's own verdict (`xray_gate_props`). Computed once, eagerly,
    right after upload/decode (before the user clicks "Analyze") so the
    Image Check card can show a real "Chest X-ray detected" answer during
    the `quality` phase, not just once analysis starts."""
    return {**quality_checks(bgr, meta), "xray_gate": xray_gate_props(gate_result)}


def build_props(
    *,
    phase: str,
    study: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """The single JSON-safe dict builder feeding the component (Phase 1C).
    Every field defaults to `None` ("Not available"/absent in the UI) rather
    than a fabricated placeholder -- callers pass only what the current
    `phase` actually has. Raw image/heatmap bytes are NOT included here (they
    go through the component call's own separate `display_image`/`heatmap`
    bytes kwargs, per the Streamlit 1.62 binary-arg behavior noted in the
    plan) -- this keeps `build_props`'s own output trivially JSON-serializable,
    checked directly by `test_app_presentation.py`."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "study": study,
        "quality": quality,
        "progress": progress,
        "result": result,
        "error": error,
    }


def result_props(result: InferenceResult, calibration: CheckpointCalibration) -> dict[str, Any]:
    """`result` -- the bulk of the props contract table
    (docs/pneumoscan_redesign_plan.md Phase 1C). Deferral/certainty/OOD
    fields are real whenever `result.deferral_threshold`/`result.ood_flags`
    are real (D2); calibration fields are real whenever `calibration.source
    == "validation"` (D1). Both degrade to `None` -> "Not available"
    independently, not as an all-or-nothing pair."""
    has_deferral = math.isfinite(result.deferral_threshold)
    focus, focus_spread = compute_focus(result.gradcam_heatmap, result.predicted_label)

    return {
        "label": result.predicted_label,
        "confidence": result.confidence,
        "prob_pneumonia": result.prob_pneumonia,
        "entropy": result.entropy,
        "abstained": result.abstained,
        "certainty": certainty_label(result.entropy, result.deferral_threshold if has_deferral else None),
        "deferred": result.deferred if has_deferral else None,
        "deferral_threshold": result.deferral_threshold if has_deferral else None,
        "image_check": image_check(result.ood_flags),
        "ood_sites": (
            {
                hospital: {"flagged": result.ood_flags[hospital], "score": result.ood_scores[hospital]}
                for hospital in result.ood_flags
            }
            if result.ood_flags
            else None
        ),
        "gradcam_target": "Pneumonia" if result.gradcam_target_class == 1 else "Normal",
        "focus": focus,
        "focus_spread": focus_spread,
        "decision_threshold": calibration.decision_threshold,
        "decision_threshold_source": calibration.source,
        "temperature": calibration.temperature,
        "abstention_half_width": calibration.abstention_half_width,
    }
