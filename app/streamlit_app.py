"""OPT-6 — Streamlit demo interface (CLAUDE.md section 16.1a; approved 2026-08-30).

PneumoScan redesign (docs/pneumoscan_redesign_plan.md), Phase 1B/1C/1D: this
file is now a thin state machine + event dispatcher around the real backend
(`app/inference.py`, `app/presentation.py`, `app/analysis_job.py`). It owns
NO presentation logic of its own -- every value shown in the UI traces to a
`presentation.py` function, never a value invented here -- and NO training,
evaluation, privacy, or federated-pipeline logic. The actual screen (New
Screening) is rendered entirely by the `pneumoscan` React component
(`app/pneumoscan_component/`); this file's only jobs are: resolve the one
fixed model configuration, run a background analysis job on a single-worker
thread pool (so Cloud's shared CPU never runs two inferences at once), and
translate JS events into session-state transitions.

Single fixed configuration, read from `conf/app.yaml`'s `active_configuration`
(owner-approved decision 3: no model selector anywhere in this UI). Switched
from `fedavg_secagg` to `fedavg_dp_eps4` on 2026-09-28 (owner-directed) so
the live public demo actually shows the project's Differential Privacy
protection, not just Secure Aggregation -- see conf/app.yaml's own comment
on that switch for the real accuracy/calibration cost.

Run from the repository root:

    uv run streamlit run app/streamlit_app.py

See docs/pneumoscan_redesign_plan.md for the full phase-by-phase design.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import streamlit as st
from omegaconf import OmegaConf

from app import inference, presentation
from app.analysis_job import AnalysisJob, run_job
from app.pneumoscan_component import pneumoscan

st.set_page_config(
    page_title="PneumoScan — AI-Assisted Chest X-ray Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# The component owns the entire viewport (sticky header, internal scroll
# regions, full-screen viewer) -- Streamlit's own chrome/padding/scrolling
# must get out of the way entirely, or the component's 100dvh layout breaks.
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stAppViewContainer"] { padding: 0 !important; }
    div[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
    div[data-testid="stMain"] { overflow: hidden !important; }
    html, body { overflow: hidden !important; }
    iframe { display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)

CFG = OmegaConf.load(REPO_ROOT / "conf" / "app.yaml")
HOSPITALS = ["A", "B", "C"]
CONFIG_BY_KEY = {cfg["key"]: cfg for cfg in CFG.configurations}
ACTIVE_CONFIG = CONFIG_BY_KEY[CFG.active_configuration]
CALIBRATION = presentation.resolve_checkpoint_calibration(ACTIVE_CONFIG)
SAMPLE_XRAY_PATH = REPO_ROOT / "app" / "assets" / "sample_xray.jpg"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
POLL_INTERVAL_SECONDS = 0.3


# --------------------------------------------------------------------------
# Cached resource loaders — each built at most once per server process.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    path = REPO_ROOT / ACTIVE_CONFIG["checkpoint"]
    if not path.exists():
        return None
    return inference.load_classifier(path, fine_tune_last_block=bool(ACTIVE_CONFIG.get("fine_tune_last_block", False)))


@st.cache_resource(show_spinner=False)
def get_frozen_model():
    # A plain, un-checkpointed backbone for the X-ray gate — see
    # `inference.check_is_xray`'s own docstring for why this must never be
    # the diagnostic checkpoint. Cached so it's built once, not per upload.
    from src.models.densenet_head import DenseNet121Head

    return DenseNet121Head()


@st.cache_resource(show_spinner=False)
def get_xray_gate():
    if not inference.XRAY_GATE_WEIGHTS_PATH.exists():
        return None
    return inference.load_xray_gate()


@st.cache_resource(show_spinner=False)
def get_ood_artifacts():
    """Loads D2's precomputed per-hospital OOD detectors (Phase 1A) — never
    the live 576MB feature cache, which Streamlit Cloud never has. Returns
    `({}, {})`, degrading every dependent prop to `None`, if the artifacts
    aren't present in this environment."""
    artifacts_dir_rel = ACTIVE_CONFIG.get("ood_artifacts_dir")
    if not artifacts_dir_rel:
        return {}, {}
    artifacts_dir = REPO_ROOT / artifacts_dir_rel
    metadata_path = artifacts_dir / "metadata.json"
    if not metadata_path.exists():
        return {}, {}
    metadata = json.loads(metadata_path.read_text())
    detectors = {}
    for hospital in HOSPITALS:
        detector_path = artifacts_dir / f"ood_detector_{hospital}.joblib"
        if detector_path.exists():
            detectors[hospital] = joblib.load(detector_path)
    thresholds = {h: float(v) for h, v in metadata.get("ood_thresholds", {}).items() if h in detectors}
    return detectors, thresholds


def get_deferral_threshold() -> float:
    value = ACTIVE_CONFIG.get("deferral_threshold")
    return float(value) if value is not None else float("inf")


@st.cache_resource(show_spinner=False)
def get_executor() -> ThreadPoolExecutor:
    # A single global worker: also serializes inference across concurrent
    # visitors, protecting Cloud's shared/limited RAM (Risk #4). A second
    # visitor's job just sits `queued: true` until this one is free.
    return ThreadPoolExecutor(max_workers=1)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def _reset_session() -> None:
    st.session_state["phase"] = "upload"
    st.session_state["study"] = None
    st.session_state["quality"] = None
    st.session_state["bgr_image"] = None
    st.session_state["display_image_bytes"] = None
    st.session_state["heatmap_bytes"] = None
    st.session_state["job"] = None
    st.session_state["result_props"] = None
    st.session_state["error"] = None


if "phase" not in st.session_state:
    _reset_session()
if "last_event_id" not in st.session_state:
    st.session_state["last_event_id"] = None


def _handle_upload(raw_bytes: bytes, filename: str, is_sample: bool) -> None:
    try:
        bgr, meta = inference.decode_uploaded_image_with_meta(raw_bytes, filename)
    except Exception as exc:  # noqa: BLE001 -- any decode failure must reach the
        # "error" phase, never crash the script (a corrupt/unsupported upload
        # is a normal, expected user input, not a bug).
        st.session_state["phase"] = "error"
        st.session_state["error"] = f"Could not read this file: {exc}"
        return

    upload_hash = hashlib.sha256(raw_bytes).hexdigest()
    study = presentation.study_props(
        meta, filename, upload_hash, datetime.now(timezone.utc).isoformat(timespec="seconds"), is_sample
    )
    study["file_size_bytes"] = len(raw_bytes)

    gate = get_xray_gate()
    gate_result = inference.check_is_xray(bgr, gate, frozen_model=get_frozen_model()) if gate is not None else None

    st.session_state["bgr_image"] = bgr
    st.session_state["display_image_bytes"] = presentation.make_display_jpeg(bgr)
    st.session_state["study"] = study
    st.session_state["quality"] = presentation.quality_props(bgr, meta, gate_result)
    st.session_state["heatmap_bytes"] = None
    st.session_state["job"] = None
    st.session_state["result_props"] = None
    st.session_state["error"] = None
    st.session_state["phase"] = "rejected" if (gate_result is not None and not gate_result.is_xray) else "quality"


def _handle_analyze() -> None:
    if st.session_state.get("bgr_image") is None:
        return
    model = get_model()
    if model is None:
        st.session_state["phase"] = "error"
        st.session_state["error"] = "The analysis model isn't available in this environment. Please try again later."
        return

    ood_detectors, ood_thresholds = get_ood_artifacts()
    job = AnalysisJob()
    get_executor().submit(
        run_job,
        job,
        st.session_state["bgr_image"],
        gate=get_xray_gate(),
        frozen_model=get_frozen_model(),
        model=model,
        deferral_threshold=get_deferral_threshold(),
        ood_detectors=ood_detectors,
        ood_thresholds=ood_thresholds,
        decision_threshold=CALIBRATION.decision_threshold,
        abstention_half_width=CALIBRATION.abstention_half_width or 0.0,
        temperature=CALIBRATION.temperature or 1.0,
    )
    st.session_state["job"] = job
    st.session_state["phase"] = "analyzing"


def _finalize_job() -> None:
    job: AnalysisJob = st.session_state["job"]
    if job.error is not None:
        st.session_state["phase"] = "error"
        st.session_state["error"] = f"Analysis failed: {job.error}"
    elif job.rejected:
        st.session_state["phase"] = "rejected"
        if job.gate_result is not None and st.session_state.get("quality") is not None:
            st.session_state["quality"]["xray_gate"] = presentation.xray_gate_props(job.gate_result)
    else:
        result = job.result
        st.session_state["heatmap_bytes"] = presentation.heatmap_to_uint8_bytes(result.gradcam_heatmap)
        st.session_state["result_props"] = presentation.result_props(result, CALIBRATION)
        st.session_state["phase"] = "review"
    st.session_state["job"] = None


def _dispatch_event(event: dict) -> None:
    event_type = event.get("type")
    payload = event.get("payload") or {}

    if event_type == "upload":
        try:
            raw = base64.b64decode(payload.get("data_b64", ""))
        except Exception:  # noqa: BLE001 -- malformed payload is a client bug, not ours to crash on
            st.session_state["phase"] = "error"
            st.session_state["error"] = "Could not read the uploaded file."
            return
        if len(raw) > MAX_UPLOAD_BYTES:
            st.session_state["phase"] = "error"
            st.session_state["error"] = "This file is larger than the 25MB upload limit."
            return
        _handle_upload(raw, payload.get("name", "upload"), is_sample=False)
    elif event_type == "use_sample":
        _handle_upload(SAMPLE_XRAY_PATH.read_bytes(), SAMPLE_XRAY_PATH.name, is_sample=True)
    elif event_type == "analyze":
        _handle_analyze()
    elif event_type in ("new_screening", "dismiss_error"):
        _reset_session()


def _current_props() -> dict:
    job = st.session_state.get("job")
    return presentation.build_props(
        phase=st.session_state["phase"],
        study=st.session_state.get("study"),
        quality=st.session_state.get("quality"),
        progress=job.snapshot() if job is not None else None,
        result=st.session_state.get("result_props"),
        error=st.session_state.get("error"),
    )


# --------------------------------------------------------------------------
# Render + dispatch — one pass per script run.
# --------------------------------------------------------------------------
event = pneumoscan(
    {
        "data": _current_props(),
        "display_image": st.session_state.get("display_image_bytes"),
        "heatmap": st.session_state.get("heatmap_bytes"),
    },
    key="pneumoscan",
    default=None,
)

if event and event.get("event_id") != st.session_state.get("last_event_id"):
    st.session_state["last_event_id"] = event["event_id"]
    _dispatch_event(event)
    st.rerun()

if st.session_state["phase"] == "analyzing":
    job = st.session_state["job"]
    if job is not None and job.is_finished():
        _finalize_job()
        st.rerun()
    else:
        time.sleep(POLL_INTERVAL_SECONDS)
        st.rerun()
