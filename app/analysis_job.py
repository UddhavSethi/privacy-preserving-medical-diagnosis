"""PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Phase 1A/1B) —
runs one image's full analysis (gate check, then `run_full_inference`) on a
background thread, with lock-protected stage-progress state a Streamlit
script can poll across reruns (`app/streamlit_app.py`'s own rerun loop, not
this module — this module is pure threading/orchestration, no Streamlit
import, same reasoning as `app/inference.py` and `app/presentation.py`).

Why a background thread + polling, not Streamlit's own blocking call: a
single `st.spinner()` around one blocking `run_full_inference()` call (the
old behavior) gives no way to show real intermediate progress, since
Streamlit's script-rerun model can't observe progress *during* one Python
call without yielding control back to it. Running the job on a thread and
polling `AnalysisJob.snapshot()` across reruns (`time.sleep` + `st.rerun()`)
lets the script report the real 5-stage progress
`app/inference.py::run_full_inference`'s `progress_callback` now provides.
"""
from __future__ import annotations

import threading

from sklearn.ensemble import IsolationForest

from app.inference import InferenceResult, check_is_xray, run_full_inference
from src.models.densenet_head import DenseNet121Head
from src.uncertainty.xray_gate import XrayGateResult

# "gate" first (runs outside run_full_inference, in check_is_xray), then the
# 4 stages run_full_inference's own progress_callback reports.
JOB_STAGES = ("gate", "preprocess", "mc_dropout", "ood", "gradcam")


class AnalysisJob:
    """One upload's analysis state. Created once per "analyze" event, never
    reused across uploads. All mutation goes through the lock so
    `snapshot()` (called from the Streamlit script's polling loop, a
    different thread than the one running `run_job`) never observes a
    torn/partial update."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queued: bool = True
        self._stage_status: dict[str, str] = {name: "pending" for name in JOB_STAGES}
        self.gate_result: XrayGateResult | None = None
        self.gate_unavailable: bool = False
        self.rejected: bool = False
        self.result: InferenceResult | None = None
        self.error: str | None = None

    def on_stage(self, stage: str, event: str) -> None:
        """Matches `app.inference.ProgressCallback`'s signature — passed
        directly as `run_full_inference(progress_callback=job.on_stage)`."""
        with self._lock:
            self._stage_status[stage] = "done" if event == "done" else "running"

    def snapshot(self) -> dict:
        """JSON-safe progress snapshot for the component's `progress` prop
        (docs/pneumoscan_redesign_plan.md Phase 1C)."""
        with self._lock:
            stages = [{"id": name, "status": self._stage_status[name]} for name in JOB_STAGES]
            done_count = sum(1 for s in stages if s["status"] == "done")
        return {
            "job_id": id(self),
            "queued": self.queued,
            "stages": stages,
            "fraction": done_count / len(JOB_STAGES),
        }

    def is_finished(self) -> bool:
        return self.result is not None or self.rejected or self.error is not None


def run_job(
    job: AnalysisJob,
    bgr_image,
    *,
    gate,
    frozen_model: DenseNet121Head,
    model: DenseNet121Head,
    deferral_threshold: float,
    ood_detectors: dict[str, IsolationForest],
    ood_thresholds: dict[str, float],
    decision_threshold: float,
    abstention_half_width: float,
    temperature: float,
) -> None:
    """Runs on a background thread (submitted by the Streamlit script to a
    single-worker `ThreadPoolExecutor` — see docs/pneumoscan_redesign_plan.md
    Phase 1B). Never raises: every failure mode is captured onto `job` for
    the polling script to render, since an uncaught exception on a
    background thread would otherwise vanish silently instead of reaching
    the user as an "error" phase."""
    job.queued = False
    try:
        job.on_stage("gate", "start")
        if gate is None:
            job.gate_unavailable = True
        else:
            job.gate_result = check_is_xray(bgr_image, gate, frozen_model=frozen_model)
            if not job.gate_result.is_xray:
                job.rejected = True
                job.on_stage("gate", "done")
                return
        job.on_stage("gate", "done")

        job.result = run_full_inference(
            model,
            bgr_image,
            deferral_threshold,
            ood_detectors,
            ood_thresholds,
            decision_threshold=decision_threshold,
            abstention_half_width=abstention_half_width,
            temperature=temperature,
            progress_callback=job.on_stage,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is the job's
        # only error boundary: any failure (decode edge case, a model quirk on
        # an unusual real upload) must reach the UI as "error", not vanish.
        job.error = f"{type(exc).__name__}: {exc}"
