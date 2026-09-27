"""PneumoScan redesign (docs/pneumoscan_redesign_plan.md, Phase 1A) — tests
for `app/analysis_job.py` using a fake inference function, so these run fast
and without needing a real checkpoint/feature cache (the real end-to-end
path is covered separately by `tests/test_app_inference.py`'s
`requires_real_artifacts`-gated tests, run manually this session against a
real checkpoint/gate/image and confirmed working).
"""
from __future__ import annotations

import numpy as np
import pytest

from app import analysis_job
from app.inference import InferenceResult
from src.uncertainty.xray_gate import XrayGateResult


def _fake_result() -> InferenceResult:
    return InferenceResult(
        rgb_image=np.zeros((224, 224, 3), dtype=np.uint8),
        predicted_label="Normal",
        predicted_class=0,
        confidence=0.9,
        prob_pneumonia=0.1,
        entropy=0.1,
        deferred=False,
        deferral_threshold=0.5,
        abstained=False,
        decision_threshold=0.5,
        gradcam_overlay_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        gradcam_heatmap=np.zeros((224, 224), dtype=np.float32),
        gradcam_target_class=0,
        ood_flags={"A": False, "B": False, "C": False},
        ood_scores={"A": 0.0, "B": 0.0, "C": 0.0},
    )


def test_snapshot_initial_state():
    job = analysis_job.AnalysisJob()
    snap = job.snapshot()
    assert snap["queued"] is True
    assert snap["fraction"] == 0.0
    assert {s["id"] for s in snap["stages"]} == set(analysis_job.JOB_STAGES)
    assert all(s["status"] == "pending" for s in snap["stages"])


def test_on_stage_updates_snapshot():
    job = analysis_job.AnalysisJob()
    job.on_stage("gate", "start")
    snap = job.snapshot()
    gate_status = next(s["status"] for s in snap["stages"] if s["id"] == "gate")
    assert gate_status == "running"

    job.on_stage("gate", "done")
    snap = job.snapshot()
    gate_status = next(s["status"] for s in snap["stages"] if s["id"] == "gate")
    assert gate_status == "done"
    assert snap["fraction"] == pytest.approx(1 / len(analysis_job.JOB_STAGES))


def test_run_job_happy_path(monkeypatch):
    """A fake `run_full_inference` and a passing gate: the job should reach a
    real result with every stage marked done, queued flipped False, and no
    error/rejection."""
    fake = _fake_result()

    def fake_run_full_inference(model, bgr_image, deferral_threshold, ood_detectors, ood_thresholds, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            for stage in ("preprocess", "mc_dropout", "ood", "gradcam"):
                callback(stage, "start")
                callback(stage, "done")
        return fake

    def fake_check_is_xray(bgr_image, gate, image_size=224, frozen_model=None):
        return XrayGateResult(is_xray=True, p_xray=0.95)

    monkeypatch.setattr(analysis_job, "run_full_inference", fake_run_full_inference)
    monkeypatch.setattr(analysis_job, "check_is_xray", fake_check_is_xray)

    job = analysis_job.AnalysisJob()
    analysis_job.run_job(
        job,
        np.zeros((224, 224, 3), dtype=np.uint8),
        gate=object(),
        frozen_model=object(),
        model=object(),
        deferral_threshold=0.5,
        ood_detectors={},
        ood_thresholds={},
        decision_threshold=0.5,
        abstention_half_width=0.0,
        temperature=1.0,
    )

    assert job.queued is False
    assert job.is_finished() is True
    assert job.error is None
    assert job.rejected is False
    assert job.result is fake
    snap = job.snapshot()
    assert snap["fraction"] == 1.0
    assert all(s["status"] == "done" for s in snap["stages"])


def test_run_job_rejects_non_xray(monkeypatch):
    def fake_check_is_xray(bgr_image, gate, image_size=224, frozen_model=None):
        return XrayGateResult(is_xray=False, p_xray=0.02)

    def fake_run_full_inference(*args, **kwargs):
        raise AssertionError("run_full_inference must not be called when the gate rejects the image")

    monkeypatch.setattr(analysis_job, "check_is_xray", fake_check_is_xray)
    monkeypatch.setattr(analysis_job, "run_full_inference", fake_run_full_inference)

    job = analysis_job.AnalysisJob()
    analysis_job.run_job(
        job,
        np.zeros((224, 224, 3), dtype=np.uint8),
        gate=object(),
        frozen_model=object(),
        model=object(),
        deferral_threshold=0.5,
        ood_detectors={},
        ood_thresholds={},
        decision_threshold=0.5,
        abstention_half_width=0.0,
        temperature=1.0,
    )

    assert job.rejected is True
    assert job.result is None
    assert job.error is None
    assert job.is_finished() is True


def test_run_job_gate_unavailable_when_no_gate(monkeypatch):
    """`gate=None` (missing weights) must proceed to inference, not reject —
    matches the pre-redesign app's own fallback behavior, just made explicit."""
    fake = _fake_result()

    def fake_run_full_inference(model, bgr_image, deferral_threshold, ood_detectors, ood_thresholds, **kwargs):
        return fake

    monkeypatch.setattr(analysis_job, "run_full_inference", fake_run_full_inference)

    job = analysis_job.AnalysisJob()
    analysis_job.run_job(
        job,
        np.zeros((224, 224, 3), dtype=np.uint8),
        gate=None,
        frozen_model=object(),
        model=object(),
        deferral_threshold=0.5,
        ood_detectors={},
        ood_thresholds={},
        decision_threshold=0.5,
        abstention_half_width=0.0,
        temperature=1.0,
    )

    assert job.gate_unavailable is True
    assert job.rejected is False
    assert job.result is fake


def test_run_job_captures_exception_into_error(monkeypatch):
    """A background-thread exception must reach `job.error`, never vanish
    silently (this is the job's only error boundary, per the module
    docstring's own stated reasoning)."""

    def fake_check_is_xray(bgr_image, gate, image_size=224, frozen_model=None):
        raise RuntimeError("simulated decode failure")

    monkeypatch.setattr(analysis_job, "check_is_xray", fake_check_is_xray)

    job = analysis_job.AnalysisJob()
    analysis_job.run_job(
        job,
        np.zeros((224, 224, 3), dtype=np.uint8),
        gate=object(),
        frozen_model=object(),
        model=object(),
        deferral_threshold=0.5,
        ood_detectors={},
        ood_thresholds={},
        decision_threshold=0.5,
        abstention_half_width=0.0,
        temperature=1.0,
    )

    assert job.error is not None
    assert "simulated decode failure" in job.error
    assert job.result is None
    assert job.is_finished() is True
