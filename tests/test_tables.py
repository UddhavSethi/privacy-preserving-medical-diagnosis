"""Stage 21 — ablation table generation. The baseline-row loaders are
deterministic (real Stage 11/12 saved JSON, no MLflow dependency) and tested
here directly; the MLflow-query functions were validated against real
campaign data as it ran (docs/SESSION_STATE.md's Stage 21 note has the
real numbers) rather than re-tested against a mocked MLflow backend, which
would test the mock's behavior, not the actual SQL filter-string dialect
MLflow's sqlite backend accepts (a real constraint discovered live: it
rejects both parenthesized boolean grouping and `LIKE ... ESCAPE`).
"""
from pathlib import Path

import pytest

from src.evaluation.tables import load_centralized_baseline_row, load_local_baseline_row

RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "results"

pytestmark = pytest.mark.skipif(
    not (RESULTS_DIR / "local_baseline.json").exists()
    or not (RESULTS_DIR / "centralized_baseline.json").exists(),
    reason="requires Stage 11/12's saved baseline results",
)


def test_load_local_baseline_row_averages_across_hospitals():
    row = load_local_baseline_row("natural")
    assert row["n_seeds"] == 3
    assert 0.0 < row["mean_auroc"] <= 1.0
    assert set(row["per_hospital"]) == {"A", "B", "C"}
    # Averaging the three hospitals' means must land between the min and max.
    per_hospital_means = list(row["per_hospital"].values())
    assert min(per_hospital_means) <= row["mean_auroc"] <= max(per_hospital_means)


def test_load_centralized_baseline_row_uses_pooled_test_set():
    row = load_centralized_baseline_row("natural")
    assert row["n_seeds"] == 3
    assert 0.0 < row["mean_auroc"] <= 1.0


def test_centralized_meets_or_exceeds_local_per_stage_12s_own_sanity_check():
    """Stage 12's own established finding (not re-derived here, just
    guarded against silent regression): the centralized model should not
    score dramatically below the local baselines' average."""
    local = load_local_baseline_row("natural")
    centralized = load_centralized_baseline_row("natural")
    assert centralized["mean_auroc"] > local["mean_auroc"] - 0.1
