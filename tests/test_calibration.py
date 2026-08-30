"""OPT-1 — real tests for src/evaluation/calibration.py, matching this project's
convention of testing calibration behavior against constructed known-answer cases
rather than only shape/smoke checks."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
    risk_coverage_curve,
)


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    # Confidence 0.9 examples: exactly 90% correct. Confidence 0.6 examples: exactly
    # 60% correct. Both bins' accuracy matches their confidence exactly.
    confidence = np.array([0.9] * 10 + [0.6] * 10)
    correct = np.array([1] * 9 + [0] * 1 + [1] * 6 + [0] * 4)
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_is_positive_for_an_overconfident_model():
    # Model claims 0.99 confidence but is only right half the time — badly overconfident.
    confidence = np.full(20, 0.99)
    correct = np.array([1] * 10 + [0] * 10)
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    assert ece == pytest.approx(0.49, abs=0.02)


def test_ece_matches_hand_computed_two_bin_example():
    # n_bins=5 (width 0.2): 0.9,0.9 -> bin [0.8,1.0), conf mean 0.9, correct=[1,0]
    # -> acc 0.5, |0.5-0.9|=0.4, weight 2/4. 0.3,0.3 -> bin [0.2,0.4), conf mean 0.3,
    # correct=[1,1] -> acc 1.0, |1.0-0.3|=0.7, weight 2/4.
    # ECE = 0.5*0.4 + 0.5*0.7 = 0.55
    confidence = np.array([0.9, 0.9, 0.3, 0.3])
    correct = np.array([1, 0, 1, 1])
    ece = expected_calibration_error(confidence, correct, n_bins=5)
    assert ece == pytest.approx(0.55, abs=1e-9)


def test_reliability_diagram_empty_bins_are_nan_not_zero():
    confidence = np.array([0.95, 0.96])
    correct = np.array([1, 1])
    data = reliability_diagram_data(confidence, correct, n_bins=10)
    assert data.bin_count[0] == 0
    assert np.isnan(data.bin_confidence[0])
    assert np.isnan(data.bin_accuracy[0])
    assert data.bin_count[9] == 2
    assert data.bin_accuracy[9] == pytest.approx(1.0)


def test_reliability_diagram_shape_mismatch_raises():
    with pytest.raises(ValueError):
        reliability_diagram_data(np.array([0.5, 0.6]), np.array([1]))


def test_brier_score_zero_for_perfect_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_one_for_maximally_wrong_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_prob) == pytest.approx(1.0)


def test_brier_score_uninformative_uniform_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    assert brier_score(y_true, y_prob) == pytest.approx(0.25)


def test_risk_coverage_curve_full_coverage_matches_overall_error_rate():
    entropy = np.array([0.1, 0.5, 0.9, 1.2, 0.3])
    correct = np.array([1, 1, 0, 0, 1])  # 3/5 correct -> error rate 0.4
    curve = risk_coverage_curve(entropy, correct)
    assert curve.coverage[-1] == pytest.approx(1.0)
    assert curve.risk[-1] == pytest.approx(0.4)


def test_risk_coverage_curve_low_coverage_isolates_low_entropy_correct_cases():
    # Two lowest-entropy examples are both correct -> risk at coverage=2/5 should be 0.
    entropy = np.array([0.1, 0.2, 0.9, 1.2, 1.5])
    correct = np.array([1, 1, 0, 0, 1])
    curve = risk_coverage_curve(entropy, correct)
    assert curve.coverage[1] == pytest.approx(2 / 5)
    assert curve.risk[1] == pytest.approx(0.0)


def test_risk_coverage_curve_shape_mismatch_raises():
    with pytest.raises(ValueError):
        risk_coverage_curve(np.array([0.1, 0.2]), np.array([1]))
