"""OPT-4 — real tests for src/uncertainty/conformal.py, checking the coverage
guarantee actually holds on constructed data (the property that matters), not
just shape/smoke checks."""
from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.conformal import (
    calibrate_conformal_threshold,
    empirical_coverage,
    mean_set_size,
    predict_conformal_sets,
    run_conformal_analysis,
    set_size_distribution,
)


def test_calibrate_threshold_matches_hand_computed_quantile():
    # 4 calibration examples, true-class probs = [0.9, 0.8, 0.7, 0.6] ->
    # scores = 1-p = [0.1, 0.2, 0.3, 0.4]. n=4, alpha=0.5 -> q_level =
    # ceil(5*0.5)/4 = ceil(2.5)/4 = 3/4 = 0.75 -> the 0.75 quantile (method="higher")
    # of [0.1,0.2,0.3,0.4] is 0.4.
    probs_cal = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    labels_cal = np.array([1, 1, 1, 1])
    threshold = calibrate_conformal_threshold(probs_cal, labels_cal, alpha=0.5)
    assert threshold == pytest.approx(0.4)


def test_calibrate_threshold_invalid_alpha_raises():
    with pytest.raises(ValueError):
        calibrate_conformal_threshold(np.array([[0.5, 0.5]]), np.array([0]), alpha=0.0)
    with pytest.raises(ValueError):
        calibrate_conformal_threshold(np.array([[0.5, 0.5]]), np.array([0]), alpha=1.0)


def test_calibrate_threshold_empty_raises():
    with pytest.raises(ValueError):
        calibrate_conformal_threshold(np.empty((0, 2)), np.empty(0, dtype=int))


def test_predict_conformal_sets_membership_matches_hand_computation():
    probs = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]])
    threshold = 0.3  # class in set iff prob >= 1 - 0.3 = 0.7
    membership = predict_conformal_sets(probs, threshold)
    # row0: probs=[0.2,0.8] -> only class1 (0.8>=0.7) in set -> [False, True]
    # row1: probs=[0.5,0.5] -> neither >= 0.7 -> [False, False] (empty set)
    # row2: probs=[0.9,0.1] -> only class0 (0.9>=0.7) in set -> [True, False]
    np.testing.assert_array_equal(membership, [[False, True], [False, False], [True, False]])


def test_empirical_coverage_and_mean_set_size():
    membership = np.array([[True, True], [True, False], [False, True], [False, False]])
    labels = np.array([1, 0, 1, 1])  # covered: row0(both) yes, row1(class0) yes,
    # row2(class1) yes, row3(class1, but empty set) no -> 3/4
    assert empirical_coverage(membership, labels) == pytest.approx(0.75)
    assert mean_set_size(membership) == pytest.approx((2 + 1 + 1 + 0) / 4)


def test_set_size_distribution_two_class():
    membership = np.array([[True, True], [True, False], [False, False], [True, False]])
    dist = set_size_distribution(membership)
    assert dist["empty"] == pytest.approx(0.25)
    assert dist["singleton"] == pytest.approx(0.5)
    assert dist["full"] == pytest.approx(0.25)


def test_conformal_coverage_guarantee_holds_on_large_synthetic_data():
    # A model whose predicted probabilities are genuinely informative but not
    # perfectly calibrated (systematically overconfident) -- conformal prediction
    # should still deliver ~90% empirical coverage on a held-out test set, drawn
    # from the SAME distribution as calibration, regardless of that miscalibration.
    rng = np.random.default_rng(0)
    n = 20000

    def make_split(n):
        labels = rng.integers(0, 2, size=n)
        # true-class prob centered higher than it "should" be (overconfident),
        # but still correlated with correctness via label-dependent noise.
        true_class_prob = np.clip(rng.normal(0.85, 0.1, size=n), 0.5, 0.999)
        probs = np.zeros((n, 2))
        probs[np.arange(n), labels] = true_class_prob
        probs[np.arange(n), 1 - labels] = 1 - true_class_prob
        return probs, labels

    probs_cal, labels_cal = make_split(n)
    probs_test, labels_test = make_split(n)

    result = run_conformal_analysis(probs_cal, labels_cal, probs_test, labels_test, alpha=0.10)
    assert result.empirical_coverage == pytest.approx(0.90, abs=0.01)
    assert result.target_coverage == pytest.approx(0.90)
    assert result.n_calibration == n
    assert result.n_test == n


def test_lower_alpha_requires_larger_or_equal_threshold():
    # Tighter target coverage (smaller alpha) should never produce a SMALLER
    # threshold than a looser target — monotonicity of the calibration quantile.
    rng = np.random.default_rng(1)
    probs_cal = np.column_stack([rng.uniform(0, 1, 500)] * 1)
    probs_cal = np.column_stack([1 - probs_cal[:, 0], probs_cal[:, 0]])
    labels_cal = rng.integers(0, 2, 500)
    threshold_90 = calibrate_conformal_threshold(probs_cal, labels_cal, alpha=0.10)
    threshold_50 = calibrate_conformal_threshold(probs_cal, labels_cal, alpha=0.50)
    assert threshold_90 >= threshold_50
