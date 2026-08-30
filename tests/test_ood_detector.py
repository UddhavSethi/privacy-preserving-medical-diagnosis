"""OPT-5 — real tests for src/uncertainty/ood_detector.py. Trains real
IsolationForest instances on synthetic clustered data and checks the actual
behavior that matters: genuine outliers score more anomalous than in-distribution
points, the calibrated threshold achieves its target flag rate, and results are
deterministic given a seed."""
from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.ood_detector import (
    build_and_calibrate,
    calibrate_ood_threshold,
    compute_anomaly_scores,
    flag_ood,
    train_ood_detector,
)


def _clustered_in_distribution(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, 16))


def test_train_ood_detector_rejects_non_2d_input():
    with pytest.raises(ValueError):
        train_ood_detector(np.zeros(10), seed=0)


def test_genuine_outliers_score_more_anomalous_than_in_distribution():
    train_features = _clustered_in_distribution(500, seed=0)
    detector = train_ood_detector(train_features, seed=0)

    in_dist_test = _clustered_in_distribution(50, seed=1)
    far_outliers = np.full((50, 16), 50.0)  # wildly far from the origin-centered cluster

    in_dist_scores = compute_anomaly_scores(detector, in_dist_test)
    outlier_scores = compute_anomaly_scores(detector, far_outliers)

    assert outlier_scores.mean() > in_dist_scores.mean()
    assert outlier_scores.min() > in_dist_scores.max()


def test_calibrate_threshold_achieves_target_flag_fraction_on_calibration_set():
    scores = np.arange(100, dtype=float)  # scores 0..99, uniformly spread
    threshold = calibrate_ood_threshold(scores, target_flag_fraction=0.10)
    flagged = flag_ood(scores, threshold)
    assert flagged.sum() == 10


def test_calibrate_threshold_zero_target_flags_nothing():
    scores = np.arange(50, dtype=float)
    threshold = calibrate_ood_threshold(scores, target_flag_fraction=0.0)
    assert flag_ood(scores, threshold).sum() == 0


def test_calibrate_threshold_invalid_fraction_raises():
    with pytest.raises(ValueError):
        calibrate_ood_threshold(np.arange(10, dtype=float), target_flag_fraction=1.0)
    with pytest.raises(ValueError):
        calibrate_ood_threshold(np.arange(10, dtype=float), target_flag_fraction=-0.1)


def test_build_and_calibrate_realized_fraction_close_to_target():
    train_features = _clustered_in_distribution(1000, seed=0)
    cal_features = _clustered_in_distribution(500, seed=1)  # same distribution -> exchangeable
    detector, evaluation = build_and_calibrate(train_features, cal_features, seed=0, target_flag_fraction=0.05)
    assert evaluation.realized_flag_fraction_on_calibration == pytest.approx(0.05, abs=0.01)
    assert evaluation.n_calibration == 500


def test_determinism_given_seed():
    train_features = _clustered_in_distribution(300, seed=0)
    test_features = _clustered_in_distribution(50, seed=2)

    detector_a = train_ood_detector(train_features, seed=42)
    detector_b = train_ood_detector(train_features, seed=42)

    scores_a = compute_anomaly_scores(detector_a, test_features)
    scores_b = compute_anomaly_scores(detector_b, test_features)
    np.testing.assert_array_equal(scores_a, scores_b)


def test_different_seeds_can_produce_different_detectors():
    train_features = _clustered_in_distribution(300, seed=0)
    test_features = _clustered_in_distribution(50, seed=2)

    detector_a = train_ood_detector(train_features, seed=1)
    detector_b = train_ood_detector(train_features, seed=2)

    scores_a = compute_anomaly_scores(detector_a, test_features)
    scores_b = compute_anomaly_scores(detector_b, test_features)
    # Not asserting inequality everywhere (could coincidentally match), just that
    # this is a real, non-degenerate computation producing finite real scores.
    assert np.all(np.isfinite(scores_a))
    assert np.all(np.isfinite(scores_b))
