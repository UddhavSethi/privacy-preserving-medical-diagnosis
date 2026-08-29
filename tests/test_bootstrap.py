import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_auroc_ci


def _synthetic_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_score = rng.random(n)
    return y_true, y_score


def test_point_estimate_matches_full_sample_auroc():
    from sklearn.metrics import roc_auc_score

    y_true, y_score = _synthetic_data()
    ci = bootstrap_auroc_ci(y_true, y_score, n_bootstrap=200, seed=42)
    assert ci.point_estimate == pytest.approx(roc_auc_score(y_true, y_score))


def test_interval_contains_point_estimate():
    y_true, y_score = _synthetic_data()
    ci = bootstrap_auroc_ci(y_true, y_score, n_bootstrap=500, seed=42)
    assert ci.lower <= ci.point_estimate <= ci.upper


def test_reproducible_given_seed():
    y_true, y_score = _synthetic_data()
    ci_a = bootstrap_auroc_ci(y_true, y_score, n_bootstrap=300, seed=7)
    ci_b = bootstrap_auroc_ci(y_true, y_score, n_bootstrap=300, seed=7)
    assert ci_a.lower == ci_b.lower
    assert ci_a.upper == ci_b.upper


def test_different_seeds_can_differ():
    y_true, y_score = _synthetic_data(n=40)  # small n so resample variance is visible
    ci_a = bootstrap_auroc_ci(y_true, y_score, n_bootstrap=100, seed=1)
    ci_b = bootstrap_auroc_ci(y_true, y_score, n_bootstrap=100, seed=2)
    assert (ci_a.lower, ci_a.upper) != (ci_b.lower, ci_b.upper)


def test_narrower_interval_with_more_data():
    rng = np.random.default_rng(0)
    y_true_small = rng.integers(0, 2, 30)
    y_score_small = rng.random(30)
    y_true_large = rng.integers(0, 2, 2000)
    y_score_large = rng.random(2000)

    ci_small = bootstrap_auroc_ci(y_true_small, y_score_small, n_bootstrap=500, seed=42)
    ci_large = bootstrap_auroc_ci(y_true_large, y_score_large, n_bootstrap=500, seed=42)
    assert (ci_large.upper - ci_large.lower) < (ci_small.upper - ci_small.lower)


def test_single_class_input_raises():
    # AUROC is undefined for single-class input; must fail clearly and immediately
    # rather than crash inside sklearn or silently return a meaningless number.
    y_true = [0, 0, 0, 0]
    y_score = [0.1, 0.4, 0.35, 0.8]
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_auroc_ci(y_true, y_score, n_bootstrap=50, seed=0)
