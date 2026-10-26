import math

import numpy as np
import pytest

from src.evaluation.mia import (
    attack_auc, best_epsilon_lower_bound, bootstrap_auc_ci, clopper_pearson, epsilon_lower_bound,
    lira_scores, logit_confidence, tpr_at_fpr,
)


def test_logit_confidence_is_finite_at_extremes_and_monotone():
    p = np.array([0.0, 1e-9, 0.5, 1 - 1e-9, 1.0])
    z = logit_confidence(p)
    assert np.all(np.isfinite(z)) and np.all(np.diff(z) > 0)
    assert z[2] == pytest.approx(0.0, abs=1e-9)


def test_attack_auc_extremes():
    assert attack_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    rng = np.random.default_rng(0)
    assert attack_auc(rng.normal(size=5000), rng.normal(size=5000)) == pytest.approx(0.5, abs=0.03)


def test_bootstrap_ci_brackets_estimate_and_shrinks_with_n():
    rng = np.random.default_rng(1)
    small = bootstrap_auc_ci(rng.normal(0.3, 1, 100), rng.normal(0, 1, 100), n_boot=200)
    large = bootstrap_auc_ci(rng.normal(0.3, 1, 3000), rng.normal(0, 1, 3000), n_boot=200)
    for auc, lo, hi in (small, large):
        assert lo <= auc <= hi
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_tpr_at_fpr_perfect_and_chance():
    perfect = tpr_at_fpr(np.arange(10, 20.0), np.arange(0, 10.0), fprs=(0.01,))
    assert perfect[0.01] == 1.0
    rng = np.random.default_rng(2)
    chance = tpr_at_fpr(rng.normal(size=20000), rng.normal(size=20000), fprs=(0.1,))
    assert chance[0.1] == pytest.approx(0.1, abs=0.02)


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 100, 0.05)
    assert lo == 0.0 and hi == pytest.approx(0.0362, abs=1e-3)
    lo, hi = clopper_pearson(50, 100, 0.05)
    assert lo < 0.5 < hi


def test_epsilon_lower_bound_is_zero_without_advantage():
    assert epsilon_lower_bound(tp=500, n_members=1000, fp=500, n_nonmembers=1000) == 0.0


def test_epsilon_lower_bound_never_exceeds_true_epsilon_on_randomized_response():
    """Calibration of the bound itself: for eps-DP randomized response the attacker's best
    TPR/FPR are e^eps/(1+e^eps) and 1/(1+e^eps); the bound must stay <= eps and approach it."""
    eps, n = 1.0, 200_000
    rng = np.random.default_rng(3)
    p = math.exp(eps) / (1 + math.exp(eps))
    members = rng.random(n) < p       # attacker says "member" on true members w.p. p
    nonmembers = rng.random(n) < 1 - p  # ... and on non-members w.p. 1-p
    lb = epsilon_lower_bound(int(members.sum()), n, int(nonmembers.sum()), n, delta=0.0)
    assert 0.9 * eps < lb <= eps


def test_best_epsilon_lower_bound_zero_for_identical_distributions():
    rng = np.random.default_rng(4)
    res = best_epsilon_lower_bound(rng.normal(size=3000), rng.normal(size=3000), max_thresholds=50)
    assert res["epsilon_lb"] < 0.5


def test_lira_scores_separate_members_from_non_members():
    rng = np.random.default_rng(5)
    n = 400
    mu = rng.normal(0, 1, n)                     # per-example difficulty
    in_stats = [mu[i] + 1.5 + rng.normal(0, 0.5, 8) for i in range(n)]
    out_stats = [mu[i] + rng.normal(0, 0.5, 8) for i in range(n)]
    member_t = mu + 1.5 + rng.normal(0, 0.5, n)
    nonmember_t = mu + rng.normal(0, 0.5, n)
    sm = lira_scores(member_t, in_stats, out_stats)
    sn = lira_scores(nonmember_t, in_stats, out_stats)
    assert attack_auc(sm, sn) > 0.9


def test_vectorised_loo_lira_matches_the_per_example_reference_implementation():
    from src.evaluation.mia import lira_loo_scores
    rng = np.random.default_rng(7)
    M, N = 9, 40
    mask = rng.random((M, N)) < 0.5
    phi = rng.normal(size=(M, N)) + 1.2 * mask
    scores, raw, truth, midx = lira_loo_scores(phi, mask)
    for t in (0, 4, 8):
        others = np.arange(M) != t
        in_s = [phi[others][mask[others][:, i], i] for i in range(N)]
        out_s = [phi[others][~mask[others][:, i], i] for i in range(N)]
        keep = np.array([len(a) > 1 and len(b) > 1 for a, b in zip(in_s, out_s)])
        want = lira_scores(phi[t][keep], [x for x, k in zip(in_s, keep) if k], [x for x, k in zip(out_s, keep) if k])
        got = scores[midx == t]
        assert got.shape == want.shape and np.allclose(got, want, atol=1e-8)
        assert np.array_equal(truth[midx == t], mask[t][keep])
