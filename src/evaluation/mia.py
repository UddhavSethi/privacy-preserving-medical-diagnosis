"""Membership-inference evaluation utilities (audit follow-up to OPT-2).

`src/evaluation/privacy_attack.py` implements the original single-threshold loss attack and
reports a bare AUROC. That is a weak instrument for two reasons the literature is explicit
about: (1) AUROC averages over the whole score range, hiding attacks that are confident on a
small subset of records (Carlini et al. 2022 argue the operating regime that matters is
true-positive rate at very low false-positive rate); (2) a single global loss threshold
ignores that some examples are intrinsically hard and others easy, so it under-estimates what
a calibrated per-example attack (LiRA) can do. This module adds:

  * `bootstrap_auc_ci`      — percentile bootstrap CI for an attack AUROC (resampling examples)
  * `tpr_at_fpr`            — attack TPR at fixed low FPR values
  * `logit_confidence`      — stable log(p/(1-p)) of the true class, the LiRA statistic
  * `lira_scores`           — per-example Gaussian likelihood-ratio score from shadow-model stats
  * `epsilon_lower_bound`   — empirical lower bound on epsilon implied by a (TPR, FPR) pair via
                              Clopper-Pearson intervals: an (eps, delta)-DP mechanism cannot allow
                              TPR > exp(eps) * FPR + delta

None of this proves a system is private. A low attack score bounds nothing; it is evidence
only against the specific attack run. `epsilon_lower_bound` can falsify a claimed epsilon
(if the bound exceeds it) but cannot certify one.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def logit_confidence(true_class_prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """log(p / (1 - p)) of the true class, computed in float64 (a float32 clip near 1.0
    collapses to exactly 1 and produces infinities)."""
    p = np.clip(np.asarray(true_class_prob, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def attack_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """AUROC of a score where HIGHER means "more likely a member"."""
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    return float(roc_auc_score(y, np.r_[member_scores, nonmember_scores]))


def bootstrap_auc_ci(
    member_scores: np.ndarray, nonmember_scores: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(auc, lo, hi): percentile bootstrap over examples, members and non-members resampled
    independently. Quantifies sampling uncertainty of the AUROC itself — separate from
    seed-to-seed (retraining) variance, which callers report over models."""
    rng = np.random.default_rng(seed)
    m, n = np.asarray(member_scores), np.asarray(nonmember_scores)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        aucs[b] = attack_auc(rng.choice(m, len(m)), rng.choice(n, len(n)))
    return attack_auc(m, n), float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray, fprs=(0.001, 0.01, 0.1)) -> dict[float, float]:
    """Attack true-positive rate at each target false-positive rate (largest TPR whose FPR
    does not exceed the target)."""
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    fpr, tpr, _ = roc_curve(y, np.r_[member_scores, nonmember_scores])
    return {f: float(tpr[fpr <= f].max()) if np.any(fpr <= f) else 0.0 for f in fprs}


def lira_scores(
    target_stat: np.ndarray,
    in_stats: list[np.ndarray],
    out_stats: list[np.ndarray],
    fixed_variance: bool = True,
) -> np.ndarray:
    """Online LiRA (Carlini et al. 2022) log-likelihood ratio per example.

    `target_stat[i]` is the logit-scaled true-class confidence of the model under attack for
    example i. `in_stats[i]` / `out_stats[i]` are the same statistic from shadow models that
    did / did not train on example i. Score = log N(t; mu_in, s_in^2) - log N(t; mu_out, s_out^2).
    `fixed_variance=True` uses one pooled standard deviation per side (more stable when each
    example has only a handful of shadow observations).
    """
    mu_in = np.array([np.mean(a) for a in in_stats])
    mu_out = np.array([np.mean(a) for a in out_stats])
    if fixed_variance:
        s_in = np.sqrt(np.mean(np.concatenate([(np.asarray(a) - m) ** 2 for a, m in zip(in_stats, mu_in)])))
        s_out = np.sqrt(np.mean(np.concatenate([(np.asarray(a) - m) ** 2 for a, m in zip(out_stats, mu_out)])))
        s_in = np.full_like(mu_in, max(s_in, 1e-6)); s_out = np.full_like(mu_out, max(s_out, 1e-6))
    else:
        s_in = np.array([max(np.std(a), 1e-6) for a in in_stats]); s_out = np.array([max(np.std(a), 1e-6) for a in out_stats])
    return stats.norm.logpdf(target_stat, mu_in, s_in) - stats.norm.logpdf(target_stat, mu_out, s_out)


def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided (1 - alpha) Clopper-Pearson interval for a binomial proportion."""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def epsilon_lower_bound(tp: int, n_members: int, fp: int, n_nonmembers: int, delta: float = 1e-5, alpha: float = 0.05) -> float:
    """Lower bound on epsilon (holding with probability >= 1 - alpha) implied by an observed
    attack operating point: an (eps, delta)-DP mechanism satisfies TPR <= e^eps * FPR + delta,
    so eps >= log((TPR_lo - delta) / FPR_hi) with TPR_lo / FPR_hi the one-sided Clopper-Pearson
    limits. Returns 0.0 when the observed advantage is not statistically separable from none.

    Caveat: valid when membership trials are independent. With examples re-used across shadow
    models the trials are dependent, so callers must label the result "indicative".
    """
    tpr_lo, _ = clopper_pearson(tp, n_members, 2 * alpha)  # one-sided via two-sided at 2*alpha
    _, fpr_hi = clopper_pearson(fp, n_nonmembers, 2 * alpha)
    num = tpr_lo - delta
    if num <= 0 or fpr_hi <= 0:
        return 0.0
    return max(0.0, math.log(num / fpr_hi))


def best_epsilon_lower_bound(
    member_scores: np.ndarray, nonmember_scores: np.ndarray, delta: float = 1e-5, alpha: float = 0.05, max_thresholds: int = 400
) -> dict:
    """Maximise `epsilon_lower_bound` over decision thresholds (the maximisation inflates the
    bound only mildly at this many thresholds; the alpha level is not corrected for it, so the
    value is indicative)."""
    m, n = np.asarray(member_scores), np.asarray(nonmember_scores)
    thr = np.unique(np.quantile(np.r_[m, n], np.linspace(0.0, 1.0, max_thresholds)))
    best = {"epsilon_lb": 0.0, "threshold": None, "tpr": 0.0, "fpr": 0.0}
    for t in thr:
        tp, fp = int((m >= t).sum()), int((n >= t).sum())
        e = epsilon_lower_bound(tp, len(m), fp, len(n), delta, alpha)
        if e > best["epsilon_lb"]:
            best = {"epsilon_lb": e, "threshold": float(t), "tpr": tp / len(m), "fpr": fp / len(n)}
    return best


def lira_loo_scores(phi: np.ndarray, mask: np.ndarray, min_obs: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised leave-one-model-out online LiRA with a pooled ("fixed") variance per side.

    `phi[m, i]` is shadow model m's logit-scaled true-class confidence on example i and
    `mask[m, i]` says whether model m trained on example i. Each model in turn is the model
    under attack and is scored using only the OTHER models' per-example in/out statistics.
    Returns flat arrays (lira_score, raw_phi, membership_truth, model_index) over every
    (model, example) pair with at least `min_obs` in- and out-observations from the other
    models. Numerically equivalent to calling `lira_scores` per target model (see tests), but
    loops over models only, not over examples.
    """
    phi = np.asarray(phi, dtype=np.float64)
    w_in = np.asarray(mask, dtype=np.float64)
    w_out = 1.0 - w_in
    M, _ = phi.shape
    tot = {"in": (w_in.sum(0), (w_in * phi).sum(0), (w_in * phi ** 2).sum(0)),
           "out": (w_out.sum(0), (w_out * phi).sum(0), (w_out * phi ** 2).sum(0))}
    ws = {"in": w_in, "out": w_out}
    out_scores, out_phi, out_truth, out_model = [], [], [], []
    for t in range(M):
        side_stats = {}
        for side in ("in", "out"):
            n = tot[side][0] - ws[side][t]
            sm = tot[side][1] - ws[side][t] * phi[t]
            q = tot[side][2] - ws[side][t] * phi[t] ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = sm / n
                var = np.maximum(q / n - mu ** 2, 0.0)
            side_stats[side] = (n, mu, var)
        valid = (side_stats["in"][0] >= min_obs) & (side_stats["out"][0] >= min_obs)
        sd = {side: max(float(np.sqrt((side_stats[side][0][valid] * side_stats[side][2][valid]).sum()
                                      / side_stats[side][0][valid].sum())), 1e-6) for side in ("in", "out")}
        sc = (stats.norm.logpdf(phi[t][valid], side_stats["in"][1][valid], sd["in"])
              - stats.norm.logpdf(phi[t][valid], side_stats["out"][1][valid], sd["out"]))
        out_scores.append(sc); out_phi.append(phi[t][valid])
        out_truth.append(np.asarray(mask[t], dtype=bool)[valid]); out_model.append(np.full(int(valid.sum()), t))
    return tuple(np.concatenate(x) for x in (out_scores, out_phi, out_truth, out_model))
