"""Bootstrap confidence intervals (Stage 10). CLAUDE.md section 11.2 requires
bootstrap 95% CIs on AUROC — single-point estimates are not credible in FL, where
run-to-run and resample-to-resample variance is high.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    n_valid_resamples: int
    seed: int


def bootstrap_auroc_ci(
    y_true,
    y_score,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Resamples (y_true, y_score) pairs with replacement `n_bootstrap` times and
    reports the percentile interval. A resample containing only one class has an
    undefined AUROC and is skipped (does not count toward `n_bootstrap`'s denominator
    in the interval, only toward the requested draw count)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true must contain both classes — AUROC (and therefore its bootstrap CI) "
            "is undefined for single-class input"
        )

    point_estimate = float(roc_auc_score(y_true, y_score))

    rng = np.random.default_rng(seed)
    scores = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_b = y_true[idx]
        if len(np.unique(y_true_b)) < 2:
            continue  # degenerate resample (single class) — undefined AUROC, skip
        scores[i] = roc_auc_score(y_true_b, y_score[idx])

    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        raise ValueError("every bootstrap resample was degenerate (single-class) — cannot form a CI")

    alpha = 1 - confidence_level
    lower = float(np.percentile(valid_scores, 100 * alpha / 2))
    upper = float(np.percentile(valid_scores, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        n_valid_resamples=len(valid_scores),
        seed=seed,
    )
