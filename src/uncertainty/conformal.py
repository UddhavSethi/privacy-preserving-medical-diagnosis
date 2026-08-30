"""OPT-4 — conformal prediction (owner-approved 2026-08-30, Phase 6 priority 4,
scoped conditionally on OPT-1's finding — see `docs/calibration.md`).

OPT-1 measured a real problem MC Dropout's raw confidence has: Expected
Calibration Error roughly quadruples the moment DP is turned on, and stays flat
across the whole epsilon sweep rather than tracking accuracy. Conformal
prediction is the principled fix precisely for this situation — it does not
require the underlying model's confidence to be well-calibrated at all. Instead,
it uses a held-out calibration set to derive a threshold such that prediction
SETS (not point predictions) contain the true label with a formal, distribution-
free marginal coverage guarantee: P(y_true in prediction_set) >= 1 - alpha,
for any alpha, as long as the calibration and test sets are exchangeable.

**Method: split conformal prediction with the LAC (Least Ambiguous set-valued
Classifier) non-conformity score** (Sadinle et al. 2019) — the standard, simplest
conformal method for classification: score(x, y) = 1 - p_model(y | x). Chosen
over the more elaborate APS (Adaptive Prediction Sets) score for simplicity and
because LAC is the textbook baseline; a stronger method is future work, not a
retracted claim here.

alpha=0.10 (90% target coverage) is the default, deliberately matching Stage 19's
own DG-10 fixed-coverage-target convention (`src/uncertainty/deferral.py`) for
direct comparability, not picked independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def calibrate_conformal_threshold(probs_cal: np.ndarray, labels_cal: np.ndarray, alpha: float = 0.10) -> float:
    """Split-conformal calibration: computes the non-conformity score
    1 - p_model(y_true | x) for every calibration example, then returns the
    finite-sample-corrected (1-alpha) quantile (Vovk's classic correction —
    ceil((n+1)(1-alpha))/n, not the naive n*(1-alpha) quantile, so the coverage
    guarantee holds exactly at finite n, not just asymptotically)."""
    probs_cal = np.asarray(probs_cal, dtype=float)
    labels_cal = np.asarray(labels_cal, dtype=int)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(labels_cal)
    if n == 0:
        raise ValueError("probs_cal/labels_cal must be non-empty")

    scores = 1.0 - probs_cal[np.arange(n), labels_cal]
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def predict_conformal_sets(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Returns an (N, C) boolean membership matrix: class c is in example i's
    prediction set iff 1 - probs[i, c] <= threshold, i.e. probs[i, c] >= 1 -
    threshold. Sets may legitimately be empty (no class confident enough) or
    contain every class (nothing ruled out) — both are valid conformal outcomes,
    not errors; an empty set is itself a strong "genuinely uncertain" signal."""
    probs = np.asarray(probs, dtype=float)
    return (1.0 - probs) <= threshold


def empirical_coverage(membership: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of examples whose TRUE label is inside their prediction set —
    the quantity the conformal guarantee bounds below by (1 - alpha)."""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    return float(membership[np.arange(n), labels].mean())


def mean_set_size(membership: np.ndarray) -> float:
    return float(membership.sum(axis=1).mean())


def set_size_distribution(membership: np.ndarray) -> dict:
    """Fraction of examples whose prediction set has size 0 (empty — maximally
    uncertain, model trusts neither class enough), 1 (a single confident
    prediction), or 2+ (ambiguous between classes) — for a 2-class problem, sizes
    are exactly {0, 1, 2}."""
    sizes = membership.sum(axis=1)
    n = len(sizes)
    return {
        "empty": float((sizes == 0).sum() / n),
        "singleton": float((sizes == 1).sum() / n),
        "full": float((sizes >= 2).sum() / n),
    }


@dataclass(frozen=True)
class ConformalResult:
    threshold: float
    target_coverage: float
    empirical_coverage: float
    mean_set_size: float
    set_size_distribution: dict
    n_calibration: int
    n_test: int

    def to_dict(self) -> dict:
        return asdict(self)


def run_conformal_analysis(
    probs_cal: np.ndarray,
    labels_cal: np.ndarray,
    probs_test: np.ndarray,
    labels_test: np.ndarray,
    alpha: float = 0.10,
) -> ConformalResult:
    threshold = calibrate_conformal_threshold(probs_cal, labels_cal, alpha)
    membership = predict_conformal_sets(probs_test, threshold)
    return ConformalResult(
        threshold=threshold,
        target_coverage=1.0 - alpha,
        empirical_coverage=empirical_coverage(membership, labels_test),
        mean_set_size=mean_set_size(membership),
        set_size_distribution=set_size_distribution(membership),
        n_calibration=len(labels_cal),
        n_test=len(labels_test),
    )
