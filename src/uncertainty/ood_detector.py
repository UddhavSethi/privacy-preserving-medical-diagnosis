"""OPT-5 — Isolation Forest OOD detection gate (owner-approved 2026-08-30, Phase 6
priority 5; concept-approved 2026-08-29, CLAUDE.md section 16.1a).

A client-side safety gate flagging chest X-rays whose frozen-backbone features are
anomalous relative to the training distribution — wrong modality, corrupted scans,
an unfamiliar population. This is a genuinely different failure mode from MC
Dropout's epistemic uncertainty (Stage 19), which implicitly assumes the input is
roughly in-distribution and can be confidently wrong on inputs that aren't.

**One `IsolationForest` per hospital, not a single federated/global detector** —
Isolation Forest is not a parametric model, so it cannot be `FedAvg`'d the way the
classifier head can; each hospital trains its own on its own cached features,
consistent with data never leaving a hospital. Trained on the full in-distribution
training feature set, BOTH classes — training on Normal-only would make it
partially redundant with the classifier itself (a density estimate correlated with
"is this Pneumonia") rather than a genuine domain-shift/corruption detector.

**This does NOT touch, and must never be made to touch, Secure Aggregation or the
FedAvg update path.** That interpretation of "anomaly detection" (flagging
anomalous federated client/model UPDATES rather than anomalous chest X-ray INPUTS)
was considered and explicitly rejected — CLAUDE.md section 16.1a and section 6:
Secure Aggregation's whole point is that the server never sees an individual
update, which is directly opposed to inspecting updates for anomalies, and
malicious-client defense is out of scope for this phase (CLAUDE.md section 16.2).
This module operates entirely client-side, on already-local cached image features,
completely independent of the federated round.

**Threshold policy**: the anomaly-flag threshold is, like Stage 19's DG-10
deferral threshold, a clinical-policy decision, not a tuning parameter — derived
from a target in-distribution flag-fraction on a held-out calibration set (the
same fixed-coverage-target design DG-10 already established,
`src/uncertainty/deferral.py`), not a hand-picked raw anomaly-score cutoff. Unlike
DG-10, no owner-approved specific rate exists yet for this gate; `TARGET_FLAG_FRACTION`
below is a reasonable placeholder requiring the same explicit clinical sign-off
DG-10 received before any real deployment — stated here, not silently treated as
already decided.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

TARGET_FLAG_FRACTION = 0.05  # placeholder default — NOT an owner-approved clinical
# policy decision like DG-10's 10% deferral rate; requires the same explicit
# sign-off before real deployment (see module docstring).


def train_ood_detector(features: np.ndarray, seed: int, n_estimators: int = 100) -> IsolationForest:
    """Trains one Isolation Forest on a hospital's full in-distribution training
    feature set (both classes — see module docstring for why). `features` is
    (N, 1024), the same pooled eval-view backbone features Stage 9's cache stores."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (N, D), got shape {features.shape}")
    detector = IsolationForest(n_estimators=n_estimators, random_state=seed)
    detector.fit(features)
    return detector


def compute_anomaly_scores(detector: IsolationForest, features: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negated so this matches the "higher = more
    concerning" convention `src/uncertainty/mc_dropout.py`'s entropy already uses —
    sklearn's own `decision_function` is the opposite sign, negative = outlier)."""
    return -detector.decision_function(np.asarray(features, dtype=float))


def calibrate_ood_threshold(calibration_scores: np.ndarray, target_flag_fraction: float = TARGET_FLAG_FRACTION) -> float:
    """Fixed-coverage-target threshold, mirroring `src.uncertainty.deferral.
    compute_deferral`'s own design: derives the anomaly-score cutoff from the
    ACTUAL score distribution on a held-out in-distribution calibration set, so
    that flagging `target_flag_fraction` of that calibration set is exactly what
    the returned threshold achieves by construction — not a hand-picked raw score
    on an otherwise-uninterpretable Isolation Forest scale."""
    if not 0.0 <= target_flag_fraction < 1.0:
        raise ValueError(f"target_flag_fraction must be in [0, 1), got {target_flag_fraction}")
    scores = np.asarray(calibration_scores, dtype=float)
    n = len(scores)
    num_flag = int(round(n * target_flag_fraction))
    if num_flag == 0:
        return float(scores.max()) + 1.0  # unreachable — nothing flagged
    sorted_scores = np.sort(scores)[::-1]
    return float(sorted_scores[num_flag - 1])


def flag_ood(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) >= threshold


@dataclass(frozen=True)
class OODEvaluation:
    threshold: float
    target_flag_fraction: float
    realized_flag_fraction_on_calibration: float
    n_calibration: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_and_calibrate(
    train_features: np.ndarray, calibration_features: np.ndarray, seed: int, target_flag_fraction: float = TARGET_FLAG_FRACTION
) -> tuple[IsolationForest, OODEvaluation]:
    """End-to-end: train on `train_features`, calibrate the threshold on
    `calibration_features` (a held-out val set — never the training set itself,
    to avoid measuring the detector's fit to its own training data as if it were
    generalization)."""
    detector = train_ood_detector(train_features, seed=seed)
    cal_scores = compute_anomaly_scores(detector, calibration_features)
    threshold = calibrate_ood_threshold(cal_scores, target_flag_fraction)
    realized = float(flag_ood(cal_scores, threshold).mean())
    evaluation = OODEvaluation(
        threshold=threshold,
        target_flag_fraction=target_flag_fraction,
        realized_flag_fraction_on_calibration=realized,
        n_calibration=len(calibration_features),
    )
    return detector, evaluation
