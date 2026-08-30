"""OPT-1 — calibration metrics (owner-approved 2026-08-30, Phase 6 priority 1).

CLAUDE.md section 10 names this project's own honest, unvalidated gap: "MC Dropout
is a known-weak uncertainty estimator and is often poorly calibrated" — asserted,
never measured. This module measures it: Expected Calibration Error (ECE), Brier
score, reliability-diagram data, and a risk-coverage curve, computed from MC
Dropout's predictive distribution (`src.uncertainty.mc_dropout`).

**Calibration convention (explicit, matching Guo et al. 2017's "On Calibration of
Modern Neural Networks", the standard ECE formulation):** confidence is the MAX
predictive probability (the probability assigned to the predicted class, from MC
Dropout's mean distribution across T passes), and accuracy is whether the predicted
class matches the true label. This is the "is the model's stated confidence
trustworthy" framing, not the "is P(pneumonia) itself well-calibrated as a
probability" framing (sklearn's `calibration_curve` convention) — the former is what
this project's deferral mechanism (Stage 19, DG-10) actually depends on: MC Dropout
defers on LOW confidence, not on any particular class's probability value. Both
conventions are legitimate; this project reports the one its own clinical mechanism
uses.

**Brier score convention:** the single positive-class formulation, mean((p_pneumonia
- y)^2), range [0, 1] — matching AUROC/AUPRC's own positive-class-probability
convention elsewhere in this project (`src/evaluation/metrics.py`), not the
multi-class sum-over-classes formulation (which would just be exactly double this
value for a 2-class problem, since P(normal) = 1 - P(pneumonia) contributes an
identical squared error term).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityDiagramData:
    bin_edges: list  # length n_bins+1
    bin_confidence: list  # mean predicted confidence per bin (nan if empty)
    bin_accuracy: list  # empirical accuracy per bin (nan if empty)
    bin_count: list  # number of samples per bin

    def to_dict(self) -> dict:
        return asdict(self)


def _bin_indices(confidence: np.ndarray, n_bins: int) -> np.ndarray:
    # Confidence for a 2-class problem lies in [0.5, 1.0] (it's the max of two
    # probabilities summing to 1), but binning over the full [0, 1] range is the
    # standard ECE convention and costs nothing — bins below 0.5 are simply always
    # empty for this model family, which is itself visible in bin_count rather than
    # silently rescaled away.
    idx = np.floor(confidence * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def reliability_diagram_data(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> ReliabilityDiagramData:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.shape != correct.shape:
        raise ValueError(f"confidence shape {confidence.shape} != correct shape {correct.shape}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = _bin_indices(confidence, n_bins)

    bin_confidence, bin_accuracy, bin_count = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bin_count.append(count)
        bin_confidence.append(float(confidence[mask].mean()) if count > 0 else float("nan"))
        bin_accuracy.append(float(correct[mask].mean()) if count > 0 else float("nan"))

    return ReliabilityDiagramData(
        bin_edges=bin_edges.tolist(),
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_count=bin_count,
    )


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) * |acc_b - conf_b| — the weighted average gap between
    confidence and accuracy across bins. 0 = perfectly calibrated; empty bins
    contribute nothing (matches the standard definition, not a special case)."""
    data = reliability_diagram_data(confidence, correct, n_bins)
    n = len(confidence)
    ece = 0.0
    for conf, acc, count in zip(data.bin_confidence, data.bin_accuracy, data.bin_count):
        if count == 0:
            continue
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob_positive: np.ndarray) -> float:
    """Single positive-class Brier score — see module docstring for the convention."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob_positive = np.asarray(y_prob_positive, dtype=float)
    if y_true.shape != y_prob_positive.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_prob_positive shape {y_prob_positive.shape}")
    return float(np.mean((y_prob_positive - y_true) ** 2))


@dataclass(frozen=True)
class RiskCoverageCurve:
    coverage: list  # fraction of examples retained, ascending from 1/N to 1.0
    risk: list  # error rate (1 - accuracy) among the retained set at that coverage
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def risk_coverage_curve(entropy: np.ndarray, correct: np.ndarray) -> RiskCoverageCurve:
    """Sorts examples by ASCENDING entropy (most-confident first) and sweeps
    coverage from 1/N to 1.0, reporting the retained set's error rate at each point
    — the standard selective-prediction curve. Generalizes DG-10's single fixed
    90%-coverage operating point (Stage 19) into the full curve: if uncertainty is
    doing its job, risk should be low at low coverage (only the most-confident,
    should-be-easiest cases retained) and rise toward the full-set error rate as
    coverage approaches 1.0."""
    entropy = np.asarray(entropy, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if entropy.shape != correct.shape:
        raise ValueError(f"entropy shape {entropy.shape} != correct shape {correct.shape}")

    n = len(entropy)
    order = np.argsort(entropy, kind="stable")  # ascending: most confident (lowest entropy) first
    sorted_correct = correct[order]
    cumulative_correct = np.cumsum(sorted_correct)

    coverage = (np.arange(1, n + 1)) / n
    risk = 1.0 - cumulative_correct / np.arange(1, n + 1)

    return RiskCoverageCurve(coverage=coverage.tolist(), risk=risk.tolist(), n_samples=n)
