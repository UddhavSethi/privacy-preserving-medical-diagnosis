"""Evaluation metrics for binary pneumonia classification (Stage 10).

AUROC is the primary metric (CLAUDE.md section 11.2) — accuracy alone is not
acceptable on this imbalanced medical data. Every later stage reports through this
module so results are directly comparable; it is built before any baseline (Stage 11)
produces a number, per CLAUDE.md's explicit warning that building it after is "the
classic mistake" (results get recomputed and tables silently disagree).

**Threshold policy (explicit, per Stage 10's flagged risk):** threshold-dependent
metrics (F1, specificity, balanced accuracy, confusion matrix) use a fixed default
decision threshold of 0.5 on the positive-class probability, unless a caller
overrides it. Any reported threshold-dependent number must state this alongside it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_TARGET_SPECIFICITY = 0.9


@dataclass(frozen=True)
class Metrics:
    auroc: float
    auprc: float
    sensitivity: float  # a.k.a. recall, at `threshold`
    specificity: float  # at `threshold`
    f1: float  # at `threshold`
    balanced_accuracy: float  # at `threshold`
    sensitivity_at_target_specificity: float
    target_specificity: float
    threshold: float
    confusion_matrix: list  # [[tn, fp], [fn, tp]]
    n_samples: int
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_binary_inputs(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_score shape {y_score.shape}")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must be binary (0/1)")


def sensitivity_at_specificity(
    y_true, y_score, target_specificity: float = DEFAULT_TARGET_SPECIFICITY
) -> float:
    """Highest sensitivity achievable at >= target_specificity, swept across every
    threshold on the ROC curve — the standard way this is reported in medical ML
    papers, and threshold-independent in that sense (no single fixed cut point)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    eligible = specificity >= target_specificity
    if not np.any(eligible):
        return 0.0
    return float(tpr[eligible].max())


def compute_metrics(
    y_true,
    y_score,
    threshold: float = DEFAULT_THRESHOLD,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
) -> Metrics:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    _validate_binary_inputs(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    n_positive = int(y_true.sum())
    n_negative = int(len(y_true) - n_positive)

    # AUROC/AUPRC/sensitivity-at-specificity are undefined with only one class present.
    if n_positive == 0 or n_negative == 0:
        auroc = float("nan")
        auprc = float("nan")
        sens_at_spec = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
        sens_at_spec = sensitivity_at_specificity(y_true, y_score, target_specificity)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return Metrics(
        auroc=auroc,
        auprc=auprc,
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        sensitivity_at_target_specificity=sens_at_spec,
        target_specificity=target_specificity,
        threshold=threshold,
        confusion_matrix=cm.tolist(),
        n_samples=int(len(y_true)),
        n_positive=n_positive,
        n_negative=n_negative,
    )
