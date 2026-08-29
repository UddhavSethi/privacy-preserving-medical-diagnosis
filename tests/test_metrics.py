import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.evaluation.metrics import compute_metrics, sensitivity_at_specificity


def test_known_input_regression():
    """The canonical sklearn roc_auc_score docstring example — AUROC is a fixed,
    well-known value (0.75) for this exact input, independent of this project's code."""
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    m = compute_metrics(y_true, y_score)
    assert m.auroc == pytest.approx(0.75)
    assert m.n_samples == 4
    assert m.n_positive == 2
    assert m.n_negative == 2


def test_auroc_and_auprc_match_sklearn_on_synthetic_data():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_score = rng.random(200)

    m = compute_metrics(y_true, y_score)
    assert m.auroc == pytest.approx(roc_auc_score(y_true, y_score))
    assert m.auprc == pytest.approx(average_precision_score(y_true, y_score))


def test_threshold_dependent_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 200)
    y_score = rng.random(200)
    threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)

    m = compute_metrics(y_true, y_score, threshold=threshold)
    assert m.f1 == pytest.approx(f1_score(y_true, y_pred))
    assert m.balanced_accuracy == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_degenerate_all_correct():
    y_true = [0, 0, 1, 1]
    y_score = [0.0, 0.1, 0.9, 1.0]
    m = compute_metrics(y_true, y_score)
    assert m.auroc == pytest.approx(1.0)
    assert m.sensitivity == pytest.approx(1.0)
    assert m.specificity == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_degenerate_all_wrong():
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 1.0, 0.0, 0.1]
    m = compute_metrics(y_true, y_score)
    assert m.auroc == pytest.approx(0.0)
    assert m.sensitivity == pytest.approx(0.0)
    assert m.specificity == pytest.approx(0.0)


def test_degenerate_single_class_present():
    """AUROC/AUPRC are mathematically undefined with only one class in y_true —
    must not crash, must report NaN rather than a fabricated number."""
    y_true = [0, 0, 0, 0]
    y_score = [0.1, 0.4, 0.35, 0.8]
    m = compute_metrics(y_true, y_score)
    assert np.isnan(m.auroc)
    assert np.isnan(m.auprc)
    assert m.n_positive == 0
    assert m.n_negative == 4
    # Threshold-dependent metrics are still well-defined even with one class present.
    assert m.specificity == pytest.approx(0.75)  # 3/4 correctly below threshold 0.5


def test_confusion_matrix_shape_and_values():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.6, 0.4, 0.9]  # one FP (0.6>=0.5), one FN (0.4<0.5)
    m = compute_metrics(y_true, y_score)
    tn, fp = m.confusion_matrix[0]
    fn, tp = m.confusion_matrix[1]
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)


def test_sensitivity_at_specificity_perfect_separation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    sens = sensitivity_at_specificity(y_true, y_score, target_specificity=0.9)
    assert sens == pytest.approx(1.0)  # perfectly separable, so full sensitivity is achievable


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5, 0.5, 0.5])


def test_non_binary_labels_raise():
    with pytest.raises(ValueError):
        compute_metrics([0, 1, 2], [0.1, 0.5, 0.9])


def test_threshold_policy_is_explicit_and_overridable():
    y_true = [0, 0, 1, 1]
    y_score = [0.2, 0.4, 0.6, 0.8]
    default = compute_metrics(y_true, y_score)
    assert default.threshold == 0.5

    strict = compute_metrics(y_true, y_score, threshold=0.7)
    assert strict.threshold == 0.7
    # A higher threshold predicts fewer positives, so sensitivity can only drop or stay.
    assert strict.sensitivity <= default.sensitivity
