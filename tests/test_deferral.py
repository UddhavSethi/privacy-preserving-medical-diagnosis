"""Stage 19 — deferral policy (DG-10). Real trained checkpoint, real pooled
test features, checking the plan's own named testing criteria: deferral rate
responds to the threshold, and accuracy on retained cases exceeds overall
accuracy (the actual "does the human-in-the-loop path help" claim).
"""
from pathlib import Path

import pytest
import torch

from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_pooled_features
from src.uncertainty.deferral import DEFAULT_TARGET_DEFER_FRACTION, compute_deferral
from src.uncertainty.mc_dropout import compute_mc_dropout_uncertainty

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "centralized_baseline" / "natural_seed42.pt"
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
HOSPITALS = ["A", "B", "C"]

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="requires a trained centralized_baseline checkpoint (Stage 12)"
)


def _load_trained_model() -> DenseNet121Head:
    model = DenseNet121Head()
    model.classifier.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    return model


def _load_real_test_features():
    if not PARTITION_PATH.exists():
        pytest.skip("requires the frozen hospitals_natural.json partition")
    pooled = load_pooled_features(PARTITION_PATH, HOSPITALS)
    return pooled.test_features, pooled.test_labels


def test_rejects_invalid_target_fraction():
    entropy = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        compute_deferral(entropy, target_defer_fraction=1.0)
    with pytest.raises(ValueError):
        compute_deferral(entropy, target_defer_fraction=-0.1)


def test_defers_approximately_the_target_fraction_on_synthetic_data():
    entropy = torch.rand(1000)
    result = compute_deferral(entropy, target_defer_fraction=0.10)
    realized_defer_fraction = result.deferred_mask.float().mean().item()
    assert abs(realized_defer_fraction - 0.10) < 0.01  # continuous random entropy -> few ties
    assert abs(result.coverage - 0.90) < 0.01


def test_deferral_rate_responds_to_the_threshold():
    """Stage 19's own testing criterion: the deferral rate must actually
    change with the target fraction, not be silently constant."""
    entropy = torch.rand(1000)
    low = compute_deferral(entropy, target_defer_fraction=0.05)
    high = compute_deferral(entropy, target_defer_fraction=0.30)

    assert high.deferred_mask.sum() > low.deferred_mask.sum()
    assert high.coverage < low.coverage
    assert high.threshold < low.threshold  # a larger deferred fraction reaches further down the ranking


def test_zero_target_fraction_defers_nothing():
    entropy = torch.rand(100)
    result = compute_deferral(entropy, target_defer_fraction=0.0)
    assert result.deferred_mask.sum() == 0
    assert result.coverage == 1.0


def test_accuracy_on_retained_cases_exceeds_overall_accuracy():
    """The actual "does the human-in-the-loop path help" claim, Stage 19's
    own testing criterion, on the real pooled test set with the
    owner-approved default coverage target (defer worst 10%)."""
    model = _load_trained_model()
    features, labels = _load_real_test_features()

    result = compute_mc_dropout_uncertainty(model, features, num_passes=20)
    deferral = compute_deferral(result.entropy, target_defer_fraction=DEFAULT_TARGET_DEFER_FRACTION)

    overall_correct = (result.predicted_class == labels).float()
    overall_accuracy = overall_correct.mean().item()

    retained_mask = ~deferral.deferred_mask
    assert retained_mask.sum() > 0
    retained_accuracy = overall_correct[retained_mask].mean().item()

    assert retained_accuracy >= overall_accuracy, (
        f"expected retained-case accuracy ({retained_accuracy:.4f}) to be at least "
        f"overall accuracy ({overall_accuracy:.4f}) after deferring the least-confident cases"
    )
