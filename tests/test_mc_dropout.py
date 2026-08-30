"""Stage 19 — Monte Carlo Dropout. Real trained checkpoint, real pooled test
features, not synthetic data — this stage's own flagged risk ("the most
common bug in this area": dropout silently not actually active at inference,
so T passes are secretly identical) is exactly the kind of thing that must be
checked against real repeated forward passes, not assumed from reading the
code.
"""
from pathlib import Path

import pytest
import torch

from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_pooled_features
from src.uncertainty.mc_dropout import (
    compute_mc_dropout_uncertainty,
    mc_dropout_predict,
    predictive_entropy,
)

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


def test_dropout_is_genuinely_active_producing_differing_passes():
    """The stage's own named most-common bug: if dropout is silently not
    active, every one of the T passes would be bit-for-bit identical."""
    model = _load_trained_model()
    features, _ = _load_real_test_features()
    sample = features[:16]

    all_probs = mc_dropout_predict(model, sample, num_passes=10)  # (T, N, C)

    first_pass = all_probs[0]
    later_passes_differ = any(not torch.allclose(all_probs[t], first_pass) for t in range(1, 10))
    assert later_passes_differ, "all T passes produced identical output — dropout is not actually active"


def test_backbone_batchnorm_stays_frozen_during_mc_dropout():
    """MC Dropout activates the classifier's Dropout via model.train(), but
    must NOT reawaken the frozen backbone's BatchNorm — verified directly,
    not assumed from DenseNet121Head.train()'s override."""
    model = _load_trained_model()
    running_mean_before = model.features.norm5.running_mean.clone()
    features, _ = _load_real_test_features()

    mc_dropout_predict(model, features[:16], num_passes=5)

    assert torch.equal(model.features.norm5.running_mean, running_mean_before)
    assert not model.features.norm5.training


def test_predictive_entropy_is_bounded_and_zero_for_a_certain_distribution():
    certain = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    uncertain = torch.tensor([[0.5, 0.5]])
    assert torch.allclose(predictive_entropy(certain), torch.zeros(2), atol=1e-6)
    assert predictive_entropy(uncertain).item() > 0.6  # ln(2) ~= 0.693 at max uncertainty


def test_uncertainty_is_higher_on_misclassified_than_correctly_classified_cases():
    """Stage 19's own testing criterion. A real, non-trivial claim: even a
    known-weak, head-only uncertainty estimator (CLAUDE.md section 10's own
    honest framing) should on average assign higher entropy to cases it gets
    wrong than cases it gets right — otherwise the confidence estimate carries
    no real signal at all."""
    model = _load_trained_model()
    features, labels = _load_real_test_features()

    result = compute_mc_dropout_uncertainty(model, features, num_passes=20)
    correct = result.predicted_class == labels
    misclassified = ~correct

    assert misclassified.sum() > 0, "expected at least some misclassifications on the real pooled test set"
    assert correct.sum() > 0

    mean_entropy_correct = result.entropy[correct].mean().item()
    mean_entropy_wrong = result.entropy[misclassified].mean().item()
    assert mean_entropy_wrong > mean_entropy_correct, (
        f"expected higher mean entropy on misclassified cases; "
        f"got correct={mean_entropy_correct:.4f} wrong={mean_entropy_wrong:.4f}"
    )
