"""OPT-2 — real tests for src/evaluation/privacy_attack.py, against constructed
known-answer cases (a model that clearly memorized its training set vs. one that
generalizes identically to both), matching this project's convention throughout."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.privacy_attack import (
    MembershipInferenceResult,
    per_example_cross_entropy_loss,
    run_membership_inference_attack,
)


def test_per_example_cross_entropy_loss_matches_hand_computation():
    probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])
    loss = per_example_cross_entropy_loss(probs, labels)
    assert loss[0] == pytest.approx(-np.log(0.9))
    assert loss[1] == pytest.approx(-np.log(0.8))


def test_per_example_cross_entropy_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_example_cross_entropy_loss(np.array([[0.5, 0.5]]), np.array([0, 1]))


def test_attack_auroc_near_half_when_member_and_nonmember_losses_are_identical():
    # A model that generalizes perfectly: member and non-member losses drawn from
    # the exact same distribution -> no membership signal -> AUROC ~ 0.5.
    rng = np.random.default_rng(0)
    member_loss = rng.normal(0.5, 0.1, size=2000)
    nonmember_loss = rng.normal(0.5, 0.1, size=2000)
    result = run_membership_inference_attack(member_loss, nonmember_loss)
    assert result.attack_auroc == pytest.approx(0.5, abs=0.03)


def test_attack_auroc_near_one_when_model_clearly_memorized_training_set():
    # A badly overfit model: near-zero loss on members, high loss on non-members ->
    # trivially separable -> attack AUROC should be ~1.0.
    member_loss = np.full(100, 0.01)
    nonmember_loss = np.full(100, 2.0)
    result = run_membership_inference_attack(member_loss, nonmember_loss)
    assert result.attack_auroc == pytest.approx(1.0, abs=1e-9)


def test_attack_auroc_is_low_when_members_have_higher_loss_than_nonmembers():
    # Pathological/inverted case (not expected in practice, but the metric should
    # still correctly report a LOW auroc — the score direction, -loss, would rank
    # non-members as more likely members here).
    member_loss = np.full(50, 2.0)
    nonmember_loss = np.full(50, 0.01)
    result = run_membership_inference_attack(member_loss, nonmember_loss)
    assert result.attack_auroc == pytest.approx(0.0, abs=1e-9)


def test_generalization_gap_sign_matches_overfitting_direction():
    member_loss = np.array([0.1, 0.2, 0.1])
    nonmember_loss = np.array([0.5, 0.6, 0.4])
    result = run_membership_inference_attack(member_loss, nonmember_loss)
    assert result.generalization_gap > 0  # nonmember loss higher -> positive gap -> overfitting signal
    assert result.mean_member_loss == pytest.approx(np.mean(member_loss))
    assert result.mean_nonmember_loss == pytest.approx(np.mean(nonmember_loss))


def test_result_counts_recorded_correctly():
    result = run_membership_inference_attack(np.full(30, 0.1), np.full(70, 0.2))
    assert result.n_members == 30
    assert result.n_nonmembers == 70


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        run_membership_inference_attack(np.array([]), np.array([0.1]))
    with pytest.raises(ValueError):
        run_membership_inference_attack(np.array([0.1]), np.array([]))


def test_result_to_dict_roundtrip():
    result = run_membership_inference_attack(np.full(10, 0.1), np.full(10, 0.3))
    d = result.to_dict()
    assert isinstance(d, dict)
    assert set(d.keys()) == set(MembershipInferenceResult.__dataclass_fields__.keys())
