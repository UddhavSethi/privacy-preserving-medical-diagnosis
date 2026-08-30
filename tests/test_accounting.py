import pytest

from src.privacy.accounting import compute_noise_multiplier, compute_total_steps


def test_compute_total_steps():
    # 1000 samples, batch 32 -> 31 steps/epoch (integer division), x2 local epochs x5 rounds
    steps = compute_total_steps(dataset_size=1000, batch_size=32, local_epochs=2, num_rounds=5)
    assert steps == (1000 // 32) * 2 * 5


def test_compute_total_steps_minimum_one_step_per_epoch():
    steps = compute_total_steps(dataset_size=5, batch_size=32, local_epochs=1, num_rounds=1)
    assert steps == 1  # dataset smaller than batch_size still counts as >=1 step/epoch


def test_noise_multiplier_increases_for_tighter_epsilon():
    """A smaller (stricter) target epsilon must require MORE noise for the same
    number of steps — this is the whole point of the privacy-utility tradeoff."""
    sigma_tight = compute_noise_multiplier(
        target_epsilon=1.0, target_delta=1e-5, sample_rate=0.01, total_steps=500
    )
    sigma_loose = compute_noise_multiplier(
        target_epsilon=8.0, target_delta=1e-5, sample_rate=0.01, total_steps=500
    )
    assert sigma_tight > sigma_loose > 0


def test_noise_multiplier_increases_with_more_steps():
    """More steps at the same target epsilon must require more noise per step —
    otherwise cumulative privacy loss over more steps would exceed the target."""
    sigma_few_steps = compute_noise_multiplier(
        target_epsilon=4.0, target_delta=1e-5, sample_rate=0.01, total_steps=100
    )
    sigma_many_steps = compute_noise_multiplier(
        target_epsilon=4.0, target_delta=1e-5, sample_rate=0.01, total_steps=2000
    )
    assert sigma_many_steps > sigma_few_steps
