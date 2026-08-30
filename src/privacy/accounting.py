"""Privacy accounting (Stage 14, ADR-2).

Wraps Opacus's accountant utilities so the noise multiplier needed for a target
(epsilon, delta) is computed once, upfront, from the TOTAL number of steps a client
will take across every round it participates in — not recomputed per round. A static
or per-round-reset epsilon is the classic silent bug this stage's own testing
criteria warn about; the accountant instance must persist across rounds (see
`src/privacy/dp.py`) for cumulative accounting to be correct.
"""
from __future__ import annotations

from opacus.accountants.utils import get_noise_multiplier


def compute_total_steps(dataset_size: int, batch_size: int, local_epochs: int, num_rounds: int) -> int:
    """Total optimizer steps one client will take over the full federated run — what
    the noise multiplier must be calibrated against for correct cross-round
    accounting."""
    steps_per_epoch = max(1, dataset_size // batch_size)
    return steps_per_epoch * local_epochs * num_rounds


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    total_steps: int,
) -> float:
    """The Gaussian noise multiplier (sigma) that achieves target_epsilon after
    `total_steps` DP-SGD steps at the given `sample_rate` (batch_size / dataset_size),
    per the pinned Opacus version's RDP accountant."""
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp",
    )
