"""Stage 19 — deferral policy (Decision Gate DG-10, owner-approved
2026-08-30): **fixed coverage target**. Defers the highest-uncertainty
fraction of predictions to human clinician review rather than acting on them
automatically. This makes the deferral rate the configured policy knob (an
explicit clinical choice — CLAUDE.md's own framing, not a hyperparameter to
silently default), and the resulting entropy cutoff a *derived* value from
each run's actual uncertainty distribution rather than a hand-picked number
on an otherwise-uninterpretable entropy scale.

Default target: defer the worst 10% by predictive entropy (owner-approved
2026-08-30). Pairs naturally with a risk-coverage curve as a future
extension (CLAUDE.md section 16.1 — explicitly a pending optional direction,
not built here).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_TARGET_DEFER_FRACTION = 0.10


@dataclass
class DeferralResult:
    threshold: float  # entropy cutoff derived from the target coverage this run
    deferred_mask: torch.Tensor  # (N,) bool — True = deferred to human review
    coverage: float  # actual fraction retained (== 1 - realized defer fraction)


def compute_deferral(entropy: torch.Tensor, target_defer_fraction: float) -> DeferralResult:
    """Defers the `target_defer_fraction` highest-entropy predictions in this
    batch. `target_defer_fraction=0.10` defers the worst 10% by uncertainty."""
    if not 0.0 <= target_defer_fraction < 1.0:
        raise ValueError(f"target_defer_fraction must be in [0, 1), got {target_defer_fraction}")

    n = entropy.shape[0]
    num_defer = int(round(n * target_defer_fraction))
    if num_defer == 0:
        threshold = float(entropy.max().item()) + 1.0  # unreachable — nothing deferred
        deferred_mask = torch.zeros(n, dtype=torch.bool)
    else:
        sorted_entropy, _ = torch.sort(entropy, descending=True)
        threshold = float(sorted_entropy[num_defer - 1].item())
        # `>=`, not top-k indices directly: ties at the threshold value can
        # legitimately defer slightly more than `num_defer` examples — that is
        # correct (never silently under-defer a tied high-uncertainty case),
        # not a bug in the realized coverage.
        deferred_mask = entropy >= threshold

    coverage = 1.0 - deferred_mask.float().mean().item()
    return DeferralResult(threshold=threshold, deferred_mask=deferred_mask, coverage=coverage)
