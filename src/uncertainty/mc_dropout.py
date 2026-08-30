"""Stage 19 — Monte Carlo Dropout (CLAUDE.md section 10, objective 6's
confidence half): the same input passed through the network T times with
dropout active, yielding a predictive distribution and an uncertainty
estimate instead of a single point prediction.

MC Dropout requires dropout layers *active* at inference — the opposite of
the usual `.eval()` behavior that disables them — while everything else
(the frozen backbone, its frozen BatchNorm) must stay exactly as it is
during ordinary evaluation. `DenseNet121Head.train()` is already overridden
(Stage 8) to force `self.features.eval()` permanently while enabling
train-mode behavior everywhere else (ADR-1) — this is precisely the mode MC
Dropout needs: calling `model.train()` activates the classifier's `Dropout`
while the backbone's BatchNorm stays frozen, with no special-case code
required here. Verified directly, not assumed: `tests/test_mc_dropout.py`
confirms repeated forward passes in this mode produce genuinely differing
outputs (this stage's own flagged "most common bug in this area" — dropout
silently not actually active).

Uncertainty metric: predictive entropy of the mean distribution across the T
passes (owner-approved 2026-08-30) — the standard MC Dropout formulation,
naturally bounded and directly interpretable.

Operates on cached pooled features (the same 1024-dim eval-view vectors
Stage 9's cache stores and Stage 11/12/18 already reuse) since dropout lives
entirely in the classifier head — no backbone forward pass is needed to
compute uncertainty for an already-cached example. For a genuinely new
image, a caller runs `DenseNet121Head.pooled_features(x)` once (execution-site
per CLAUDE.md section 9/10 — client-side, wherever the image already is) and
passes the result here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.densenet_head import DenseNet121Head


def mc_dropout_predict(
    model: DenseNet121Head,
    features: torch.Tensor,  # (N, 1024) eval-view pooled features
    num_passes: int,
) -> torch.Tensor:
    """T stochastic forward passes with dropout active. Returns raw (not
    averaged) softmax probabilities, shape (T, N, num_classes), so callers
    can compute whatever uncertainty metric they need from the full
    predictive distribution rather than only its mean."""
    model.train()  # activates classifier Dropout; backbone stays eval (see module docstring)
    all_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model.classifier(features)
            all_probs.append(F.softmax(logits, dim=1))
    return torch.stack(all_probs, dim=0)  # (T, N, C)


def predictive_entropy(mean_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy of the MEAN predictive distribution across T passes (not the
    mean of T per-pass entropies) — the standard MC Dropout formulation."""
    return -(mean_probs * torch.log(mean_probs.clamp_min(eps))).sum(dim=-1)


@dataclass
class MCDropoutResult:
    mean_probs: torch.Tensor  # (N, num_classes)
    predicted_class: torch.Tensor  # (N,)
    entropy: torch.Tensor  # (N,)


def compute_mc_dropout_uncertainty(
    model: DenseNet121Head,
    features: torch.Tensor,
    num_passes: int,
) -> MCDropoutResult:
    all_probs = mc_dropout_predict(model, features, num_passes)
    mean_probs = all_probs.mean(dim=0)
    return MCDropoutResult(
        mean_probs=mean_probs,
        predicted_class=mean_probs.argmax(dim=1),
        entropy=predictive_entropy(mean_probs),
    )
