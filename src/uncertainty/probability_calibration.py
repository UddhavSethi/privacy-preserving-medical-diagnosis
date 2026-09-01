"""Probability calibration (temperature scaling), added 2026-09-01 following a
real owner-directed investigation into round 9's fine-tuned checkpoint (ADR-1
GroupNorm fallback, docs/adr1_groupnorm_fallback.md sec. 10): does the "100%
confidence" complaint indicate the underlying probability is genuinely
overconfident, or was it a display artifact?

Measured, not assumed: `src/evaluation/calibration.py` (OPT-1) already measures
ECE/Brier but this project had no calibration *correction* anywhere. Fitting
temperature scaling (Guo et al. 2017) on round 9's real validation set found
T=0.9349 -- ECE moved 0.0155 -> 0.0129, both already in the "well calibrated"
range. Since T < 1 *sharpens* the distribution (raises confidence) rather than
softening it, this is the opposite correction the "near-1.0 is misleading"
premise assumed: MC Dropout's own T-pass averaging already calibrates this
checkpoint reasonably well, and the real source of the "100%" complaint was a
separate display-rounding bug in `app/components.py::confidence_meter` (fixed
2026-09-01, independent of this module). Temperature scaling is still wired in
here because the small improvement it does measure is real, not because it
addresses overconfidence -- see docs/adr1_groupnorm_fallback.md for the full
investigation and numbers.

Operates on MC Dropout's *mean* predictive distribution (`mean_probs`, T=20
passes already averaged) rather than a single raw logit, since that mean
distribution is what the deployed app actually computes and thresholds
end-to-end (app/inference.py) -- calibrating anything else would calibrate a
distribution the app never uses. `log(mean_probs)` is treated as a pseudo-logit,
a standard practical adaptation of temperature scaling for an already-averaged
probability vector.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-8


def fit_temperature(mean_probs: torch.Tensor, y_true: torch.Tensor, max_iter: int = 200) -> float:
    """Fits a single scalar temperature T minimizing cross-entropy NLL on the
    given (mean_probs, y_true) pairs via LBFGS, matching Guo et al. 2017's
    standard method. Optimized in log-space so T is always > 0. Intended to be
    fit once on a validation set and reused as a fixed constant at inference
    (never fit on live data, and never on the test set -- see the module
    docstring's investigation)."""
    pseudo_logits = torch.log(mean_probs.clamp_min(_EPS))
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(pseudo_logits / T, y_true)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_T).item())


def apply_temperature(mean_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Applies a fitted (or default 1.0 = no-op) temperature to a predictive
    distribution. Monotonic in the positive-class margin, so this never changes
    which class is favored at any given decision threshold or AUROC -- it only
    reshapes how confident the displayed probability is."""
    pseudo_logits = torch.log(mean_probs.clamp_min(_EPS))
    return F.softmax(pseudo_logits / temperature, dim=-1)
