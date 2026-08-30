"""Opacus DP-SGD integration (Stage 14, ADR-2): sample-level differential privacy —
per-sample gradient clipping to a fixed norm, calibrated Gaussian noise, and a formal
(RDP) accountant reporting (epsilon, delta).

DP-SGD trains on the deterministic (eval-style) cached feature view only, not the K
augmented views Stage 13's plain FedAvg path cycles through. Deliberate
simplification, not an oversight: DP-SGD's per-sample clipping already regularizes
strongly on its own, and Opacus's Poisson-sampling `DataLoader` (required for its
privacy accounting to be valid) doesn't compose simply with "a different random view
per epoch" the way Stage 13's manual batching loop does.

`secure_mode=False` (Opacus's default, unchanged here): acceptable for an academic
research prototype per Opacus's own guidance — faster, but not cryptographically
secure RNG. Stated honestly as a limitation, not silently assumed away (CLAUDE.md
section 15's culture of naming limitations rather than concealing them).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from src.models.densenet_head import DenseNet121Head
from src.training.trainer import compute_class_weights


@dataclass(frozen=True)
class DPConfig:
    target_epsilon: float
    target_delta: float
    max_grad_norm: float = 1.0


def make_privacy_engine() -> PrivacyEngine:
    """RDP accountant, matching `src/privacy/accounting.py::compute_noise_multiplier`'s
    choice — using different accountant types to calibrate noise vs. report spent
    epsilon would be internally inconsistent."""
    return PrivacyEngine(accountant="rdp")


def _strip_opacus_prefix(state_dict: dict) -> dict:
    """Opacus's GradSampleModule wraps the module and prefixes every state_dict key
    with `_module.` — strip it so the resulting state dict loads cleanly into a plain
    (unwrapped) `DenseNet121Head.classifier`."""
    return {k.removeprefix("_module."): v.clone() for k, v in state_dict.items()}


def train_local_round_dp(
    model: DenseNet121Head,
    train_features: torch.Tensor,  # (N, 1024) — eval-style view only, see module docstring
    train_labels: torch.Tensor,  # (N,)
    seed: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    noise_multiplier: float,
    max_grad_norm: float,
    target_delta: float,
    privacy_engine: PrivacyEngine,
) -> dict:
    """A DP-SGD local training round. `privacy_engine` must be the SAME object across
    every call for a given client across every round it participates in — that's what
    makes the accountant's spent-budget accumulate correctly across rounds instead of
    silently resetting each time.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    class_weights = compute_class_weights(train_labels)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, generator=generator)

    dp_model, dp_opt, dp_loader = privacy_engine.make_private(
        module=model.classifier,
        optimizer=opt,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    dp_model.train()
    total_loss = 0.0
    n_examples = 0
    for _ in range(local_epochs):
        for x, y in dp_loader:
            out = dp_model(x)
            loss = F.cross_entropy(out, y, weight=class_weights)
            dp_opt.zero_grad()
            loss.backward()
            dp_opt.step()
            total_loss += loss.item() * len(y)
            n_examples += len(y)

    try:
        epsilon_spent = privacy_engine.get_epsilon(delta=target_delta)
    except OverflowError:
        # noise_multiplier ~ 0 makes epsilon mathematically infinite (no noise = no
        # privacy); Opacus's accountants can overflow computing that limit rather
        # than returning it, so treat the overflow as its own answer instead of
        # crashing the training call.
        epsilon_spent = float("inf")

    return {
        "classifier_state": _strip_opacus_prefix(dp_model.state_dict()),
        "num_examples": len(train_labels),
        "train_loss": total_loss / max(n_examples, 1),
        "epsilon_spent": epsilon_spent,
    }
