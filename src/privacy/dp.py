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
from src.training.trainer import resolve_class_weights


@dataclass(frozen=True)
class DPConfig:
    target_epsilon: float
    target_delta: float
    max_grad_norm: float = 1.0


def dp_training_loss(logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor, mode: str) -> torch.Tensor:
    """Training loss for DP-SGD.

    DP-SGD's sensitivity argument needs every per-sample gradient to be a function of
    that sample alone. `"legacy_weighted_mean"` (the loss the original Stage 14 code and
    the published ablation checkpoints used) is `F.cross_entropy(..., weight=w)` with
    mean reduction, which divides by sum(w[y_j]) over the WHOLE batch — so one sample's
    gradient is rescaled by whichever other samples share its batch (measured: a 60%
    relative change in a sample's per-sample gradient when its batch companions swap
    class; docs/audit_2026-10.md F4). Clipping hides this only for samples that are
    clipped anyway. `"per_sample_mean"` computes mean_j(w[y_j] * ce_j) with a constant
    normalizer, so per-sample gradients are batch-independent as the analysis requires.
    """
    if mode == "legacy_weighted_mean":
        return F.cross_entropy(logits, y, weight=class_weights)
    if mode == "per_sample_mean":
        return (F.cross_entropy(logits, y, reduction="none") * class_weights[y]).mean()
    raise ValueError(f"unknown dp loss mode {mode!r}")


def make_privacy_engine() -> PrivacyEngine:
    """RDP accountant, matching `src/privacy/accounting.py::compute_noise_multiplier`'s
    choice — using different accountant types to calibrate noise vs. report spent
    epsilon would be internally inconsistent."""
    return PrivacyEngine(accountant="rdp")


def _strip_opacus_prefix(state_dict: dict) -> dict:
    """Opacus's GradSampleModule wraps the module and prefixes every state_dict key
    with `_module.` — strip it so the resulting state dict loads cleanly into a plain
    (unwrapped) `DenseNet121Head.classifier`."""
    return {k.removeprefix("_module."): v.detach().cpu().clone() for k, v in state_dict.items()}


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
    loss_mode: str = "legacy_weighted_mean",
    class_weight_mode: str = "local",
    optimizer: str = "adam",
    momentum: float = 0.0,
    device: str = "cpu",
) -> dict:
    """A DP-SGD local training round. `privacy_engine` must be the SAME object across
    every call for a given client across every round it participates in — that's what
    makes the accountant's spent-budget accumulate correctly across rounds instead of
    silently resetting each time. A stateless client (fresh engine per round) must restore
    the accountant history first — see `src/privacy/accountant_state.py` (audit F11).
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model.classifier.to(device)
    class_weights = resolve_class_weights(train_labels, class_weight_mode).to(device)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    elif optimizer == "sgd":
        # Plain (momentum-free by default) SGD: Adam divides by the running RMS of the
        # gradient, which under DP is dominated by injected noise, so every coordinate
        # takes a full-size random step (audit F6). SGD's step scales with the actual
        # noisy gradient, the standard DP-SGD choice.
        opt = torch.optim.SGD(model.classifier.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

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
            x, y = x.to(device), y.to(device)
            out = dp_model(x)
            loss = dp_training_loss(out, y, class_weights, loss_mode)
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
