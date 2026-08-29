"""Backbone-freezing utilities (ADR-1).

Freezing means two things together, not one: (1) `requires_grad=False` on every
backbone parameter, and (2) every BatchNorm submodule forced into `eval()` mode with
its running statistics frozen. Only both together make frozen BatchNorm a fixed affine
transform — safe for Opacus per-sample gradients — rather than a per-batch statistic
mixer that would silently break the DP guarantee.
"""
from __future__ import annotations

import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    """Set requires_grad=False for every parameter in `module`."""
    for param in module.parameters():
        param.requires_grad = False


def freeze_batchnorm(module: nn.Module) -> None:
    """Force every BatchNorm submodule of `module` into eval() mode. Combined with
    `freeze_module`, this is what makes frozen BatchNorm per-sample-gradient-safe."""
    for submodule in module.modules():
        if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            submodule.eval()


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_total_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
