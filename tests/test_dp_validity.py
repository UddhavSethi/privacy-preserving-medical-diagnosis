"""Validity tests for the DP implementation (audit follow-up, docs/audit_2026-10.md F4/F8).

These test the properties a (epsilon, delta) claim rests on, not just that code runs:
  * per-sample gradients are functions of that sample alone (sensitivity argument)
  * injected noise has standard deviation sigma * C
  * the accountant's epsilon agrees with independent computations
"""
import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import PRVAccountant, RDPAccountant
from scipy.optimize import brentq
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset

from src.models.densenet_head import DenseNet121Head
from src.privacy.dp import dp_training_loss, make_privacy_engine

WEIGHTS = torch.tensor([0.6, 1.9])  # class weights of a 73%-pneumonia hospital


def _per_sample_grad_of_first_sample(others_y: int, mode: str) -> torch.Tensor:
    torch.manual_seed(0)
    base = DenseNet121Head(pretrained=False).classifier
    for m in base.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0  # keep train mode (Opacus hooks need it) but remove dropout randomness
    x0, y0 = torch.randn(1, 1024), torch.tensor([1])
    ox = torch.randn(7, 1024)
    gm = GradSampleModule(copy.deepcopy(base), loss_reduction="mean")
    gm.train()
    x, y = torch.cat([x0, ox]), torch.cat([y0, torch.full((7,), others_y)])
    dp_training_loss(gm(x), y, WEIGHTS, mode).backward()
    return torch.cat([p.grad_sample[0].flatten() for p in gm.parameters()]) * len(y)


def test_legacy_weighted_mean_loss_makes_per_sample_gradient_depend_on_batch_companions():
    """Documents the defect: with F.cross_entropy(weight=...), mean reduction divides by the
    batch's total weight, so sample 0's gradient changes when its companions' classes change."""
    a = _per_sample_grad_of_first_sample(0, "legacy_weighted_mean")
    b = _per_sample_grad_of_first_sample(1, "legacy_weighted_mean")
    assert (a - b).norm() / a.norm() > 0.1


def test_per_sample_mean_loss_gives_batch_independent_per_sample_gradients():
    a = _per_sample_grad_of_first_sample(0, "per_sample_mean")
    b = _per_sample_grad_of_first_sample(1, "per_sample_mean")
    assert (a - b).norm() / a.norm() < 1e-5


def test_unknown_loss_mode_raises():
    with pytest.raises(ValueError):
        dp_training_loss(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), WEIGHTS, "nope")


@pytest.mark.parametrize("sigma,clip", [(1.0, 1.0), (2.5, 0.5)])
def test_injected_noise_std_equals_sigma_times_clip(sigma, clip):
    """Opacus adds N(0, (sigma*C)^2) to the summed clipped gradient; verify on 262k coordinates."""
    torch.manual_seed(0)
    x, y = torch.randn(64, 1024) * 3, torch.randint(0, 2, (64,))
    model = DenseNet121Head(pretrained=False)
    opt = torch.optim.SGD(model.classifier.parameters(), lr=1.0)
    dm, do, dl = make_privacy_engine().make_private(
        module=model.classifier, optimizer=opt, data_loader=DataLoader(TensorDataset(x, y), batch_size=64),
        noise_multiplier=sigma, max_grad_norm=clip, poisson_sampling=False,
    )
    dm.train()
    xb, yb = next(iter(dl))
    F.cross_entropy(dm(xb), yb).backward()
    do.clip_and_accumulate()
    p = next(dm.parameters())
    summed = p.summed_grad.clone()
    do.add_noise()
    noise = p.grad - summed
    assert noise.std().item() == pytest.approx(sigma * clip, rel=0.02)
    assert abs(noise.mean().item()) < 0.01 * sigma * clip + 0.005


def test_clipped_neighbouring_batches_differ_by_at_most_clip_norm():
    """Sensitivity: removing one sample changes the summed clipped gradient by at most C."""
    torch.manual_seed(1)
    base = DenseNet121Head(pretrained=False).classifier
    for m in base.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    x, y = torch.randn(9, 1024) * 5, torch.randint(0, 2, (9,))
    clip = 1.0

    def summed_clipped(xs, ys):
        gm = GradSampleModule(copy.deepcopy(base), loss_reduction="sum")  # sum: no 1/B anywhere
        gm.train()
        F.cross_entropy(gm(xs), ys, reduction="sum").backward()
        flat = torch.cat([p.grad_sample.flatten(1) for p in gm.parameters()], dim=1)
        scale = (clip / (flat.norm(dim=1) + 1e-6)).clamp(max=1.0)
        return (flat * scale[:, None]).sum(0)

    full, minus_one = summed_clipped(x, y), summed_clipped(x[1:], y[1:])
    assert (full - minus_one).norm().item() <= clip * (1 + 1e-4)


def _eps_rdp(sigma, q, steps, delta):
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    return acc.get_epsilon(delta)


def _eps_prv(sigma, q, steps, delta):
    acc = PRVAccountant()
    acc.history = [(sigma, q, steps)]
    return acc.get_epsilon(delta)


def _eps_exact_gaussian(sigma, steps, delta):
    """Exact (eps, delta) of the T-fold composition of the un-subsampled Gaussian mechanism:
    a Gaussian with noise sigma/sqrt(T) (Balle & Wang 2018, analytic Gaussian mechanism)."""
    s = sigma / math.sqrt(steps)
    f = lambda e: norm.cdf(-e * s + 1 / (2 * s)) - math.exp(e) * norm.cdf(-e * s - 1 / (2 * s)) - delta
    return brentq(f, 0, 500)


def test_rdp_accountant_is_a_valid_but_not_vacuous_upper_bound_on_exact_gaussian_epsilon():
    exact = _eps_exact_gaussian(5.0, 50, 1e-5)
    rdp = _eps_rdp(5.0, 1.0, 50, 1e-5)
    assert rdp >= exact  # an upper bound must not undercut the truth
    assert rdp <= 1.2 * exact  # and must be reasonably tight


def test_rdp_and_prv_accountants_agree_on_project_like_setting():
    """Hospital-A-like setting: sample rate ~ 32/4180, 2600 steps. PRV is the tighter accountant;
    the project's RDP epsilon must be at or above it and within 25%."""
    rdp, prv = _eps_rdp(0.9, 0.00766, 2600, 1e-5), _eps_prv(0.9, 0.00766, 2600, 1e-5)
    assert prv <= rdp <= 1.25 * prv


def test_epsilon_is_monotone_in_noise_and_steps():
    assert _eps_rdp(1.5, 0.01, 1000, 1e-5) < _eps_rdp(1.0, 0.01, 1000, 1e-5)
    assert _eps_rdp(1.0, 0.01, 500, 1e-5) < _eps_rdp(1.0, 0.01, 1000, 1e-5)
