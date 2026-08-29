"""Stage 8's real deliverable: proving ADR-1 (frozen backbone + DP-SGD on a small
head) actually works before anything is built on top of it. Failure here is meant to
be cheap; failure discovered at Stage 14 (DP integration) would not be.
"""
import torch
import torch.nn as nn
import pytest
from opacus import GradSampleModule
from opacus.validators import ModuleValidator

from src.models.densenet_head import DenseNet121Head
from src.models.freezing import count_total_parameters, count_trainable_parameters


@pytest.fixture(scope="module")
def model():
    return DenseNet121Head()


def test_backbone_fully_frozen(model):
    assert count_trainable_parameters(model.features) == 0


def test_trainable_parameter_count_in_expected_range(model):
    n = count_trainable_parameters(model)
    assert 1e5 <= n <= 1e6, f"expected ~1e5-1e6 trainable params, got {n}"


def test_total_parameter_count_matches_densenet121(model):
    # DenseNet121 has ~7.0-7.2M parameters total (backbone + head combined) —
    # confirms the backbone itself wasn't accidentally shrunk or duplicated.
    n = count_total_parameters(model)
    assert 6_900_000 <= n <= 7_300_000


def test_forward_pass_output_shape(model):
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


def test_opacus_module_validator_accepts_model(model):
    """The actual ADR-1 gate: if this fails, the GroupNorm fallback requires owner
    approval before adopting — do not silently switch to it."""
    errors = ModuleValidator.validate(model, strict=False)
    assert errors == []
    assert ModuleValidator.is_valid(model)


def test_per_sample_gradients_computed_only_for_head_params():
    model = DenseNet121Head()
    gs_model = GradSampleModule(model)

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    out = gs_model(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()

    head_params_with_grad_sample = 0
    for name, p in gs_model.named_parameters():
        if "features" in name:
            assert not p.requires_grad, f"backbone param {name} should not require grad"
            continue
        if p.requires_grad:
            assert hasattr(p, "grad_sample") and p.grad_sample is not None, (
                f"head param {name} missing per-sample gradients"
            )
            assert p.grad_sample.shape[0] == 4  # batch size
            head_params_with_grad_sample += 1

    assert head_params_with_grad_sample == 4  # 2 Linear layers x (weight, bias)


def test_batchnorm_running_stats_unchanged_after_training_step():
    model = DenseNet121Head()

    bn_before = {
        name: (m.running_mean.clone(), m.running_var.clone())
        for name, m in model.features.named_modules()
        if isinstance(m, nn.BatchNorm2d)
    }
    assert len(bn_before) > 0

    model.train()  # a real training loop calling .train() must not undo the freeze
    gs_model = GradSampleModule(model)
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    loss = nn.functional.cross_entropy(gs_model(x), y)
    loss.backward()
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()

    for name, m in model.features.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            mean_before, var_before = bn_before[name]
            assert torch.equal(mean_before, m.running_mean), f"{name} running_mean changed"
            assert torch.equal(var_before, m.running_var), f"{name} running_var changed"


def test_train_override_keeps_backbone_in_eval_mode():
    model = DenseNet121Head()
    model.train()
    assert model.features.training is False
    assert model.training is True  # the head/top-level module does train normally


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VRAM spike test requires CUDA")
def test_dp_sgd_step_fits_in_4gb_vram():
    """Spike test: one DP-SGD-style step (per-sample gradients) at the intended
    batch size must fit the 4GB RTX 3050 budget (CLAUDE.md section 7)."""
    device = torch.device("cuda")
    model = DenseNet121Head().to(device)
    model.train()
    gs_model = GradSampleModule(model)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    batch_size = 32  # matches conf/data/transforms.yaml's dataloader.batch_size
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)

    loss = nn.functional.cross_entropy(gs_model(x), y)
    loss.backward()
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    assert peak_gb < 4.0, f"peak VRAM {peak_gb:.2f}GB exceeds the 4GB budget"
