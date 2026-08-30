import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.densenet_head import DenseNet121Head
from src.privacy.dp import make_privacy_engine, train_local_round_dp


def _synthetic_features(n=64, seed=0):
    rng = torch.Generator().manual_seed(seed)
    features = torch.randn(n, 1024, generator=rng) * 10  # scaled up: large natural gradients
    labels = torch.randint(0, 2, (n,), generator=rng)
    return features, labels


def _one_clipped_step(max_grad_norm: float, noise_multiplier: float, seed: int = 0):
    """Runs exactly one Opacus-wrapped step by hand (bypassing Adam's adaptive
    normalization, which otherwise obscures raw-gradient-magnitude comparisons) and
    returns (summed_grad_norm_after_clip, applied_grad_norm_after_noise, batch_size)
    for the first classifier weight tensor."""
    torch.manual_seed(seed)
    features, labels = _synthetic_features(seed=seed)
    model = DenseNet121Head(pretrained=False)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=len(features))  # one full-batch step
    opt = torch.optim.SGD(model.classifier.parameters(), lr=1.0)

    pe = make_privacy_engine()
    dp_model, dp_opt, dp_loader = pe.make_private(
        module=model.classifier, optimizer=opt, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
    )
    dp_model.train()
    x, y = next(iter(dp_loader))
    loss = F.cross_entropy(dp_model(x), y)
    dp_opt.zero_grad()
    loss.backward()
    dp_opt.clip_and_accumulate()

    param = next(dp_model.parameters())
    per_sample_norms = param.grad_sample.flatten(1).norm(dim=1)
    summed_norm = param.summed_grad.norm().item()

    dp_opt.add_noise()
    noisy_norm = param.grad.norm().item()

    return per_sample_norms, summed_norm, noisy_norm, len(features)


def test_gradient_clipping_bounds_the_summed_gradient():
    """Direct, mathematical test of 'gradients are genuinely clipped to the
    configured norm': each per-sample gradient is clipped to <= max_grad_norm before
    summing, so the batch-summed gradient's norm must be <= max_grad_norm *
    batch_size (triangle inequality) — and the natural (pre-clip) per-sample norms
    must actually exceed max_grad_norm, or this would be vacuously true."""
    max_grad_norm = 0.01
    per_sample_norms, summed_norm, _, batch_size = _one_clipped_step(
        max_grad_norm=max_grad_norm, noise_multiplier=0.0
    )

    assert per_sample_norms.max().item() > max_grad_norm * 10  # clipping is non-trivial here
    assert summed_norm <= max_grad_norm * batch_size * 1.01  # small tolerance for fp error


def test_looser_clip_norm_permits_larger_summed_gradient():
    _, summed_norm_tight, _, _ = _one_clipped_step(max_grad_norm=0.01, noise_multiplier=0.0)
    _, summed_norm_loose, _, _ = _one_clipped_step(max_grad_norm=1000.0, noise_multiplier=0.0)
    assert summed_norm_loose > summed_norm_tight


def test_noise_scales_correctly_with_multiplier():
    """Direct test of 'noise scales correctly with the multiplier': the actual noise
    injected (applied gradient norm minus the clipped, noise-free summed gradient)
    must grow with noise_multiplier, at a fixed clip norm."""
    max_grad_norm = 1.0
    _, summed_norm, noisy_norm_low, _ = _one_clipped_step(
        max_grad_norm=max_grad_norm, noise_multiplier=0.5, seed=1
    )
    _, _, noisy_norm_high, _ = _one_clipped_step(
        max_grad_norm=max_grad_norm, noise_multiplier=10.0, seed=1
    )
    noise_added_low = abs(noisy_norm_low - summed_norm)
    noise_added_high = abs(noisy_norm_high - summed_norm)
    assert noise_added_high > noise_added_low


def test_accountant_consumes_budget_monotonically_across_rounds():
    """The stage's own flagged risk: a static epsilon across rounds is the classic
    silent bug. A fresh model per round (matching client_app.py's real usage — Opacus
    refuses to re-attach hooks to an already-wrapped model, which is itself a useful
    guard) but the SAME PrivacyEngine across rounds must show epsilon strictly
    increasing, never resetting."""
    features, labels = _synthetic_features()
    privacy_engine = make_privacy_engine()

    classifier_state = DenseNet121Head(pretrained=False).classifier.state_dict()
    epsilons = []
    for round_num in range(5):
        model = DenseNet121Head(pretrained=False)
        model.classifier.load_state_dict(classifier_state)
        result = train_local_round_dp(
            model, features, labels, seed=round_num, local_epochs=1, lr=0.01, batch_size=16,
            noise_multiplier=1.0, max_grad_norm=1.0, target_delta=1e-5,
            privacy_engine=privacy_engine,
        )
        classifier_state = result["classifier_state"]
        epsilons.append(result["epsilon_spent"])

    assert all(epsilons[i] < epsilons[i + 1] for i in range(len(epsilons) - 1)), epsilons


def test_zero_noise_still_trains_sensibly():
    """'Setting sigma to zero should reproduce Stage 13': not bit-identical (per-sample
    clipping is still active, unlike Stage 13's unclipped plain Adam), but training
    with clipping-only DP-SGD must still sensibly reduce loss on separable data."""
    rng = torch.Generator().manual_seed(0)
    neg = torch.randn(40, 1024, generator=rng) - 3.0
    pos = torch.randn(40, 1024, generator=rng) + 3.0
    features = torch.cat([neg, pos])
    labels = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(40, dtype=torch.long)])

    privacy_engine = make_privacy_engine()
    classifier_state = DenseNet121Head(pretrained=False).classifier.state_dict()
    losses = []
    for round_num in range(10):
        model = DenseNet121Head(pretrained=False)
        model.classifier.load_state_dict(classifier_state)
        result = train_local_round_dp(
            model, features, labels, seed=round_num, local_epochs=1, lr=0.1, batch_size=16,
            noise_multiplier=0.0, max_grad_norm=10.0, target_delta=1e-5,
            privacy_engine=privacy_engine,
        )
        classifier_state = result["classifier_state"]
        losses.append(result["train_loss"])

    assert losses[-1] < losses[0]


def test_classifier_state_loads_cleanly_without_opacus_prefix():
    """Regression guard: Opacus's GradSampleModule prefixes state_dict keys with
    `_module.` — the returned classifier_state must have that stripped so it loads
    directly into a plain (unwrapped) DenseNet121Head.classifier elsewhere in the
    pipeline (evaluate_classifier, the next round's client, etc.)."""
    features, labels = _synthetic_features()
    model = DenseNet121Head(pretrained=False)
    result = train_local_round_dp(
        model, features, labels, seed=0, local_epochs=1, lr=0.01, batch_size=16,
        noise_multiplier=1.0, max_grad_norm=1.0, target_delta=1e-5,
        privacy_engine=make_privacy_engine(),
    )
    assert not any(k.startswith("_module.") for k in result["classifier_state"])

    fresh_model = DenseNet121Head(pretrained=False)
    fresh_model.classifier.load_state_dict(result["classifier_state"])  # must not raise
