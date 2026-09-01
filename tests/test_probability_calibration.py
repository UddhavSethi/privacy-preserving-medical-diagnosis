import torch

from src.uncertainty.probability_calibration import apply_temperature, fit_temperature


def test_apply_temperature_identity_at_one():
    probs = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
    out = apply_temperature(probs, temperature=1.0)
    assert torch.allclose(out, probs, atol=1e-6)


def test_apply_temperature_preserves_ranking():
    probs = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    for T in [0.5, 0.9349, 1.5, 2.0]:
        calibrated = apply_temperature(probs, temperature=T)
        raw_order = probs[:, 1].argsort()
        cal_order = calibrated[:, 1].argsort()
        assert torch.equal(raw_order, cal_order)


def test_apply_temperature_below_one_sharpens():
    probs = torch.tensor([[0.35, 0.65]])
    sharpened = apply_temperature(probs, temperature=0.5)
    assert sharpened[0, 1] > probs[0, 1]


def test_apply_temperature_above_one_softens():
    probs = torch.tensor([[0.2, 0.8]])
    softened = apply_temperature(probs, temperature=2.0)
    assert softened[0, 1] < probs[0, 1]


def test_fit_temperature_recovers_near_one_for_already_calibrated_data():
    torch.manual_seed(0)
    n = 2000
    true_probs = torch.rand(n)
    y_true = (torch.rand(n) < true_probs).long()
    mean_probs = torch.stack([1 - true_probs, true_probs], dim=1)

    T = fit_temperature(mean_probs, y_true)
    assert 0.85 < T < 1.15


def test_fit_temperature_softens_an_overconfident_distribution():
    torch.manual_seed(1)
    n = 2000
    # Genuinely overconfident: predicted prob is always far more extreme than
    # the true generating probability actually justifies.
    true_probs = torch.rand(n) * 0.4 + 0.3  # in [0.3, 0.7], mild true signal
    y_true = (torch.rand(n) < true_probs).long()
    overconfident = true_probs.clamp(0.01, 0.99) ** 3
    overconfident = overconfident / (overconfident + (1 - true_probs).clamp(0.01, 0.99) ** 3)
    mean_probs = torch.stack([1 - overconfident, overconfident], dim=1)

    T = fit_temperature(mean_probs, y_true)
    assert T > 1.0  # should soften an overconfident distribution
