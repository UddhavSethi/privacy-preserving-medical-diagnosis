"""Regression for the accountant-reset defect (audit F11): the reported epsilon must accumulate across
rounds even when every round gets a brand-new PrivacyEngine (a stateless client process)."""
import torch

from src.models.densenet_head import DenseNet121Head
from src.privacy.accountant_state import apply_history_json, history_to_json
from src.privacy.dp import make_privacy_engine, train_local_round_dp


def _round(engine, state, seed):
    m = DenseNet121Head(pretrained=False)
    m.classifier.load_state_dict(state)
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(96, 1024, generator=g), torch.randint(0, 2, (96,), generator=g)
    out = train_local_round_dp(m, x, y, seed=seed, local_epochs=1, lr=1e-3, batch_size=16, noise_multiplier=1.0,
                               max_grad_norm=1.0, target_delta=1e-5, privacy_engine=engine)
    return out["classifier_state"], out["epsilon_spent"]


def test_a_fresh_engine_each_round_without_persistence_resets_epsilon_the_defect():
    state = DenseNet121Head(pretrained=False).classifier.state_dict()
    eps = []
    for r in range(3):
        state, e = _round(make_privacy_engine(), state, r)  # new engine per round, nothing carried over
        eps.append(e)
    assert max(eps) - min(eps) < 1e-6  # constant: the reported epsilon never grows


def test_persisting_the_history_makes_epsilon_accumulate_exactly_like_one_long_lived_engine():
    init = DenseNet121Head(pretrained=False).classifier.state_dict()

    state, saved, eps_fresh = init, None, []
    for r in range(3):
        engine = make_privacy_engine()
        if saved is not None:
            apply_history_json(engine, saved)
        state, e = _round(engine, state, r)
        saved = history_to_json(engine)
        eps_fresh.append(e)

    long_lived = make_privacy_engine()
    state, eps_long = init, []
    for r in range(3):
        state, e = _round(long_lived, state, r)
        eps_long.append(e)

    assert all(eps_fresh[i] < eps_fresh[i + 1] for i in range(2))
    assert all(abs(a - b) < 1e-9 for a, b in zip(eps_fresh, eps_long))
