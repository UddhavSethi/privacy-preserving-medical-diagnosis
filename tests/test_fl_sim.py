"""Tests for the in-process FedAvg harness (src/experiments/fl_sim.py)."""
from pathlib import Path

import pytest
import torch

from src.experiments.fl_sim import (
    DPSimConfig, SimConfig, _LightHead, _reference_classifier, evaluate_state, fedavg_aggregate, run_fedavg_sim,
)
from src.training.trainer import HospitalFeatures, train_local_round


def _synthetic_hospitals(n_train=(60, 90, 70), seed=0):
    g = torch.Generator().manual_seed(seed)
    data = {}
    for h, n in zip("ABC", n_train):
        def split(m, views=None):
            shape = (m, views, 1024) if views else (m, 1024)
            y = torch.randint(0, 2, (m,), generator=g)
            x = torch.randn(*shape, generator=g) + (y.float() * 1.5).view(-1, *([1] * (len(shape) - 1)))
            return x, y
        trx, try_ = split(n, views=3)
        vax, vay = split(40)
        tex, tey = split(40)
        data[h] = HospitalFeatures(trx, try_, vax, vay, tex, tey)
    return data


def _init_state():
    torch.manual_seed(0)
    return {k: v.clone() for k, v in _reference_classifier().state_dict().items()}


def _cfg(**kw):
    return SimConfig(partition_path=Path("unused"), rounds=2, batch_size=16, **kw)


def test_fedavg_aggregate_is_sample_weighted_mean():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([3.0, 6.0])}
    out = fedavg_aggregate([a, b], [1, 2])
    assert torch.allclose(out["w"], torch.tensor([2.0, 4.0]))


def test_one_round_equals_manual_fedavg_of_the_shared_client_trainer():
    """The harness adds no training logic of its own: round 1 must equal calling the shared
    client trainer per hospital with the same seeds and averaging by example count."""
    data, init = _synthetic_hospitals(), _init_state()
    cfg = SimConfig(partition_path=Path("unused"), rounds=1, batch_size=16, seed=7)
    got = run_fedavg_sim(cfg, data=data, initial_state=init).final_state
    states, ns = [], []
    for pid, h in enumerate("ABC"):
        m = _LightHead(_reference_classifier())
        m.classifier.load_state_dict(init)
        out = train_local_round(m, data[h].train_features, data[h].train_labels, seed=7 + pid, local_epochs=1, lr=1e-3, batch_size=16)
        states.append(out["classifier_state"]); ns.append(out["num_examples"])
    want = fedavg_aggregate(states, ns)
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-6)


def test_run_is_deterministic_for_a_seed():
    data, init = _synthetic_hospitals(), _init_state()
    a = run_fedavg_sim(_cfg(seed=3), data=data, initial_state=init).final_state
    b = run_fedavg_sim(_cfg(seed=3), data=data, initial_state=init).final_state
    c = run_fedavg_sim(_cfg(seed=4), data=data, initial_state=init).final_state
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_class_weight_and_view_options_change_training_but_default_matches_legacy():
    data, init = _synthetic_hospitals(), _init_state()
    base = run_fedavg_sim(_cfg(seed=1), data=data, initial_state=init).final_state
    explicit = run_fedavg_sim(_cfg(seed=1, class_weight_mode="local", eval_view_only=False), data=data, initial_state=init).final_state
    nocw = run_fedavg_sim(_cfg(seed=1, class_weight_mode="none"), data=data, initial_state=init).final_state
    ev = run_fedavg_sim(_cfg(seed=1, eval_view_only=True), data=data, initial_state=init).final_state
    assert all(torch.equal(base[k], explicit[k]) for k in base)
    assert any(not torch.equal(base[k], nocw[k]) for k in base)
    assert any(not torch.equal(base[k], ev[k]) for k in base)


def test_history_records_validation_metrics_only():
    res = run_fedavg_sim(_cfg(seed=1), data=_synthetic_hospitals(), initial_state=_init_state())
    assert [r["round"] for r in res.history] == [1, 2]
    assert all(k.startswith("val_") or k == "round" for r in res.history for k in r)


def test_dp_path_spends_increasing_epsilon_near_target_and_per_hospital_noise():
    data, init = _synthetic_hospitals(n_train=(120, 160, 140)), _init_state()
    cfg = SimConfig(partition_path=Path("unused"), rounds=3, batch_size=16, seed=1,
                    dp=DPSimConfig(target_epsilon=4.0, loss_mode="per_sample_mean", optimizer="sgd"), lr=0.1)
    res = run_fedavg_sim(cfg, data=data, initial_state=init)
    assert set(res.epsilon_spent) == set("ABC") and set(res.noise_multiplier) == set("ABC")
    for h, eps in res.epsilon_spent.items():
        assert 0 < eps <= 4.0 * 1.05, (h, eps)  # accountant lands at (not above) the target


def test_evaluate_state_returns_metrics_dict():
    d = _synthetic_hospitals()["A"]
    m = evaluate_state(_init_state(), d.test_features, d.test_labels)
    assert 0.0 <= m["auroc"] <= 1.0 and m["n_samples"] == 40
