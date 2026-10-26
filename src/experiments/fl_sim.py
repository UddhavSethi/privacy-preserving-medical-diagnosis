"""In-process FedAvg simulation over the cached features, for fast controlled experiments.

This is a research harness, not a second client implementation: every client's local
training is a call to the SAME functions the Flower ClientApp uses
(`src.training.trainer.train_local_round`, `src.privacy.dp.train_local_round_dp`), and the
server step is the same sample-count-weighted FedAvg. What it removes is Flower's
process/IPC overhead, so a 20-round run takes seconds (non-DP) to ~2 minutes (DP) instead
of several minutes — which is what makes validation-only hyper-parameter selection and
shadow-model privacy audits affordable on this machine. Fidelity to the Flower path is
checked by `scripts/validate_fl_sim_parity.py` (final weights vs. the saved Flower
checkpoint) and by `tests/test_fl_sim.py`.

Discipline: this module never touches test data. `run_fedavg_sim` evaluates on the
validation split only; callers that need a test number call `evaluate_state` explicitly,
in a separate, clearly-labelled reporting phase.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.metrics import compute_metrics
from src.models.densenet_head import DenseNet121Head
from src.privacy.accounting import compute_noise_multiplier, compute_total_steps
from src.privacy.dp import make_privacy_engine, train_local_round_dp
from src.training.trainer import HospitalFeatures, load_hospital_features, train_local_round

HOSPITALS = ["A", "B", "C"]


@dataclass(frozen=True)
class DPSimConfig:
    target_epsilon: float
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    loss_mode: str = "legacy_weighted_mean"  # see src.privacy.dp.dp_training_loss
    optimizer: str = "adam"  # "adam" (project default) | "sgd"
    momentum: float = 0.0


@dataclass(frozen=True)
class SimConfig:
    partition_path: Path
    seed: int = 42
    rounds: int = 20
    local_epochs: int = 1
    lr: float = 1e-3
    lr_decay: float = 1.0  # per-round multiplicative decay (1.0 = constant, the project default)
    batch_size: int = 32
    class_weight_mode: str = "local"  # "local" (project default) | "none"
    eval_view_only: bool = False  # non-DP only: train on the deterministic view (no augmentation)
    dp: DPSimConfig | None = None
    hospitals: tuple[str, ...] = tuple(HOSPITALS)
    device: str = "cpu"  # "cuda" moves the (eval-view) training features and client training to the GPU
    track_val: bool = True  # False skips per-round validation scoring (shadow-model audits don't need it)


@dataclass
class SimResult:
    history: list[dict] = field(default_factory=list)  # one dict per round: val metrics (pooled + per hospital)
    final_state: dict | None = None
    epsilon_spent: dict[str, float] = field(default_factory=dict)  # hospital -> epsilon at the final round
    noise_multiplier: dict[str, float] = field(default_factory=dict)


class _LightHead(nn.Module):
    """Duck-typed stand-in for `DenseNet121Head` (the trainers only touch `.classifier`):
    same classifier architecture, no backbone, so building one per client per round is free."""

    def __init__(self, reference: nn.Sequential):
        super().__init__()
        self.classifier = copy.deepcopy(reference)


def flower_parity_initial_state(seed: int) -> dict:
    """The exact initial classifier the Flower ServerApp broadcasts for this seed
    (`torch.manual_seed(seed); DenseNet121Head()`), so the harness and the Flower path
    start from identical weights."""
    torch.manual_seed(seed)
    head = DenseNet121Head()
    return {k: v.clone() for k, v in head.classifier.state_dict().items()}


def _reference_classifier() -> nn.Sequential:
    return DenseNet121Head(pretrained=False).classifier


def load_all_hospitals(partition_path: Path, hospitals=tuple(HOSPITALS)) -> dict[str, HospitalFeatures]:
    return {h: load_hospital_features(Path(partition_path), h) for h in hospitals}


def fedavg_aggregate(states: list[dict], weights: list[float]) -> dict:
    total = float(sum(weights))
    out = {}
    for k in states[0]:
        out[k] = sum(s[k].double() * (w / total) for s, w in zip(states, weights)).to(states[0][k].dtype)
    return out


def predict_probs(state: dict, features: torch.Tensor, reference: nn.Sequential | None = None) -> np.ndarray:
    clf = copy.deepcopy(reference if reference is not None else _reference_classifier())
    clf.load_state_dict(state)
    clf.eval()
    with torch.no_grad():
        return F.softmax(clf(features), dim=1)[:, 1].numpy()


def evaluate_state(state: dict, features: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    """Metrics for a classifier state on the given split. Callers choose the split — this
    function is split-agnostic by design; the *caller* is responsible for not using a test
    split to make decisions."""
    return compute_metrics(labels.numpy(), predict_probs(state, features), threshold=threshold).to_dict()


def _val_report(state: dict, data: dict[str, HospitalFeatures], reference: nn.Sequential) -> dict:
    probs = {h: predict_probs(state, d.val_features, reference) for h, d in data.items()}
    y = {h: d.val_labels.numpy() for h, d in data.items()}
    row = {"val_auroc_pooled": compute_metrics(np.concatenate(list(y.values())), np.concatenate(list(probs.values()))).auroc}
    for h in data:
        row[f"val_auroc_{h}"] = compute_metrics(y[h], probs[h]).auroc
    return row


def run_fedavg_sim(
    cfg: SimConfig,
    data: dict[str, HospitalFeatures] | None = None,
    initial_state: dict | None = None,
    verbose: bool = False,
) -> SimResult:
    """FedAvg over `cfg.rounds` rounds with every hospital participating every round
    (the project's configuration), evaluated on VALIDATION data only."""
    data = data if data is not None else load_all_hospitals(cfg.partition_path, cfg.hospitals)
    state = initial_state if initial_state is not None else flower_parity_initial_state(cfg.seed)
    state = {k: v.clone() for k, v in state.items()}
    reference = _reference_classifier()
    result = SimResult()

    # GPU path: only the deterministic view is moved (a few hundred MB for all hospitals), so
    # several worker processes can share a 4 GB card. Augmented multi-view training stays on CPU.
    gpu = cfg.device != "cpu"
    if gpu and cfg.dp is None and not cfg.eval_view_only:
        raise ValueError("device != cpu requires eval_view_only=True for non-DP runs (multi-view features are too large to share the GPU)")
    train_dev = {h: (d.train_features[:, -1:, :].to(cfg.device) if gpu else d.train_features) for h, d in data.items()}
    labels_dev = {h: (d.train_labels.to(cfg.device) if gpu else d.train_labels) for h, d in data.items()}

    engines: dict[str, tuple] = {}
    if cfg.dp is not None:
        for h, d in data.items():
            n = len(d.train_labels)
            steps = compute_total_steps(n, cfg.batch_size, cfg.local_epochs, cfg.rounds)
            sigma = compute_noise_multiplier(cfg.dp.target_epsilon, cfg.dp.target_delta, cfg.batch_size / n, steps)
            engines[h] = (make_privacy_engine(), sigma)
            result.noise_multiplier[h] = sigma

    for rnd in range(1, cfg.rounds + 1):
        lr = cfg.lr * (cfg.lr_decay ** (rnd - 1))
        client_states, client_n = [], []
        for pid, h in enumerate(cfg.hospitals):
            d = data[h]
            model = _LightHead(reference)
            model.classifier.load_state_dict(state)
            seed = cfg.seed + pid  # same per-client seed rule as client_app.py
            if cfg.dp is None:
                out = train_local_round(
                    model.to(cfg.device), train_dev[h], labels_dev[h], seed=seed, local_epochs=cfg.local_epochs, lr=lr,
                    batch_size=cfg.batch_size, class_weight_mode=cfg.class_weight_mode,
                    eval_view_only=cfg.eval_view_only,
                )
            else:
                engine, sigma = engines[h]
                out = train_local_round_dp(
                    model, d.train_features[:, -1, :], d.train_labels, seed=seed, local_epochs=cfg.local_epochs, device=cfg.device,
                    lr=lr, batch_size=cfg.batch_size, noise_multiplier=sigma, max_grad_norm=cfg.dp.max_grad_norm,
                    target_delta=cfg.dp.target_delta, privacy_engine=engine, loss_mode=cfg.dp.loss_mode,
                    class_weight_mode=cfg.class_weight_mode, optimizer=cfg.dp.optimizer, momentum=cfg.dp.momentum,
                )
                result.epsilon_spent[h] = out["epsilon_spent"]
            client_states.append(out["classifier_state"])
            client_n.append(out["num_examples"])
        state = fedavg_aggregate(client_states, client_n)
        row = {"round": rnd, **(_val_report(state, data, reference) if cfg.track_val else {})}
        result.history.append(row)
        if verbose:
            print(f"round {rnd}: " + (f"val_auroc_pooled={row['val_auroc_pooled']:.4f}" if cfg.track_val else "done"))

    result.final_state = state
    return result
