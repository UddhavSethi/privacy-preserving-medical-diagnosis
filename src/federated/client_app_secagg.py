"""Stage 15 — Flower ClientApp for the SecAgg+ ablation row (ablation row 4:
FedAvg + Secure Aggregation, no DP — see `server_app_secagg.py`'s docstring).

Why this is a separate module from `client_app.py`, not a runtime branch in it:
Flower's client-side SecAgg+ modifier (`flwr.client.mod.secaggplus_mod`) drives a
four-stage handshake (setup / share keys / collect masked vectors / unmask) whose
messages are built and consumed through Flower's legacy `NumPyClient` compat glue
(`flwr.compat`), which packs/unpacks `RecordDict`s under `fitins.parameters` /
`fitres.parameters` keys — a different wire format from `client_app.py`'s
Stage 13/14 Message-API convention (`msg.content["arrays"]`,
`RecordDict({"arrays": ..., "metrics": ...})`). A `ClientApp` can only be built
with `client_fn` (legacy) or the new `@app.train()`/`@app.evaluate()` decorators,
never both at once (see `ClientApp.__init__`) — so there's no way to make this a
config flag inside the existing decorator-based `client_app.py`. Verified against
flwr==1.35.0's actual installed source (not memory — ADR-5) and cross-checked
against Flower's own `examples/flower-secure-aggregation` reference app, which
uses this exact `NumPyClient` + `mods=[secaggplus_mod]` pattern.

Trains with Stage 13's plain `train_local_round` (no DP — that's row 5/Stage 14,
row 6 is the eventual full-system combination) on the same cached pooled
features, over the same head-only classifier (ADR-1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from src.evaluation.metrics import compute_metrics
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import HospitalFeatures, load_hospital_features, train_local_round

PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}


def _state_to_ndarrays(state: dict) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state.values()]


def _ndarrays_to_state(ndarrays: list[np.ndarray], reference_keys: list[str]) -> dict:
    return {k: torch.tensor(arr) for k, arr in zip(reference_keys, ndarrays, strict=True)}


class SecAggClient(NumPyClient):
    """Plain local training, wrapped by `secaggplus_mod` (registered on the
    `ClientApp`, not here) for the masking/unmasking handshake."""

    def __init__(
        self,
        features: HospitalFeatures,
        seed: int,
        local_epochs: int,
        lr: float,
        batch_size: int,
    ) -> None:
        self.features = features
        self.seed = seed
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = DenseNet121Head()

    def fit(self, parameters, config):
        reference_keys = list(self.model.classifier.state_dict().keys())
        self.model.classifier.load_state_dict(_ndarrays_to_state(parameters, reference_keys))
        result = train_local_round(
            self.model,
            self.features.train_features,
            self.features.train_labels,
            seed=self.seed,
            local_epochs=self.local_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
        )
        return (
            _state_to_ndarrays(result["classifier_state"]),
            result["num_examples"],
            {"train_loss": result["train_loss"]},
        )

    def evaluate(self, parameters, config):
        reference_keys = list(self.model.classifier.state_dict().keys())
        self.model.classifier.load_state_dict(_ndarrays_to_state(parameters, reference_keys))
        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model.classifier(self.features.val_features), dim=1)[:, 1].numpy()
        m = compute_metrics(self.features.val_labels.numpy(), probs)
        auroc = m.auroc if m.auroc == m.auroc else 0.0  # NaN guard
        return float(1.0 - auroc), len(self.features.val_labels), {"val_auroc": auroc}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    hospital = PARTITION_TO_HOSPITAL[partition_id]
    features = load_hospital_features(
        Path(context.run_config["partition-path"]),
        hospital,
        feature_cache_dir=Path(context.run_config["feature-cache-dir"]),
    )
    seed = int(context.run_config["seed"]) + partition_id
    return SecAggClient(
        features,
        seed=seed,
        local_epochs=int(context.run_config["local-epochs"]),
        lr=float(context.run_config["learning-rate"]),
        batch_size=int(context.run_config["batch-size"]),
    ).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
