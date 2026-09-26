"""Stage 15 (+ ablation row 6 extension) — Flower ClientApp for the SecAgg+
app pair. Row 4 (FedAvg + SecAgg, no DP) is `dp-enabled=false` (this pyproject
config's default); row 6 (full system: FedAvg + SecAgg + DP + TLS/auth) is the
same app pair run with `dp-enabled=true` plus the deployment engine's TLS/auth
flags (ADR-4) — see `docs/SESSION_STATE.md`'s row-6 note for the exact
invocation. TLS/auth itself needs no code here (`security.py`'s docstring:
it's a pure CLI/deployment-runtime concern), so this module's only job in the
row-6 extension is adding the DP-SGD branch alongside the existing SecAgg
wrapping.

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

DP-SGD branch mirrors `client_app.py`'s own `dp-enabled` path exactly (same
`train_local_round_dp`, same deterministic eval-view-only features, same
per-hospital cached `PrivacyEngine` so the accountant's spent epsilon
accumulates across rounds rather than resetting) — see `src/privacy/dp.py`'s
docstring for why DP trains on the eval view only.

`_resolve_config` (ported from `client_app.py`): this app pair never needed
node-config overrides while it only ran in simulation (Stage 15), but the
Docker Compose deployment (Stage 17) gives each hospital container its own
`--node-config "feature-cache-dir=... partition-path=..."` pointing at an
isolated per-hospital data directory — without this override, every hospital
container would silently fall back to the run-level (pooled) path instead of
its own isolated shard, breaking the per-hospital data-isolation invariant.
Added here specifically for the row-6 deployment extension.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from opacus import PrivacyEngine

from src.evaluation.metrics import compute_metrics
from src.evaluation.overhead import classifier_payload_size_bytes, measure_wall_clock
from src.models.densenet_head import DenseNet121Head
from src.privacy.accounting import compute_noise_multiplier, compute_total_steps
from src.privacy.dp import make_privacy_engine, train_local_round_dp
from src.training.trainer import HospitalFeatures, load_hospital_features, train_local_round

PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}


def _resolve_config(context: Context, key: str) -> str:
    if key in context.node_config:
        return str(context.node_config[key])
    return str(context.run_config[key])


def _state_to_ndarrays(state: dict) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state.values()]


def _ndarrays_to_state(ndarrays: list[np.ndarray], reference_keys: list[str]) -> dict:
    return {k: torch.tensor(arr) for k, arr in zip(reference_keys, ndarrays, strict=True)}


class SecAggClient(NumPyClient):
    """Local training, wrapped by `secaggplus_mod` (registered on the
    `ClientApp`, not here) for the masking/unmasking handshake. Plain
    (no-DP) training when `dp_enabled=False` (ablation row 4); Opacus
    DP-SGD when `dp_enabled=True` (ablation row 6)."""

    def __init__(
        self,
        features: HospitalFeatures,
        seed: int,
        local_epochs: int,
        lr: float,
        batch_size: int,
        dp_enabled: bool = False,
        target_epsilon: float = 4.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        num_server_rounds: int = 1,
    ) -> None:
        self.features = features
        self.seed = seed
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = DenseNet121Head()
        self.dp_enabled = dp_enabled
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.num_server_rounds = num_server_rounds
        # Cached lazily and reused across every `fit()` call this same
        # long-lived client process makes — required for the RDP accountant's
        # spent-epsilon to accumulate correctly across rounds (same
        # requirement as client_app.py's module-level cache; here an instance
        # attribute suffices since one SecAggClient instance IS the one
        # long-lived process, unlike the Message-API's stateless @app.train()).
        self._privacy_engine: PrivacyEngine | None = None
        self._noise_multiplier: float | None = None

    def _get_privacy_engine(self) -> tuple[PrivacyEngine, float]:
        if self._privacy_engine is None:
            dataset_size = len(self.features.train_labels)
            total_steps = compute_total_steps(
                dataset_size=dataset_size,
                batch_size=self.batch_size,
                local_epochs=self.local_epochs,
                num_rounds=self.num_server_rounds,
            )
            sample_rate = self.batch_size / dataset_size
            self._noise_multiplier = compute_noise_multiplier(
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                sample_rate=sample_rate,
                total_steps=total_steps,
            )
            self._privacy_engine = make_privacy_engine()
        return self._privacy_engine, self._noise_multiplier

    def fit(self, parameters, config):
        reference_keys = list(self.model.classifier.state_dict().keys())
        self.model.classifier.load_state_dict(_ndarrays_to_state(parameters, reference_keys))
        # Stage 20: local-training-only wall-clock (mirrors client_app.py's
        # instrumentation) — the SecAgg masking/quantization/multi-stage
        # handshake itself happens OUTSIDE this call, inside Flower's own
        # secaggplus_mod wrapping it, so this number isolates compute
        # overhead from this stage's own SecAgg-protocol overhead (measured
        # instead by comparing whole-round wall-clock against the plain
        # FedAvg app, per this stage's own note on that limitation).
        metrics: dict = {}
        with measure_wall_clock() as timing:
            if self.dp_enabled:
                eval_view_features = self.features.train_features[:, -1, :]  # deterministic view only
                privacy_engine, noise_multiplier = self._get_privacy_engine()
                result = train_local_round_dp(
                    self.model,
                    eval_view_features,
                    self.features.train_labels,
                    seed=self.seed,
                    local_epochs=self.local_epochs,
                    lr=self.lr,
                    batch_size=self.batch_size,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                    target_delta=self.target_delta,
                    privacy_engine=privacy_engine,
                )
                metrics["epsilon_spent"] = result["epsilon_spent"]
                metrics["noise_multiplier"] = noise_multiplier
            else:
                result = train_local_round(
                    self.model,
                    self.features.train_features,
                    self.features.train_labels,
                    seed=self.seed,
                    local_epochs=self.local_epochs,
                    lr=self.lr,
                    batch_size=self.batch_size,
                )
        metrics["train_loss"] = result["train_loss"]
        metrics["wall_clock_seconds"] = timing.wall_clock_seconds
        metrics["payload_bytes"] = classifier_payload_size_bytes(result["classifier_state"])
        return (
            _state_to_ndarrays(result["classifier_state"]),
            result["num_examples"],
            metrics,
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
        Path(_resolve_config(context, "partition-path")),
        hospital,
        feature_cache_dir=Path(_resolve_config(context, "feature-cache-dir")),
    )
    seed = int(context.run_config["seed"]) + partition_id
    return SecAggClient(
        features,
        seed=seed,
        local_epochs=int(context.run_config["local-epochs"]),
        lr=float(context.run_config["learning-rate"]),
        batch_size=int(context.run_config["batch-size"]),
        dp_enabled=bool(context.run_config.get("dp-enabled", False)),
        target_epsilon=float(context.run_config.get("target-epsilon", 4.0)),
        target_delta=float(context.run_config.get("target-delta", 1e-5)),
        max_grad_norm=float(context.run_config.get("max-grad-norm", 1.0)),
        num_server_rounds=int(context.run_config["num-server-rounds"]),
    ).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
