"""Stage 13/14 — Flower ClientApp: FedAvg, with Differential Privacy as a
config-switchable layer (`dp-enabled` in `[tool.flwr.app.config]`, default `false`).

One hospital per simulated node (partition-id 0/1/2 -> A/B/C). Trains only the
classifier — Stage 8's frozen backbone is never serialized or transmitted (ADR-1's
federated payload is head-only) — on Stage 9's cached pooled features.

With DP disabled (default): Stage 13's original path — plain Adam, cycling through
the K augmented + eval views per epoch. Verified working end-to-end (real 20-round
run); untouched by Stage 14's addition, so it stays reproducible.

With DP enabled: Opacus DP-SGD (Stage 14, ADR-2) on the deterministic eval-style view
only (see `src/privacy/dp.py`'s docstring for why). The same `PrivacyEngine` instance
is cached per hospital across every round that hospital participates in — required
for the accountant's spent-epsilon to accumulate correctly rather than resetting each
round (this stage's own flagged risk).

Written once, per ADR-8: this same code runs under both the simulation engine (used
here, for the measured ablation results) and the deployment engine (Stage 17), since
Flower's ClientApp abstraction doesn't fork based on execution mode.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from opacus import PrivacyEngine

from src.evaluation.metrics import compute_metrics
from src.federated.serialization import array_record_to_classifier_state, classifier_state_to_array_record
from src.models.densenet_head import DenseNet121Head
from src.privacy.accounting import compute_noise_multiplier, compute_total_steps
from src.privacy.dp import make_privacy_engine, train_local_round_dp
from src.training.trainer import HospitalFeatures, load_hospital_features, train_local_round

PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}

app = ClientApp()

# Module-level caches: a simulated node's process is reused across rounds within one
# run, so both the loaded features and (when DP is enabled) the PrivacyEngine persist
# across @app.train() calls rather than being rebuilt/reset every round.
_feature_cache: dict[tuple[str, str, str], HospitalFeatures] = {}
_privacy_engine_cache: dict[str, tuple[PrivacyEngine, float]] = {}  # hospital -> (engine, noise_multiplier)


def _get_hospital_features(hospital: str, partition_path: str, feature_cache_dir: str) -> HospitalFeatures:
    key = (hospital, partition_path, feature_cache_dir)
    if key not in _feature_cache:
        _feature_cache[key] = load_hospital_features(
            Path(partition_path), hospital, feature_cache_dir=Path(feature_cache_dir)
        )
    return _feature_cache[key]


def _get_privacy_engine(hospital: str, context: Context, dataset_size: int) -> tuple[PrivacyEngine, float]:
    if hospital not in _privacy_engine_cache:
        total_steps = compute_total_steps(
            dataset_size=dataset_size,
            batch_size=int(context.run_config["batch-size"]),
            local_epochs=int(context.run_config["local-epochs"]),
            num_rounds=int(context.run_config["num-server-rounds"]),
        )
        sample_rate = int(context.run_config["batch-size"]) / dataset_size
        noise_multiplier = compute_noise_multiplier(
            target_epsilon=float(context.run_config["target-epsilon"]),
            target_delta=float(context.run_config["target-delta"]),
            sample_rate=sample_rate,
            total_steps=total_steps,
        )
        _privacy_engine_cache[hospital] = (make_privacy_engine(), noise_multiplier)
    return _privacy_engine_cache[hospital]


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    hospital = PARTITION_TO_HOSPITAL[partition_id]
    features = _get_hospital_features(
        hospital, context.run_config["partition-path"], context.run_config["feature-cache-dir"]
    )
    seed = int(context.run_config["seed"]) + partition_id
    local_epochs = int(context.run_config["local-epochs"])
    lr = float(msg.content["config"]["lr"])
    batch_size = int(context.run_config["batch-size"])

    model = DenseNet121Head()
    model.classifier.load_state_dict(array_record_to_classifier_state(msg.content["arrays"]))

    dp_enabled = bool(context.run_config.get("dp-enabled", False))
    metrics = {}

    if dp_enabled:
        eval_view_features = features.train_features[:, -1, :]  # deterministic view only
        privacy_engine, noise_multiplier = _get_privacy_engine(
            hospital, context, dataset_size=len(features.train_labels)
        )
        target_delta = float(context.run_config["target-delta"])
        result = train_local_round_dp(
            model,
            eval_view_features,
            features.train_labels,
            seed=seed,
            local_epochs=local_epochs,
            lr=lr,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            max_grad_norm=float(context.run_config["max-grad-norm"]),
            target_delta=target_delta,
            privacy_engine=privacy_engine,
        )
        metrics["epsilon_spent"] = result["epsilon_spent"]
        metrics["noise_multiplier"] = noise_multiplier
    else:
        result = train_local_round(
            model,
            features.train_features,
            features.train_labels,
            seed=seed,
            local_epochs=local_epochs,
            lr=lr,
            batch_size=batch_size,
        )

    metrics["train_loss"] = result["train_loss"]
    metrics["num-examples"] = result["num_examples"]

    content = RecordDict(
        {
            "arrays": classifier_state_to_array_record(result["classifier_state"]),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    hospital = PARTITION_TO_HOSPITAL[partition_id]
    features = _get_hospital_features(
        hospital, context.run_config["partition-path"], context.run_config["feature-cache-dir"]
    )

    model = DenseNet121Head()
    model.classifier.load_state_dict(array_record_to_classifier_state(msg.content["arrays"]))
    model.eval()

    with torch.no_grad():
        probs = F.softmax(model.classifier(features.val_features), dim=1)[:, 1].numpy()
    m = compute_metrics(features.val_labels.numpy(), probs)
    auroc = m.auroc if m.auroc == m.auroc else 0.0  # NaN guard (degenerate val split)

    content = RecordDict(
        {"metrics": MetricRecord({"val_auroc": auroc, "num-examples": len(features.val_labels)})}
    )
    return Message(content=content, reply_to=msg)
