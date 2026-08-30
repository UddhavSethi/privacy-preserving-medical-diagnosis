"""Stage 13 — Flower ClientApp (FedAvg, no privacy layers yet).

One hospital per simulated node (partition-id 0/1/2 -> A/B/C). Trains only the
classifier — Stage 8's frozen backbone is never serialized or transmitted (ADR-1's
federated payload is head-only) — on Stage 9's cached pooled features.

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

from src.evaluation.metrics import compute_metrics
from src.federated.serialization import array_record_to_classifier_state, classifier_state_to_array_record
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import HospitalFeatures, load_hospital_features, train_local_round

PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}

app = ClientApp()

# Module-level cache: a simulated node's process is reused across rounds within one
# run, so loading each hospital's cached features once (not per-round) matters for
# wall-clock time even though the features themselves are already fast to load.
_feature_cache: dict[tuple[str, str, str], HospitalFeatures] = {}


def _get_hospital_features(hospital: str, partition_path: str, feature_cache_dir: str) -> HospitalFeatures:
    key = (hospital, partition_path, feature_cache_dir)
    if key not in _feature_cache:
        _feature_cache[key] = load_hospital_features(
            Path(partition_path), hospital, feature_cache_dir=Path(feature_cache_dir)
        )
    return _feature_cache[key]


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    hospital = PARTITION_TO_HOSPITAL[partition_id]
    features = _get_hospital_features(
        hospital, context.run_config["partition-path"], context.run_config["feature-cache-dir"]
    )

    model = DenseNet121Head()
    model.classifier.load_state_dict(array_record_to_classifier_state(msg.content["arrays"]))

    result = train_local_round(
        model,
        features.train_features,
        features.train_labels,
        seed=int(context.run_config["seed"]) + partition_id,
        local_epochs=int(context.run_config["local-epochs"]),
        lr=float(msg.content["config"]["lr"]),
        batch_size=int(context.run_config["batch-size"]),
    )

    content = RecordDict(
        {
            "arrays": classifier_state_to_array_record(result["classifier_state"]),
            "metrics": MetricRecord(
                {"train_loss": result["train_loss"], "num-examples": result["num_examples"]}
            ),
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
