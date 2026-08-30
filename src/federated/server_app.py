"""Stage 13 — Flower ServerApp: FedAvg strategy over the head-only federated payload.

No privacy layers yet — DP is Stage 14, SecAgg is Stage 15, TLS is Stage 16. This
stage isolates federated-learning correctness on its own, per CLAUDE.md's strict
ordering rule (no DP/SecAgg/TLS before FedAvg is verified working DP-free here
first). Strategy is Flower's built-in FedAvg — CLAUDE.md section 8 prohibits
substituting FedProx, FedBN, or anything else without approval.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from src.evaluation.metrics import compute_metrics
from src.federated.serialization import array_record_to_classifier_state, classifier_state_to_array_record
from src.federated.strategy import build_fedavg_strategy
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_pooled_features

HOSPITALS = ["A", "B", "C"]

app = ServerApp()


def _make_centralized_evaluate_fn(partition_path: str, feature_cache_dir: str):
    """Server-side evaluation on the pooled test set (Stage 12's same evaluation
    protocol) — separate from the client-side federated evaluation (val sets), and
    what makes "does the federated model beat the local/centralized baselines"
    directly comparable to Stages 11/12's numbers."""
    pooled = load_pooled_features(Path(partition_path), HOSPITALS, feature_cache_dir=Path(feature_cache_dir))

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = DenseNet121Head()
        model.classifier.load_state_dict(array_record_to_classifier_state(arrays))
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model.classifier(pooled.test_features), dim=1)[:, 1].numpy()
        m = compute_metrics(pooled.test_labels.numpy(), probs)
        auroc = m.auroc if m.auroc == m.auroc else 0.0
        return MetricRecord({"pooled_test_auroc": auroc})

    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    partition_path = context.run_config["partition-path"]
    feature_cache_dir = context.run_config["feature-cache-dir"]
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    seed = int(context.run_config["seed"])

    torch.manual_seed(seed)
    global_model = DenseNet121Head()
    initial_arrays = classifier_state_to_array_record(global_model.classifier.state_dict())

    strategy = build_fedavg_strategy(fraction_evaluate=fraction_evaluate, min_available_nodes=len(HOSPITALS))

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"lr": lr}),
        evaluate_fn=_make_centralized_evaluate_fn(partition_path, feature_cache_dir),
    )

    print("\n=== Per-round summary ===")
    for round_num in sorted(result.evaluate_metrics_serverapp):
        pooled_auroc = dict(result.evaluate_metrics_serverapp[round_num]).get("pooled_test_auroc")
        val_auroc = None
        if round_num in result.evaluate_metrics_clientapp:
            val_auroc = dict(result.evaluate_metrics_clientapp[round_num]).get("val_auroc")
        print(f"round {round_num}: pooled_test_auroc={pooled_auroc} client_val_auroc={val_auroc}")

    out_path = Path(
        context.run_config.get("output-checkpoint", "outputs/checkpoints/federated/fedavg_final.pt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(array_record_to_classifier_state(result.arrays), out_path)
    print(f"\nFinal global classifier saved: {out_path}")
