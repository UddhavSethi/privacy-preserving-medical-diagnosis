"""Stage 13 — Flower ServerApp: FedAvg strategy over the head-only federated payload.

No privacy layers yet — DP is Stage 14, SecAgg is Stage 15, TLS is Stage 16. This
stage isolates federated-learning correctness on its own, per CLAUDE.md's strict
ordering rule (no DP/SecAgg/TLS before FedAvg is verified working DP-free here
first). Strategy is Flower's built-in FedAvg — CLAUDE.md section 8 prohibits
substituting FedProx, FedBN, or anything else without approval.

Stage 20/21: MLflow logging (CLAUDE.md section 12 — "a result that is not in
MLflow does not exist") is wired in, but only activates when
`context.run_config["mlflow-tracking-uri"]` is set — Stages 13-17's own
TLS/deployment validation runs never set it and must keep working unmodified.
The tracking URI must be an ABSOLUTE path threaded through config (the same
pattern `partition-path`/`feature-cache-dir` already use): Flower's simulation
runtime executes this ServerApp from an isolated copy of the app
(`~/.flwr/apps/<hash>/`), so a relative sqlite path here would silently create
a disconnected mlruns.db inside that directory instead of the project's real one.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import mlflow
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

    dp_enabled = bool(context.run_config.get("dp-enabled", False))
    if dp_enabled:
        print(
            f"Differential Privacy ENABLED (Stage 14, ADR-2): "
            f"target_epsilon={context.run_config['target-epsilon']}, "
            f"target_delta={context.run_config['target-delta']}, "
            f"max_grad_norm={context.run_config['max-grad-norm']}. "
            f"Per-client noise_multiplier is calibrated to each hospital's own "
            f"dataset size — see client-reported 'noise_multiplier' and "
            f"'epsilon_spent' in the aggregated train metrics below."
        )
    else:
        print("Differential Privacy disabled (Stage 13's plain FedAvg path).")

    mlflow_uri = context.run_config.get("mlflow-tracking-uri")
    if mlflow_uri:
        mlflow.set_tracking_uri(str(mlflow_uri))
        mlflow.set_experiment(str(context.run_config.get("mlflow-experiment-name", "federated")))
        # Derived from config already present, rather than a dedicated
        # mlflow-run-name key — `--run-config` can only override keys already
        # declared in [tool.flwr.app.config] (found via this stage's own
        # smoke test: overriding an undeclared key hard-fails with "Invalid
        # run configuration"), and every campaign run already varies exactly
        # these values.
        regime = Path(partition_path).stem.removeprefix("hospitals_")
        run_label = f"fedavg_{regime}_seed{seed}"
        if dp_enabled:
            run_label += f"_dp_eps{context.run_config.get('target-epsilon')}"
        run_cm = mlflow.start_run(run_name=run_label)
    else:
        run_cm = nullcontext()

    with run_cm:
        if mlflow_uri:
            mlflow.log_params(
                {
                    "partition_path": partition_path,
                    "num_server_rounds": num_rounds,
                    "learning_rate": lr,
                    "fraction_evaluate": fraction_evaluate,
                    "seed": seed,
                    "dp_enabled": dp_enabled,
                    **(
                        {
                            "target_epsilon": context.run_config.get("target-epsilon"),
                            "target_delta": context.run_config.get("target-delta"),
                            "max_grad_norm": context.run_config.get("max-grad-norm"),
                        }
                        if dp_enabled
                        else {}
                    ),
                }
            )

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
            server_metrics = dict(result.evaluate_metrics_serverapp[round_num])
            pooled_auroc = server_metrics.get("pooled_test_auroc")
            val_auroc = None
            client_metrics = {}
            if round_num in result.evaluate_metrics_clientapp:
                client_metrics = dict(result.evaluate_metrics_clientapp[round_num])
                val_auroc = client_metrics.get("val_auroc")
            train_metrics = {}
            if round_num in result.train_metrics_clientapp:
                train_metrics = dict(result.train_metrics_clientapp[round_num])
            print(f"round {round_num}: pooled_test_auroc={pooled_auroc} client_val_auroc={val_auroc}")

            if mlflow_uri and round_num > 0:  # round 0 is the pre-training initial eval
                if pooled_auroc is not None:
                    mlflow.log_metric("pooled_test_auroc", pooled_auroc, step=round_num)
                if val_auroc is not None:
                    mlflow.log_metric("client_val_auroc", val_auroc, step=round_num)
                for key in ("wall_clock_seconds", "payload_bytes", "epsilon_spent", "noise_multiplier"):
                    if key in train_metrics:
                        mlflow.log_metric(key, train_metrics[key], step=round_num)

        out_path = Path(
            context.run_config.get("output-checkpoint", "outputs/checkpoints/federated/fedavg_final.pt")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(array_record_to_classifier_state(result.arrays), out_path)
        print(f"\nFinal global classifier saved: {out_path}")

        if mlflow_uri:
            final_round = max(result.evaluate_metrics_serverapp)
            final_auroc = dict(result.evaluate_metrics_serverapp[final_round]).get("pooled_test_auroc")
            if final_auroc is not None:
                mlflow.log_metric("final_pooled_test_auroc", final_auroc)
