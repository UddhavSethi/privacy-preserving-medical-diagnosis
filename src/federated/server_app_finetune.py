"""ADR-1 GroupNorm fallback -- federated fine-tuning ServerApp, added 2026-08-31.

Pairs with `client_app_finetune.py` (see that module's docstring for the full
rationale). Mirrors the canonical `server_app.py`'s FedAvg orchestration exactly
(same `build_fedavg_strategy`, same per-round summary/MLflow logging shape) --
the only real differences are: the global model is
`DenseNet121Head(fine_tune_last_block=True)`, the transmitted/aggregated state is
`trainable_state_dict()` (classifier + denseblock4 + norm5) instead of
classifier-only, and best-checkpoint selection happens over raw CLAHE-cached
images instead of Stage 9's cached pooled features (which assume a frozen
backbone).

**Best-checkpoint selection, and why it does NOT happen inside `evaluate_fn`
(found live, across two real stalls, not by inspection).** The first version
scored every round against the pooled validation set from inside `evaluate_fn`
and saved whenever a round improved. That stalled for 1h45m+ on round 1 with GPU
at 100% but no forward progress; switching every DataLoader to `num_workers=0`
(ruling out a CUDA-post-fork hazard) did not fix a second attempt, which stalled
2 hours with 0% GPU utilization. Inspecting the stuck run's own Ray worker logs
found the real mechanism: a `ray::ClientAppActor` process -- the SAME actor-pool
process type used for client GPU training -- showed a single task
(`CoreWorker.MarkActorTaskArgsReady`) reported "1 active, 1 running" for ~140
minutes straight. Flower's simulation engine evidently dispatches the
ServerApp's `evaluate_fn` through the same Ray actor pool used for client
training, and reusing a GPU-bound actor for a second heavy CUDA workload
immediately after training hangs -- almost certainly CUDA context/stream
contention on actor reuse. The canonical `server_app.py`'s own `evaluate_fn`
never exercises this: it only touches lightweight cached CPU tensors, no GPU
DataLoader work, so it never stresses actor reuse this way. A standalone,
non-Ray call to the exact same evaluation logic completed in 6.9 minutes,
confirming the code itself is fine -- the hang is specific to running it inside
Flower/Ray's actor-managed context.

**Fix: keep `evaluate_fn` itself cheap (CPU-only, no model construction, no GPU,
no DataLoader) so it can never trigger actor-reuse contention.** It just persists
every round's arrays to its own numbered checkpoint file. All real evaluation
(computing pooled test/val AUROC per round, picking the best) happens in `main()`
AFTER `strategy.start()` returns -- back in the plain driver process, entirely
outside Ray's actor system, exactly matching the standalone call that worked.
Per-round checkpoint files are cleaned up after the best one is identified and
copied to the real output path.
"""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from torch.utils.data import DataLoader

from src.data.raw_image_dataset import RawImageDataset, records_for
from src.data.transforms import build_eval_transform
from src.evaluation.metrics import compute_metrics
from src.federated.serialization import array_record_to_classifier_state, classifier_state_to_array_record
from src.federated.strategy import build_fedavg_strategy
from src.models.densenet_head import DenseNet121Head

HOSPITALS = ["A", "B", "C"]
IMAGE_SIZE = 224

app = ServerApp()


def _auroc(model: DenseNet121Head, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())
    m = compute_metrics(np.array(all_labels), np.array(all_probs))
    return m.auroc if m.auroc == m.auroc else 0.0


def _make_checkpoint_saving_evaluate_fn(round_checkpoints_dir: Path):
    """Deliberately does no GPU work and builds no model -- see module docstring
    for why. Just writes the round's raw trainable state to its own file so
    `_evaluate_saved_rounds` (called post-hoc, outside Ray) can score every round
    afterward."""
    round_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if server_round > 0:
            state = {k: v.cpu() for k, v in array_record_to_classifier_state(arrays).items()}
            torch.save(state, round_checkpoints_dir / f"round_{server_round}.pt")
        return MetricRecord({})

    return evaluate_fn


def _evaluate_saved_rounds(
    round_checkpoints_dir: Path, partition_path: str, clahe_cache_dir: str, num_rounds: int
) -> dict[int, dict[str, float]]:
    """Runs entirely in the plain driver process, after `strategy.start()` has
    returned and every Ray actor from this run is done -- no actor-reuse hazard
    here. Scores every round's saved checkpoint against pooled test AND pooled
    validation (never selecting on test -- see server_app_finetune.py's original
    docstring note, preserved in spirit here)."""
    partition = json.loads(Path(partition_path).read_text())
    eval_transform = build_eval_transform(image_size=IMAGE_SIZE)
    test_loader = DataLoader(
        RawImageDataset(records_for(partition, HOSPITALS, "test"), eval_transform, Path(clahe_cache_dir)),
        batch_size=64, shuffle=False, num_workers=0,
    )
    val_loader = DataLoader(
        RawImageDataset(records_for(partition, HOSPITALS, "val"), eval_transform, Path(clahe_cache_dir)),
        batch_size=64, shuffle=False, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for round_num in range(1, num_rounds + 1):
        ckpt_path = round_checkpoints_dir / f"round_{round_num}.pt"
        if not ckpt_path.exists():
            continue
        model = DenseNet121Head(fine_tune_last_block=True).to(device)
        model.load_trainable_state_dict({k: v.to(device) for k, v in torch.load(ckpt_path, map_location="cpu", weights_only=True).items()})
        test_auroc = _auroc(model, test_loader, device)
        val_auroc = _auroc(model, val_loader, device)
        results[round_num] = {"pooled_test_auroc": test_auroc, "pooled_val_auroc": val_auroc}
        print(f"round {round_num}: pooled_test_auroc={test_auroc:.4f} pooled_val_auroc={val_auroc:.4f}")

    return results


@app.main()
def main(grid: Grid, context: Context) -> None:
    partition_path = context.run_config["partition-path"]
    clahe_cache_dir = context.run_config["clahe-cache-dir"]
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    seed = int(context.run_config["seed"])

    mlflow_uri = context.run_config.get("mlflow-tracking-uri")
    if mlflow_uri:
        mlflow.set_tracking_uri(str(mlflow_uri))
        mlflow.set_experiment(str(context.run_config.get("mlflow-experiment-name", "federated")))
        run_cm = mlflow.start_run(run_name=f"fedavg_finetune_natural_seed{seed}")
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
                    "fine_tune_last_block": True,
                }
            )

        torch.manual_seed(seed)
        global_model = DenseNet121Head(fine_tune_last_block=True)
        initial_arrays = classifier_state_to_array_record(global_model.trainable_state_dict())

        out_path = Path(
            context.run_config.get("output-checkpoint", "outputs/checkpoints/finetuned/fedavg_final.pt")
        )
        round_checkpoints_dir = out_path.parent / f"{out_path.stem}_rounds_tmp"

        strategy = build_fedavg_strategy(fraction_evaluate=fraction_evaluate, min_available_nodes=len(HOSPITALS))
        evaluate_fn = _make_checkpoint_saving_evaluate_fn(round_checkpoints_dir)

        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            train_config=ConfigRecord({"lr": lr}),
            evaluate_fn=evaluate_fn,
        )

        print("\n=== Per-round client-side metrics (training loss, wall clock, payload, val AUROC) ===")
        for round_num in sorted(result.train_metrics_clientapp):
            train_metrics = dict(result.train_metrics_clientapp[round_num])
            client_val_auroc = None
            if round_num in result.evaluate_metrics_clientapp:
                client_val_auroc = dict(result.evaluate_metrics_clientapp[round_num]).get("val_auroc")
            print(f"round {round_num}: client_val_auroc={client_val_auroc} {train_metrics}")
            if mlflow_uri and round_num > 0:
                if client_val_auroc is not None:
                    mlflow.log_metric("client_val_auroc", client_val_auroc, step=round_num)
                for key in ("wall_clock_seconds", "payload_bytes"):
                    if key in train_metrics:
                        mlflow.log_metric(key, train_metrics[key], step=round_num)

        print("\n=== Evaluating every saved round (post-hoc, outside Ray) ===")
        per_round = _evaluate_saved_rounds(round_checkpoints_dir, partition_path, clahe_cache_dir, num_rounds)
        if not per_round:
            raise RuntimeError(f"No round checkpoints found under {round_checkpoints_dir} -- nothing to evaluate.")

        best_round = max(per_round, key=lambda r: per_round[r]["pooled_val_auroc"])
        best_val_auroc = per_round[best_round]["pooled_val_auroc"]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        best_state = torch.load(round_checkpoints_dir / f"round_{best_round}.pt", map_location="cpu", weights_only=True)
        torch.save(best_state, out_path)

        for round_num in per_round:
            marker = " <-- BEST (saved)" if round_num == best_round else ""
            print(f"round {round_num}: {per_round[round_num]}{marker}")
            if mlflow_uri:
                mlflow.log_metric("pooled_test_auroc", per_round[round_num]["pooled_test_auroc"], step=round_num)
                mlflow.log_metric("pooled_val_auroc", per_round[round_num]["pooled_val_auroc"], step=round_num)

        print(f"\nBest checkpoint: round {best_round}, pooled_val_auroc={best_val_auroc:.4f}")
        print(f"Saved to: {out_path}")

        for round_num in per_round:
            (round_checkpoints_dir / f"round_{round_num}.pt").unlink(missing_ok=True)
        if round_checkpoints_dir.exists() and not any(round_checkpoints_dir.iterdir()):
            round_checkpoints_dir.rmdir()

        if mlflow_uri:
            mlflow.log_metric("best_round", best_round)
            mlflow.log_metric("best_pooled_val_auroc", best_val_auroc)
