"""Stage 12 — centralized pooled baseline (ablation row 2).

Pools all hospitals' training data together (per partition regime) and trains under
the identical protocol as Stage 11 (same architecture, hyperparameters, seeds,
class-imbalance handling) — this is the privacy-free ceiling every later
federated/private result gets compared against.

Reports two things per partition regime, per Stage 12's own flagged risk that this
row must use exactly the same evaluation set and protocol as every other row:
  - **Primary number**: AUROC on the *pooled* test set (all hospitals' test data
    combined) — the fair global comparison point for ablation row 2.
  - **Per-hospital breakdown**: the same centralized model evaluated separately on
    each hospital's own test set — the same test sets Stage 11's local models used,
    which is what makes the "centralized should at least match local" sanity check
    (this stage's own testing criterion) a fair comparison rather than an
    apples-to-oranges one.

Usage: uv run python scripts/train_centralized.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from src.evaluation.reporting import aggregate_metrics_over_seeds, save_results
from src.training.trainer import (
    evaluate_classifier,
    load_hospital_features,
    load_pooled_features,
    train_classifier,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "centralized_baseline.json"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "centralized_baseline"
LOCAL_RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "local_baseline.json"

HOSPITALS = ["A", "B", "C"]
PARTITION_FILES = {
    "natural": PARTITIONS_DIR / "hospitals_natural.json",
    "balanced": PARTITIONS_DIR / "hospitals_natural_balanced.json",
}
SEEDS = [42, 123, 2024]

NUM_EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 32
PATIENCE = 5


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = OmegaConf.load(REPO_ROOT / "conf" / "config.yaml")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("centralized_baseline")

    local_results = None
    if LOCAL_RESULTS_PATH.exists():
        local_results = json.loads(LOCAL_RESULTS_PATH.read_text())

    all_results: dict = {}

    for partition_name, partition_path in PARTITION_FILES.items():
        print(f"\n=== {partition_name} (pooled A+B+C) ===")
        pooled = load_pooled_features(partition_path, HOSPITALS)
        per_hospital_test = {
            h: load_hospital_features(partition_path, h) for h in HOSPITALS
        }
        print(
            f"train={len(pooled.train_labels)} val={len(pooled.val_labels)} "
            f"test={len(pooled.test_labels)} (pooled)"
        )

        pooled_test_metrics_per_seed = []
        per_hospital_metrics_per_seed = {h: [] for h in HOSPITALS}

        for seed in SEEDS:
            with mlflow.start_run(run_name=f"centralized_{partition_name}_seed{seed}"):
                mlflow.log_params(
                    {
                        "partition": partition_name,
                        "seed": seed,
                        "num_epochs": NUM_EPOCHS,
                        "lr": LR,
                        "batch_size": BATCH_SIZE,
                        "patience": PATIENCE,
                        "n_train": len(pooled.train_labels),
                        "n_val": len(pooled.val_labels),
                        "n_test": len(pooled.test_labels),
                    }
                )

                result = train_classifier(
                    pooled.train_features,
                    pooled.train_labels,
                    pooled.val_features,
                    pooled.val_labels,
                    seed=seed,
                    num_epochs=NUM_EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    patience=PATIENCE,
                    device=device,
                )

                pooled_test_metrics = evaluate_classifier(
                    result["classifier_state"], pooled.test_features, pooled.test_labels, device=device
                )
                for k, v in pooled_test_metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        mlflow.log_metric(f"pooled_test_{k}", v)
                pooled_test_metrics_per_seed.append(pooled_test_metrics)

                for h in HOSPITALS:
                    hm = evaluate_classifier(
                        result["classifier_state"],
                        per_hospital_test[h].test_features,
                        per_hospital_test[h].test_labels,
                        device=device,
                    )
                    mlflow.log_metric(f"hospital_{h}_test_auroc", hm["auroc"])
                    per_hospital_metrics_per_seed[h].append(hm)

                ckpt_path = CHECKPOINT_DIR / f"{partition_name}_seed{seed}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(result["classifier_state"], ckpt_path)

                print(
                    f"  seed={seed}: pooled_test_auroc={pooled_test_metrics['auroc']:.4f} "
                    + " ".join(f"{h}={per_hospital_metrics_per_seed[h][-1]['auroc']:.4f}" for h in HOSPITALS)
                )

        pooled_agg = aggregate_metrics_over_seeds(pooled_test_metrics_per_seed)
        per_hospital_agg = {h: aggregate_metrics_over_seeds(v) for h, v in per_hospital_metrics_per_seed.items()}

        print(
            f"\n  {partition_name} pooled: AUROC = {pooled_agg['auroc']['mean']:.4f} "
            f"+/- {pooled_agg['auroc']['std']:.4f}"
        )
        for h in HOSPITALS:
            centralized_auroc = per_hospital_agg[h]["auroc"]["mean"]
            line = f"  {partition_name}/{h} (centralized model): AUROC = {centralized_auroc:.4f}"
            if local_results and partition_name in local_results and h in local_results[partition_name]:
                local_auroc = local_results[partition_name][h]["auroc"]["mean"]
                flag = "OK" if centralized_auroc >= local_auroc - 0.02 else "BELOW LOCAL — investigate"
                line += f"  (local baseline was {local_auroc:.4f}) [{flag}]"
            print(line)

        all_results[partition_name] = {
            "pooled_test": pooled_agg,
            "per_hospital_test": per_hospital_agg,
        }

    save_results(all_results, RESULTS_PATH)
    print(f"\nResults written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
