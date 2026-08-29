"""Stage 11 — local single-hospital baseline (ablation row 1).

Trains an independent classifier per hospital, per partition regime (natural /
balanced, per DG-3's "report both" resolution), over 3 seeds, and reports mean+/-std
test metrics via Stage 10's evaluation module, logged to MLflow.

Usage: uv run python scripts/train_local.py
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from src.evaluation.reporting import aggregate_metrics_over_seeds, save_results
from src.training.trainer import evaluate_classifier, load_hospital_features, train_classifier

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "local_baseline.json"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "local_baseline"

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
    mlflow.set_experiment("local_baseline")

    all_results: dict = {}

    for partition_name, partition_path in PARTITION_FILES.items():
        all_results[partition_name] = {}
        for hospital in HOSPITALS:
            print(f"\n=== {partition_name}/{hospital} ===")
            features = load_hospital_features(partition_path, hospital)
            print(
                f"train={len(features.train_labels)} val={len(features.val_labels)} "
                f"test={len(features.test_labels)}"
            )

            per_seed_test_metrics = []
            for seed in SEEDS:
                with mlflow.start_run(run_name=f"local_{partition_name}_{hospital}_seed{seed}"):
                    mlflow.log_params(
                        {
                            "partition": partition_name,
                            "hospital": hospital,
                            "seed": seed,
                            "num_epochs": NUM_EPOCHS,
                            "lr": LR,
                            "batch_size": BATCH_SIZE,
                            "patience": PATIENCE,
                            "n_train": len(features.train_labels),
                            "n_val": len(features.val_labels),
                            "n_test": len(features.test_labels),
                        }
                    )

                    result = train_classifier(
                        features.train_features,
                        features.train_labels,
                        features.val_features,
                        features.val_labels,
                        seed=seed,
                        num_epochs=NUM_EPOCHS,
                        lr=LR,
                        batch_size=BATCH_SIZE,
                        patience=PATIENCE,
                        device=device,
                    )
                    test_metrics = evaluate_classifier(
                        result["classifier_state"],
                        features.test_features,
                        features.test_labels,
                        device=device,
                    )

                    for epoch_row in result["history"]:
                        mlflow.log_metric("train_loss", epoch_row["train_loss"], step=epoch_row["epoch"])
                        mlflow.log_metric("val_auroc", epoch_row["val_auroc"], step=epoch_row["epoch"])
                    for k, v in test_metrics.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            mlflow.log_metric(f"test_{k}", v)

                    ckpt_path = CHECKPOINT_DIR / f"{partition_name}_{hospital}_seed{seed}.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(result["classifier_state"], ckpt_path)

                    print(
                        f"  seed={seed}: best_val_auroc={result['best_val_auroc']:.4f} "
                        f"test_auroc={test_metrics['auroc']:.4f} epochs={len(result['history'])}"
                    )
                    per_seed_test_metrics.append(test_metrics)

            aggregated = aggregate_metrics_over_seeds(per_seed_test_metrics)
            all_results[partition_name][hospital] = aggregated
            print(
                f"  {partition_name}/{hospital}: AUROC = "
                f"{aggregated['auroc']['mean']:.4f} +/- {aggregated['auroc']['std']:.4f} "
                f"(n={aggregated['auroc']['n_seeds']} seeds)"
            )

    save_results(all_results, RESULTS_PATH)
    print(f"\nResults written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
