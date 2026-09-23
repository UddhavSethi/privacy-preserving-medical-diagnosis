"""ADR-1 GroupNorm fallback -- local per-hospital fine-tuned baseline (ablation
row 1), added 2026-09-21 as part of scaling the fine-tuning pilot into a proper
3-seed campaign (owner-approved scope: rows 1-3, natural regime -- see CLAUDE.md's
pending decision 3).

No fine-tuned version of this row existed before this script: `scripts/
train_local.py` (Stage 11, the frozen-backbone version) trains off Stage 9's
cached pooled features, which are a frozen-backbone-only artifact and invalid once
`denseblock4`/`norm5` become trainable (same reason `train_centralized_finetune.py`
already had to read raw CLAHE-cached images instead of the feature cache).

Trains one `DenseNet121Head(fine_tune_last_block=True)` per hospital per seed,
reusing `train_centralized_finetune.py`'s exact training/eval protocol and
hyperparameters (differential head/backbone LR, 8-epoch budget, patience 3) --
only the record set changes, from pooled (all 3 hospitals) to one hospital's own
train/val/test split. Natural regime only, matching this campaign's scope
(balanced regime and DP/SecAgg rows are explicitly out of scope).

Usage: uv run python scripts/train_local_finetune.py
"""
from __future__ import annotations

from pathlib import Path

import json
import mlflow
import torch
from omegaconf import OmegaConf

from src.evaluation.reporting import aggregate_metrics_over_seeds, save_results
from train_centralized_finetune import (
    HOSPITALS,
    LABEL_TO_INDEX,
    PARTITION_PATH,
    SEEDS,
    _records_for,
    _train_one_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "local_finetune_multiseed.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = OmegaConf.load(REPO_ROOT / "conf" / "config.yaml")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("local_finetune")

    partition = json.loads(PARTITION_PATH.read_text())

    all_results: dict = {}
    for hospital in HOSPITALS:
        print(f"\n=== hospital {hospital} (fine-tuned, natural) ===")
        train_records = _records_for(partition, [hospital], "train")
        val_records = _records_for(partition, [hospital], "val")
        test_records = _records_for(partition, [hospital], "test")
        print(f"  train={len(train_records)} val={len(val_records)} test={len(test_records)}")

        per_seed_test_metrics = []
        for seed in SEEDS:
            print(f"  --- seed {seed} ---")
            with mlflow.start_run(run_name=f"local_finetune_natural_{hospital}_seed{seed}"):
                mlflow.log_params(
                    {
                        "hospital": hospital,
                        "seed": seed,
                        "fine_tune_last_block": True,
                        "n_train": len(train_records),
                        "n_val": len(val_records),
                        "n_test": len(test_records),
                    }
                )
                # Explicit distinct checkpoint_name per hospital+seed -- 2026-09-21
                # post-mortem: an earlier version relied on _train_one_seed's
                # default shared filename plus a rename-after-the-fact, which let
                # this loop silently overwrite row 2's (centralized) checkpoints
                # for the same seed values before they'd been renamed away,
                # permanently losing those three files. Passing the name directly
                # removes the collision instead of racing to rename around it.
                result = _train_one_seed(
                    seed, train_records, val_records, test_records, device,
                    checkpoint_name=f"local_natural_{hospital}_seed{seed}.pt",
                )
                per_seed_test_metrics.append(result["pooled_test"])
                print(f"  seed={seed}: test_auroc={result['pooled_test']['auroc']:.4f} test_accuracy={result['pooled_test']['accuracy']:.4f}")

        aggregated = aggregate_metrics_over_seeds(per_seed_test_metrics)
        all_results[hospital] = aggregated
        print(
            f"  {hospital}: AUROC = {aggregated['auroc']['mean']:.4f} +/- {aggregated['auroc']['std']:.4f} "
            f"(n={aggregated['auroc']['n_seeds']} seeds)"
        )

    save_results({"natural": all_results}, RESULTS_PATH)
    print(f"\nResults written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
