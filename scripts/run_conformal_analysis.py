"""OPT-4 — conformal prediction analysis (Phase 6, priority 4, owner-approved
2026-08-30). Post-hoc analysis over Stage 21's already-trained checkpoints, same
pattern as OPT-1/OPT-2 — no new training.

Directly tests the claim `docs/calibration.md` (OPT-1) motivates this extension
with: DP damages MC Dropout's raw confidence calibration (ECE ~4x worse), but
conformal prediction's coverage guarantee does not depend on the underlying model
being well-calibrated at all. For each configuration, calibrates a conformal
threshold on the pooled natural VAL set (held out, never used for training or for
OPT-1/OPT-2/OPT-3's test-set evaluations) at alpha=0.10 (matching DG-10's 90%
target coverage), then checks whether the resulting prediction sets actually
achieve ~90% empirical coverage on the pooled natural TEST set — even for the DP
configurations OPT-1 found have badly miscalibrated raw confidence.

Usage: uv run python scripts/run_conformal_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.reporting import aggregate_over_seeds
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_pooled_features
from src.uncertainty.conformal import run_conformal_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "conformal.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
HOSPITALS = ["A", "B", "C"]
SEEDS = [42, 123, 2024]
ALPHA = 0.10  # matches DG-10's 90% target coverage (src/uncertainty/deferral.py)

CONFIGURATIONS = {
    "centralized (natural, ceiling)": [
        CHECKPOINT_DIR / "centralized_baseline" / f"natural_seed{s}.pt" for s in SEEDS
    ],
    "FedAvg (natural, no DP)": [
        CHECKPOINT_DIR / "ablation" / f"fedavg_natural_seed{s}.pt" for s in SEEDS
    ],
    "FedAvg + DP (epsilon=1.0)": [
        CHECKPOINT_DIR / "ablation" / f"dp_eps1.0_seed{s}.pt" for s in SEEDS
    ],
    "FedAvg + DP (epsilon=2.0)": [
        CHECKPOINT_DIR / "ablation" / f"dp_eps2.0_seed{s}.pt" for s in SEEDS
    ],
    "FedAvg + DP (epsilon=4.0)": [
        CHECKPOINT_DIR / "ablation" / f"dp_eps4.0_seed{s}.pt" for s in SEEDS
    ],
    "FedAvg + DP (epsilon=8.0)": [
        CHECKPOINT_DIR / "ablation" / f"dp_eps8.0_seed{s}.pt" for s in SEEDS
    ],
    "FedAvg + SecAgg": [
        CHECKPOINT_DIR / "ablation" / f"secagg_seed{s}.pt" for s in SEEDS
    ],
}


def _predict_probs(ckpt_path: Path, features: torch.Tensor) -> np.ndarray:
    model = DenseNet121Head()
    model.classifier.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model.classifier(features), dim=1).numpy()
    return probs


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("loading natural-regime pooled val (calibration) + test features...")
    pooled = load_pooled_features(PARTITION_PATH, HOSPITALS)
    cal_features, cal_labels = pooled.val_features, pooled.val_labels.numpy()
    test_features, test_labels = pooled.test_features, pooled.test_labels.numpy()
    print(f"calibration (val): {len(cal_labels)}   test: {len(test_labels)}   target coverage: {1 - ALPHA}")

    per_config_seed_results: dict[str, list[dict]] = {}
    for name, ckpt_paths in CONFIGURATIONS.items():
        per_config_seed_results[name] = []
        for ckpt_path in ckpt_paths:
            if not ckpt_path.exists():
                print(f"  MISSING checkpoint: {ckpt_path} — skipping")
                continue
            probs_cal = _predict_probs(ckpt_path, cal_features)
            probs_test = _predict_probs(ckpt_path, test_features)
            result = run_conformal_analysis(probs_cal, cal_labels, probs_test, test_labels, alpha=ALPHA)
            per_config_seed_results[name].append(result.to_dict())
            print(
                f"  {name} / {ckpt_path.stem}: coverage={result.empirical_coverage:.4f} "
                f"mean_set_size={result.mean_set_size:.4f} threshold={result.threshold:.4f}"
            )

    aggregated = {}
    for name, seed_results in per_config_seed_results.items():
        if not seed_results:
            continue
        aggregated[name] = {
            "empirical_coverage": aggregate_over_seeds([r["empirical_coverage"] for r in seed_results]).to_dict(),
            "mean_set_size": aggregate_over_seeds([r["mean_set_size"] for r in seed_results]).to_dict(),
            "threshold": aggregate_over_seeds([r["threshold"] for r in seed_results]).to_dict(),
            "empty_set_fraction": aggregate_over_seeds(
                [r["set_size_distribution"]["empty"] for r in seed_results]
            ).to_dict(),
            "full_set_fraction": aggregate_over_seeds(
                [r["set_size_distribution"]["full"] for r in seed_results]
            ).to_dict(),
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(aggregated, indent=2))
    print(f"\nresults written: {RESULTS_PATH}")

    print(f"\n=== Conformal prediction summary (target coverage={1 - ALPHA}, mean +/- std over seeds) ===")
    for name, row in aggregated.items():
        print(
            f"{name}: coverage={row['empirical_coverage']['mean']:.4f}+/-{row['empirical_coverage']['std']:.4f}  "
            f"mean_set_size={row['mean_set_size']['mean']:.4f}+/-{row['mean_set_size']['std']:.4f}  "
            f"full_set_frac={row['full_set_fraction']['mean']:.4f}  n_seeds={row['empirical_coverage']['n_seeds']}"
        )

    _plot_coverage_and_set_size(aggregated)


def _plot_coverage_and_set_size(aggregated: dict) -> None:
    names = list(aggregated.keys())
    coverage = [aggregated[n]["empirical_coverage"]["mean"] for n in names]
    coverage_std = [aggregated[n]["empirical_coverage"]["std"] for n in names]
    set_size = [aggregated[n]["mean_set_size"]["mean"] for n in names]
    set_size_std = [aggregated[n]["mean_set_size"]["std"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    y_pos = range(len(names))

    axes[0].barh(y_pos, coverage, xerr=coverage_std, capsize=3, color="#8c564b")
    axes[0].axvline(1 - ALPHA, color="green", linestyle=":", label=f"target coverage ({1 - ALPHA})")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Empirical coverage")
    axes[0].set_title("Conformal coverage — does the guarantee hold under DP?")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(y_pos, set_size, xerr=set_size_std, capsize=3, color="#e377c2")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean prediction set size")
    axes[1].set_title("Conformal set size — cost of federation and privacy")
    axes[1].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    out = FIGURES_DIR / "conformal_coverage_and_set_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
