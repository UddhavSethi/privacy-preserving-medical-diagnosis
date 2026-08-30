"""OPT-1 — calibration analysis (Phase 6, priority 1, owner-approved 2026-08-30).

Post-hoc analysis over Stage 21's already-trained, already-saved checkpoints — no
new training, no new federated runs. For each configuration (centralized ceiling,
FedAvg no-DP, and the full DP epsilon sweep {1,2,4,8}), loads each seed's saved
classifier state, runs MC Dropout (T=20, matching Stage 19's own default) on the
natural-regime pooled test set, and computes:
  - Expected Calibration Error (ECE) and Brier score, aggregated mean +/- std over
    the 3 seeds {42, 123, 2024} (CLAUDE.md section 11.2's own seed-aggregation
    convention, applied here for the first time to a calibration metric)
  - reliability-diagram data and a risk-coverage curve from the seed with the
    median ECE, for the figures (not seed-averaged directly — a bin-by-bin average
    across differently-shaped per-seed diagrams is not a meaningful object; a single
    representative run's curve is the standard way this is shown)

Answers this project's own honest, previously-unmeasured question (CLAUDE.md
section 10): is MC Dropout's confidence actually trustworthy, and does it degrade as
the DP epsilon tightens (CLAUDE.md section 6's second named architectural tension)?

Usage: uv run python scripts/run_calibration_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
    risk_coverage_curve,
)
from src.evaluation.reporting import aggregate_over_seeds
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_pooled_features
from src.uncertainty.mc_dropout import compute_mc_dropout_uncertainty

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "calibration.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
HOSPITALS = ["A", "B", "C"]
SEEDS = [42, 123, 2024]
NUM_PASSES = 20  # T, matches conf/experiment/uncertainty.yaml's Stage 19 default

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


def _evaluate_checkpoint(ckpt_path: Path, features: torch.Tensor, labels: torch.Tensor) -> dict:
    model = DenseNet121Head()
    model.classifier.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    result = compute_mc_dropout_uncertainty(model, features, num_passes=NUM_PASSES)
    confidence = result.mean_probs.max(dim=1).values.numpy()
    predicted = result.predicted_class.numpy()
    correct = (predicted == labels.numpy()).astype(float)
    prob_positive = result.mean_probs[:, 1].numpy()

    ece = expected_calibration_error(confidence, correct)
    brier = brier_score(labels.numpy(), prob_positive)
    diagram = reliability_diagram_data(confidence, correct)
    rc_curve = risk_coverage_curve(result.entropy.numpy(), correct)

    return {
        "ece": ece,
        "brier": brier,
        "accuracy": float(correct.mean()),
        "reliability_diagram": diagram.to_dict(),
        "risk_coverage": rc_curve.to_dict(),
    }


def _plot_reliability_diagrams(per_config_representative: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(per_config_representative)))
    for (name, data), color in zip(per_config_representative.items(), colors):
        diagram = data["reliability_diagram"]
        conf = np.array(diagram["bin_confidence"])
        acc = np.array(diagram["bin_accuracy"])
        valid = ~np.isnan(conf)
        ax.plot(conf[valid], acc[valid], marker="o", color=color, label=f"{name} (ECE={data['ece']:.3f})")
    ax.set_xlabel("Mean predicted confidence (bin)")
    ax.set_ylabel("Empirical accuracy (bin)")
    ax.set_title("Reliability diagrams — MC Dropout confidence (T=20)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "reliability_diagrams.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


def _plot_ece_vs_epsilon(aggregated: dict) -> None:
    eps_rows = {
        float(name.split("epsilon=")[1].rstrip(")")): row
        for name, row in aggregated.items()
        if "epsilon=" in name
    }
    if not eps_rows:
        return
    epsilons = sorted(eps_rows)
    ece_means = [eps_rows[e]["ece"]["mean"] for e in epsilons]
    ece_stds = [eps_rows[e]["ece"]["std"] for e in epsilons]
    no_dp = aggregated.get("FedAvg (natural, no DP)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(epsilons, ece_means, yerr=ece_stds, marker="o", capsize=4, color="#d62728", label="FedAvg + DP")
    if no_dp:
        ax.axhline(
            no_dp["ece"]["mean"], color="#888888", linestyle="--",
            label=f"FedAvg, no DP (ECE={no_dp['ece']['mean']:.3f})",
        )
    ax.set_xscale("log")
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel("Target epsilon (privacy budget, log scale)")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Calibration vs. privacy budget (3 seeds/point)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "calibration_vs_epsilon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


def _plot_risk_coverage(per_config_representative: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(per_config_representative)))
    for (name, data), color in zip(per_config_representative.items(), colors):
        rc = data["risk_coverage"]
        ax.plot(rc["coverage"], rc["risk"], color=color, label=name, linewidth=1.2)
    ax.set_xlabel("Coverage (fraction of predictions retained)")
    ax.set_ylabel("Risk (error rate among retained predictions)")
    ax.set_title("Risk-coverage curves (selective prediction via MC Dropout entropy)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "risk_coverage_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("loading natural-regime pooled test features...")
    pooled = load_pooled_features(PARTITION_PATH, HOSPITALS)
    test_features, test_labels = pooled.test_features, pooled.test_labels
    print(f"test set: {len(test_labels)} examples")

    per_config_seed_results: dict[str, list[dict]] = {}
    for name, ckpt_paths in CONFIGURATIONS.items():
        per_config_seed_results[name] = []
        for ckpt_path in ckpt_paths:
            if not ckpt_path.exists():
                print(f"  MISSING checkpoint: {ckpt_path} — skipping")
                continue
            result = _evaluate_checkpoint(ckpt_path, test_features, test_labels)
            per_config_seed_results[name].append(result)
            print(f"  {name} / {ckpt_path.stem}: ECE={result['ece']:.4f} Brier={result['brier']:.4f}")

    aggregated = {}
    representative = {}
    for name, seed_results in per_config_seed_results.items():
        if not seed_results:
            continue
        aggregated[name] = {
            "ece": aggregate_over_seeds([r["ece"] for r in seed_results]).to_dict(),
            "brier": aggregate_over_seeds([r["brier"] for r in seed_results]).to_dict(),
            "accuracy": aggregate_over_seeds([r["accuracy"] for r in seed_results]).to_dict(),
        }
        # Representative run for the figures: the seed with the median ECE — avoids
        # cherry-picking the best-looking seed while still showing one coherent,
        # real curve rather than an artificial per-bin average across seeds.
        median_idx = int(np.argsort([r["ece"] for r in seed_results])[len(seed_results) // 2])
        representative[name] = seed_results[median_idx]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(aggregated, indent=2))
    print(f"\nresults written: {RESULTS_PATH}")

    print("\n=== Calibration summary (mean +/- std over seeds) ===")
    for name, row in aggregated.items():
        print(
            f"{name}: ECE={row['ece']['mean']:.4f}+/-{row['ece']['std']:.4f}  "
            f"Brier={row['brier']['mean']:.4f}+/-{row['brier']['std']:.4f}  "
            f"n_seeds={row['ece']['n_seeds']}"
        )

    _plot_reliability_diagrams(representative)
    _plot_ece_vs_epsilon(aggregated)
    _plot_risk_coverage(representative)


if __name__ == "__main__":
    main()
