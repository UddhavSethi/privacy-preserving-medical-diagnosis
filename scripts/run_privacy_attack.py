"""OPT-2 — empirical privacy attack analysis (Phase 6, priority 2, owner-approved
2026-08-30). Post-hoc analysis over Stage 21's already-trained checkpoints, exactly
like OPT-1's calibration analysis — no new training.

For each configuration, evaluates each of the 3 seeds' saved classifier on the
pooled natural TRAIN set (member — what the model was trained on) and pooled
natural TEST set (non-member — held out, never seen during training), computes
per-example cross-entropy loss, and runs the loss-based membership inference
attack (`src.evaluation.privacy_attack`). Reports attack AUROC aggregated over
seeds — the headline question: does turning DP on, and tightening epsilon, actually
reduce measured membership-inference leakage, as the architecture claims it should?

Usage: uv run python scripts/run_privacy_attack.py
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

from src.evaluation.privacy_attack import per_example_cross_entropy_loss, run_membership_inference_attack
from src.evaluation.reporting import aggregate_over_seeds
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_pooled_features

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "privacy_attack.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
HOSPITALS = ["A", "B", "C"]
SEEDS = [42, 123, 2024]

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
}


def _predict_probs(ckpt_path: Path, features: torch.Tensor) -> np.ndarray:
    model = DenseNet121Head()
    model.classifier.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()  # point-estimate attack — deliberately NOT MC Dropout (standard MIA uses the
    # model's actual deployed decision boundary, a single deterministic forward pass)
    with torch.no_grad():
        probs = F.softmax(model.classifier(features), dim=1).numpy()
    return probs


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("loading natural-regime pooled train (members) + test (non-members) features...")
    pooled = load_pooled_features(PARTITION_PATH, HOSPITALS)
    member_features, member_labels = pooled.train_features[:, -1, :], pooled.train_labels
    nonmember_features, nonmember_labels = pooled.test_features, pooled.test_labels
    print(f"members (train): {len(member_labels)}   non-members (test): {len(nonmember_labels)}")

    per_config_seed_results: dict[str, list[dict]] = {}
    for name, ckpt_paths in CONFIGURATIONS.items():
        per_config_seed_results[name] = []
        for ckpt_path in ckpt_paths:
            if not ckpt_path.exists():
                print(f"  MISSING checkpoint: {ckpt_path} — skipping")
                continue
            member_probs = _predict_probs(ckpt_path, member_features)
            nonmember_probs = _predict_probs(ckpt_path, nonmember_features)
            member_loss = per_example_cross_entropy_loss(member_probs, member_labels.numpy())
            nonmember_loss = per_example_cross_entropy_loss(nonmember_probs, nonmember_labels.numpy())
            result = run_membership_inference_attack(member_loss, nonmember_loss)
            per_config_seed_results[name].append(result.to_dict())
            print(
                f"  {name} / {ckpt_path.stem}: attack_auroc={result.attack_auroc:.4f} "
                f"gen_gap={result.generalization_gap:.4f}"
            )

    aggregated = {}
    representative_losses = {}
    for name, seed_results in per_config_seed_results.items():
        if not seed_results:
            continue
        aggregated[name] = {
            "attack_auroc": aggregate_over_seeds([r["attack_auroc"] for r in seed_results]).to_dict(),
            "generalization_gap": aggregate_over_seeds([r["generalization_gap"] for r in seed_results]).to_dict(),
            "mean_member_loss": aggregate_over_seeds([r["mean_member_loss"] for r in seed_results]).to_dict(),
            "mean_nonmember_loss": aggregate_over_seeds([r["mean_nonmember_loss"] for r in seed_results]).to_dict(),
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(aggregated, indent=2))
    print(f"\nresults written: {RESULTS_PATH}")

    print("\n=== Membership inference attack summary (mean +/- std over seeds) ===")
    for name, row in aggregated.items():
        print(
            f"{name}: attack_AUROC={row['attack_auroc']['mean']:.4f}+/-{row['attack_auroc']['std']:.4f}  "
            f"gen_gap={row['generalization_gap']['mean']:.4f}+/-{row['generalization_gap']['std']:.4f}  "
            f"n_seeds={row['attack_auroc']['n_seeds']}"
        )

    _plot_attack_auroc_vs_epsilon(aggregated)
    _plot_attack_auroc_bar(aggregated)


def _plot_attack_auroc_vs_epsilon(aggregated: dict) -> None:
    eps_rows = {
        float(name.split("epsilon=")[1].rstrip(")")): row
        for name, row in aggregated.items()
        if "epsilon=" in name
    }
    if not eps_rows:
        return
    epsilons = sorted(eps_rows)
    means = [eps_rows[e]["attack_auroc"]["mean"] for e in epsilons]
    stds = [eps_rows[e]["attack_auroc"]["std"] for e in epsilons]
    no_dp = aggregated.get("FedAvg (natural, no DP)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(epsilons, means, yerr=stds, marker="o", capsize=4, color="#9467bd", label="FedAvg + DP")
    if no_dp:
        ax.axhline(
            no_dp["attack_auroc"]["mean"], color="#888888", linestyle="--",
            label=f"FedAvg, no DP (AUROC={no_dp['attack_auroc']['mean']:.3f})",
        )
    ax.axhline(0.5, color="green", linestyle=":", label="no leakage (AUROC=0.5)")
    ax.set_xscale("log")
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel("Target epsilon (privacy budget, log scale)")
    ax.set_ylabel("Membership inference attack AUROC")
    ax.set_title("Empirical privacy leakage vs. privacy budget (3 seeds/point)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "privacy_attack_vs_epsilon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


def _plot_attack_auroc_bar(aggregated: dict) -> None:
    names = list(aggregated.keys())
    means = [aggregated[n]["attack_auroc"]["mean"] for n in names]
    stds = [aggregated[n]["attack_auroc"]["std"] for n in names]

    # Every value here sits within a few thousandths of 0.5 (see docs/privacy_attack.md)
    # — plotting from x=0 would make every bar look visually identical and hide the
    # real, if small, signal. Zoom to the range the actual data occupies instead.
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    pad = max(0.01, (hi - lo) * 0.3)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(names))
    ax.barh(y_pos, means, xerr=stds, capsize=3, color="#9467bd")
    ax.axvline(0.5, color="green", linestyle=":", label="no leakage (AUROC=0.5)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Membership inference attack AUROC (zoomed — see docs/privacy_attack.md)")
    ax.set_title("Empirical privacy leakage across configurations (3 seeds each)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIGURES_DIR / "privacy_attack_all_configs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
