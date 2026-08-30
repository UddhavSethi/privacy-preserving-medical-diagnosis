"""Stage 23 — generates the paper/README figures from Stage 21's real
campaign data (MLflow + outputs/results/*.json), not synthetic placeholders.

Usage: uv run python scripts/generate_result_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.tables import build_ablation_table

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def plot_privacy_utility_curve(rows: list[dict]) -> None:
    dp_rows = {
        float(r["row"].split("epsilon=")[1].rstrip(")")): r
        for r in rows
        if "FedAvg + DP" in r["row"] and r["mean_auroc"] is not None
    }
    if not dp_rows:
        print("no DP rows with data yet — skipping privacy-utility curve")
        return

    epsilons = sorted(dp_rows)
    means = [dp_rows[e]["mean_auroc"] for e in epsilons]
    stds = [dp_rows[e]["std_auroc"] for e in epsilons]
    baseline = next((r for r in rows if r["row"].startswith("3. FedAvg (natural)")), None)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(epsilons, means, yerr=stds, marker="o", capsize=4, label="FedAvg + DP", color="#1f77b4")
    if baseline and baseline["mean_auroc"] is not None:
        ax.axhline(
            baseline["mean_auroc"], color="#888888", linestyle="--",
            label=f"FedAvg, no DP ({baseline['mean_auroc']:.3f})",
        )
    ax.set_xscale("log")
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel("Target epsilon (privacy budget, log scale)")
    ax.set_ylabel("Pooled test AUROC")
    ax.set_title("Privacy-utility tradeoff (Stage 21, 3 seeds/point)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIGURES_DIR / "privacy_utility_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"written: {out_path}")


def plot_dirichlet_heterogeneity(rows: list[dict]) -> None:
    dirichlet_rows = {
        float(r["row"].split("alpha=")[1].rstrip(")")): r
        for r in rows
        if "Dirichlet" in r["row"] and r["mean_auroc"] is not None
    }
    natural_row = next((r for r in rows if r["row"].startswith("3. FedAvg (natural)")), None)
    if not dirichlet_rows:
        print("no Dirichlet rows with data yet — skipping heterogeneity plot")
        return

    alphas = sorted(dirichlet_rows)
    means = [dirichlet_rows[a]["mean_auroc"] for a in alphas]
    stds = [dirichlet_rows[a]["std_auroc"] for a in alphas]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(a) for a in alphas], means, yerr=stds, capsize=4, color="#ff7f0e")
    if natural_row and natural_row["mean_auroc"] is not None:
        ax.axhline(
            natural_row["mean_auroc"], color="#888888", linestyle="--",
            label=f"Natural partition ({natural_row['mean_auroc']:.3f})",
        )
        ax.legend()
    ax.set_xlabel("Dirichlet alpha (lower = more non-IID)")
    ax.set_ylabel("Pooled test AUROC")
    ax.set_title("Synthetic non-IID heterogeneity sweep (Stage 21, supplementary)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = FIGURES_DIR / "dirichlet_heterogeneity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"written: {out_path}")


def plot_ablation_bar_chart(rows: list[dict]) -> None:
    labeled = [(r["row"], r["mean_auroc"], r["std_auroc"]) for r in rows if r["mean_auroc"] is not None]
    labels = [l for l, _, _ in labeled]
    means = [m for _, m, _ in labeled]
    stds = [s for _, _, s in labeled]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(labels))
    ax.barh(y_pos, means, xerr=stds, capsize=3, color="#2ca02c")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pooled test AUROC")
    ax.set_title("Full ablation table (Stage 21, real live runs, 3 seeds each)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out_path = FIGURES_DIR / "ablation_table_chart.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"written: {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_ablation_table()
    plot_privacy_utility_curve(rows)
    plot_dirichlet_heterogeneity(rows)
    plot_ablation_bar_chart(rows)


if __name__ == "__main__":
    main()
