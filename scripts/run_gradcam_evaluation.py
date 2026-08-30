"""OPT-3 — quantitative Grad-CAM evaluation (Phase 6, priority 3, owner-approved
2026-08-30). CLAUDE.md section 15, item 6: "Grad-CAM is evaluated qualitatively" —
this converts that into a measured result using RSNA's real bounding-box
annotations (RSNA-only; Kermany carries none — CLAUDE.md's own named limitation).

For each configuration (centralized ceiling, FedAvg no-DP, DP epsilon sweep
{1,2,4,8}, SecAgg — the same checkpoint set OPT-1/OPT-2 used), each of the 3
seeds' saved classifier is combined with the frozen backbone to compute a real
Grad-CAM heatmap (targeting the Pneumonia class) for a fixed, seeded subsample of
RSNA test-set pneumonia-positive images that carry real bounding boxes, and scores
it with the pointing game and IoU (`src.evaluation.gradcam_eval`).

Subsample, not the full 902 eligible images, for tractability across 7 configs x 3
seeds x real Grad-CAM computation (each requires a full backbone forward+backward
pass, unlike OPT-1/OPT-2's cached-feature analyses) — a fixed seeded subsample is
standard practice for this kind of evaluation and is applied identically across
every configuration for a fair comparison.

Usage: uv run python scripts/run_gradcam_evaluation.py
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.data.transforms import build_eval_transform
from src.evaluation.gradcam_eval import iou_against_boxes, pointing_game_hit, rescale_boxes, summarize_localization
from src.evaluation.reporting import aggregate_over_seeds
from src.explain.gradcam import PNEUMONIA_CLASS_INDEX, compute_gradcam_heatmap
from src.models.densenet_head import DenseNet121Head

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
RSNA_LABELS_CSV = REPO_ROOT / "data" / "raw" / "rsna" / "stage_2_train_labels.csv"
CLAHE_CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "gradcam_localization.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

SEEDS = [42, 123, 2024]
SAMPLE_SIZE = 300  # fixed, seeded subsample of the 902 eligible boxed test images
SAMPLE_SEED = 42
IMAGE_SIZE = 224
IOU_THRESHOLD_FRACTION = 0.5

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


def _load_boxes_by_patient() -> dict[str, list[tuple[float, float, float, float]]]:
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    with open(RSNA_LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["Target"] == "1" and row["x"]:
                boxes.setdefault(row["patientId"], []).append(
                    (float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"]))
                )
    return boxes


def _select_sample_records() -> list[dict]:
    """RSNA test-set pneumonia-positive records (hospitals B, C) with real
    bounding boxes, subsampled to SAMPLE_SIZE with a fixed seed."""
    partition = json.loads(PARTITION_PATH.read_text())
    boxes_by_patient = _load_boxes_by_patient()

    eligible = []
    for hospital in ("B", "C"):
        for r in partition["hospitals"][hospital]:
            if r["frozen_split"] != "test" or r["source"] != "rsna" or r["label"] != "Pneumonia":
                continue
            raw_patient_id = r["patient_id"].removeprefix("rsna-")
            if raw_patient_id in boxes_by_patient:
                eligible.append({**r, "boxes": boxes_by_patient[raw_patient_id]})

    print(f"eligible RSNA test-set boxed positives: {len(eligible)}")
    rng = random.Random(SAMPLE_SEED)
    sample = eligible if len(eligible) <= SAMPLE_SIZE else rng.sample(eligible, SAMPLE_SIZE)
    print(f"evaluating on a fixed subsample of {len(sample)} images (seed={SAMPLE_SEED})")
    return sample


def _evaluate_checkpoint(ckpt_path: Path, records: list[dict]) -> tuple[list[bool], list[float]]:
    model = DenseNet121Head()
    model.classifier.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    transform = build_eval_transform(image_size=IMAGE_SIZE)
    hits, ious = [], []
    for r in records:
        cache_path = cache_path_for(CLAHE_CACHE_DIR, r["source"], r["relative_path"], ClaheParams())
        if not cache_path.exists():
            continue
        image_rgb = load_from_cache(cache_path)
        orig_h, orig_w = image_rgb.shape[:2]
        tensor = transform(image_rgb).unsqueeze(0)

        heatmap = compute_gradcam_heatmap(model, tensor, PNEUMONIA_CLASS_INDEX)
        boxes_224 = rescale_boxes(r["boxes"], (orig_w, orig_h), (IMAGE_SIZE, IMAGE_SIZE))

        hits.append(pointing_game_hit(heatmap, boxes_224))
        ious.append(iou_against_boxes(heatmap, boxes_224, threshold_fraction=IOU_THRESHOLD_FRACTION))

    return hits, ious


def _plot_localization_vs_epsilon(aggregated: dict) -> None:
    eps_rows = {
        float(name.split("epsilon=")[1].rstrip(")")): row
        for name, row in aggregated.items()
        if "epsilon=" in name
    }
    if not eps_rows:
        return
    epsilons = sorted(eps_rows)
    pg_means = [eps_rows[e]["pointing_game_accuracy"]["mean"] for e in epsilons]
    pg_stds = [eps_rows[e]["pointing_game_accuracy"]["std"] for e in epsilons]
    iou_means = [eps_rows[e]["mean_iou"]["mean"] for e in epsilons]
    iou_stds = [eps_rows[e]["mean_iou"]["std"] for e in epsilons]
    no_dp = aggregated.get("FedAvg (natural, no DP)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(epsilons, pg_means, yerr=pg_stds, marker="o", capsize=4, color="#e377c2")
    axes[1].errorbar(epsilons, iou_means, yerr=iou_stds, marker="o", capsize=4, color="#17becf")
    if no_dp:
        axes[0].axhline(no_dp["pointing_game_accuracy"]["mean"], color="#888888", linestyle="--", label="no DP")
        axes[1].axhline(no_dp["mean_iou"]["mean"], color="#888888", linestyle="--", label="no DP")
    for ax, title, ylabel in zip(
        axes, ["Pointing game", "Mean IoU"], ["Pointing-game accuracy", "Mean IoU vs. GT boxes"]
    ):
        ax.set_xscale("log")
        ax.set_xticks(epsilons)
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.set_xlabel("Target epsilon (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Grad-CAM localization quality vs. privacy budget (RSNA boxes, 3 seeds/point)")
    fig.tight_layout()
    out = FIGURES_DIR / "gradcam_localization_vs_epsilon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


def _plot_localization_bar(aggregated: dict) -> None:
    names = list(aggregated.keys())
    pg = [aggregated[n]["pointing_game_accuracy"]["mean"] for n in names]
    pg_std = [aggregated[n]["pointing_game_accuracy"]["std"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(names))
    ax.barh(y_pos, pg, xerr=pg_std, capsize=3, color="#e377c2")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pointing-game accuracy (RSNA bounding boxes)")
    ax.set_title("Grad-CAM localization across configurations (3 seeds each)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIGURES_DIR / "gradcam_localization_all_configs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    records = _select_sample_records()

    per_config_seed_results: dict[str, list[dict]] = {}
    for name, ckpt_paths in CONFIGURATIONS.items():
        per_config_seed_results[name] = []
        for ckpt_path in ckpt_paths:
            if not ckpt_path.exists():
                print(f"  MISSING checkpoint: {ckpt_path} — skipping")
                continue
            hits, ious = _evaluate_checkpoint(ckpt_path, records)
            summary = summarize_localization(hits, ious)
            per_config_seed_results[name].append(summary.to_dict())
            print(
                f"  {name} / {ckpt_path.stem}: pointing_game={summary.pointing_game_accuracy:.4f} "
                f"mean_iou={summary.mean_iou:.4f} (n={summary.n_images})"
            )

    aggregated = {}
    for name, seed_results in per_config_seed_results.items():
        if not seed_results:
            continue
        aggregated[name] = {
            "pointing_game_accuracy": aggregate_over_seeds(
                [r["pointing_game_accuracy"] for r in seed_results]
            ).to_dict(),
            "mean_iou": aggregate_over_seeds([r["mean_iou"] for r in seed_results]).to_dict(),
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(aggregated, indent=2))
    print(f"\nresults written: {RESULTS_PATH}")

    print("\n=== Grad-CAM localization summary (mean +/- std over seeds) ===")
    for name, row in aggregated.items():
        print(
            f"{name}: pointing_game={row['pointing_game_accuracy']['mean']:.4f}"
            f"+/-{row['pointing_game_accuracy']['std']:.4f}  "
            f"mean_iou={row['mean_iou']['mean']:.4f}+/-{row['mean_iou']['std']:.4f}  "
            f"n_seeds={row['pointing_game_accuracy']['n_seeds']}"
        )

    _plot_localization_vs_epsilon(aggregated)
    _plot_localization_bar(aggregated)


if __name__ == "__main__":
    main()
