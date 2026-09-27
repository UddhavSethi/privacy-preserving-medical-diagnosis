#!/usr/bin/env python3
"""D1 (docs/pneumoscan_redesign_plan.md, Phase 1A): derives real decision-policy
calibration (decision_threshold, temperature, abstention_half_width) for one
of the Streamlit app's checkpoints.

Replicates, as a reusable committed script, the exact validation-set
methodology already applied ad hoc for the app's other two checkpoints
(docs/adr1_groupnorm_fallback.md sections 10 and 17 -- "sweep the decision
threshold 0.10-0.90 on the validation set only [...] recommend an operating
threshold from that data rather than the arbitrary 0.5 default", then fit
temperature scaling, then sweep the abstention half-width toward this
project's own ~10% deferral-target convention, DG-10). Neither of those two
prior runs had a committed script -- this one exists so a third checkpoint
(and any future one) doesn't repeat that gap.

Order matches sections 10.2 -> 10.3 -> 10.4 exactly: threshold chosen first
on raw (uncalibrated) probabilities, temperature fit second, then abstention
swept using the now-calibrated probabilities at the already-chosen threshold.

Usage:
    uv run python scripts/derive_app_decision_policy.py \
        --checkpoint outputs/checkpoints/ablation/secagg_seed42.pt \
        --output-name fedavg_secagg

Writes outputs/results/<output-name>_threshold_calibration_analysis.json
(the full sweep, same convention as round 9's own analysis file) and prints
the three recommended values to copy into conf/app.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from app.inference import load_classifier
from src.evaluation.metrics import compute_metrics
from src.training.trainer import FEATURE_KEY, load_pooled_features
from src.uncertainty.mc_dropout import compute_mc_dropout_uncertainty
from src.uncertainty.probability_calibration import apply_temperature, fit_temperature

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
FEATURE_CACHE_DIR = REPO_ROOT / "data" / "feature_cache"
HOSPITALS = ["A", "B", "C"]

# 17 points, 0.10-0.90, matching section 10.2's own sweep exactly.
THRESHOLD_SWEEP = [round(0.10 + 0.05 * i, 2) for i in range(17)]
ABSTENTION_HALF_WIDTH_SWEEP = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
TARGET_ABSTAIN_RATE = 0.10  # DG-10's own 10% deferral-target convention (Stage 19)
SEED = 42
NUM_MC_PASSES = 20


def _sweep_threshold(y_true: np.ndarray, prob_pneumonia: np.ndarray) -> tuple[list[dict], float]:
    rows = []
    best_f1, best_threshold = -1.0, 0.5
    for threshold in THRESHOLD_SWEEP:
        m = compute_metrics(y_true, prob_pneumonia, threshold=threshold)
        y_pred = (prob_pneumonia >= threshold).astype(int)
        accuracy = float((y_pred == y_true).mean())
        tn, fp, fn, tp = np.array(m.confusion_matrix).ravel()
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")
        rows.append(
            {
                "threshold": threshold,
                "sensitivity": m.sensitivity,
                "specificity": m.specificity,
                "precision": (float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")),
                "f1": m.f1,
                "accuracy": accuracy,
                "fnr": fnr,
                "fn": int(fn),
                "fp": int(fp),
            }
        )
        # Youden's-J and best-F1 coincided for round 9 (section 10.2); F1 is used
        # directly here as the tie-breaking criterion, matching that precedent.
        if m.f1 > best_f1:
            best_f1, best_threshold = m.f1, threshold
    return rows, best_threshold


def _sweep_abstention(y_true: np.ndarray, calibrated_probs: np.ndarray, threshold: float) -> tuple[list[dict], float]:
    rows = []
    best_half_width, best_distance = ABSTENTION_HALF_WIDTH_SWEEP[0], float("inf")
    for half_width in ABSTENTION_HALF_WIDTH_SWEEP:
        lo, hi = threshold - half_width, threshold + half_width
        abstained = (calibrated_probs >= lo) & (calibrated_probs <= hi)
        retained = ~abstained
        abstain_rate = float(abstained.mean())

        if retained.sum() > 0:
            retained_true = y_true[retained]
            retained_probs = calibrated_probs[retained]
            retained_pred = (retained_probs >= threshold).astype(int)
            retained_accuracy = float((retained_pred == retained_true).mean())
            retained_m = compute_metrics(retained_true, retained_probs, threshold=threshold)
            tn, fp, fn, tp = np.array(retained_m.confusion_matrix).ravel()
            retained_fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")
            retained_f1 = retained_m.f1
        else:
            retained_accuracy = retained_f1 = retained_fnr = float("nan")

        rows.append(
            {
                "half_width": half_width,
                "abstain_rate": abstain_rate,
                "retained_accuracy": retained_accuracy,
                "retained_f1": retained_f1,
                "retained_fnr": retained_fnr,
            }
        )
        distance = abs(abstain_rate - TARGET_ABSTAIN_RATE)
        if distance < best_distance:
            best_distance, best_half_width = distance, half_width
    return rows, best_half_width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-name", required=True, help="Names the output JSON and the conf/app.yaml key this is for."
    )
    parser.add_argument(
        "--fine-tune-last-block",
        action="store_true",
        help="Pass for an ADR-1 GroupNorm-fallback checkpoint (denseblock4+norm5 unfrozen); "
        "omit for the default frozen-backbone architecture.",
    )
    args = parser.parse_args()

    model = load_classifier(args.checkpoint, fine_tune_last_block=args.fine_tune_last_block)
    model.eval()

    pooled = load_pooled_features(PARTITION_PATH, HOSPITALS, FEATURE_CACHE_DIR, FEATURE_KEY)
    y_true = pooled.val_labels.numpy()

    torch.manual_seed(SEED)  # section 10.2's own reproducibility fix: MC Dropout's
    # masks are otherwise unseeded, giving a different sweep on every run.
    mc_result = compute_mc_dropout_uncertainty(model, pooled.val_features, num_passes=NUM_MC_PASSES)
    prob_pneumonia = mc_result.mean_probs[:, 1].numpy()

    threshold_rows, recommended_threshold = _sweep_threshold(y_true, prob_pneumonia)

    temperature = fit_temperature(mc_result.mean_probs, torch.tensor(y_true, dtype=torch.long))
    calibrated_probs = apply_temperature(mc_result.mean_probs, temperature)[:, 1].numpy()

    abstention_rows, recommended_half_width = _sweep_abstention(y_true, calibrated_probs, recommended_threshold)

    result = {
        "checkpoint": str(args.checkpoint),
        "output_name": args.output_name,
        "seed": SEED,
        "num_mc_passes": NUM_MC_PASSES,
        "n_val": int(len(y_true)),
        "threshold_sweep": threshold_rows,
        "recommended_decision_threshold": recommended_threshold,
        "temperature": temperature,
        "abstention_sweep": abstention_rows,
        "recommended_abstention_half_width": recommended_half_width,
        "target_abstain_rate": TARGET_ABSTAIN_RATE,
    }

    out_path = REPO_ROOT / "outputs" / "results" / f"{args.output_name}_threshold_calibration_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Wrote {out_path}")
    print(f"\nRecommended values for conf/app.yaml's '{args.output_name}' entry:")
    print(f"  decision_threshold: {recommended_threshold}")
    print(f"  temperature: {round(temperature, 4)}")
    print(f"  abstention_half_width: {recommended_half_width}")


if __name__ == "__main__":
    main()
