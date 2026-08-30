"""OPT-5 — build and validate per-hospital Isolation Forest OOD detectors (Phase 6,
priority 5, owner-approved 2026-08-30). One detector per hospital, trained on that
hospital's own cached training features (both classes) — see
`src/uncertainty/ood_detector.py`'s module docstring for the full design rationale
and the explicit note that this must never touch Secure Aggregation or the FedAvg
update path.

Validation, per the plan's own testing criteria:
  1. False-positive rate on held-out in-distribution data (the hospital's own val
     set for calibration, then an independent check on its test set) — should sit
     close to the target flag fraction, by construction for val and as a genuine
     generalization check for test.
  2. Synthetic non-chest-X-ray inputs score as anomalous at a much higher rate.
     No real natural-image dataset is available in this project (and downloading
     one is out of scope — CLAUDE.md's own dependency/dataset governance), so two
     synthetic stand-ins are used instead, run through the REAL frozen backbone
     (not faked at the feature level): uniform random pixel noise, and a
     structured synthetic pattern (colored geometric shapes) clearly unlike a
     chest X-ray's grayscale anatomy — both explicitly labeled as synthetic
     surrogates in the results, not claimed to be real natural photographs.

Usage: uv run python scripts/build_ood_detector.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.transforms import build_eval_transform
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_hospital_features
from src.uncertainty.ood_detector import build_and_calibrate, compute_anomaly_scores, flag_ood

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "ood_detector.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

HOSPITALS = ["A", "B", "C"]
DETECTOR_SEED = 42
TARGET_FLAG_FRACTION = 0.05
NUM_SYNTHETIC_PER_KIND = 100
IMAGE_SIZE = 224


def _synthetic_ood_features(kind: str, n: int, seed: int) -> np.ndarray:
    """Generates `n` synthetic non-chest-X-ray images of `kind`, runs them
    through the REAL frozen DenseNet121 backbone (not a shortcut at the feature
    level), and returns their real pooled 1024-dim features."""
    rng = np.random.default_rng(seed)
    model = DenseNet121Head()
    transform = build_eval_transform(image_size=IMAGE_SIZE)

    images = []
    for _ in range(n):
        if kind == "random_noise":
            img = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        elif kind == "structured_pattern":
            # Colored geometric shapes on a random background — clearly unlike a
            # chest X-ray's grayscale anatomy, without needing any external image.
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            img[:] = rng.integers(0, 256, size=3, dtype=np.uint8)
            for _ in range(5):
                color = rng.integers(0, 256, size=3, dtype=np.uint8)
                cx, cy = rng.integers(0, IMAGE_SIZE, size=2)
                radius = rng.integers(10, 60)
                y_grid, x_grid = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2
                img[mask] = color
        else:
            raise ValueError(f"unknown synthetic kind: {kind}")
        images.append(img)

    features = []
    with torch.no_grad():
        for img in images:
            tensor = transform(img).unsqueeze(0)
            features.append(model.pooled_features(tensor).numpy()[0])
    return np.stack(features, axis=0)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print(f"generating {NUM_SYNTHETIC_PER_KIND} synthetic images per kind via the real frozen backbone...")
    synthetic_features = {
        "random_noise": _synthetic_ood_features("random_noise", NUM_SYNTHETIC_PER_KIND, seed=0),
        "structured_pattern": _synthetic_ood_features("structured_pattern", NUM_SYNTHETIC_PER_KIND, seed=1),
    }

    for hospital in HOSPITALS:
        print(f"\n=== Hospital {hospital} ===")
        features = load_hospital_features(PARTITION_PATH, hospital)
        train_features = features.train_features[:, -1, :].numpy()  # eval-view, both classes
        val_features = features.val_features.numpy()
        test_features = features.test_features.numpy()
        print(f"train={len(train_features)} val={len(val_features)} test={len(test_features)}")

        detector, calibration = build_and_calibrate(
            train_features, val_features, seed=DETECTOR_SEED, target_flag_fraction=TARGET_FLAG_FRACTION
        )
        print(
            f"  calibration (val): target={calibration.target_flag_fraction:.3f} "
            f"realized={calibration.realized_flag_fraction_on_calibration:.3f}"
        )

        test_scores = compute_anomaly_scores(detector, test_features)
        test_flag_rate = float(flag_ood(test_scores, calibration.threshold).mean())
        print(f"  independent check (test, in-distribution): flag_rate={test_flag_rate:.4f}")

        synthetic_flag_rates = {}
        for kind, feats in synthetic_features.items():
            scores = compute_anomaly_scores(detector, feats)
            rate = float(flag_ood(scores, calibration.threshold).mean())
            synthetic_flag_rates[kind] = rate
            print(f"  synthetic OOD ({kind}): flag_rate={rate:.4f}")

        results[hospital] = {
            "n_train": len(train_features),
            "n_val": len(val_features),
            "n_test": len(test_features),
            "calibration": calibration.to_dict(),
            "test_flag_rate": test_flag_rate,
            "synthetic_flag_rates": synthetic_flag_rates,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nresults written: {RESULTS_PATH}")

    _plot_flag_rates(results)


def _plot_flag_rates(results: dict) -> None:
    hospitals = list(results.keys())
    val_rates = [results[h]["calibration"]["realized_flag_fraction_on_calibration"] for h in hospitals]
    test_rates = [results[h]["test_flag_rate"] for h in hospitals]
    noise_rates = [results[h]["synthetic_flag_rates"]["random_noise"] for h in hospitals]
    pattern_rates = [results[h]["synthetic_flag_rates"]["structured_pattern"] for h in hospitals]

    x = np.arange(len(hospitals))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5 * width, val_rates, width, label="val (in-distribution, calibration)", color="#2ca02c")
    ax.bar(x - 0.5 * width, test_rates, width, label="test (in-distribution, held out)", color="#98df8a")
    ax.bar(x + 0.5 * width, noise_rates, width, label="synthetic: random noise", color="#d62728")
    ax.bar(x + 1.5 * width, pattern_rates, width, label="synthetic: structured pattern", color="#ff9896")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Hospital {h}" for h in hospitals])
    ax.set_ylabel("OOD flag rate")
    ax.set_title(f"Isolation Forest OOD gate — flag rates (target {TARGET_FLAG_FRACTION:.0%} on in-distribution val)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGURES_DIR / "ood_detector_flag_rates.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
