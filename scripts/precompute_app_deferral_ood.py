#!/usr/bin/env python3
"""D2 (docs/pneumoscan_redesign_plan.md, Phase 1A): precomputes and commits a
checkpoint's deferral threshold and per-hospital OOD detectors, so the public
Streamlit Cloud deployment has real "How sure is the AI" / "Image check"
values instead of permanently showing "Not available".

Why this is needed at all: `app/inference.py::calibrate_deferral_threshold`
and `build_ood_detectors` are real, already-tested functions -- but both need
`data/feature_cache/` (576MB, gitignored, never deployed to Cloud) to run.
Running them ONCE here, locally, and committing just their small outputs
(one float + 3 IsolationForest models, a few hundred KB total) gives Cloud
the same real deferral/OOD behavior a local run already has, without
committing the 576MB cache itself.

`joblib` is scikit-learn's own hard dependency (not a new package this
project doesn't already have installed) and is the library's own documented
way to persist a fitted estimator -- used here rather than raw pickle for
that reason, matching the approved plan's own choice.

Usage:
    uv run python scripts/precompute_app_deferral_ood.py --output-name fedavg_secagg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from app.inference import build_ood_detectors, calibrate_deferral_threshold, load_classifier

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
FEATURE_CACHE_DIR = REPO_ROOT / "data" / "feature_cache"
HOSPITALS = ["A", "B", "C"]

TARGET_DEFER_FRACTION = 0.10  # DG-10, Stage 19 -- same convention every other checkpoint uses
TARGET_FLAG_FRACTION = 0.05  # OPT-5's own placeholder (conf/experiment/ood.yaml)
OOD_SEED = 42
NUM_MC_PASSES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-name", required=True, help="Names the output directory and conf/app.yaml key.")
    parser.add_argument("--fine-tune-last-block", action="store_true")
    args = parser.parse_args()

    model = load_classifier(args.checkpoint, fine_tune_last_block=args.fine_tune_last_block)
    model.eval()

    deferral_threshold = calibrate_deferral_threshold(
        model,
        PARTITION_PATH,
        FEATURE_CACHE_DIR,
        target_defer_fraction=TARGET_DEFER_FRACTION,
        num_mc_passes=NUM_MC_PASSES,
    )

    detectors, thresholds = build_ood_detectors(
        PARTITION_PATH,
        FEATURE_CACHE_DIR,
        HOSPITALS,
        seed=OOD_SEED,
        target_flag_fraction=TARGET_FLAG_FRACTION,
    )

    out_dir = REPO_ROOT / "outputs" / "app_artifacts" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for hospital, detector in detectors.items():
        joblib.dump(detector, out_dir / f"ood_detector_{hospital}.joblib")

    metadata = {
        "checkpoint": str(args.checkpoint),
        "output_name": args.output_name,
        "deferral_threshold": deferral_threshold,
        "target_defer_fraction": TARGET_DEFER_FRACTION,
        "ood_thresholds": thresholds,
        "target_flag_fraction": TARGET_FLAG_FRACTION,
        "ood_seed": OOD_SEED,
        "num_mc_passes": NUM_MC_PASSES,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {out_dir}/ (metadata.json + 3 ood_detector_<hospital>.joblib files)")
    print(f"\nRecommended value for conf/app.yaml's '{args.output_name}' entry:")
    print(f"  deferral_threshold: {deferral_threshold}")


if __name__ == "__main__":
    main()
