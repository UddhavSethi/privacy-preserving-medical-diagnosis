"""Stage 18 — Grad-CAM explainability, batch generation.

Loads Stage 12's trained centralized global-model checkpoint, evaluates it on
the pooled test set to find real true positive / false positive / true
negative / false negative examples (threshold=0.5, Stage 10's default
policy), generates a Grad-CAM overlay for each selected example targeting its
predicted class, and saves them to outputs/explanations/ plus logs them to
MLflow as artifacts — "Explanation artifacts logged to MLflow, and figures
for the paper" (docs/IMPLEMENTATION_PLAN.md's Stage 18 write-up).

Execution site note: this script runs centrally (over the pooled test set,
for reporting/paper-figure purposes) rather than "client-side, on the
received global model" as CLAUDE.md section 9 describes for the *deployed*
system — `src/explain/gradcam.py` itself is execution-site-agnostic (a plain
function over a model + image) and is exactly what a client process would
call; this script is simply this stage's own reporting/validation harness,
analogous to how Stage 11/12's train_local.py / train_centralized.py run
centrally to produce baseline numbers without contradicting the federated
architecture.

Usage: uv run python scripts/generate_explanations.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.data.feature_cache import cache_file_path, load_feature_bank
from src.explain.gradcam import NORMAL_CLASS_INDEX, PNEUMONIA_CLASS_INDEX, generate_overlay
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import FEATURE_CACHE_DIR, FEATURE_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
CLAHE_CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"
CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "centralized_baseline" / "natural_seed42.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "explanations"

HOSPITALS = ["A", "B", "C"]
THRESHOLD = 0.5  # Stage 10's default decision threshold, applied explicitly (not implicit)
EXAMPLES_PER_BUCKET = 3
SHUFFLE_SEED = 42  # records are collected Hospital A (Kermany) first, then B/C (RSNA
# shards); without shuffling, `items[:EXAMPLES_PER_BUCKET]` picked almost
# entirely Kermany examples, since Kermany's classifier is highly accurate and
# its TN/FP/FN buckets fill up before any RSNA record is ever reached — found
# by inspecting the actual saved output files, not assumed. A fixed seed keeps
# the selection reproducible run to run.


def _collect_test_records() -> list[dict]:
    """One entry per pooled test-set image, with everything needed to both
    re-derive the model's prediction and load the raw CLAHE'd image:
    source, relative_path, true label, and its 1024-dim eval-view feature."""
    partition = json.loads(PARTITION_PATH.read_text())
    records = []
    for hospital in HOSPITALS:
        for r in partition["hospitals"][hospital]:
            if r["frozen_split"] == "test":
                records.append(r)

    by_source: dict[str, list[dict]] = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    enriched = []
    for source, source_records in by_source.items():
        bank_path = cache_file_path(FEATURE_CACHE_DIR, source, "test", FEATURE_KEY)
        bank = load_feature_bank(bank_path)
        id_to_idx = {rid: i for i, rid in enumerate(bank["record_ids"])}
        for r in source_records:
            idx = id_to_idx.get(r["patient_id"])
            if idx is None:
                continue
            enriched.append(
                {
                    **r,
                    "feature": bank["features"][idx],  # (1, 1024) eval-view
                    "true_label": bank["labels"][idx],
                }
            )
    random.Random(SHUFFLE_SEED).shuffle(enriched)  # see SHUFFLE_SEED's comment
    return enriched


def _bucket_examples(model: DenseNet121Head, records: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"TP": [], "FP": [], "TN": [], "FN": []}
    model.eval()
    with torch.no_grad():
        for r in records:
            # Test-split banks store one eval-only view per record, so a single
            # scalar index already returns shape (1, 1024) — batch-ready as-is.
            logits = model.classifier(r["feature"])
            prob_pneumonia = torch.softmax(logits, dim=1)[0, PNEUMONIA_CLASS_INDEX].item()
            predicted = PNEUMONIA_CLASS_INDEX if prob_pneumonia >= THRESHOLD else NORMAL_CLASS_INDEX
            true_label = r["true_label"]

            if predicted == PNEUMONIA_CLASS_INDEX and true_label == PNEUMONIA_CLASS_INDEX:
                bucket = "TP"
            elif predicted == PNEUMONIA_CLASS_INDEX and true_label == NORMAL_CLASS_INDEX:
                bucket = "FP"
            elif predicted == NORMAL_CLASS_INDEX and true_label == NORMAL_CLASS_INDEX:
                bucket = "TN"
            else:
                bucket = "FN"
            buckets[bucket].append({**r, "predicted": predicted})
    return buckets


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "conf" / "config.yaml")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("gradcam_explanations")

    model = DenseNet121Head()
    model.classifier.load_state_dict(torch.load(CHECKPOINT, weights_only=True))

    print("Collecting pooled test-set records...")
    records = _collect_test_records()
    print(f"  {len(records)} test-set images")

    print("Bucketing by prediction vs. ground truth...")
    buckets = _bucket_examples(model, records)
    for name, items in buckets.items():
        print(f"  {name}: {len(items)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="gradcam_batch"):
        mlflow.log_param("checkpoint", str(CHECKPOINT.relative_to(REPO_ROOT)))
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("examples_per_bucket", EXAMPLES_PER_BUCKET)
        for name, items in buckets.items():
            mlflow.log_metric(f"count_{name}", len(items))

        for bucket_name, items in buckets.items():
            bucket_dir = OUTPUT_DIR / bucket_name
            bucket_dir.mkdir(parents=True, exist_ok=True)
            for r in items[:EXAMPLES_PER_BUCKET]:
                cache_path = cache_path_for(CLAHE_CACHE_DIR, r["source"], r["relative_path"], ClaheParams())
                if not cache_path.exists():
                    continue
                image_rgb = load_from_cache(cache_path)
                overlay = generate_overlay(model, image_rgb, target_class=r["predicted"])

                out_name = f"{r['patient_id']}.png"
                out_path = bucket_dir / out_name
                Image.fromarray(overlay.overlay_rgb).save(out_path)
                mlflow.log_artifact(str(out_path), artifact_path=f"gradcam/{bucket_name}")

        print(f"\nExplanation overlays written to {OUTPUT_DIR.relative_to(REPO_ROOT)}/ and logged to MLflow.")


if __name__ == "__main__":
    main()
