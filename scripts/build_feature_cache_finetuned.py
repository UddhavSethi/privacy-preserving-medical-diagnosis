"""Builds a pooled-feature cache from round 9's fine-tuned backbone (ADR-1
GroupNorm fallback, docs/adr1_groupnorm_fallback.md), so deferral calibration
and per-hospital OOD detection can be re-enabled for it (both were disabled
at deployment time — sections 8/10/16 — because Stage 9's original cache is a
frozen-backbone artifact, invalid the moment part of the backbone becomes
trainable, and using it would calibrate a threshold/detector against features
this checkpoint's own backbone never actually produces).

Mirrors `scripts/build_feature_cache.py` (Stage 9) exactly in format and
reuses its cache read/write functions unmodified (`src/data/feature_cache.py`)
-- only three things differ: the model (round 9's checkpoint, not a fresh
frozen backbone), the output directory, and `NUM_AUGMENTED_VIEWS=0` instead
of 5. The 5 augmented views Stage 9 caches exist for training-time
regularization; nothing this cache is used for (`build_ood_detectors`,
`calibrate_deferral_threshold` -- both only ever read `[:, -1, :]`, the
eval-style view) touches them, so building them would be pure wasted compute
here -- roughly a 5x cost cut on the train split, no functional difference for
this purpose. `FeatureCacheKey.hash_suffix()` already folds `num_augmented_views`
into the cache path, so this cache lands in its own subdirectory automatically,
never colliding with Stage 9's.

Usage: uv run python scripts/build_feature_cache_finetuned.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from src.data.datasets import LABEL_TO_INDEX, load_split_records
from src.data.feature_cache import (
    FeatureCacheKey,
    cache_file_path,
    compute_pooled_features,
    save_feature_bank,
)
from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.data.transforms import build_eval_transform
from src.models.densenet_head import DenseNet121Head

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAHE_CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"
FEATURE_CACHE_DIR = REPO_ROOT / "data" / "feature_cache_finetuned"
CHECKPOINT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "finetuned" / "fedavg_natural_seed42.pt"
CLAHE_PARAMS = ClaheParams()

NUM_AUGMENTED_VIEWS = 0  # unlike Stage 9 -- see module docstring
KEY = FeatureCacheKey(
    image_size=224,
    num_augmented_views=NUM_AUGMENTED_VIEWS,
    rotation_degrees=10.0,
    brightness=0.1,
    contrast=0.1,
)


def build_for_source_split(model, source: str, split: str) -> None:
    records = load_split_records(source, split)
    if not records:
        return

    eval_transform = build_eval_transform(KEY.image_size)
    all_features = torch.zeros(len(records), 1, 1024, dtype=torch.float32)
    record_ids, labels = [], []

    t0 = time.time()
    for i, r in enumerate(records):
        cache_path = cache_path_for(CLAHE_CACHE_DIR, source, r["relative_path"], CLAHE_PARAMS)
        image_rgb = load_from_cache(cache_path)
        all_features[i, 0] = compute_pooled_features(model, image_rgb, eval_transform)
        record_ids.append(r.get("patient_id", r["relative_path"]))
        labels.append(LABEL_TO_INDEX[r["label"]])
        if (i + 1) % 2000 == 0:
            print(f"  {source}/{split}: {i + 1}/{len(records)}", flush=True)

    elapsed = time.time() - t0
    print(f"{source}/{split}: {len(records)} images in {elapsed:.1f}s")

    out_path = cache_file_path(FEATURE_CACHE_DIR, source, split, KEY)
    save_feature_bank(out_path, all_features, record_ids, labels)
    print(f"written: {out_path}\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = DenseNet121Head(fine_tune_last_block=True).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_trainable_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()

    for source in ("kermany", "rsna"):
        for split in ("train", "val", "test"):
            build_for_source_split(model, source, split)

    print(f"Feature cache dir: {FEATURE_CACHE_DIR}")
    print(f"Cache key: {KEY}")


if __name__ == "__main__":
    main()
