"""Build the frozen-backbone feature cache (Stage 9, DG-5: K=5 augmented views).

Usage: uv run python scripts/build_feature_cache.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from src.data.datasets import LABEL_TO_INDEX, load_split_records
from src.data.feature_cache import (
    FEATURE_DIM,
    FeatureCacheKey,
    cache_file_path,
    compute_pooled_features,
    save_feature_bank,
)
from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.densenet_head import DenseNet121Head

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAHE_CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"
FEATURE_CACHE_DIR = REPO_ROOT / "data" / "feature_cache"
CLAHE_PARAMS = ClaheParams()

NUM_AUGMENTED_VIEWS = 5  # DG-5 resolution
KEY = FeatureCacheKey(
    image_size=224,
    num_augmented_views=NUM_AUGMENTED_VIEWS,
    rotation_degrees=10.0,
    brightness=0.1,
    contrast=0.1,
)
VIEW_SEED = 1000  # data_partition_seed (conf/config.yaml), reused for view determinism


def build_for_source_split(model, source: str, split: str, num_views: int) -> None:
    records = load_split_records(source, split)
    if not records:
        return

    train_transform = build_train_transform(
        KEY.image_size, KEY.rotation_degrees, KEY.brightness, KEY.contrast
    )
    eval_transform = build_eval_transform(KEY.image_size)

    all_features = torch.zeros(len(records), num_views + 1, FEATURE_DIM, dtype=torch.float32)
    record_ids, labels = [], []

    t0 = time.time()
    for i, r in enumerate(records):
        cache_path = cache_path_for(CLAHE_CACHE_DIR, source, r["relative_path"], CLAHE_PARAMS)
        image_rgb = load_from_cache(cache_path)

        for v in range(num_views):
            torch.manual_seed(VIEW_SEED + i * 1000 + v)  # deterministic per (image, view)
            all_features[i, v] = compute_pooled_features(model, image_rgb, train_transform)
        all_features[i, num_views] = compute_pooled_features(model, image_rgb, eval_transform)

        record_ids.append(r.get("patient_id", r["relative_path"]))
        labels.append(LABEL_TO_INDEX[r["label"]])

        if (i + 1) % 2000 == 0:
            print(f"  {source}/{split}: {i + 1}/{len(records)}", flush=True)

    elapsed = time.time() - t0
    print(f"{source}/{split}: {len(records)} images x {num_views + 1} views in {elapsed:.1f}s")

    out_path = cache_file_path(FEATURE_CACHE_DIR, source, split, KEY)
    save_feature_bank(out_path, all_features, record_ids, labels)
    print(f"written: {out_path}\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = DenseNet121Head().to(device)
    model.eval()

    for source in ("kermany", "rsna"):
        for split in ("train", "val", "test"):
            num_views = NUM_AUGMENTED_VIEWS if split == "train" else 0
            build_for_source_split(model, source, split, num_views)

    print(f"Feature cache dir: {FEATURE_CACHE_DIR}")
    print(f"Cache key: {KEY}")


if __name__ == "__main__":
    main()
