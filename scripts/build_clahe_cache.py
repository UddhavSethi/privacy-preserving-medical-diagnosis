"""Precompute and cache CLAHE-enhanced images for both dataset sources (Stage 6).

Removes CLAHE from the per-epoch hot path (CLAUDE.md ADR-6) by writing every image's
enhanced RGB output to disk once, keyed by source + a hash of the CLAHE parameters +
the image's relative path — a parameter change produces a disjoint cache directory
rather than silently reusing stale output.

Also writes a small before/after visual sample to data/manifests/clahe_sample/ and logs
it to MLflow as an artifact (Stage 6's "visual inspection artifact" requirement).

Usage: uv run python scripts/build_clahe_cache.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import mlflow
from omegaconf import OmegaConf

from src.data.preprocessing import (
    ClaheParams,
    cache_path_for,
    load_dicom_as_uint8_bgr,
    load_jpeg_bgr,
    preprocess_to_rgb,
    save_to_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"
SAMPLE_DIR = REPO_ROOT / "data" / "manifests" / "clahe_sample"
RSNA_RAW_ROOT = REPO_ROOT / "data" / "raw" / "rsna"

# Fixed, logged CLAHE parameters (ADR-6) — standard values for chest X-ray contrast
# enhancement in the literature; not tuned per-image.
PARAMS = ClaheParams(clip_limit=2.0, tile_grid_size=(8, 8))


def _kermany_chest_xray_root() -> Path:
    roots = list((REPO_ROOT / "data" / "raw" / "kermany").rglob("chest_xray"))
    if not roots:
        raise FileNotFoundError(
            "Kermany chest_xray/ not found — run scripts/download_kermany.py first"
        )
    return roots[0]


def _iter_source_records(source: str) -> list[dict]:
    data = json.loads((PARTITIONS_DIR / f"{source}_splits.json").read_text())
    records = []
    for recs in data["splits"].values():
        records.extend(recs)
    return records


def build_cache_for_source(source: str, raw_root: Path, loader) -> tuple[int, int]:
    records = _iter_source_records(source)
    hits, misses = 0, 0
    for i, r in enumerate(records):
        cache_path = cache_path_for(CACHE_DIR, source, r["relative_path"], PARAMS)
        if cache_path.exists():
            hits += 1
            continue
        bgr = loader(raw_root / r["relative_path"])
        rgb = preprocess_to_rgb(bgr, PARAMS)
        save_to_cache(rgb, cache_path)
        misses += 1
        if (i + 1) % 2000 == 0:
            print(f"  {source}: {i + 1}/{len(records)} processed ({hits} hits, {misses} built)", flush=True)
    print(f"{source}: {hits} cache hits, {misses} newly built, {len(records)} total")
    return hits, misses


def save_before_after_samples(source: str, raw_root: Path, loader, n: int = 3) -> None:
    records = _iter_source_records(source)[:n]
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        bgr = loader(raw_root / r["relative_path"])
        rgb = preprocess_to_rgb(bgr, PARAMS)
        cv2.imwrite(str(SAMPLE_DIR / f"{source}_{i}_before.png"), bgr)
        cv2.imwrite(str(SAMPLE_DIR / f"{source}_{i}_after.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    kermany_root = _kermany_chest_xray_root()
    build_cache_for_source("kermany", kermany_root, load_jpeg_bgr)
    build_cache_for_source("rsna", RSNA_RAW_ROOT, load_dicom_as_uint8_bgr)

    save_before_after_samples("kermany", kermany_root, load_jpeg_bgr)
    save_before_after_samples("rsna", RSNA_RAW_ROOT, load_dicom_as_uint8_bgr)

    cfg = OmegaConf.load(REPO_ROOT / "conf" / "config.yaml")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("clahe_cache")
    with mlflow.start_run(run_name="build_clahe_cache"):
        mlflow.log_param("clip_limit", PARAMS.clip_limit)
        mlflow.log_param("tile_grid_size", str(PARAMS.tile_grid_size))
        mlflow.log_artifacts(str(SAMPLE_DIR), artifact_path="clahe_before_after_samples")

    print(f"\nCLAHE params: clip_limit={PARAMS.clip_limit}, tile_grid_size={PARAMS.tile_grid_size}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Sample dir: {SAMPLE_DIR} (logged to MLflow experiment 'clahe_cache')")


if __name__ == "__main__":
    main()
