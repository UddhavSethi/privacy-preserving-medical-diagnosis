"""torchvision Dataset + DataLoader wiring, reading from the Stage 6 CLAHE cache
(Stage 7). Nothing here touches OpenCV or raw source images — only the cached,
already-CLAHE'd RGB PNGs produced by `scripts/build_clahe_cache.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.utils.seeding import make_generator, seed_worker

REPO_ROOT = Path(__file__).resolve().parents[2]
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"

LABEL_TO_INDEX = {"Normal": 0, "Pneumonia": 1}


class ChestXrayDataset(Dataset):
    """Reads CLAHE-cached images for a given list of records (already filtered to
    one source/split/hospital by the caller — this class has no partitioning logic
    of its own)."""

    def __init__(
        self,
        records: list[dict],
        source: str,
        transform: Optional[Callable] = None,
        cache_dir: Path = CACHE_DIR,
        clahe_params: ClaheParams = ClaheParams(),
    ) -> None:
        self.records = records
        self.source = source
        self.transform = transform
        self.cache_dir = cache_dir
        self.clahe_params = clahe_params

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        cache_path = cache_path_for(
            self.cache_dir, self.source, record["relative_path"], self.clahe_params
        )
        image = load_from_cache(cache_path)  # RGB uint8 numpy array

        if self.transform is not None:
            image = self.transform(image)

        label = LABEL_TO_INDEX[record["label"]]
        return image, label


def load_split_records(
    source: str, split: str, partitions_dir: Path = PARTITIONS_DIR
) -> list[dict]:
    """Load one source's frozen Stage 4 split ("train"/"val"/"test")."""
    data = json.loads((partitions_dir / f"{source}_splits.json").read_text())
    return data["splits"][split]


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """A DataLoader with seeded shuffling and seeded worker processes (Stage 7's
    flagged risk: forgotten worker seeding is a classic silent nondeterminism source).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed) if shuffle else None,
    )
