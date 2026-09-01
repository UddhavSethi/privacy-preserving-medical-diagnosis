"""Raw (uncached-feature) CLAHE-cached image dataset -- for training paths that
need actual images through a (partially) trainable backbone, unlike Stage 9's
pooled-feature cache which assumes a fully frozen backbone (see
`DenseNet121Head`'s `fine_tune_last_block` docstring). Added 2026-08-31 alongside
the ADR-1 GroupNorm fallback pilot; shared by the centralized fine-tuning script
and the federated fine-tuning client app so both read images identically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache

LABEL_TO_INDEX = {"Normal": 0, "Pneumonia": 1}


class RawImageDataset(Dataset):
    """Reads CLAHE-cached images directly for a list of `hospitals_natural.json`-
    style records. Mixed sources allowed -- each record carries its own `source`
    (unlike `src/data/datasets.py::ChestXrayDataset`, which fixes one source per
    instance)."""

    def __init__(self, records: list[dict], transform: Callable, cache_dir: Path) -> None:
        self.records = records
        self.transform = transform
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        cache_path = cache_path_for(self.cache_dir, record["source"], record["relative_path"], ClaheParams())
        image = load_from_cache(cache_path)
        tensor = self.transform(image)
        label = LABEL_TO_INDEX[record["label"]]
        return tensor, label


def records_for(partition: dict, hospitals: list[str], split: str) -> list[dict]:
    out = []
    for h in hospitals:
        out += [r for r in partition["hospitals"][h] if r["frozen_split"] == split]
    return out
