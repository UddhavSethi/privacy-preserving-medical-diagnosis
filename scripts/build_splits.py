"""Build harmonized, patient-grouped train/val/test splits for both dataset sources
(Stage 4). Writes data/partitions/{kermany,rsna}_splits.json.

Split fractions and the seed are fixed here rather than in Hydra config because these
splits are meant to be built once and frozen (CLAUDE.md section 12: "a script that
acquires the data, verifies checksums, builds the partition, and commits the resulting
partition indices"); re-running with different fractions should be a deliberate,
visible change to this file, not an accidental config override.

Usage: uv run python scripts/build_splits.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.labels import load_kermany_records, load_rsna_records
from src.data.splitting import assert_no_patient_overlap, grouped_stratified_split

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"

SEED = 1000  # matches conf/config.yaml's data_partition_seed
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def build_and_write(source_name: str, records: list[dict]) -> dict:
    splits = grouped_stratified_split(records, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED)
    assert_no_patient_overlap(splits)

    summary = {
        split_name: {
            "num_images": len(recs),
            "num_patients": len({r["patient_id"] for r in recs}),
            "label_counts": dict(Counter(r["label"] for r in recs)),
        }
        for split_name, recs in splits.items()
    }

    out = {
        "source": source_name,
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "summary": summary,
        "splits": splits,
    }
    out_path = PARTITIONS_DIR / f"{source_name}_splits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"{source_name}: {json.dumps(summary, indent=2)}")
    print(f"written: {out_path}\n")
    return summary


def main() -> None:
    build_and_write("kermany", load_kermany_records())
    build_and_write("rsna", load_rsna_records())


if __name__ == "__main__":
    main()
