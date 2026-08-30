"""Stage 17 — per-hospital data isolation for the Docker Compose demonstration.

The committed feature cache (`data/feature_cache/`) stores one bank per *source*
(`kermany_{train,val,test}.pt`, `rsna_{train,val,test}.pt`) — Hospital B and C are
both patient-disjoint shards of the *same* `rsna_*` files, distinguished only by
patient_id membership in the partition JSON. Mounting that shared cache into every
hospital's container would mean each container physically *could* read another
hospital's records; only in-process filtering would stop it. That is not the
"each container can access only its own data" claim Stage 17's own testing
criterion asks for.

This script pre-slices each hospital's real train/val/test feature bank once,
using the exact same filtering `src/training/trainer.py::load_hospital_features`
already does, and writes it to its own directory (`data/deployment_shards/hospital_X/`)
alongside a partition JSON scoped to *only* that hospital. Docker Compose then
bind-mounts one such directory, read-only, into each hospital's container — so
Hospital B's container has no path at which Hospital C's (or Hospital A's) data
could even exist, not merely a promise that the code won't read it.

No change to the federated training/aggregation code: `client_app.py`'s
`_resolve_config()` (added this stage) just points `feature-cache-dir` /
`partition-path` at a per-node path via `--node-config`, and
`load_hospital_features()` runs completely unchanged against these smaller,
single-hospital directories.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.feature_cache import cache_file_path, load_feature_bank, save_feature_bank
from src.training.trainer import FEATURE_CACHE_DIR, FEATURE_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
OUTPUT_ROOT = REPO_ROOT / "data" / "deployment_shards"
HOSPITALS = ["A", "B", "C"]


def _extract_hospital(hospital: str, partition: dict) -> None:
    records = partition["hospitals"][hospital]
    if not records:
        raise ValueError(f"No records for hospital {hospital} in {PARTITION_PATH}")
    source = records[0]["source"]

    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in records:
        by_split[r["frozen_split"]].add(r["patient_id"])

    out_dir = OUTPUT_ROOT / f"hospital_{hospital}"
    hash_dir = FEATURE_KEY.hash_suffix()

    for split in ("train", "val", "test"):
        bank_path = cache_file_path(FEATURE_CACHE_DIR, source, split, FEATURE_KEY)
        bank = load_feature_bank(bank_path)
        keep_idx = [i for i, rid in enumerate(bank["record_ids"]) if rid in by_split[split]]
        if not keep_idx:
            raise ValueError(f"Hospital {hospital}, split {split}: 0 records extracted from {bank_path}")

        out_path = out_dir / hash_dir / f"{source}_{split}.pt"
        save_feature_bank(
            out_path,
            features=bank["features"][keep_idx],
            record_ids=[bank["record_ids"][i] for i in keep_idx],
            labels=[bank["labels"][i] for i in keep_idx],
        )
        print(f"  {split}: {len(keep_idx)} records -> {out_path.relative_to(REPO_ROOT)}")

    scoped_partition = {"hospitals": {hospital: records}}
    partition_out = out_dir / "partition.json"
    partition_out.write_text(json.dumps(scoped_partition, indent=2))
    print(f"  scoped partition -> {partition_out.relative_to(REPO_ROOT)}")


def main() -> None:
    partition = json.loads(PARTITION_PATH.read_text())
    for hospital in HOSPITALS:
        print(f"Hospital {hospital}:")
        _extract_hospital(hospital, partition)


if __name__ == "__main__":
    main()
