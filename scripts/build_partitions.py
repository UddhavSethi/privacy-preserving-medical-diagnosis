"""Build and freeze hospital partitions (Stage 5).

Writes three files, per Decision Gate DG-3's "report both" resolution (owner-approved
2026-08-29):
  - data/partitions/hospitals_natural.json — Kermany=A, RSNA patient-disjoint shards
    B/C, full natural ~4.5x imbalance. The headline, realistic regime.
  - data/partitions/hospitals_natural_balanced.json — same A/B/C assignment, but B and
    C are subsampled down to Hospital A's size (no upsampling of A). A companion result
    isolating the effect of client-size imbalance from everything else.
  - data/partitions/hospitals_dirichlet_alpha_sweep.json — a demonstration sweep over
    several alpha values, pooling both sources, independent of DG-3.

Usage: uv run python scripts/build_partitions.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.partitioning import (
    assert_no_patient_overlap_across_hospitals,
    dirichlet_partition,
    natural_shard_rsna,
    per_client_stats,
    subsample_to_size,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"

RSNA_SHARD_SEED = 1000
BALANCE_SEED = 1500
DIRICHLET_SEED = 2000
DIRICHLET_NUM_CLIENTS = 5
DIRICHLET_ALPHAS = [0.1, 0.5, 1.0, 10.0]


def _load_source_records(source: str) -> list[dict]:
    data = json.loads((PARTITIONS_DIR / f"{source}_splits.json").read_text())
    records = []
    for split_name, recs in data["splits"].items():
        for r in recs:
            records.append({**r, "frozen_split": split_name})
    return records


def build_natural() -> dict[str, list[dict]]:
    kermany_records = _load_source_records("kermany")
    rsna_records = _load_source_records("rsna")

    rsna_shards = natural_shard_rsna(rsna_records, seed=RSNA_SHARD_SEED)
    hospitals = {"A": kermany_records, "B": rsna_shards["B"], "C": rsna_shards["C"]}
    assert_no_patient_overlap_across_hospitals(hospitals)

    out = {
        "scheme": "natural",
        "note": (
            "Full natural imbalance — Hospital A (Kermany) is far smaller than B/C "
            "(RSNA shards). DG-3 resolution: report both this and the size-balanced "
            "variant (hospitals_natural_balanced.json) — this is the unbalanced half."
        ),
        "rsna_shard_seed": RSNA_SHARD_SEED,
        "summary": per_client_stats(hospitals),
        "hospitals": hospitals,
    }
    out_path = PARTITIONS_DIR / "hospitals_natural.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("natural:", json.dumps(out["summary"], indent=2))
    print(f"written: {out_path}\n")
    return hospitals


def build_natural_balanced(natural_hospitals: dict[str, list[dict]]) -> None:
    target_size = len(natural_hospitals["A"])  # Hospital A (Kermany) is the smallest
    hospitals = {
        "A": natural_hospitals["A"],
        "B": subsample_to_size(natural_hospitals["B"], target_size, seed=BALANCE_SEED),
        "C": subsample_to_size(natural_hospitals["C"], target_size, seed=BALANCE_SEED),
    }
    assert_no_patient_overlap_across_hospitals(hospitals)

    out = {
        "scheme": "natural_balanced",
        "note": (
            "DG-3 resolution: report both. Same A/B/C assignment as hospitals_natural.json, "
            "but Hospitals B and C are label-stratified-subsampled down to Hospital A's "
            "size (never upsampled) so all three hospitals contribute comparable amounts "
            "of data — isolates the effect of client-size imbalance from other factors."
        ),
        "rsna_shard_seed": RSNA_SHARD_SEED,
        "balance_seed": BALANCE_SEED,
        "target_size": target_size,
        "summary": per_client_stats(hospitals),
        "hospitals": hospitals,
    }
    out_path = PARTITIONS_DIR / "hospitals_natural_balanced.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("natural_balanced:", json.dumps(out["summary"], indent=2))
    print(f"written: {out_path}\n")


def build_dirichlet_sweep() -> None:
    pooled = _load_source_records("kermany") + _load_source_records("rsna")

    sweep = {}
    for alpha in DIRICHLET_ALPHAS:
        parts = dirichlet_partition(
            pooled, num_clients=DIRICHLET_NUM_CLIENTS, alpha=alpha, seed=DIRICHLET_SEED
        )
        assert_no_patient_overlap_across_hospitals(parts)
        sweep[str(alpha)] = per_client_stats(parts)

    out = {
        "scheme": "dirichlet_sweep",
        "note": "Per-client label distribution at each alpha, pooling both sources. Demonstrates the sweep required by Stage 5's testing criteria; not a headline result.",
        "num_clients": DIRICHLET_NUM_CLIENTS,
        "seed": DIRICHLET_SEED,
        "alphas": DIRICHLET_ALPHAS,
        "summary_by_alpha": sweep,
    }
    out_path = PARTITIONS_DIR / "hospitals_dirichlet_alpha_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("dirichlet sweep summary:", json.dumps(sweep, indent=2))
    print(f"written: {out_path}\n")


def main() -> None:
    natural_hospitals = build_natural()
    build_natural_balanced(natural_hospitals)
    build_dirichlet_sweep()


if __name__ == "__main__":
    main()
