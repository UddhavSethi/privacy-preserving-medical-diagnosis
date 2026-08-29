"""Build and freeze hospital partitions (Stage 5).

Writes data/partitions/hospitals_natural.json (Kermany=A, RSNA patient-disjoint shards
B/C, full natural imbalance — DG-3's "keep the natural imbalance" half; whether to also
produce a size-balanced variant is still open, see CLAUDE.md section 14) and
data/partitions/hospitals_dirichlet_alpha_sweep.json (a demonstration sweep over several
alpha values, pooling both sources).

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
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"

RSNA_SHARD_SEED = 1000
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


def build_natural() -> None:
    kermany_records = _load_source_records("kermany")
    rsna_records = _load_source_records("rsna")

    rsna_shards = natural_shard_rsna(rsna_records, seed=RSNA_SHARD_SEED)
    hospitals = {"A": kermany_records, "B": rsna_shards["B"], "C": rsna_shards["C"]}
    assert_no_patient_overlap_across_hospitals(hospitals)

    out = {
        "scheme": "natural",
        "note": (
            "Full natural imbalance — Hospital A (Kermany) is far smaller than B/C "
            "(RSNA shards). This is DG-3's 'keep the natural imbalance' option; a "
            "size-balanced variant is a separate, not-yet-built config pending owner "
            "input (CLAUDE.md section 14)."
        ),
        "rsna_shard_seed": RSNA_SHARD_SEED,
        "summary": per_client_stats(hospitals),
        "hospitals": hospitals,
    }
    out_path = PARTITIONS_DIR / "hospitals_natural.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("natural:", json.dumps(out["summary"], indent=2))
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
    build_natural()
    build_dirichlet_sweep()


if __name__ == "__main__":
    main()
