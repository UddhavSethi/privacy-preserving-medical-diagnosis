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

# RSNA_SHARD_SEED must NEVER equal conf/config.yaml's data_partition_seed (1000), which
# Stage 4 (src/data/splitting.py) already used for a sorted-list-then-shuffle split over
# this same patient population. Reusing that seed here reproduces the identical
# permutation, so cutting it in half for B/C silently reproduces Stage 4's test/val/train
# cut points instead of producing an independent shard split — a real bug found and
# fixed during Stage 11 (Hospital C ended up with zero val/test records; Hospital B
# absorbed all of RSNA's val+test). See tests/test_partitioning.py's regression test.
RSNA_SHARD_SEED = 5000
BALANCE_SEED = 1500
DIRICHLET_SEED = 2000
DIRICHLET_NUM_CLIENTS = 5
DIRICHLET_ALPHAS = [0.1, 0.5, 1.0, 10.0]

# Stage 21 (owner-approved 2026-08-30): the actual trainable Dirichlet ablation
# partitions, distinct from build_dirichlet_sweep()'s demonstration-only summary
# above (which never persisted per-client records, only aggregate stats, and used
# different parameters). num_clients=3 for direct comparability with the natural/
# balanced regimes' client count; alpha in {0.1, 1.0} = strong vs. mild synthetic
# heterogeneity, a secondary/supplementary result per CLAUDE.md's own framing that
# natural non-IID is preferred.
DIRICHLET_ABLATION_SEED = 2100  # deliberately different from DIRICHLET_SEED above
DIRICHLET_ABLATION_NUM_CLIENTS = 3
DIRICHLET_ABLATION_ALPHAS = [0.1, 1.0]


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


def build_dirichlet_ablation_partitions() -> None:
    """Stage 21's actual trainable Dirichlet partitions (see the module-level
    constants' comment for why this is separate from build_dirichlet_sweep()).
    Writes one full hospitals_dirichlet_alpha<alpha>.json per alpha, in the same
    {"hospitals": {...}} format as hospitals_natural.json, usable directly by
    load_hospital_features (which as of Stage 21 supports multi-source hospitals —
    Dirichlet pools both Kermany and RSNA before assigning patients to clients, so
    a given synthetic client can and often does span both sources).

    Client keys are remapped from dirichlet_partition()'s own "client-0"/"client-1"/
    "client-2" to "A"/"B"/"C" — `src/federated/client_app.py`'s
    `PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}` is hardcoded to those names
    (matching every other partition file), and changing that well-tested mapping to
    accommodate one partition scheme was judged riskier than renaming keys here.
    """
    pooled = _load_source_records("kermany") + _load_source_records("rsna")
    client_names = ["A", "B", "C"]

    for alpha in DIRICHLET_ABLATION_ALPHAS:
        raw = dirichlet_partition(
            pooled,
            num_clients=DIRICHLET_ABLATION_NUM_CLIENTS,
            alpha=alpha,
            seed=DIRICHLET_ABLATION_SEED,
        )
        hospitals = dict(zip(client_names, raw.values(), strict=True))
        assert_no_patient_overlap_across_hospitals(hospitals)

        out = {
            "scheme": "dirichlet",
            "note": (
                f"Stage 21 synthetic non-IID ablation partition, alpha={alpha}, "
                f"{DIRICHLET_ABLATION_NUM_CLIENTS} clients, pooling both sources. "
                "Secondary/supplementary to the natural regime per CLAUDE.md's own "
                "framing (natural non-IID is preferred, this is 'a controlled sweep')."
            ),
            "alpha": alpha,
            "num_clients": DIRICHLET_ABLATION_NUM_CLIENTS,
            "seed": DIRICHLET_ABLATION_SEED,
            "summary": per_client_stats(hospitals),
            "hospitals": hospitals,
        }
        out_path = PARTITIONS_DIR / f"hospitals_dirichlet_alpha{alpha}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"dirichlet alpha={alpha}:", json.dumps(out["summary"], indent=2))
        print(f"written: {out_path}\n")


def main() -> None:
    natural_hospitals = build_natural()
    build_natural_balanced(natural_hospitals)
    build_dirichlet_sweep()
    build_dirichlet_ablation_partitions()


if __name__ == "__main__":
    main()
