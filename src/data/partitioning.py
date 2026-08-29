"""Hospital partitioning (Stage 5): turn the two data sources into simulated FL clients.

Two schemes, per docs/IMPLEMENTATION_PLAN.md Stage 5:
  - **natural**: Kermany = Hospital A; RSNA's patient pool is split into two
    patient-disjoint, label-stratified shards for Hospitals B and C. Reflects genuine
    cross-institutional heterogeneity (different population, equipment, label
    semantics) rather than an artificial split.
  - **dirichlet**: a configurable-alpha Dirichlet partition over a pooled patient set,
    for controlled non-IID sweeps independent of the natural source boundary.

Both operate on Stage 4's frozen per-source records — a patient's train/val/test split
assignment is fixed by Stage 4 (`src/data/splitting.py`) and is never changed here; only
the *hospital* (client) assignment is decided in this module. `src/data/splitting.py` is
intentionally left untouched so the already-committed, frozen
`data/partitions/{kermany,rsna}_splits.json` stay reproducible.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


def natural_shard_rsna(records: list[dict], seed: int = 1000) -> dict[str, list[dict]]:
    """Split RSNA's full patient pool into two patient-disjoint, label-stratified
    shards: Hospital B and Hospital C. A patient's Stage 4 train/val/test assignment
    is preserved regardless of which shard they land in.
    """
    by_patient: dict[str, list[dict]] = defaultdict(list)
    patient_label: dict[str, str] = {}
    for r in records:
        pid = r["patient_id"]
        by_patient[pid].append(r)
        patient_label[pid] = r["label"]

    patients_by_label: dict[str, list[str]] = defaultdict(list)
    for pid, label in patient_label.items():
        patients_by_label[label].append(pid)

    rng = random.Random(seed)
    shard_of_patient: dict[str, str] = {}
    for _, patients in sorted(patients_by_label.items()):
        patients = sorted(patients)
        rng.shuffle(patients)
        half = len(patients) // 2
        for pid in patients[:half]:
            shard_of_patient[pid] = "B"
        for pid in patients[half:]:
            shard_of_patient[pid] = "C"

    result: dict[str, list[dict]] = {"B": [], "C": []}
    for pid, recs in by_patient.items():
        result[shard_of_patient[pid]].extend(recs)
    return result


def dirichlet_partition(
    records: list[dict],
    num_clients: int,
    alpha: float,
    seed: int = 2000,
) -> dict[str, list[dict]]:
    """Synthetic non-IID partition: for each label, draw a Dirichlet(alpha) proportion
    vector over `num_clients` clients and assign that label's patients accordingly.
    Lower alpha => more skewed (non-IID) per-client label distributions; alpha -> inf
    approaches IID. Patient-grouped: every patient's records go to exactly one client.

    Standard technique — see Hsu, Qi & Brown (2019), "Measuring the Effects of
    Non-Identical Data Distribution for Federated Visual Classification."
    """
    by_patient: dict[str, list[dict]] = defaultdict(list)
    patient_label: dict[str, str] = {}
    for r in records:
        pid = r["patient_id"]
        by_patient[pid].append(r)
        patient_label[pid] = r["label"]

    patients_by_label: dict[str, list[str]] = defaultdict(list)
    for pid, label in patient_label.items():
        patients_by_label[label].append(pid)

    rng = np.random.default_rng(seed)
    client_names = [f"client-{i}" for i in range(num_clients)]
    client_patients: dict[str, list[str]] = {name: [] for name in client_names}

    for _, patients in sorted(patients_by_label.items()):
        patients = sorted(patients)
        rng.shuffle(patients)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = (proportions * len(patients)).astype(int)
        counts[-1] += len(patients) - counts.sum()  # fix rounding remainder
        idx = 0
        for name, count in zip(client_names, counts):
            client_patients[name].extend(patients[idx : idx + int(count)])
            idx += int(count)

    result: dict[str, list[dict]] = {name: [] for name in client_names}
    for name, patients in client_patients.items():
        for pid in patients:
            result[name].extend(by_patient[pid])
    return result


def subsample_to_size(
    records: list[dict],
    target_num_images: int,
    seed: int = 1500,
) -> list[dict]:
    """Patient-grouped, label-stratified subsample down to ~target_num_images images.

    Used for DG-3's "report both" resolution: the natural partition (see
    `natural_shard_rsna`) is kept as the unbalanced headline, and this produces a
    size-balanced companion by shrinking the larger hospitals rather than growing the
    smaller one (no synthetic/duplicated data). Sampling is patient-grouped (a
    patient's images are taken or left together) and stratified per label so the
    hospital's original class balance is preserved in the subsample.

    If `target_num_images` is at or above the input's size, returns all records
    unchanged (never upsamples).
    """
    by_patient: dict[str, list[dict]] = defaultdict(list)
    patient_label: dict[str, str] = {}
    for r in records:
        pid = r["patient_id"]
        by_patient[pid].append(r)
        patient_label[pid] = r["label"]

    total_images = len(records)
    if target_num_images >= total_images:
        return list(records)

    patients_by_label: dict[str, list[str]] = defaultdict(list)
    label_image_counts: dict[str, int] = defaultdict(int)
    for pid, recs in by_patient.items():
        label = patient_label[pid]
        patients_by_label[label].append(pid)
        label_image_counts[label] += len(recs)

    rng = random.Random(seed)
    kept: list[dict] = []
    for label, patients in sorted(patients_by_label.items()):
        label_target = round(target_num_images * (label_image_counts[label] / total_images))
        patients = sorted(patients)
        rng.shuffle(patients)
        running = 0
        for pid in patients:
            if running >= label_target:
                break
            kept.extend(by_patient[pid])
            running += len(by_patient[pid])
    return kept


def assert_no_patient_overlap_across_hospitals(hospitals: dict[str, list[dict]]) -> None:
    patient_sets = {name: {r["patient_id"] for r in recs} for name, recs in hospitals.items()}
    names = list(patient_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = patient_sets[names[i]] & patient_sets[names[j]]
            if overlap:
                raise AssertionError(
                    f"Patient overlap between hospital {names[i]} and {names[j]}: {overlap}"
                )


def per_client_stats(hospitals: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        name: {
            "num_images": len(recs),
            "num_patients": len({r["patient_id"] for r in recs}),
            "label_counts": dict(Counter(r["label"] for r in recs)),
        }
        for name, recs in hospitals.items()
    }
