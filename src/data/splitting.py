"""Patient-grouped, label-stratified train/val/test splitting (ADR-7): a patient's
images never span more than one split.
"""
from __future__ import annotations

import random
from collections import defaultdict


def grouped_stratified_split(
    records: list[dict],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1000,
) -> dict[str, list[dict]]:
    """Split records into train/val/test with zero patient_id crossing a split boundary.

    Grouped and stratified: every patient's entire group of images goes to exactly one
    split, and the split is carried out independently per label so class balance is
    preserved per split. This is safe because, in both approved sources, a given
    patient_id maps to exactly one label (true by construction for RSNA; verified
    empirically for Kermany — zero patient ids shared between its NORMAL and PNEUMONIA
    directories).
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    by_patient: dict[str, list[dict]] = defaultdict(list)
    patient_label: dict[str, str] = {}
    for r in records:
        pid = r["patient_id"]
        by_patient[pid].append(r)
        if pid in patient_label and patient_label[pid] != r["label"]:
            raise ValueError(f"patient_id {pid} has conflicting labels across records")
        patient_label[pid] = r["label"]

    patients_by_label: dict[str, list[str]] = defaultdict(list)
    for pid, label in patient_label.items():
        patients_by_label[label].append(pid)

    rng = random.Random(seed)
    split_of_patient: dict[str, str] = {}
    for _, patients in sorted(patients_by_label.items()):
        patients = sorted(patients)  # deterministic order before shuffling
        rng.shuffle(patients)
        n = len(patients)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        for pid in patients[:n_test]:
            split_of_patient[pid] = "test"
        for pid in patients[n_test : n_test + n_val]:
            split_of_patient[pid] = "val"
        for pid in patients[n_test + n_val :]:
            split_of_patient[pid] = "train"

    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for pid, recs in by_patient.items():
        result[split_of_patient[pid]].extend(recs)
    return result


def assert_no_patient_overlap(splits: dict[str, list[dict]]) -> None:
    patient_sets = {name: {r["patient_id"] for r in recs} for name, recs in splits.items()}
    names = list(patient_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = patient_sets[names[i]] & patient_sets[names[j]]
            if overlap:
                raise AssertionError(f"Patient overlap between {names[i]} and {names[j]}: {overlap}")
