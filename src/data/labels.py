"""Label mapping and patient-identifier extraction for both dataset sources (Stage 4).

Unifies Kermany and RSNA into one binary schema — {"Normal", "Pneumonia"} — and attaches
a patient_id to every record so splitting.py can enforce patient-level separation (ADR-7).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

KERMANY_FILENAME_RE = re.compile(r"^([A-Za-z]+)-(\d+)-(\d+)\.jpe?g$")


def load_kermany_records(raw_root: Path | None = None) -> list[dict]:
    """One record per Kermany image, with a patient_id parsed from the filename.

    This Mendeley release's filenames are `<CLASS>-<accession-id>-<seq>.jpeg` for both
    NORMAL and PNEUMONIA — the accession id repeats across a patient's own images in
    *both* classes, so patient-level grouping is fully available here. This is a
    different, better-structured naming scheme than the third-party Kaggle mirror's
    `personNNN_bacteria_NNN.jpeg` / `IM-NNNN-NNNN.jpeg` convention that CLAUDE.md's
    original limitation note was written against — verified empirically: zero id
    collisions between NORMAL and PNEUMONIA, and zero between the source's own
    train/test split, across the full 5,856-file corpus.
    """
    raw_root = raw_root or REPO_ROOT / "data" / "raw" / "kermany"
    chest_xray_roots = list(raw_root.rglob("chest_xray"))
    if not chest_xray_roots:
        raise FileNotFoundError(f"No chest_xray/ found under {raw_root}")
    chest_xray_root = chest_xray_roots[0]

    records = []
    for split_dir in ("train", "test"):
        for class_dir, label in (("NORMAL", "Normal"), ("PNEUMONIA", "Pneumonia")):
            d = chest_xray_root / split_dir / class_dir
            for path in sorted(d.glob("*.jpeg")):
                m = KERMANY_FILENAME_RE.match(path.name)
                if not m:
                    raise ValueError(f"Unrecognized Kermany filename pattern: {path.name}")
                accession_id = m.group(2)
                records.append(
                    {
                        "source": "kermany",
                        "patient_id": f"kermany-{accession_id}",
                        "label": label,
                        "relative_path": str(path.relative_to(chest_xray_root)),
                        "source_native_split": split_dir,
                    }
                )
    return records


def load_rsna_records(raw_root: Path | None = None) -> list[dict]:
    """One record per RSNA *labeled* image.

    `stage_2_test_images/` (3,000 DICOMs) is Kaggle's held-out competition test set —
    its ground truth was never publicly released — so it is excluded here rather than
    silently treated as unlabeled data. All usable records come from
    `stage_2_train_images/`, which is exactly the set covered by
    `stage_2_train_labels.csv` (verified: identical patientId sets, zero overlap with
    the unlabeled test set).

    Label mapping resolves Decision Gate DG-2 (owner-approved 2026-08-29, option a):
    RSNA's native Target column is kept as-is. Target=1 -> Pneumonia; Target=0 covers
    both true "Normal" and "No Lung Opacity / Not Normal" (abnormal-but-not-pneumonia)
    — a known, intentional limitation to be stated honestly in the paper, not
    engineered around (CLAUDE.md section 15).

    `stage_2_train_labels.csv` has one row per bounding box, so a Pneumonia patient
    with multiple opacities appears multiple times; this dedupes to one record per
    patientId (each patientId maps to exactly one DICOM file in this dataset).
    """
    raw_root = raw_root or REPO_ROOT / "data" / "raw" / "rsna"
    labels_csv = raw_root / "stage_2_train_labels.csv"

    target_by_patient: dict[str, str] = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            target_by_patient[row["patientId"]] = row["Target"]

    records = []
    for patient_id, target in sorted(target_by_patient.items()):
        label = "Pneumonia" if target == "1" else "Normal"
        records.append(
            {
                "source": "rsna",
                "patient_id": f"rsna-{patient_id}",
                "label": label,
                "relative_path": f"stage_2_train_images/{patient_id}.dcm",
                "source_native_split": "train",
            }
        )
    return records
