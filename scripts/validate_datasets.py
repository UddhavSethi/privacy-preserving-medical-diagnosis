"""Validate acquired datasets against their manifests (CLAUDE.md section 11.3, Stage 3).

Checks, per dataset:
  - every manifest entry's checksum still matches the file on disk
  - image counts (total, per split, per class)
  - class balance per split
  - corrupt / unreadable files
  - image size (dimensions) distribution

Writes a JSON report to data/manifests/<dataset>_validation_report.json and prints a
human-readable summary. Exits non-zero if any checksum mismatch or corrupt file is found.

Usage: uv run python scripts/validate_datasets.py [--dataset kermany|rsna|all]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pydicom
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_kermany() -> dict:
    manifest_path = MANIFEST_DIR / "kermany_checksums.json"
    if not manifest_path.exists():
        return {"dataset": "kermany", "status": "skipped", "reason": "no manifest found"}

    manifest = json.loads(manifest_path.read_text())
    roots = list((REPO_ROOT / "data" / "raw" / "kermany").rglob("chest_xray"))
    if not roots:
        return {"dataset": "kermany", "status": "error", "reason": "extracted chest_xray/ not found"}
    root = roots[0]

    checksum_mismatches: list[str] = []
    corrupt_files: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    counts: dict[str, dict[str, int]] = {}

    for entry in manifest["files"]:
        path = root / entry["relative_path"]
        split, label = entry["split"], entry["label"]
        counts.setdefault(split, {}).setdefault(label, 0)
        counts[split][label] += 1

        if not path.exists():
            checksum_mismatches.append(entry["relative_path"] + " (missing)")
            continue
        if _sha256(path) != entry["sha256"]:
            checksum_mismatches.append(entry["relative_path"])

        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as exc:  # noqa: BLE001 — any decode failure counts as corrupt
            corrupt_files.append(f"{entry['relative_path']}: {exc}")

    report = {
        "dataset": "kermany",
        "status": "ok" if not checksum_mismatches and not corrupt_files else "issues_found",
        "num_files_manifested": len(manifest["files"]),
        "counts_per_split_per_class": counts,
        "checksum_mismatches": checksum_mismatches,
        "corrupt_files": corrupt_files,
        "image_size": {
            "width": {"min": min(widths), "max": max(widths), "mean": round(statistics.mean(widths), 1)},
            "height": {"min": min(heights), "max": max(heights), "mean": round(statistics.mean(heights), 1)},
        } if widths else None,
        "notes": [
            "No official validation split in this source (train/test only) — "
            "a val split must be carved from train at Stage 4/5, consistent with "
            "CLAUDE.md's note that Kermany's official val split is unusably small anyway.",
            "Normal-class filenames do not carry a patient identifier (only "
            "pneumonia-class filenames do, as 'personNNN_bacteria/virus_NNN.jpeg') — "
            "patient-level grouping is only partially possible for this dataset (ADR-7 limitation).",
        ],
    }
    return report


def _label_balance(raw_root: Path) -> dict | None:
    labels_csv = raw_root / "stage_2_train_labels.csv"
    class_csv = raw_root / "stage_2_detailed_class_info.csv"
    if not labels_csv.exists() or not class_csv.exists():
        return None

    with open(labels_csv, newline="") as f:
        target_by_patient = {row["patientId"]: row["Target"] for row in csv.DictReader(f)}
    with open(class_csv, newline="") as f:
        class_by_patient = {row["patientId"]: row["class"] for row in csv.DictReader(f)}

    return {
        "unique_patients": len(target_by_patient),
        "target_counts": dict(Counter(target_by_patient.values())),
        "detailed_class_counts": dict(Counter(class_by_patient.values())),
        "dg2_note": (
            "Target=0 (20,672 rows) combines two semantically different classes: "
            "'Normal' (8,851) and 'No Lung Opacity / Not Normal' (11,821, i.e. abnormal "
            "but not pneumonia). Decision Gate DG-2 (docs/IMPLEMENTATION_PLAN.md Stage 4) "
            "asks whether to keep this RSNA-native grouping (matches the original "
            "challenge, but teaches the model 'abnormal-but-not-pneumonia' = 'normal', "
            "which is clinically questionable) or exclude 'No Lung Opacity / Not Normal' "
            "from the negative class (cleaner label semantics, smaller/costlier negative "
            "class, diverges from the published challenge framing). Unresolved — do not "
            "act on it without the project owner's decision."
        ),
    }


def validate_rsna() -> dict:
    manifest_path = MANIFEST_DIR / "rsna_checksums.json"
    if not manifest_path.exists():
        return {"dataset": "rsna", "status": "skipped", "reason": "no manifest found (RSNA not yet downloaded)"}

    manifest = json.loads(manifest_path.read_text())
    raw_root = REPO_ROOT / "data" / "raw" / "rsna"

    checksum_mismatches: list[str] = []
    corrupt_files: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    photometric: Counter = Counter()
    rescale_present = 0
    rescale_absent = 0

    for entry in manifest["files"]:
        path = raw_root / entry["relative_path"]
        if not path.exists():
            checksum_mismatches.append(entry["relative_path"] + " (missing)")
            continue
        if _sha256(path) != entry["sha256"]:
            checksum_mismatches.append(entry["relative_path"])

        try:
            ds = pydicom.dcmread(path)
            _ = ds.pixel_array  # forces pixel data decode, catches corrupt/truncated files
            rows.append(int(ds.Rows))
            cols.append(int(ds.Columns))
            photometric[str(getattr(ds, "PhotometricInterpretation", "MISSING"))] += 1
            if hasattr(ds, "RescaleSlope") or hasattr(ds, "RescaleIntercept"):
                rescale_present += 1
            else:
                rescale_absent += 1
        except Exception as exc:  # noqa: BLE001 — any decode failure counts as corrupt
            corrupt_files.append(f"{entry['relative_path']}: {exc}")

    report = {
        "dataset": "rsna",
        "status": "ok" if not checksum_mismatches and not corrupt_files else "issues_found",
        "num_files_manifested": len(manifest["files"]),
        "checksum_mismatches": checksum_mismatches,
        "corrupt_files": corrupt_files,
        "image_size": {
            "rows": {"min": min(rows), "max": max(rows), "mean": round(statistics.mean(rows), 1)},
            "columns": {"min": min(cols), "max": max(cols), "mean": round(statistics.mean(cols), 1)},
        } if rows else None,
        "photometric_interpretation_counts": dict(photometric),
        "rescale_slope_intercept": {
            "present": rescale_present,
            "absent": rescale_absent,
            "note": (
                "Absent means the DICOM stores raw pixel values with no linear "
                "rescale to apply; present means RescaleSlope/RescaleIntercept must "
                "be applied before display or CLAHE (ADR-6) — verify which case "
                "applies before Stage 6 preprocessing, per CLAUDE.md's flagged "
                "DICOM pixel-handling risk."
            ),
        },
        "label_balance": _label_balance(raw_root),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kermany", "rsna", "all"], default="all")
    args = parser.parse_args()

    reports = []
    if args.dataset in ("kermany", "all"):
        reports.append(validate_kermany())
    if args.dataset in ("rsna", "all"):
        reports.append(validate_rsna())

    any_issues = False
    for report in reports:
        out_path = MANIFEST_DIR / f"{report['dataset']}_validation_report.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n=== {report['dataset']} ===")
        print(f"status: {report['status']}")
        for key, value in report.items():
            if key in ("dataset", "status", "checksum_mismatches", "corrupt_files"):
                continue
            print(f"{key}: {json.dumps(value, indent=2) if isinstance(value, (dict, list)) else value}")
        if report.get("checksum_mismatches") or report.get("corrupt_files"):
            any_issues = True
            print(f"checksum_mismatches: {len(report['checksum_mismatches'])}")
            print(f"corrupt_files: {len(report['corrupt_files'])}")
        print(f"report written: {out_path}")

    return 1 if any_issues else 0


if __name__ == "__main__":
    sys.exit(main())
