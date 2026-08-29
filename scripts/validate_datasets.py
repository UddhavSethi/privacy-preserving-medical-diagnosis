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
import hashlib
import json
import statistics
import sys
from pathlib import Path

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


def validate_rsna() -> dict:
    manifest_path = MANIFEST_DIR / "rsna_checksums.json"
    if not manifest_path.exists():
        return {"dataset": "rsna", "status": "skipped", "reason": "no manifest found (RSNA not yet downloaded)"}
    # Populated once scripts/download_rsna.py has been run.
    manifest = json.loads(manifest_path.read_text())
    return {"dataset": "rsna", "status": "manifest_present", "num_files_manifested": manifest["num_files"]}


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
