"""Download the RSNA Pneumonia Detection Challenge dataset (Hospitals B & C).

Source: Kaggle competition `rsna-pneumonia-detection-challenge`.

This is gated behind a Kaggle account with the competition rules accepted — that is a
manual action only the project owner can take (CLAUDE.md section 14, docs/IMPLEMENTATION_PLAN.md
Stage 3). This script does not attempt to bypass that gate; it fails fast with instructions
if credentials are missing, rather than silently doing nothing.

Setup (one-time, by the project owner):
  1. Create a Kaggle account at https://www.kaggle.com
  2. Visit https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules and accept the
     competition rules — the API refuses downloads otherwise, even with valid credentials.
  3. Kaggle account settings -> Create New API Token -> downloads kaggle.json
  4. Place it at ~/.kaggle/kaggle.json with mode 600 (`chmod 600 ~/.kaggle/kaggle.json`)

Usage: uv run python scripts/download_rsna.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "rsna"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
DOWNLOAD_DIR = REPO_ROOT / "data" / "_downloads"
COMPETITION = "rsna-pneumonia-detection-challenge"
CHUNK_SIZE = 1024 * 1024


def check_kaggle_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise SystemExit(
            "No Kaggle API token found at ~/.kaggle/kaggle.json.\n"
            "This is a manual, owner-only step — see the module docstring in this file "
            "for setup instructions. Not attempting to proceed without it."
        )


def download_via_kaggle_api(dest_dir: Path) -> None:
    # Imported lazily so `check_kaggle_credentials` can fail with a clear message before
    # the kaggle package tries (and fails less clearly) to read the missing token itself.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading competition files for {COMPETITION} ...")
    api.competition_download_files(COMPETITION, path=str(dest_dir), quiet=False)


def extract_and_manifest() -> None:
    zips = list(DOWNLOAD_DIR.glob(f"{COMPETITION}*.zip"))
    if not zips:
        raise RuntimeError(f"Expected a downloaded zip in {DOWNLOAD_DIR}, found none.")
    zip_path = zips[0]

    print(f"Extracting {zip_path.name} ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)

    print("Building SHA-256 manifest of extracted DICOM files ...")
    dicoms = sorted(RAW_DIR.rglob("*.dcm"))
    entries = []
    for path in dicoms:
        rel = path.relative_to(RAW_DIR)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"relative_path": str(rel), "sha256": h, "size_bytes": path.stat().st_size})

    manifest = {"dataset": "rsna", "num_files": len(entries), "files": entries}
    out_path = MANIFEST_DIR / "rsna_checksums.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {out_path} ({len(entries)} files)")


def main() -> None:
    check_kaggle_credentials()
    download_via_kaggle_api(DOWNLOAD_DIR)
    extract_and_manifest()


if __name__ == "__main__":
    sys.exit(main())
