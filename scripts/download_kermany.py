"""Download and verify the Kermany chest X-ray dataset (Hospital A).

Source: Mendeley Data, "Large Dataset of Labeled Optical Coherence Tomography (OCT) and
Chest X-Ray Images" (Kermany, Zhang, Goldbaum; doi:10.17632/rscbjbr9sj.3). As of dataset
version 3, Mendeley bundles the chest X-ray images together with an unrelated OCT dataset
in a single ~8.4GB zip — there is no standalone chest-only download from the authoritative
source any more. This script downloads the full zip, verifies it against the SHA-256 the
Mendeley API publishes, and extracts only the `chest_xray/` subtree, discarding the OCT
images. This is preferred over a third-party Kaggle re-upload because the hash is
independently verifiable and traceable to the original DOI (CLAUDE.md section 12).

Usage: uv run python scripts/download_kermany.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "kermany"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
DOWNLOAD_DIR = REPO_ROOT / "data" / "_downloads"

MENDELEY_DATASET_API = "https://data.mendeley.com/public-api/datasets/rscbjbr9sj"
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def fetch_dataset_metadata() -> dict:
    resp = requests.get(MENDELEY_DATASET_API, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_with_resume(url: str, dest: Path, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing == expected_size:
        print(f"Already downloaded: {dest} ({existing} bytes)")
        return

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = existing + int(resp.headers.get("content-length", 0))
        downloaded = existing
        with open(dest, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (200 * CHUNK_SIZE) < CHUNK_SIZE:
                    pct = 100 * downloaded / total if total else 0
                    print(f"  {downloaded / 1e9:.2f} GB / {total / 1e9:.2f} GB ({pct:.1f}%)", flush=True)
    print(f"Download complete: {dest} ({dest.stat().st_size} bytes)")


def verify_sha256(path: Path, expected_hex: str) -> None:
    print(f"Verifying SHA-256 of {path.name} ...")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_hex:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected_hex}, got {actual}. "
            "Refusing to extract a corrupted/tampered archive."
        )
    print("Checksum OK.")


def extract_chest_xray_only(zip_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.rglob("*.jpeg")):
        print(f"Already extracted: {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting chest_xray/ subtree (ignoring OCT images) ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.namelist()
            if "chest_xray" in m.lower() and not m.endswith("/")
        ]
        if not members:
            raise RuntimeError("No chest_xray/ entries found in archive — layout may have changed.")
        for i, member in enumerate(members):
            zf.extract(member, dest_dir)
            if i % 2000 == 0:
                print(f"  extracted {i}/{len(members)}")
    print(f"Extracted {len(members)} files to {dest_dir}")


def build_manifest(chest_xray_root: Path, out_path: Path, source_meta: dict) -> None:
    print("Building SHA-256 manifest of extracted chest X-ray images ...")
    files = sorted(chest_xray_root.rglob("*.jpeg")) + sorted(chest_xray_root.rglob("*.jpg"))
    entries = []
    for path in files:
        rel = path.relative_to(chest_xray_root)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        parts = rel.parts  # (split, class, filename) e.g. ("train", "PNEUMONIA", "person1_bacteria_1.jpeg")
        entries.append(
            {
                "relative_path": str(rel),
                "split": parts[0] if len(parts) > 0 else None,
                "label": parts[1] if len(parts) > 1 else None,
                "sha256": h,
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "dataset": "kermany",
        "source_doi": source_meta.get("doi", {}).get("id"),
        "source_version": source_meta.get("version"),
        "archive_sha256": source_meta["files"][0]["content_details"]["sha256_hash"],
        "num_files": len(entries),
        "files": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {out_path} ({len(entries)} files)")


def main() -> None:
    meta = fetch_dataset_metadata()
    file_info = meta["files"][0]
    zip_path = DOWNLOAD_DIR / file_info["filename"]

    download_with_resume(file_info["content_details"]["download_url"], zip_path, file_info["size"])
    verify_sha256(zip_path, file_info["content_details"]["sha256_hash"])
    extract_chest_xray_only(zip_path, RAW_DIR)

    # The extracted layout is data/raw/kermany/CellData/chest_xray/{train,test,val}/...
    chest_xray_roots = list(RAW_DIR.rglob("chest_xray"))
    if not chest_xray_roots:
        raise RuntimeError("Could not locate extracted chest_xray/ directory.")
    chest_xray_root = chest_xray_roots[0]

    build_manifest(chest_xray_root, MANIFEST_DIR / "kermany_checksums.json", meta)

    print("\nDone. Zip retained at", zip_path, "for provenance; delete manually if disk space is needed.")


if __name__ == "__main__":
    sys.exit(main())
