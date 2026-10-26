"""Write docs/audit_raw_outputs_manifest.md: SHA-256 and size of every raw output the audit's tables depend on.

`outputs/` is gitignored (it holds model checkpoints and logs), so the raw evidence behind docs/audit_tables.md is NOT in
the repository. This manifest is the committed fingerprint: someone re-running the pipeline can compare their files against
it. Deterministic file ordering; no timestamps (so it only changes when a file does).

Usage: uv run python scripts/make_audit_manifest.py
"""
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERNS = ["outputs/results/audit_*.json", "outputs/results/exp_*.json*", "outputs/results/selected_dp_config.json",
            "outputs/results/verified_epsilon.json", "outputs/mia/*/analysis.json", "outputs/mia/*/config.json",
            "outputs/comm/*.jsonl", "outputs/checkpoints/exp2026/*.pt", "outputs/ablation_logs/secagg_*_seed*.log"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted({f for pat in PATTERNS for f in REPO.glob(pat)})
    lines = ["# Audit raw-output manifest (generated)", "",
             "`outputs/` is gitignored, so these files are not in the repository; this manifest fingerprints them "
             "(`scripts/make_audit_manifest.py`). GPU training is not bit-reproducible, so a re-run will not match the checkpoint "
             "hashes; the result JSON files are what the tables are built from.", "",
             "| file | bytes | sha256 |", "|---|---|---|"]
    lines += [f"| `{f.relative_to(REPO)}` | {f.stat().st_size} | `{sha256(f)[:16]}…` |" for f in files]
    (REPO / "docs/audit_raw_outputs_manifest.md").write_text("\n".join(lines) + "\n")
    print(f"{len(files)} files fingerprinted")


if __name__ == "__main__":
    main()
