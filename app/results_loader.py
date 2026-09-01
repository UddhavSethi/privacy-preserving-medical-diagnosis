"""OPT-6 — loads this project's real, already-generated research artifacts for
the Streamlit "Research Results" tab. Never invents or hardcodes a number: every
value here is read from `outputs/results/*.json` (OPT-1-5's own analysis scripts)
or from `src.evaluation.tables.build_ablation_table()` (Stage 21's live MLflow
query). Every loader degrades gracefully (returns `None`, never raises) so a
fresh checkout without the generated artifacts still shows a working app with
clear "not yet generated" messaging instead of crashing — CLAUDE.md's own
"handle missing checkpoints/results gracefully" requirement.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def _load_json(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_calibration_results() -> dict | None:
    return _load_json("calibration")


def load_privacy_attack_results() -> dict | None:
    return _load_json("privacy_attack")


def load_gradcam_localization_results() -> dict | None:
    return _load_json("gradcam_localization")


def load_conformal_results() -> dict | None:
    return _load_json("conformal")


def load_ood_detector_results() -> dict | None:
    return _load_json("ood_detector")


def load_ablation_table() -> list[dict] | None:
    """Reuses `src.evaluation.tables.build_ablation_table()` directly — the exact
    live-MLflow-backed function `scripts/generate_result_figures.py` and
    `docs/results.md` already depend on. Returns None (not a partial/fake table)
    if the MLflow database isn't reachable, e.g. a fresh checkout with no
    `mlruns.db`."""
    try:
        from src.evaluation.tables import build_ablation_table

        return build_ablation_table()
    except Exception:
        return None


def figure_path(name: str) -> Path | None:
    path = FIGURES_DIR / name
    return path if path.exists() else None
