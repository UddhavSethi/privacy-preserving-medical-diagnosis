"""ADR-1 GroupNorm fallback -- federated fine-tuning campaign (ablation row 3).

Runs `client_app_finetune.py`/`server_app_finetune.py` (see those modules'
docstrings) via `flwr run`, using the same temporary `[tool.flwr.app.components]`
swap-and-revert procedure `scripts/run_ablation.py` already established for
Stage 15's SecAgg app pair (CLAUDE.md's resolved decision 7).

Protocol matches the original single-seed pilot exactly (see
docs/adr1_groupnorm_fallback.md section 6 for the rationale): num-server-rounds=10
(half of Stage 13's canonical 20 -- one local epoch/round on raw images is far more
expensive than on Stage 9's cached features), local-epochs=1, batch-size=32,
learning-rate=0.001, natural partition.

**2026-09-21: scaled from the original single-seed pilot to a proper 3-seed
campaign** (owner-approved scope: rows 1-3, natural regime, fine-tuning -- see
CLAUDE.md's pending decision 3), matching CLAUDE.md section 11.2's own evidentiary
bar. The original pilot's checkpoint (`fedavg_natural_seed42.pt`) is left in place;
this script writes fresh per-seed checkpoints (`fedavg_natural_seed{seed}.pt`,
overwriting the pilot's seed-42 file with a re-run under this unified script for a
single consistent source, per CLAUDE.md section 12) and a new aggregated results
file, `federated_finetune_multiseed.json`.

`server_app_finetune.py`'s own MLflow logging (experiment "federated_ablation",
inherited from pyproject.toml's [tool.flwr.app.config] defaults) only reports
`pooled_test_auroc`/`pooled_val_auroc` per round -- the full sensitivity/
specificity/F1/accuracy breakdown is not persisted there (same gap already found
and worked around for the DP ablation sweep). This script closes that gap for row
3 by loading each seed's best-round checkpoint after training and re-evaluating it
against the pooled test set directly (raw images through the fine-tuned model --
Stage 9's cached pooled features are frozen-backbone-only and invalid here),
reusing `train_centralized_finetune.py`'s own raw-image dataset/eval helpers so
there is exactly one evaluation implementation for this fine-tuned architecture.

Usage: uv run python scripts/run_federated_finetune_pilot.py
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import torch

from src.evaluation.reporting import aggregate_metrics_over_seeds, save_results
from src.models.densenet_head import DenseNet121Head
from src.data.transforms import build_eval_transform
from torch.utils.data import DataLoader

# Sibling-module import (not `scripts.train_centralized_finetune`): `scripts/` is
# not an installed package (pyproject.toml's [tool.hatch.build.targets.wheel] only
# packages `src`) -- only `src.*` imports resolve from an arbitrary cwd. Running
# this file directly puts its own directory on sys.path[0], so the plain sibling
# import below is what actually works.
from train_centralized_finetune import RawImageDataset, _evaluate, _records_for

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOG_DIR = REPO_ROOT / "outputs" / "ablation_logs"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "federated_finetune_multiseed.json"

NUM_ROUNDS = 10  # matches the original pilot; see module docstring
SEEDS = [42, 123, 2024]
HOSPITALS = ["A", "B", "C"]
IMAGE_SIZE = 224
PARTITION_NATURAL = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "finetuned"

CANONICAL_COMPONENTS = (
    'serverapp = "src.federated.server_app:app"\n'
    'clientapp = "src.federated.client_app:app"'
)
FINETUNE_COMPONENTS = (
    'serverapp = "src.federated.server_app_finetune:app"\n'
    'clientapp = "src.federated.client_app_finetune:app"'
)
_COMPONENTS_PATTERN = re.compile(
    r'serverapp = "src\.federated\.server_app(?:_secagg|_finetune)?:app"\n'
    r'clientapp = "src\.federated\.client_app(?:_secagg|_finetune)?:app"'
)


def _swap_components(target: str) -> None:
    text = PYPROJECT.read_text()
    new_text, n = _COMPONENTS_PATTERN.subn(target, text, count=1)
    if n != 1:
        raise RuntimeError("could not find [tool.flwr.app.components] block to swap")
    PYPROJECT.write_text(new_text)


def _run_one_seed(seed: int) -> tuple[bool, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"fedavg_finetune_seed{seed}.log"
    ckpt_path = CHECKPOINT_DIR / f"fedavg_natural_seed{seed}.pt"

    run_config = (
        f'num-server-rounds={NUM_ROUNDS} seed={seed} partition-path="{PARTITION_NATURAL}" '
        f'output-checkpoint="{ckpt_path}"'
    )
    cmd = [
        "uv", "run", "flwr", "run", ".",
        "--run-config", run_config,
        "--federation-config", "num-supernodes=3",
        "--stream",
    ]

    print(f"[seed {seed}] run-config: {run_config}")
    start = time.monotonic()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    ok = result.returncode == 0
    print(f"[seed {seed}] {'OK' if ok else 'FAILED (see ' + str(log_path) + ')'} in {elapsed:.1f}s")
    return ok, ckpt_path


def _evaluate_checkpoint(ckpt_path: Path, test_records: list, device: torch.device) -> dict:
    model = DenseNet121Head(fine_tune_last_block=True)
    model.load_trainable_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device)
    test_ds = RawImageDataset(test_records, build_eval_transform(image_size=IMAGE_SIZE))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    return _evaluate(model, test_loader, device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition = json.loads(PARTITION_NATURAL.read_text())
    test_records = _records_for(partition, HOSPITALS, "test")
    print(f"pooled test set: n={len(test_records)}")

    per_seed_test_metrics = []
    failures = []

    _swap_components(FINETUNE_COMPONENTS)
    try:
        for seed in SEEDS:
            ok, ckpt_path = _run_one_seed(seed)
            if not ok:
                failures.append(seed)
                continue
            metrics = _evaluate_checkpoint(ckpt_path, test_records, device)
            print(f"[seed {seed}] pooled test AUROC={metrics['auroc']:.4f} accuracy={metrics['accuracy']:.4f}")
            per_seed_test_metrics.append(metrics)
    finally:
        _swap_components(CANONICAL_COMPONENTS)
        current = PYPROJECT.read_text()
        assert CANONICAL_COMPONENTS in current, "pyproject.toml components swap failed to revert!"
        print("[tool.flwr.app.components] reverted to the canonical app.")

    if failures:
        print(f"\nWARNING: seeds failed and were excluded from aggregation: {failures}")

    if not per_seed_test_metrics:
        print("No successful runs -- nothing to aggregate.")
        return

    aggregated = aggregate_metrics_over_seeds(per_seed_test_metrics)
    print(
        f"\nFedAvg fine-tune (natural), pooled test: AUROC = "
        f"{aggregated['auroc']['mean']:.4f} +/- {aggregated['auroc']['std']:.4f} "
        f"(n={aggregated['auroc']['n_seeds']} seeds)"
    )
    for k in ("accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy"):
        if k in aggregated:
            print(f"  {k}: {aggregated[k]['mean']:.4f} +/- {aggregated[k]['std']:.4f}")

    save_results({"per_seed": per_seed_test_metrics, "pooled_test": aggregated, "failed_seeds": failures}, RESULTS_PATH)
    print(f"\nResults written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
