"""ADR-1 fine-tuning x ADR-2 DP -- federated single-seed diagnostic.

CLAUDE.md pending decision 4's "clean comparison" follow-up: the existing DP
epsilon sweep (`client_app.py`, `docs/results.md` row 5) is head-only + federated;
`scripts/train_centralized_finetune_dp.py`'s result is fine-tuned + centralized.
Neither isolates the architecture effect (head-only vs. fine-tuned) while holding
topology fixed. This script runs `client_app_finetune_dp.py` (federated,
fine-tuned, DP) so it can be compared directly against the existing federated DP
sweep at the same epsilon, holding topology fixed at FEDERATED -- this project's
actual thesis, not the centralized ceiling.

Single seed (42), num-server-rounds=10 (matches the existing no-DP federated
fine-tuning pilot/campaign's own protocol -- `run_federated_finetune_pilot.py`,
`docs/adr1_groupnorm_fallback.md`'s resolved decision 12 -- so this result is
comparable to that one too), epsilon=4/delta=1e-5 (DG-7's project default, same as
every other DP result in this project). This is a single-seed diagnostic, not a
3-seed campaign -- CLAUDE.md section 11.2's evidentiary bar is not claimed by this
run alone -- and it does NOT touch pending decision 3 (whether to scale
fine-tuning into the paper's headline ablation table or the app's default
checkpoint).

Usage: uv run python scripts/run_federated_finetune_dp_pilot.py
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.transforms import build_eval_transform
from src.evaluation.reporting import load_results, save_results
from src.models.densenet_head import DenseNet121Head

# Sibling-module import, not `scripts.train_centralized_finetune` -- see that
# script's own note (or `run_federated_finetune_pilot.py`'s) on why: `scripts/` is
# not an installed package, only `src.*` resolves from an arbitrary cwd.
from train_centralized_finetune import RawImageDataset, _evaluate, _records_for

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOG_DIR = REPO_ROOT / "outputs" / "ablation_logs"
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "federated_finetune_dp_singleseed.json"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "finetuned"

SEED = 42
NUM_ROUNDS = 10  # matches the existing no-DP federated fine-tuning pilot/campaign
TARGET_EPSILON = 4.0  # DG-7 project default
TARGET_DELTA = 1e-5  # DG-7 project default
HOSPITALS = ["A", "B", "C"]
IMAGE_SIZE = 224
PARTITION_NATURAL = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"

CANONICAL_COMPONENTS = (
    'serverapp = "src.federated.server_app:app"\n'
    'clientapp = "src.federated.client_app:app"'
)
FINETUNE_DP_COMPONENTS = (
    'serverapp = "src.federated.server_app_finetune:app"\n'
    'clientapp = "src.federated.client_app_finetune_dp:app"'
)
_COMPONENTS_PATTERN = re.compile(
    r'serverapp = "src\.federated\.server_app(?:_secagg|_finetune)?:app"\n'
    r'clientapp = "src\.federated\.client_app(?:_secagg|_finetune|_finetune_dp)?:app"'
)


def _swap_components(target: str) -> None:
    text = PYPROJECT.read_text()
    new_text, n = _COMPONENTS_PATTERN.subn(target, text, count=1)
    if n != 1:
        raise RuntimeError("could not find [tool.flwr.app.components] block to swap")
    PYPROJECT.write_text(new_text)


def _run() -> tuple[bool, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"fedavg_finetune_dp_seed{SEED}.log"
    ckpt_path = CHECKPOINT_DIR / f"fedavg_dp_natural_seed{SEED}.pt"

    run_config = (
        f'num-server-rounds={NUM_ROUNDS} seed={SEED} partition-path="{PARTITION_NATURAL}" '
        f'output-checkpoint="{ckpt_path}" target-epsilon={TARGET_EPSILON} target-delta={TARGET_DELTA}'
    )
    cmd = [
        "uv", "run", "flwr", "run", ".",
        "--run-config", run_config,
        "--federation-config", "num-supernodes=3",
        "--stream",
    ]

    print(f"run-config: {run_config}")
    start = time.monotonic()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    ok = result.returncode == 0
    print(f"{'OK' if ok else 'FAILED (see ' + str(log_path) + ')'} in {elapsed:.1f}s")
    return ok, ckpt_path


def _evaluate_checkpoint(ckpt_path: Path, test_records: list, device: torch.device) -> dict:
    model = DenseNet121Head(fine_tune_last_block=True)
    model.load_trainable_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.to(device)
    test_ds = RawImageDataset(test_records, build_eval_transform(image_size=IMAGE_SIZE))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    return _evaluate(model, test_loader, device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition = json.loads(PARTITION_NATURAL.read_text())
    test_records = _records_for(partition, HOSPITALS, "test")
    print(f"pooled test set: n={len(test_records)}")

    _swap_components(FINETUNE_DP_COMPONENTS)
    try:
        ok, ckpt_path = _run()
    finally:
        _swap_components(CANONICAL_COMPONENTS)
        current = PYPROJECT.read_text()
        assert CANONICAL_COMPONENTS in current, "pyproject.toml components swap failed to revert!"
        print("[tool.flwr.app.components] reverted to the canonical app.")

    if not ok:
        print("Run failed -- nothing to evaluate.")
        return

    metrics = _evaluate_checkpoint(ckpt_path, test_records, device)
    print(f"\npooled test AUROC={metrics['auroc']:.4f} accuracy={metrics['accuracy']:.4f} "
          f"sensitivity={metrics.get('sensitivity')} specificity={metrics.get('specificity')}")

    # server_app_finetune.py (Stage: full per-round metrics) writes every round's
    # complete metric breakdown (not just AUROC) to this file before the
    # round-checkpoint files themselves are cleaned up -- load it here so this
    # run's own results JSON carries the full per-round trajectory, not only the
    # single best-round checkpoint re-evaluated above.
    per_round_path = ckpt_path.parent / f"{ckpt_path.stem}_per_round_metrics.json"
    per_round = load_results(per_round_path) if per_round_path.exists() else {}
    if not per_round:
        print(f"WARNING: no per-round metrics file found at {per_round_path}")

    save_results(
        {
            "seed": SEED,
            "num_server_rounds": NUM_ROUNDS,
            "target_epsilon": TARGET_EPSILON,
            "target_delta": TARGET_DELTA,
            "checkpoint": str(ckpt_path),
            "pooled_test": metrics,
            "per_round": per_round,
        },
        RESULTS_PATH,
    )
    print(f"Results written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
