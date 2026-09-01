"""ADR-1 GroupNorm fallback -- federated fine-tuning pilot runner, added 2026-08-31.

Runs `client_app_finetune.py`/`server_app_finetune.py` (see those modules'
docstrings) via `flwr run`, using the same temporary `[tool.flwr.app.components]`
swap-and-revert procedure `scripts/run_ablation.py` already established for
Stage 15's SecAgg app pair (CLAUDE.md's resolved decision 7).

Scoped as a bounded pilot, matching `scripts/train_centralized_finetune.py`'s own
scoping rationale (see `docs/adr1_groupnorm_fallback.md`): num-server-rounds=10
(half of Stage 13's canonical 20 -- one local epoch/round on raw images is far
more expensive than on Stage 9's cached features, so this session bounds it
rather than committing to the full protocol's ~4-hour cost blind), one seed (42,
matching every other baseline this pilot is compared against). Otherwise matches
the canonical FedAvg-no-DP protocol exactly: local-epochs=1, batch-size=32,
learning-rate=0.001, natural partition.

Usage: uv run python scripts/run_federated_finetune_pilot.py
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOG_DIR = REPO_ROOT / "outputs" / "ablation_logs"

NUM_ROUNDS = 10  # pilot-scoped, see module docstring; canonical FedAvg-no-DP uses 20
SEED = 42
PARTITION_NATURAL = str(REPO_ROOT / "data" / "partitions" / "hospitals_natural.json")
CHECKPOINT_OUT = REPO_ROOT / "outputs" / "checkpoints" / "finetuned" / "fedavg_natural_seed42.pt"

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


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "fedavg_finetune_seed42.log"

    run_config = (
        f'num-server-rounds={NUM_ROUNDS} seed={SEED} partition-path="{PARTITION_NATURAL}" '
        f'output-checkpoint="{CHECKPOINT_OUT}"'
    )
    cmd = [
        "uv", "run", "flwr", "run", ".",
        "--run-config", run_config,
        "--federation-config", "num-supernodes=3",
        "--stream",
    ]

    print(f"run-config: {run_config}")
    _swap_components(FINETUNE_COMPONENTS)
    try:
        start = time.monotonic()
        with open(log_path, "w") as f:
            result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - start
        ok = result.returncode == 0
        print(f"{'OK' if ok else 'FAILED (see ' + str(log_path) + ')'} in {elapsed:.1f}s")
    finally:
        _swap_components(CANONICAL_COMPONENTS)
        current = PYPROJECT.read_text()
        assert CANONICAL_COMPONENTS in current, "pyproject.toml components swap failed to revert!"
        print("[tool.flwr.app.components] reverted to the canonical app.")


if __name__ == "__main__":
    main()
