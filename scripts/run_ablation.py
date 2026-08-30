"""Stage 21 — full ablation campaign batch runner (CLAUDE.md section 11.1:
"the ablation table is the paper").

Scope, all owner-approved 2026-08-30:
  - Seeds: {42, 123, 2024} (Stage 11/12's own precedent).
  - 20 rounds per federated run (Stage 13's own precedent) — Stage 20's real
    timing showed this is affordable even for the 16.7x-slower DP rows.
  - Row 3 (FedAvg): natural + balanced partitions, 3 seeds each.
  - Row 4 (FedAvg + SecAgg): natural partition, 3 seeds. Uses the separate
    legacy-API app pair (Stage 15) — this script temporarily swaps
    `pyproject.toml`'s [tool.flwr.app.components] before these runs and
    restores it immediately after, exactly Stage 15's established procedure.
  - Row 5 (FedAvg + DP): natural partition, epsilon sweep {1, 2, 4, 8} (DG-7),
    3 seeds each.
  - Dirichlet synthetic non-IID (supplementary, not one of the 6 official
    ablation rows): alpha in {0.1, 1.0}, 3 clients, 3 seeds each — CLAUDE.md's
    own framing that natural non-IID is preferred, Dirichlet is "a controlled
    sweep."
  - Row 6 (full combined system) explicitly deferred — needs real integration
    work (reconciling Stage 15's SecAgg app with the canonical DP-capable app)
    not yet done; not part of this campaign.
  - Rows 1 (local) and 2 (centralized) already have real results from Stages
    11/12 (outputs/results/{local,centralized}_baseline.json) — not re-run.

Every run is tracked to MLflow (experiment "federated_ablation") via the
Stage 21 instrumentation added to server_app.py/server_app_secagg.py — that
is the authoritative record; this script's own stdout/log files are for
progress visibility and debugging, not a second source of truth.

A single run's failure does not abort the campaign — each run's outcome is
recorded and printed in the final summary, so a real defect discovered
mid-campaign is visible rather than silently absent from the results table.

Usage: uv run python scripts/run_ablation.py
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "outputs" / "ablation_logs"
PYPROJECT = REPO_ROOT / "pyproject.toml"

NUM_ROUNDS = 20
SEEDS = [42, 123, 2024]
EPSILON_SWEEP = [1.0, 2.0, 4.0, 8.0]
DIRICHLET_ALPHAS = [0.1, 1.0]

PARTITION_NATURAL = str(REPO_ROOT / "data" / "partitions" / "hospitals_natural.json")
PARTITION_BALANCED = str(REPO_ROOT / "data" / "partitions" / "hospitals_natural_balanced.json")

CANONICAL_COMPONENTS = (
    'serverapp = "src.federated.server_app:app"\n'
    'clientapp = "src.federated.client_app:app"'
)
SECAGG_COMPONENTS = (
    'serverapp = "src.federated.server_app_secagg:app"\n'
    'clientapp = "src.federated.client_app_secagg:app"'
)

_COMPONENTS_PATTERN = re.compile(
    r'serverapp = "src\.federated\.server_app(?:_secagg)?:app"\n'
    r'clientapp = "src\.federated\.client_app(?:_secagg)?:app"'
)


def _swap_components(target: str) -> None:
    text = PYPROJECT.read_text()
    new_text, n = _COMPONENTS_PATTERN.subn(target, text, count=1)
    if n != 1:
        raise RuntimeError("could not find [tool.flwr.app.components] block to swap")
    PYPROJECT.write_text(new_text)


def _run_flwr(run_config: str, log_name: str) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    cmd = [
        "uv", "run", "flwr", "run", ".",
        "--run-config", run_config,
        "--federation-config", "num-supernodes=3",
        "--stream",  # load-bearing, not cosmetic: without --stream, `flwr run`
        # submits the run and returns immediately rather than blocking until
        # it finishes — found via this script's own first real launch, not by
        # inspection: all 27 runs were submitted to the same persistent
        # SuperLink within seconds of each other instead of running one at a
        # time, corrupting the whole campaign (had to be killed, its MLflow
        # runs deleted, and relaunched with this fix).
    ]
    print(f"=== {log_name} ===  run-config: {run_config}")
    start = time.monotonic()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    ok = result.returncode == 0
    print(f"  {'OK' if ok else 'FAILED (see ' + str(log_path) + ')'} in {elapsed:.1f}s")
    return ok


def main() -> None:
    results: dict[str, bool] = {}

    checkpoint_dir = REPO_ROOT / "outputs" / "checkpoints" / "ablation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n##### Row 3 — FedAvg (natural + balanced), 3 seeds each #####")
    for regime, partition_path in [("natural", PARTITION_NATURAL), ("balanced", PARTITION_BALANCED)]:
        for seed in SEEDS:
            name = f"fedavg_{regime}_seed{seed}"
            ckpt = checkpoint_dir / f"{name}.pt"
            run_config = (
                f'num-server-rounds={NUM_ROUNDS} seed={seed} partition-path="{partition_path}" '
                f'output-checkpoint="{ckpt}"'
            )
            results[name] = _run_flwr(run_config, name)

    print("\n##### Row 5 — FedAvg + DP, epsilon sweep {1,2,4,8}, 3 seeds each #####")
    for epsilon in EPSILON_SWEEP:
        for seed in SEEDS:
            name = f"dp_eps{epsilon}_seed{seed}"
            ckpt = checkpoint_dir / f"{name}.pt"
            run_config = (
                f'num-server-rounds={NUM_ROUNDS} seed={seed} partition-path="{PARTITION_NATURAL}" '
                f'dp-enabled=true target-epsilon={epsilon} output-checkpoint="{ckpt}"'
            )
            results[name] = _run_flwr(run_config, name)

    print("\n##### Dirichlet synthetic non-IID (supplementary), alpha in {0.1, 1.0}, 3 seeds each #####")
    for alpha in DIRICHLET_ALPHAS:
        partition_path = str(REPO_ROOT / "data" / "partitions" / f"hospitals_dirichlet_alpha{alpha}.json")
        for seed in SEEDS:
            name = f"dirichlet_alpha{alpha}_seed{seed}"
            ckpt = checkpoint_dir / f"{name}.pt"
            run_config = (
                f'num-server-rounds={NUM_ROUNDS} seed={seed} partition-path="{partition_path}" '
                f'output-checkpoint="{ckpt}"'
            )
            results[name] = _run_flwr(run_config, name)

    print("\n##### Row 4 — FedAvg + SecAgg (natural), 3 seeds #####")
    _swap_components(SECAGG_COMPONENTS)
    try:
        for seed in SEEDS:
            name = f"secagg_seed{seed}"
            ckpt = checkpoint_dir / f"{name}.pt"
            run_config = f'num-server-rounds={NUM_ROUNDS} seed={seed} output-checkpoint="{ckpt}"'
            results[name] = _run_flwr(run_config, name)
    finally:
        _swap_components(CANONICAL_COMPONENTS)
        current = PYPROJECT.read_text()
        assert CANONICAL_COMPONENTS in current, "pyproject.toml components swap failed to revert!"
        print("  [tool.flwr.app.components] reverted to the canonical app.")

    print("\n##### Campaign summary #####")
    failed = [name for name, ok in results.items() if not ok]
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    print(f"\n{len(results) - len(failed)}/{len(results)} runs succeeded.")
    if failed:
        print(f"FAILED runs (see outputs/ablation_logs/<name>.log): {failed}")


if __name__ == "__main__":
    main()
