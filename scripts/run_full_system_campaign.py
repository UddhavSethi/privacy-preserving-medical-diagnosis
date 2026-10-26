"""Ablation rows 4 and 6 through the REAL Flower SecAgg+ pipeline, 3 seeds, with communication logged.

Row 4' : FedAvg + SecAgg+            (matched non-private recipe: no class weights)
Row 6  : FedAvg + SecAgg+ + DP       (validation-selected DP recipe, epsilon = 4, delta = 1e-5)

Simulation can exercise SecAgg+ masking and DP; it has no network, so TLS and client
authentication are NOT active in these runs (docs/audit_2026-10.md, "which mechanisms were
active"). They are exercised separately by scripts/run_deployment.sh (Docker Compose, real TLS +
registered node keys). Per-round, per-node SecAgg+ message bytes are written by
src/federated/comm_accounting.py to outputs/comm/<run>.jsonl (PNEUMO_COMM_LOG).

Like scripts/run_ablation.py this temporarily points [tool.flwr.app.components] at the SecAgg app
pair and restores it in a `finally`.

Usage: uv run python scripts/run_full_system_campaign.py [--rounds 20] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ablation as ra  # noqa: E402  (reuses the audited components-swap + `flwr run` helpers)

REPO = ra.REPO_ROOT
COMM_DIR = REPO / "outputs" / "comm"
CK_DIR = REPO / "outputs" / "checkpoints" / "exp2026"
SEEDS = [42, 123, 2024]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", choices=["row4", "row6", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--suffix", default="", help="appended to run/checkpoint/log names so a re-run never overwrites earlier outputs")
    a = ap.parse_args()
    sel = json.loads((REPO / "outputs/results/selected_dp_config.json").read_text())["selected"]
    COMM_DIR.mkdir(parents=True, exist_ok=True); CK_DIR.mkdir(parents=True, exist_ok=True)

    dp_cfg = (f'dp-enabled=true target-epsilon=4.0 dp-loss-mode="{sel["loss"]}" dp-optimizer="{sel["opt"]}" '
              f'max-grad-norm={sel["clip"]} learning-rate={sel["lr"]} class-weights="{sel["cw"]}" device="{a.device}"')
    base = f'num-server-rounds={a.rounds} partition-path="{ra.PARTITION_NATURAL}"'
    jobs = []
    for seed in a.seeds:
        if a.only in ("row4", "both"):
            jobs.append((f"secagg_matched{a.suffix}_seed{seed}", f'{base} seed={seed} class-weights="none" learning-rate=0.001'))
        if a.only in ("row6", "both"):
            jobs.append((f"secagg_dp_eps4{a.suffix}_seed{seed}", f"{base} seed={seed} {dp_cfg}"))

    results = {}
    ra._swap_components(ra.SECAGG_COMPONENTS)
    try:
        for name, run_config in jobs:
            os.environ["PNEUMO_COMM_LOG"] = str(COMM_DIR / f"{name}.jsonl")
            Path(os.environ["PNEUMO_COMM_LOG"]).unlink(missing_ok=True)
            # Ray hides the GPU from simulated clients unless each is allocated a fraction of it
            fed_cfg = "num-supernodes=3" + (" client-resources-num-gpus=0.3" if a.device == "cuda" else "")
            results[name] = ra._run_flwr(f'{run_config} output-checkpoint="{CK_DIR / (name + ".pt")}"', name, fed_cfg)
    finally:
        ra._swap_components(ra.CANONICAL_COMPONENTS)
        assert ra.CANONICAL_COMPONENTS in ra.PYPROJECT.read_text(), "pyproject.toml components swap failed to revert!"
        print("[tool.flwr.app.components] reverted to the canonical app.")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
