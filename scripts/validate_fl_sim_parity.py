"""Parity check: the in-process FedAvg harness must reproduce the saved Flower-simulation
checkpoint (same seed, same config) to numerical tolerance, otherwise results from the
harness cannot be compared with the published ablation rows.

Usage: uv run python scripts/validate_fl_sim_parity.py [--dp]
"""
import argparse
from pathlib import Path

import torch

from src.experiments.fl_sim import DPSimConfig, SimConfig, run_fedavg_sim

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = SimConfig(
        partition_path=REPO / "data/partitions/hospitals_natural.json", seed=a.seed,
        dp=DPSimConfig(target_epsilon=4.0) if a.dp else None,
    )
    ref_path = REPO / "outputs/checkpoints/ablation" / (f"dp_eps4.0_seed{a.seed}.pt" if a.dp else f"fedavg_natural_seed{a.seed}.pt")
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    res = run_fedavg_sim(cfg, verbose=True)
    diffs = {k: (res.final_state[k].float() - ref[k].float()).abs().max().item() for k in ref}
    print("max |sim - flower| per tensor:", {k: f"{v:.2e}" for k, v in diffs.items()})
    print("MAX ABS DIFF:", max(diffs.values()), "vs reference", ref_path.name)
    if a.dp:
        print("epsilon_spent per hospital:", res.epsilon_spent, "noise_multiplier:", res.noise_multiplier)


if __name__ == "__main__":
    main()
