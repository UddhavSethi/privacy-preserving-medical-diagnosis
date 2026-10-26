"""Controlled FL-vs-centralized diagnosis, VALIDATION SPLIT ONLY (no test data is read).

Question (docs/audit_2026-10.md F2/F3): why does FedAvg trail centralized training on the
pooled metric, and how much of that is within-hospital discrimination versus the pooled
metric's sensitivity to between-hospital score offsets? Factors varied, one at a time or
crossed, all else identical (same seeds, rounds, lr, batch size, partition):

  class_weights: "local" (project default: each client balances its own classes) vs "none"
  views:         "all"   (project default: augmented + eval views) vs "eval" (deterministic only)

Usage: uv run python scripts/exp_fl_diagnosis.py
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PARTITION = REPO / "data/partitions/hospitals_natural.json"
OUT = REPO / "outputs/results/exp_fl_diagnosis_val.json"
SEEDS = [42, 123, 2024]
GRID = list(itertools.product(["local", "none"], [False, True]))  # (class_weight_mode, eval_view_only)


def _job(args):
    cw, eval_only, seed = args
    torch.set_num_threads(2)
    from src.experiments.fl_sim import SimConfig, run_fedavg_sim
    res = run_fedavg_sim(SimConfig(partition_path=PARTITION, seed=seed, class_weight_mode=cw, eval_view_only=eval_only))
    h = res.history
    last5 = lambda k: float(np.mean([r[k] for r in h[-5:]]))
    return {"cw": cw, "eval_only": eval_only, "seed": seed,
            **{f"final_{k}": h[-1][k] for k in h[-1] if k.startswith("val_")},
            **{f"last5_{k}": last5(k) for k in h[-1] if k.startswith("val_")}}


def main() -> None:
    jobs = [(cw, ev, s) for cw, ev in GRID for s in SEEDS]
    with mp.get_context("spawn").Pool(6, maxtasksperchild=1) as pool:
        rows = pool.map(_job, jobs)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"{'class_weights':14s}{'views':7s} | " + " | ".join(f"{k:>12s}" for k in ("pooled", "A", "B", "C")) + "   (mean over 3 seeds; VALIDATION AUROC, mean of last 5 rounds)")
    for cw, ev in GRID:
        sel = [r for r in rows if r["cw"] == cw and r["eval_only"] == ev]
        vals = [np.mean([r[f"last5_val_auroc_{k}"] for r in sel]) for k in ("pooled", "A", "B", "C")]
        print(f"{cw:14s}{'eval' if ev else 'all':7s} | " + " | ".join(f"{v:12.4f}" for v in vals))


if __name__ == "__main__":
    main()
