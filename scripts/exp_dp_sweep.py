"""Validation-only DP-FedAvg hyper-parameter study at target epsilon = 4 (no test data read).

Stage A isolates two implementation choices at the project defaults (C=1, lr=1e-3, 20 rounds):
  loss_mode    legacy_weighted_mean  vs  per_sample_mean   (DP-validity of the loss, audit F4)
  class_weights  local               vs  none              (label-count-derived weights are
                                                             outside the privacy accountant, F5)
Stage B grids plain-SGD learning rate x clipping norm x rounds (DP-valid loss, no class weights); Stage C tries smaller Adam learning rates.
Selection criterion is declared up front: macro-average over hospitals of the VALIDATION AUROC
at the final round (the model that would actually be released). Screening uses one seed (42); the top candidates are re-run on 3 seeds. JSONL is appended
as jobs finish so a partial run is still usable.

Usage: uv run python scripts/exp_dp_sweep.py [--seeds 42 123] [--workers 6]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PARTITION = REPO / "data/partitions/hospitals_natural.json"
OUT = REPO / "outputs/results/exp_dp_sweep_val.jsonl"


def _job(spec):
    torch.set_num_threads(1)
    from src.experiments.fl_sim import DPSimConfig, SimConfig, run_fedavg_sim
    cfg = SimConfig(partition_path=PARTITION, seed=spec["seed"], rounds=spec["rounds"], lr=spec["lr"],
                    class_weight_mode=spec["cw"], device=spec.get("device", "cpu"), track_val=True,
                    dp=DPSimConfig(target_epsilon=spec.get("eps", 4.0), max_grad_norm=spec["clip"], loss_mode=spec["loss"],
                                   optimizer=spec["opt"]))
    res = run_fedavg_sim(cfg)
    return {**spec, "history": res.history, "epsilon_spent": res.epsilon_spent, "noise_multiplier": res.noise_multiplier}


KEYS = ("stage", "seed", "rounds", "lr", "clip", "cw", "loss", "opt")


def specs(seeds, stages="ABC"):
    out = []
    for s in seeds:
        if "D" in stages:
            # Stage D (added after A-C showed the best SGD cells sit at the small-lr / small-clip edge of the grid):
            # refine below the edge. Selection still uses validation only.
            for lr, clip, rounds in itertools.product([0.03, 0.1], [0.1, 0.25, 0.5], [5, 10]):
                out.append(dict(stage="D", seed=s, rounds=rounds, lr=lr, clip=clip, cw="none", loss="per_sample_mean", opt="sgd"))
        if "E" in stages:
            # Stage E (GPU): stage C showed Adam is viable at small lr (1e-4 gave the best pooled validation AUROC); grid it properly.
            for lr, clip, rounds in itertools.product([3e-5, 1e-4, 3e-4], [0.25, 0.5, 1.0], [10, 20]):
                out.append(dict(stage="E", seed=s, rounds=rounds, lr=lr, clip=clip, cw="none", loss="per_sample_mean", opt="adam", device="cuda"))
        if not set("ABC") & set(stages):
            continue
        # Stage A: the two implementation choices at the project defaults (Adam, C=1, lr=1e-3, 20 rounds)
        for loss, cw in itertools.product(["legacy_weighted_mean", "per_sample_mean"], ["local", "none"]):
            out.append(dict(stage="A", seed=s, rounds=20, lr=1e-3, clip=1.0, cw=cw, loss=loss, opt="adam"))
        # Stage B: plain SGD (DP-valid loss, no label-derived weights)
        for lr, clip, rounds in itertools.product([0.1, 0.3, 1.0], [0.5, 1.0], [10, 20]):
            out.append(dict(stage="B", seed=s, rounds=rounds, lr=lr, clip=clip, cw="none", loss="per_sample_mean", opt="sgd"))
        # Stage C: Adam with smaller learning rates
        for lr in (1e-4, 3e-4):
            out.append(dict(stage="C", seed=s, rounds=20, lr=lr, clip=1.0, cw="none", loss="per_sample_mean", opt="adam"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stages", default="ABC")
    ap.add_argument("--device", default=None, help="override the device of every spec (e.g. cuda)")
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line); done.add(json.dumps({k: r[k] for k in KEYS}, sort_keys=True))
    all_specs = specs(a.seeds, a.stages)
    if a.device:
        for sp in all_specs:
            sp["device"] = a.device
    todo = [s for s in all_specs if json.dumps({k: s[k] for k in KEYS}, sort_keys=True) not in done]
    print(f"{len(todo)} jobs to run ({len(done)} already done)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_job, todo), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            h = r["history"][-1]
            print(f"[{i}/{len(todo)}] {r['stage']} seed{r['seed']} C={r['clip']} lr={r['lr']} R={r['rounds']} cw={r['cw']} {r['loss']}: "
                  f"val_pooled={h['val_auroc_pooled']:.4f} A={h['val_auroc_A']:.3f} B={h['val_auroc_B']:.3f} C={h['val_auroc_C']:.3f}", flush=True)


if __name__ == "__main__":
    main()
