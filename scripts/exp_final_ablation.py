"""Report-phase ablation under ONE evaluation protocol (src/experiments/evaluation_protocol.py).

Reads the DP configuration chosen on VALIDATION data (outputs/results/selected_dp_config.json,
written by scripts/select_dp_config.py) and then, for the first time, scores on the test splits.
Rows:
  existing checkpoints (re-scored, not retrained): local A/B/C, centralized, FedAvg (project
    recipe), FedAvg+SecAgg, FedAvg+DP (original recipe) at eps 1/2/4/8
  new in-process runs (harness validated against Flower; 3 seeds):
    fedavg_matched   FedAvg with the DP path's data view/loss (eval view only, no class weights):
                     the correct non-private counterpart of the DP rows
    dp_tuned_eps*    DP-FedAvg with the validation-selected recipe, eps 1/2/4/8

Usage: uv run python scripts/exp_final_ablation.py [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PARTITION = REPO / "data/partitions/hospitals_natural.json"
CK = REPO / "outputs/checkpoints"
NEW_CK = CK / "exp2026"
OUT = REPO / "outputs/results/exp_final_ablation.json"
SEEDS = [42, 123, 2024]
EPS = [1.0, 2.0, 4.0, 8.0]


def _sim_job(job):
    torch.set_num_threads(1)
    from src.experiments.fl_sim import DPSimConfig, SimConfig, run_fedavg_sim
    name, seed, eps, sel = job
    path = NEW_CK / f"{name}_seed{seed}.pt"
    if eps is None:
        cfg = SimConfig(partition_path=PARTITION, seed=seed, class_weight_mode="none", eval_view_only=True, rounds=sel["rounds"],
                        lr=1e-3, track_val=False, device="cuda")
    else:
        cfg = SimConfig(partition_path=PARTITION, seed=seed, rounds=sel["rounds"], lr=sel["lr"], class_weight_mode=sel["cw"],
                        dp=DPSimConfig(target_epsilon=eps, max_grad_norm=sel["clip"], loss_mode=sel["loss"], optimizer=sel["opt"]), track_val=False, device="cuda")
    res = run_fedavg_sim(cfg)
    NEW_CK.mkdir(parents=True, exist_ok=True)
    torch.save(res.final_state, path)
    return name, seed, max(res.epsilon_spent.values()) if res.epsilon_spent else None


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    sel = json.loads((REPO / "outputs/results/selected_dp_config.json").read_text())["selected"]
    print("selected DP recipe (validation-chosen):", sel, flush=True)

    jobs = [("fedavg_matched", s, None, sel) for s in SEEDS] + [(f"dp_tuned_eps{e}", s, e, sel) for e in EPS for s in SEEDS]
    jobs = [j for j in jobs if not (NEW_CK / f"{j[0]}_seed{j[1]}.pt").exists()]
    eps_spent: dict = {}
    if jobs:
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
            for name, seed, eps in pool.imap_unordered(_sim_job, jobs):
                eps_spent[f"{name}_seed{seed}"] = eps; print("trained", name, seed, "max eps_spent", eps, flush=True)

    from src.evaluation.reporting import aggregate_over_seeds
    from src.experiments.evaluation_protocol import evaluate_model
    from src.experiments.fl_sim import load_all_hospitals
    data = load_all_hospitals(PARTITION)
    rows = {}
    for h in "ABC":
        rows[f"local_{h}"] = {s: CK / f"local_baseline/natural_{h}_seed{s}.pt" for s in SEEDS}
    rows["centralized"] = {s: CK / f"centralized_baseline/natural_seed{s}.pt" for s in SEEDS}
    rows["fedavg_original"] = {s: CK / f"ablation/fedavg_natural_seed{s}.pt" for s in SEEDS}
    rows["fedavg_secagg_original"] = {s: CK / f"ablation/secagg_seed{s}.pt" for s in SEEDS}
    rows["fedavg_matched"] = {s: NEW_CK / f"fedavg_matched_seed{s}.pt" for s in SEEDS}
    # real Flower SecAgg+ pipeline, scripts/run_full_system_campaign.py (rows 4' and 6; simulation: no TLS/auth)
    rows["secagg_matched"] = {s: NEW_CK / f"secagg_matched_seed{s}.pt" for s in SEEDS}
    rows["secagg_dp_eps4_row6"] = {s: NEW_CK / f"secagg_dp_eps4_seed{s}.pt" for s in SEEDS}  # first campaign (accountant-reset reporting defect, same training)
    rows["secagg_dp_eps4_row6_v2"] = {s: NEW_CK / f"secagg_dp_eps4_v2_seed{s}.pt" for s in SEEDS}  # after the F11 fix: correct cumulative epsilon
    for e in EPS:
        rows[f"dp_original_eps{e:g}"] = {s: CK / f"ablation/dp_eps{e}_seed{s}.pt" for s in SEEDS}
        rows[f"dp_tuned_eps{e:g}"] = {s: NEW_CK / f"dp_tuned_eps{e}_seed{s}.pt" for s in SEEDS}
    result = {"selected_dp_recipe": sel, "rows": {}}
    for name, paths in rows.items():
        per_seed = [evaluate_model(torch.load(p, map_location="cpu", weights_only=True), data) for p in paths.values() if p.exists()]
        if per_seed:
            result["rows"][name] = {"per_seed": per_seed, **{k: aggregate_over_seeds([r[k] for r in per_seed]).to_dict() for k in per_seed[0]}}
    OUT.write_text(json.dumps(result, indent=2))
    cols = ["auroc_macro", "auroc_pooled", "auroc_A", "auroc_B", "auroc_C", "sensitivity_spec90val", "specificity_spec90val", "sensitivity_t0.5"]
    print(f"\n{'row':26s}" + "".join(f"{c:>22s}" for c in cols))
    for name, r in result["rows"].items():
        print(f"{name:26s}" + "".join(f"{r[c]['mean']:>14.4f}±{r[c]['sample_std']:<6.4f}" if r[c]['n_seeds'] > 1 else f"{r[c]['mean']:>22.4f}" for c in cols))


if __name__ == "__main__":
    main()
