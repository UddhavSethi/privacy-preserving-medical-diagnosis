"""Shadow-model membership-inference audit (LiRA, Carlini et al. 2022) of the training RECIPE.

Why this exists (docs/audit_2026-10.md F7): the original loss-threshold attack scores ~0.50 on
every configuration, including the unprotected one, and its own negative control (val vs test)
scores about the same, so it cannot distinguish "no leakage" from "attack too weak".
LiRA calibrates the attack per example using models trained with and without that example, and
is evaluated at low false-positive rates.

Design. D = pooled (train ∪ val) per hospital, deterministic eval-view features. Each shadow
model is a full federated run (the same FedAvg harness, same recipe) in which every example is
included independently with probability 1/2, per hospital. The score statistic is the
logit-scaled true-class confidence of the final global model on EVERY example of D, so each
(model, example) pair has a known ground-truth membership bit. Each model is then attacked in
leave-one-model-out fashion using the other models' per-example in/out statistics.

Note the audited quantity is the RECIPE on ~half-size data (each shadow trains on ~50% of D),
and ε targets apply to that half-size dataset. This is stated wherever the result is reported.

Usage:
  uv run python scripts/exp_mia_lira.py run     --tag nondp  [--n-shadow 32]
  uv run python scripts/exp_mia_lira.py run     --tag dp_eps4 --dp-eps 4 --opt sgd --lr 0.3 --clip 1.0 --rounds 10
  uv run python scripts/exp_mia_lira.py analyze --tag dp_eps4
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
OUT_DIR = REPO / "outputs/mia"
HOSPITALS = ["A", "B", "C"]


def _pooled_D():
    """(train ∪ val) eval-view features and labels per hospital. Cached once as a small .pt:
    re-reading the full multi-view feature banks in every shadow job dominated the run time."""
    cache = OUT_DIR / "D_cache.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)
    from src.experiments.fl_sim import load_all_hospitals
    data = load_all_hospitals(PARTITION)
    D = {h: (torch.cat([d.train_features[:, -1, :], d.val_features]), torch.cat([d.train_labels, d.val_labels])) for h, d in data.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(D, cache)
    return D


def _shadow(args):
    m, tag, cfg_kw = args
    torch.set_num_threads(1)
    from dataclasses import replace
    from src.experiments.fl_sim import DPSimConfig, SimConfig, predict_probs, run_fedavg_sim
    from src.evaluation.mia import logit_confidence
    from src.training.trainer import HospitalFeatures
    path = OUT_DIR / tag / f"model_{m:03d}.npz"
    if path.exists():
        return m
    D = _pooled_D()
    rng = np.random.default_rng(10_000 + m)
    masks = {h: rng.random(len(D[h][1])) < 0.5 for h in HOSPITALS}
    sub = {}
    for h in HOSPITALS:
        x, y = D[h]; k = torch.from_numpy(masks[h])
        sub[h] = HospitalFeatures(x[k][:, None, :], y[k], x[:5], y[:5], x[:5], y[:5])  # val/test slots unused (track_val=False)
    dp = DPSimConfig(target_epsilon=cfg_kw["dp_eps"], max_grad_norm=cfg_kw["clip"], loss_mode="per_sample_mean", optimizer=cfg_kw["opt"]) if cfg_kw["dp_eps"] else None
    cfg = SimConfig(partition_path=PARTITION, seed=m, rounds=cfg_kw["rounds"], lr=cfg_kw["lr"], class_weight_mode="none",
                    eval_view_only=True, dp=dp, track_val=False, device="cuda")
    res = run_fedavg_sim(cfg, data=sub)
    phi = {}
    for h in HOSPITALS:
        x, y = D[h]
        p = predict_probs(res.final_state, x).astype(np.float64)
        phi[h] = logit_confidence(np.where(y.numpy() == 1, p, 1 - p))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"phi_{h}": phi[h] for h in HOSPITALS}, **{f"mask_{h}": masks[h] for h in HOSPITALS},
             eps=json.dumps(res.epsilon_spent))
    return m


def run(a) -> None:
    kw = dict(dp_eps=a.dp_eps, clip=a.clip, opt=a.opt, lr=a.lr, rounds=a.rounds)
    (OUT_DIR / a.tag).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / a.tag / "config.json").write_text(json.dumps({**kw, "n_shadow": a.n_shadow}, indent=2))
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for i, m in enumerate(pool.imap_unordered(_shadow, [(m, a.tag, kw) for m in range(a.n_shadow)]), 1):
            print(f"[{i}/{a.n_shadow}] shadow model {m} done", flush=True)


def analyze(a) -> None:
    from src.evaluation.mia import attack_auc, best_epsilon_lower_bound, lira_scores, tpr_at_fpr
    files = sorted((OUT_DIR / a.tag).glob("model_*.npz"))
    cfg = json.loads((OUT_DIR / a.tag / "config.json").read_text())
    Z = [np.load(f) for f in files]
    phi = np.stack([np.concatenate([z[f"phi_{h}"] for h in HOSPITALS]) for z in Z])    # (M, N)
    mask = np.stack([np.concatenate([z[f"mask_{h}"] for h in HOSPITALS]) for z in Z])  # (M, N) bool
    M, N = phi.shape
    print(f"{a.tag}: {M} shadow models x {N} examples; mean examples/model in: {mask.sum(1).mean():.0f}")
    from src.evaluation.mia import lira_loo_scores
    lira, loss_like, truth, _ = lira_loo_scores(phi, mask)
    res = {"tag": a.tag, "config": cfg, "n_models": M, "n_pairs": int(len(truth))}
    for name, s in (("global_threshold_attack", loss_like), ("lira_online_fixed_var", lira)):
        mem, non = s[truth], s[~truth]
        res[name] = {"auroc": attack_auc(mem, non), "tpr_at_fpr": tpr_at_fpr(mem, non, (0.001, 0.01, 0.1)),
                     "epsilon_lower_bound_indicative": best_epsilon_lower_bound(mem, non)}
        print(f"{name:26s} AUROC={res[name]['auroc']:.4f}  TPR@0.1%/1%/10% FPR = "
              + "/".join(f"{res[name]['tpr_at_fpr'][f]:.3f}" for f in (0.001, 0.01, 0.1))
              + f"  indicative eps_lb={res[name]['epsilon_lower_bound_indicative']['epsilon_lb']:.2f}")
    (OUT_DIR / a.tag / "analysis.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-shadow", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dp-eps", type=float, default=0.0)
    ap.add_argument("--opt", default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else analyze(args)
