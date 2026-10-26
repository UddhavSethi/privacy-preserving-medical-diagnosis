"""Re-audit of the existing membership-inference results (no training) — docs/audit_2026-10.md F7.

For every already-trained checkpoint it reports, with bootstrap CIs and per-seed values:
  * attack AUROC, train (members) vs TEST (non-members)   — the original OPT-2 setting
  * attack AUROC, train vs VAL                            — a second, independent non-member set
  * NEGATIVE CONTROL: val vs test                          — both are non-members, so a valid
                                                             attack must score ~0.5 here; any
                                                             departure measures split-to-split
                                                             distribution shift, which would
                                                             contaminate the main number
  * class-balanced AUROC (average of the within-class AUROCs) — removes label-prior differences
  * TPR at FPR in {0.1%, 1%, 10%}                          — the low-FPR regime AUROC hides
Score = logit-scaled true-class confidence (monotone in the original -loss score, so AUROC is
unchanged by the transform; computed in float64).

Usage: uv run python scripts/audit_mia_existing.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.mia import attack_auc, bootstrap_auc_ci, logit_confidence, tpr_at_fpr
from src.experiments.fl_sim import HOSPITALS, load_all_hospitals, predict_probs
from src.evaluation.reporting import aggregate_over_seeds

REPO = Path(__file__).resolve().parents[1]
CK = REPO / "outputs/checkpoints"
OUT = REPO / "outputs/results/audit_mia_existing.json"
SEEDS = [42, 123, 2024]
CONFIGS = {
    "centralized": lambda s: CK / f"centralized_baseline/natural_seed{s}.pt",
    "fedavg": lambda s: CK / f"ablation/fedavg_natural_seed{s}.pt",
    "fedavg_dp_eps1": lambda s: CK / f"ablation/dp_eps1.0_seed{s}.pt",
    "fedavg_dp_eps4": lambda s: CK / f"ablation/dp_eps4.0_seed{s}.pt",
    "fedavg_dp_eps8": lambda s: CK / f"ablation/dp_eps8.0_seed{s}.pt",
}


def class_balanced_auc(ms, ml, ns, nl) -> float:
    return float(np.mean([attack_auc(ms[ml == c], ns[nl == c]) for c in (0, 1)]))


def main() -> None:
    data = load_all_hospitals(REPO / "data/partitions/hospitals_natural.json")
    cat = lambda f: torch.cat([f(data[h]) for h in HOSPITALS])
    tr_x, tr_y = cat(lambda d: d.train_features[:, -1, :]), cat(lambda d: d.train_labels).numpy()
    va_x, va_y = cat(lambda d: d.val_features), cat(lambda d: d.val_labels).numpy()
    te_x, te_y = cat(lambda d: d.test_features), cat(lambda d: d.test_labels).numpy()
    print(f"members(train)={len(tr_y)} val={len(va_y)} test={len(te_y)}; pneumonia fraction train/val/test = "
          f"{tr_y.mean():.3f}/{va_y.mean():.3f}/{te_y.mean():.3f}")
    result = {}
    for name, path_fn in CONFIGS.items():
        per_seed = []
        for s in SEEDS:
            st = torch.load(path_fn(s), map_location="cpu", weights_only=True)
            def score(x, y):
                p = predict_probs(st, x).astype(np.float64)
                return logit_confidence(np.where(y == 1, p, 1 - p))
            sm, sv, st_ = score(tr_x, tr_y), score(va_x, va_y), score(te_x, te_y)
            auc_te, lo, hi = bootstrap_auc_ci(sm, st_, n_boot=200, seed=s)
            row = {
                "auc_train_vs_test": auc_te, "auc_train_vs_test_ci": [lo, hi],
                "auc_train_vs_val": attack_auc(sm, sv),
                "auc_val_vs_test_NEGATIVE_CONTROL": attack_auc(sv, st_),
                "auc_class_balanced_train_vs_test": class_balanced_auc(sm, tr_y, st_, te_y),
                "tpr_at_fpr_train_vs_test": tpr_at_fpr(sm, st_),
            }
            per_seed.append(row)
        keys = ["auc_train_vs_test", "auc_train_vs_val", "auc_val_vs_test_NEGATIVE_CONTROL", "auc_class_balanced_train_vs_test"]
        result[name] = {"per_seed": per_seed, **{k: aggregate_over_seeds([r[k] for r in per_seed]).to_dict() for k in keys},
                        "tpr_at_1pct_fpr": aggregate_over_seeds([r["tpr_at_fpr_train_vs_test"][0.01] for r in per_seed]).to_dict()}
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n{'config':16s} {'tr-vs-te':>9s} {'tr-vs-val':>9s} {'val-vs-te (control)':>20s} {'class-bal':>9s} {'TPR@1%FPR':>10s}   (mean over 3 seeds)")
    for n, r in result.items():
        print(f"{n:16s} {r['auc_train_vs_test']['mean']:9.4f} {r['auc_train_vs_val']['mean']:9.4f} "
              f"{r['auc_val_vs_test_NEGATIVE_CONTROL']['mean']:20.4f} {r['auc_class_balanced_train_vs_test']['mean']:9.4f} {r['tpr_at_1pct_fpr']['mean']:10.4f}")


if __name__ == "__main__":
    main()
