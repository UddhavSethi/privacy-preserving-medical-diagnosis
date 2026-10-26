"""One evaluation protocol for every ablation row (docs/audit_2026-10.md F1, F3).

Why: rows of the original ablation table were scored differently (row 1 on each hospital's
own test split, rows 2-5 on the pooled test split), and the pooled AUROC mixes hospitals with
very different pneumonia prevalence (A ~72%, B/C ~22%), so it also rewards a model for telling
the hospitals apart. Every model is therefore scored on the SAME splits in three ways:

  macro AUROC     mean over hospitals of the AUROC on that hospital's own test split — the
                  headline: it asks "how well does the model rank patients within a hospital"
  per-hospital    A, B, C individually
  pooled AUROC    kept for continuity with the earlier table, labelled as pooled

Threshold-dependent numbers use thresholds chosen on the POOLED VALIDATION split only and then
applied unchanged to test: the project default 0.5, and the highest-sensitivity threshold that
still achieves >= 90% validation specificity. No test label influences any threshold.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_metrics
from src.experiments.fl_sim import predict_probs


def threshold_for_specificity(val_y: np.ndarray, val_score: np.ndarray, target_specificity: float = 0.9) -> float:
    """Lowest threshold whose validation specificity is >= target (maximises sensitivity)."""
    neg = np.sort(val_score[val_y == 0])
    k = int(np.ceil(target_specificity * len(neg))) - 1
    return float(np.nextafter(neg[min(max(k, 0), len(neg) - 1)], np.inf))


def evaluate_model(state: dict, data: dict, hospitals=("A", "B", "C")) -> dict:
    vp = {h: predict_probs(state, data[h].val_features) for h in hospitals}
    vy = {h: data[h].val_labels.numpy() for h in hospitals}
    tp = {h: predict_probs(state, data[h].test_features) for h in hospitals}
    ty = {h: data[h].test_labels.numpy() for h in hospitals}
    pv_y, pv_s = np.concatenate([vy[h] for h in hospitals]), np.concatenate([vp[h] for h in hospitals])
    pt_y, pt_s = np.concatenate([ty[h] for h in hospitals]), np.concatenate([tp[h] for h in hospitals])
    thr90 = threshold_for_specificity(pv_y, pv_s, 0.9)

    out = {"auroc_pooled": compute_metrics(pt_y, pt_s).auroc, "val_auroc_pooled": compute_metrics(pv_y, pv_s).auroc,
           "threshold_spec90_val": thr90}
    for h in hospitals:
        out[f"auroc_{h}"] = compute_metrics(ty[h], tp[h]).auroc
        out[f"val_auroc_{h}"] = compute_metrics(vy[h], vp[h]).auroc
    out["auroc_macro"] = float(np.mean([out[f"auroc_{h}"] for h in hospitals]))
    out["val_auroc_macro"] = float(np.mean([out[f"val_auroc_{h}"] for h in hospitals]))
    for name, thr in (("t0.5", 0.5), ("spec90val", thr90)):
        m = compute_metrics(pt_y, pt_s, threshold=thr)
        out[f"sensitivity_{name}"], out[f"specificity_{name}"], out[f"f1_{name}"] = m.sensitivity, m.specificity, m.f1
        out[f"balanced_accuracy_{name}"] = m.balanced_accuracy
    out["auprc_pooled"] = compute_metrics(pt_y, pt_s).auprc

    # Per-hospital operating point: each hospital derives its own >=90%-specificity threshold on ITS
    # validation split (no data leaves the hospital) and applies it to its own test split. Unlike the
    # pooled number this cannot be inflated by between-hospital score offsets.
    sens, spec = [], []
    for h in hospitals:
        thr_h = threshold_for_specificity(vy[h], vp[h], 0.9)
        m = compute_metrics(ty[h], tp[h], threshold=thr_h)
        out[f"sensitivity_spec90local_{h}"], out[f"specificity_spec90local_{h}"] = m.sensitivity, m.specificity
        sens.append(m.sensitivity); spec.append(m.specificity)
    out["sensitivity_spec90local_macro"], out["specificity_spec90local_macro"] = float(np.mean(sens)), float(np.mean(spec))
    return out
