#!/usr/bin/env python3
"""Re-analysis of the EXISTING filtered-retrain checkpoints (no training). Scores on GPU.
Separates: (1) model effect of filtering (threshold-free AUROCs, full/subgroup), (2) threshold choice,
(3) operating-point/class-balance shift, (4) calibration. Every threshold and temperature is fitted on
VALIDATION data only, then frozen and applied to the unchanged original test set (n=4838).
Writes outputs/results/rsna_ambiguity/subgroup_reanalysis.json (refuses to overwrite).
Usage: PYTHONPATH=. uv run python scripts/rsna_ambiguity_reanalysis.py"""
import json
from pathlib import Path
import numpy as np, torch
from app.inference import load_classifier
from src.evaluation.rsna_ambiguity import (HARD_NEGATIVE, KERMANY_NORMAL, KERMANY_PNEUMONIA, LUNG_OPACITY, NORMAL,
    expected_calibration_error, fit_temperature, load_rsna_class_info, subgroup_metrics,
    test_subgroups_in_pooled_order, threshold_at_val_specificity, threshold_at_val_youden)
from src.training.trainer import FEATURE_KEY, load_pooled_features

R = Path('.'); SEEDS = [42, 123, 2024]; OUT = R / 'outputs/results/rsna_ambiguity/subgroup_reanalysis.json'
assert not OUT.exists(), 'refusing to overwrite'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev)
PART = R / 'data/partitions/hospitals_natural.json'
part = json.load(open(PART)); cls = load_rsna_class_info(R / 'data/raw/rsna/stage_2_detailed_class_info.csv')
p = load_pooled_features(PART, list('ABC'), R / 'data/feature_cache', FEATURE_KEY)
ty, vy = p.test_labels.numpy(), p.val_labels.numpy()
tsub = test_subgroups_in_pooled_order(part, cls, 'test'); vsub = test_subgroups_in_pooled_order(part, cls, 'val')
assert len(tsub) == len(ty) == 4838 and len(vsub) == len(vy)
# protocol guard: the test split is exactly the frozen one (counts per original class), never altered
EXPECT = {KERMANY_NORMAL: 235, KERMANY_PNEUMONIA: 600, LUNG_OPACITY: 902, HARD_NEGATIVE: 1761, NORMAL: 1340}
assert {g: int((tsub == g).sum()) for g in EXPECT} == EXPECT
clear_t, clear_v = tsub != HARD_NEGATIVE, vsub != HARD_NEGATIVE


@torch.no_grad()
def logits(ckpt, feats):
    m = load_classifier(ckpt, fine_tune_last_block=False).eval().to(dev)
    return m.classifier(feats.to(dev)).cpu().numpy().astype(np.float64)


def sm1(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return (e / e.sum(1, keepdims=True))[:, 1]


def run(ckpt):
    zv, zt = logits(ckpt, p.val_features), logits(ckpt, p.test_features)
    T = fit_temperature(vy, zv)                       # validation-only
    pv, pt, pvT, ptT = sm1(zv), sm1(zt), sm1(zv / T), sm1(zt / T)
    thr = {'fixed_0.5': 0.5,
           'val_spec90_full': threshold_at_val_specificity(vy, pv, .9),
           'val_youden_full': threshold_at_val_youden(vy, pv),
           'val_spec90_clearlabel_only': threshold_at_val_specificity(vy[clear_v], pv[clear_v], .9)}
    out = {'temperature_val': T, 'thresholds_frozen_from_val': thr, 'by_threshold': {k: subgroup_metrics(ty, pt, tsub, t) for k, t in thr.items()}}
    out['calibration_test'] = {
        'full': {'ece_raw': expected_calibration_error(ty, pt), 'ece_temp_scaled': expected_calibration_error(ty, ptT),
                 'brier_raw': float(((pt - ty) ** 2).mean()), 'brier_temp_scaled': float(((ptT - ty) ** 2).mean())},
        'clear_label': {'ece_raw': expected_calibration_error(ty[clear_t], pt[clear_t]), 'ece_temp_scaled': expected_calibration_error(ty[clear_t], ptT[clear_t])}}
    out['temp_scaled_at_val_spec90'] = subgroup_metrics(ty, ptT, tsub, threshold_at_val_specificity(vy, pvT, .9))  # T is monotone: same ranking
    out['mean_score_test_by_subgroup'] = {g: float(pt[tsub == g].mean()) for g in EXPECT}
    out['auroc_clear_label'] = subgroup_metrics(ty[clear_t], pt[clear_t], tsub[clear_t], .5)['full']['auroc']
    return out


res = {'device': dev, 'n_test': int(len(ty)), 'test_counts': EXPECT, 'arms': {}}
for arm, tmpl in (('all_data', 'outputs/checkpoints/ablation/dp_eps4.0_seed{}.pt'), ('filtered', 'outputs/checkpoints/filtered/dp_eps4_filtered_seed{}.pt')):
    res['arms'][arm] = {str(s): run(R / tmpl.format(s)) for s in SEEDS}


def agg(arm, path):
    v = []
    for s in SEEDS:
        d = res['arms'][arm][str(s)]
        for k in path: d = d[k]
        v.append(d)
    return float(np.mean(v)), float(np.std(v, ddof=1))


S = {}
for arm in res['arms']:
    S[arm] = {'auroc_full': agg(arm, ['by_threshold', 'fixed_0.5', 'full', 'auroc']), 'auroc_clear_label': agg(arm, ['auroc_clear_label']),
              'lo_vs_normal': agg(arm, ['by_threshold', 'fixed_0.5', 'lung_opacity_vs_normal', 'auroc']),
              'lo_vs_hardneg': agg(arm, ['by_threshold', 'fixed_0.5', 'lung_opacity_vs_hard_negative', 'auroc']),
              'normal_vs_hardneg_scores': agg(arm, ['by_threshold', 'fixed_0.5', 'normal_vs_hard_negative_score_auroc']),
              'ece_raw_full': agg(arm, ['calibration_test', 'full', 'ece_raw']), 'ece_temp_full': agg(arm, ['calibration_test', 'full', 'ece_temp_scaled'])}
    for t in res['arms'][arm]['42']['by_threshold']:
        for m in ('sensitivity', 'specificity', 'balanced_accuracy'):
            S[arm][f'{t}/full/{m}'] = agg(arm, ['by_threshold', t, 'full', m])
        for g in (NORMAL, HARD_NEGATIVE):
            S[arm][f'{t}/spec/{g}'] = agg(arm, ['by_threshold', t, 'by_subgroup', g, 'specificity'])
        S[arm][f'{t}/sens/{LUNG_OPACITY}'] = agg(arm, ['by_threshold', t, 'by_subgroup', LUNG_OPACITY, 'sensitivity'])
res['summary_mean_std_n3'] = S
for arm in S:
    print('==', arm)
    for k, (m, s) in S[arm].items(): print(f'  {k:55s} {m:.4f} ± {s:.4f}')
OUT.write_text(json.dumps(res, indent=1, default=float))
