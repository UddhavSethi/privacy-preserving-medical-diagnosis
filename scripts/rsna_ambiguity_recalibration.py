#!/usr/bin/env python3
"""Companion to rsna_ambiguity_reanalysis.py: fits Platt (slope+intercept) on VALIDATION logit margins only,
freezes it, applies to test. Distinguishes a pure bias/prior shift (intercept fixes it) from a ranking change.
Writes outputs/results/rsna_ambiguity/platt_recalibration.json (refuses to overwrite)."""
import json
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from app.inference import load_classifier
from src.evaluation.rsna_ambiguity import (HARD_NEGATIVE, NORMAL, LUNG_OPACITY, expected_calibration_error, load_rsna_class_info,
                                           subgroup_metrics, test_subgroups_in_pooled_order, threshold_at_val_specificity)
from src.training.trainer import FEATURE_KEY, load_pooled_features
R = Path('.'); OUT = R / 'outputs/results/rsna_ambiguity/platt_recalibration.json'; assert not OUT.exists()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
PART = R / 'data/partitions/hospitals_natural.json'; part = json.load(open(PART))
cls = load_rsna_class_info(R / 'data/raw/rsna/stage_2_detailed_class_info.csv')
p = load_pooled_features(PART, list('ABC'), R / 'data/feature_cache', FEATURE_KEY)
ty, vy = p.test_labels.numpy(), p.val_labels.numpy(); tsub = test_subgroups_in_pooled_order(part, cls, 'test')
res = {}
for arm, tm in (('all_data', 'outputs/checkpoints/ablation/dp_eps4.0_seed{}.pt'), ('filtered', 'outputs/checkpoints/filtered/dp_eps4_filtered_seed{}.pt')):
    res[arm] = {}
    for s in (42, 123, 2024):
        with torch.no_grad():
            m = load_classifier(R / tm.format(s)).eval().to(dev)
            mv, mt = [(lambda z: (z[:, 1] - z[:, 0]).double().cpu().numpy())(m.classifier(f.to(dev))) for f in (p.val_features, p.test_features)]
        lr = LogisticRegression(C=1e6).fit(mv[:, None], vy)           # validation only
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0]); sv, st = 1 / (1 + np.exp(-(a * mv + b))), 1 / (1 + np.exp(-(a * mt + b)))
        thr05 = subgroup_metrics(ty, st, tsub, .5); thr90 = subgroup_metrics(ty, st, tsub, threshold_at_val_specificity(vy, sv, .9))
        res[arm][str(s)] = {'slope': a, 'intercept': b, 'ece_test': expected_calibration_error(ty, st), 'at_0.5': thr05, 'at_val_spec90': thr90}
def g(arm, *k):
    v = []
    for s in res[arm]:
        d = res[arm][s]
        for x in k: d = d[x]
        v.append(d)
    return round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)
res['summary_mean_std_n3'] = {arm: {'slope': g(arm, 'slope'), 'intercept': g(arm, 'intercept'), 'ece_test': g(arm, 'ece_test'),
   **{f'{t}/{m}': g(arm, t, 'full', m) for t in ('at_0.5', 'at_val_spec90') for m in ('sensitivity', 'specificity', 'balanced_accuracy')},
   **{f'{t}/spec/{q}': g(arm, t, 'by_subgroup', q, 'specificity') for t in ('at_0.5', 'at_val_spec90') for q in (NORMAL, HARD_NEGATIVE)}} for arm in ('all_data', 'filtered')}
for arm, d in res['summary_mean_std_n3'].items():
    print(arm); [print(f'  {k:40s} {v}') for k, v in d.items()]
OUT.write_text(json.dumps(res, indent=1, default=float))
