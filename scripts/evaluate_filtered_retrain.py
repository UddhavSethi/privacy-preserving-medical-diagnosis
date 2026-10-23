#!/usr/bin/env python3
"""Supplementary experiment (NOT an ablation-table row): 3-seed comparison of the
head-only federated DP model (eps=4) trained on all data vs. retrained on the
natural partition with RSNA "No Lung Opacity / Not Normal" patients removed.
Both are scored on the SAME original natural test set: full, and clear-label
(ambiguous subgroup excluded from scoring). Deterministic forward pass, no
calibration, threshold 0.5 -- so numbers match the ablation table's convention.
Usage: PYTHONPATH=. uv run python scripts/evaluate_filtered_retrain.py
Writes outputs/results/filtered_retrain_3seed.json."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from app.inference import load_classifier
from src.training.trainer import FEATURE_KEY, load_pooled_features

R = Path('.'); SEEDS = [42, 123, 2024]; PART = R / 'data/partitions/hospitals_natural.json'
cls = pd.read_csv(R / 'data/raw/rsna/stage_2_detailed_class_info.csv').drop_duplicates('patientId').set_index('patientId')['class']
part = json.load(open(PART)); sub = []
for h in 'ABC':
    by = defaultdict(list)
    for r in part['hospitals'][h]:
        if r['frozen_split'] == 'test': by[r['source']].append(r)
    for s, rs in sorted(by.items()):
        for r in rs: sub.append(cls.get(r['patient_id'].removeprefix('rsna-'), 'kermany') if s == 'rsna' else 'kermany')
amb = np.array(sub) == 'No Lung Opacity / Not Normal'
p = load_pooled_features(PART, list('ABC'), R / 'data/feature_cache', FEATURE_KEY)
y = p.test_labels.numpy(); assert len(amb) == len(y)

def score(ckpt):
    m = load_classifier(ckpt, fine_tune_last_block=False).eval()
    with torch.no_grad(): pr = torch.softmax(m.classifier(p.test_features), 1)[:, 1].numpy()
    out = {}
    for name, k in (('full', np.ones(len(y), bool)), ('clear_label', ~amb)):
        yy, pp = y[k], pr[k]; tn, fp, fn, tp = confusion_matrix(yy, pp >= .5, labels=[0, 1]).ravel()
        out[name] = dict(n=int(k.sum()), auroc=roc_auc_score(yy, pp), auprc=average_precision_score(yy, pp),
                         sens=tp / (tp + fn), spec=tn / (tn + fp), bal_acc=(tp / (tp + fn) + tn / (tn + fp)) / 2)
    return out

res = {'all_data': {}, 'filtered': {}}
for s in SEEDS:
    res['all_data'][s] = score(R / f'outputs/checkpoints/ablation/dp_eps4.0_seed{s}.pt')
    res['filtered'][s] = score(R / f'outputs/checkpoints/filtered/dp_eps4_filtered_seed{s}.pt')
summ = {}
for arm in res:
    for sc in ('full', 'clear_label'):
        for mt in ('auroc', 'auprc', 'sens', 'spec', 'bal_acc'):
            v = [res[arm][s][sc][mt] for s in SEEDS]
            summ[f'{arm}/{sc}/{mt}'] = (float(np.mean(v)), float(np.std(v, ddof=1)))
res['summary_mean_std'] = summ
for k, (m_, s_) in summ.items(): print(f'{k:32s} {m_:.4f} ± {s_:.4f}')
json.dump(res, open(R / 'outputs/results/filtered_retrain_3seed.json', 'w'), indent=1, default=float)
