#!/usr/bin/env python3
"""Evaluation-only sensitivity analysis for the deployed fedavg_dp_eps4 checkpoint:
re-scores existing test predictions with the ambiguous RSNA "No Lung Opacity / Not Normal"
subgroup excluded (no retraining). Values below mirror conf/app.yaml.
Usage: PYTHONPATH=. uv run python scripts/evaluate_excluding_ambiguous.py
Writes outputs/results/fedavg_dp_eps4_excl_ambiguous_report.json."""
import numpy as np, torch, json, pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from app.inference import load_classifier
from src.training.trainer import FEATURE_KEY, load_pooled_features
from src.uncertainty.mc_dropout import compute_mc_dropout_uncertainty
from src.uncertainty.probability_calibration import apply_temperature
R=Path('.'); T=2.253361225128174; thr=0.35; hw=0.05; dth=0.6728980541229248
part=json.load(open(R/'data/partitions/hospitals_natural.json'))
cls=pd.read_csv(R/'data/raw/rsna/stage_2_detailed_class_info.csv').drop_duplicates('patientId').set_index('patientId')['class']
sub=[]
for h in 'ABC':
    by=defaultdict(list)
    for r in part['hospitals'][h]:
        if r['frozen_split']=='test': by[r['source']].append(r)
    for s,rs in sorted(by.items()):
        for r in rs: sub.append(cls.get(r['patient_id'].removeprefix('rsna-'),'kermany') if s=='rsna' else 'kermany')
sub=np.array(sub)
m=load_classifier(R/'outputs/checkpoints/ablation/dp_eps4.0_seed42.pt',fine_tune_last_block=False).eval()
p=load_pooled_features(R/'data/partitions/hospitals_natural.json',list('ABC'),R/'data/feature_cache',FEATURE_KEY)
y=p.test_labels.numpy(); assert len(sub)==len(y),(len(sub),len(y))
torch.manual_seed(42)
mc=compute_mc_dropout_uncertainty(m,p.test_features,num_passes=20)
pr=apply_temperature(mc.mean_probs,T)[:,1].numpy(); ent=mc.entropy.numpy()
print('subgroup counts',dict(zip(*np.unique(sub,return_counts=True))))
def met(k,t=thr):
    yy,pp=y[k],pr[k]; yp=(pp>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(yy,yp,labels=[0,1]).ravel(); se=tp/(tp+fn); sp=tn/(tn+fp); pc=tp/(tp+fp)
    rng=np.random.default_rng(42); b=[]
    for _ in range(1000):
        i=rng.integers(0,len(yy),len(yy)); b.append(roc_auc_score(yy[i],pp[i]))
    return dict(n=int(k.sum()),prev=float(yy.mean()),auroc=roc_auc_score(yy,pp),ci=list(np.percentile(b,[2.5,97.5])),auprc=average_precision_score(yy,pp),sens=se,spec=sp,prec=pc,npv=tn/(tn+fn),f1=2*pc*se/(pc+se),acc=(tp+tn)/len(yy),bal=(se+sp)/2,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
amb=sub=='No Lung Opacity / Not Normal'
out={'full':met(np.ones(len(y),bool)),'excl_ambiguous':met(~amb),'ambiguous_only_fpr':float(((pr[amb&(y==0)]>=thr)).mean())}
for name in set(sub):
    k=(sub==name)&(y==0)
    out.setdefault('fpr_by_group',{})[name]=[int(k.sum()),float((pr[k]>=thr).mean())]
ab=(pr>=thr-hw)&(pr<=thr+hw); df=ent>=dth; keep=~amb
for nm,k in(('excl_amb+abstain',keep&~ab),('excl_amb+abstain+defer',keep&~ab&~df)):
    q=met(k); q['coverage']=float(k.sum()/keep.sum()); out[nm]=q
print(json.dumps(out,indent=1,default=float))
json.dump(out,open('outputs/results/fedavg_dp_eps4_excl_ambiguous_report.json','w'),indent=1,default=float)
