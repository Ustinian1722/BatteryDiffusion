#!/usr/bin/env python3
"""Compare real-only, equal-budget classical and RA-CDiff on one frozen TS0330 fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_virtual_vehicle_baseline import feature_matrix

REPS=('temperature','pressure_absolute','pressure_shape','fusion_absolute','fusion_shape')


def balanced_anchor(y,n,rng):
    if n<=0: return np.empty(0,dtype=int)
    classes=np.unique(y); seq=np.resize(classes,n); rng.shuffle(seq); out=[]
    for c in seq:
        pool=np.flatnonzero(y==c); out.append(int(rng.choice(pool)))
    return np.asarray(out,dtype=int)


def classical_augment(x,y,n,rng,temp_cap=150.0):
    if n<=0: return np.empty((0,*x.shape[1:]),np.float32),np.empty(0,dtype=int)
    std=np.maximum(np.std(x,axis=(0,2)),1e-6); xs=[]; ys=[]; attempts=0
    # Balanced desired classes; if a proposal violates the frozen task cap, retry.
    while len(xs)<n and attempts<max(200,n*50):
        attempts+=1; desired=len(xs)%2; pool=np.flatnonzero(y==desired)
        if not len(pool): pool=np.arange(len(y))
        idx=int(rng.choice(pool)); s=x[idx].astype(np.float64).copy()
        # Local baseline-preserving amplitude jitter, identical conceptual family
        # to the classical comparator used in the prior Hanyang study.
        for c in range(s.shape[0]):
            baseline=s[c,0]; amp=rng.uniform(0.96,1.04); s[c]=baseline+amp*(s[c]-baseline)
            s[c]+=rng.normal(0,std[c]*0.003,size=s.shape[1])
        if not np.isfinite(s).all() or np.max(s[0])>temp_cap: continue
        xs.append(s.astype(np.float32)); ys.append(int(y[idx]))
    return (np.stack(xs) if xs else np.empty((0,*x.shape[1:]),np.float32),np.asarray(ys,dtype=int))


def fit_eval(xtr,ytr,xte,yte,rep):
    m=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026))])
    m.fit(feature_matrix(xtr,rep),ytr); p=m.predict_proba(feature_matrix(xte,rep))[:,1]; pred=(p>=0.5).astype(int)
    return {'auroc':float(roc_auc_score(yte,p)),'auprc':float(average_precision_score(yte,p)),
            'f1':float(f1_score(yte,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold-dir',type=Path,required=True); ap.add_argument('--synthetic',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(args.seed)
    tr=np.load(args.fold_dir/'train_real.npz'); te=np.load(args.fold_dir/'test_real.npz'); sy=np.load(args.synthetic)
    xtr=tr['x_raw'].astype(np.float32); ytr=tr['y'].astype(int); xte=te['x_raw'].astype(np.float32); yte=te['y'].astype(int)
    xs=sy['x_raw'].astype(np.float32); ys=sy['y'].astype(int); xc,yc=classical_augment(xtr,ytr,len(xs),rng)
    summary=json.loads((args.fold_dir/'summary.json').read_text(encoding='utf-8'))
    methods={'real_only':(xtr,ytr),'classical':(np.concatenate([xtr,xc]),np.concatenate([ytr,yc])),'racdiff':(np.concatenate([xtr,xs]),np.concatenate([ytr,ys]))}
    rows=[]
    for method,(xa,ya) in methods.items():
        for rep in REPS:
            rows.append({'heldout':summary['heldout'],'train_experiments':'+'.join(summary['train_experiments']),'method':method,'representation':rep,
                         'n_real_train_windows':len(ytr),'n_aug':0 if method=='real_only' else (len(yc) if method=='classical' else len(ys)),
                         'n_test_windows':len(yte),'test_positive':int(yte.sum()),**fit_eval(xa,ya,xte,yte,rep)})
    df=pd.DataFrame(rows); df.to_csv(args.out,index=False)
    audit={'heldout':summary['heldout'],'train_experiments':summary['train_experiments'],'real_train_windows':len(ytr),'real_train_positive':int(ytr.sum()),'test_windows':len(yte),'test_positive':int(yte.sum()),'racdiff_aug':len(ys),'racdiff_positive':int(ys.sum()),'classical_aug':len(yc),'classical_positive':int(yc.sum()),'seed':args.seed}
    args.out.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf-8'); print(df.to_string(index=False)); print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
