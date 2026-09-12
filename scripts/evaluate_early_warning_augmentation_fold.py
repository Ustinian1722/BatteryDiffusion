#!/usr/bin/env python3
"""Evaluate augmentation utility on one frozen early-warning LOEO fold.

All model fitting uses the fold's real training windows plus, when requested,
training-only augmentation. The held-out real experiment is untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_early_warning_baseline import window_descriptors, select_modality


def classical_augment(x: np.ndarray,y: np.ndarray,n: int,temp_cap: float,rng: np.random.Generator):
    """Mild label-preserving amplitude/noise augmentation with balanced classes.

    The requested synthetic budget is split as evenly as possible across the
    observed warning classes before perturbation. A candidate that violates the
    frozen temperature guard is retried within the same target class, avoiding
    class drift caused by rejection sampling.
    """
    if n<=0:
        return np.empty((0,*x.shape[1:]),dtype=np.float32),np.empty(0,dtype=np.int64)
    classes=np.unique(y)
    targets=np.tile(classes,int(np.ceil(n/len(classes))))[:n].copy()
    rng.shuffle(targets)
    ch_std=np.std(x,axis=(0,2)); generated=[]; labels=[]
    for target in targets:
        pool=np.flatnonzero(y==target)
        accepted=False
        for _ in range(30):
            idx=int(rng.choice(pool)); s=x[idx].astype(np.float64).copy()
            for c in range(s.shape[0]):
                a=rng.uniform(0.96,1.04); baseline=s[c,0]; s[c]=baseline+a*(s[c]-baseline)
                s[c]+=rng.normal(0,max(ch_std[c]*0.003,1e-4),size=s.shape[1])
            if np.max(s[0])>temp_cap or np.min(s[0])< -20 or not np.isfinite(s).all():
                continue
            generated.append(s.astype(np.float32)); labels.append(int(target)); accepted=True; break
        # A failed target is intentionally not replaced by the opposite class.
        # This preserves class balance as far as the physical task guard permits.
        if not accepted:
            continue
    return np.stack(generated) if generated else np.empty((0,*x.shape[1:]),dtype=np.float32),np.asarray(labels,dtype=np.int64)


def fit_eval(xtr,ytr,xte,yte,mode):
    ftr=window_descriptors(select_modality(xtr,mode)); fte=window_descriptors(select_modality(xte,mode))
    model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026))])
    model.fit(ftr,ytr); p=model.predict_proba(fte)[:,1]; pred=(p>=0.5).astype(int)
    return {'auroc':float(roc_auc_score(yte,p)),'auprc':float(average_precision_score(yte,p)),'f1':float(f1_score(yte,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold-dir',type=Path,required=True); ap.add_argument('--synthetic',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--temp-cap',type=float,default=120.0)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(args.seed)
    tr=np.load(args.fold_dir/'train_real.npz'); te=np.load(args.fold_dir/'test_real.npz'); sy=np.load(args.synthetic)
    xtr=tr['x_raw'].astype(np.float32); ytr=tr['y'].astype(int); xte=te['x_raw'].astype(np.float32); yte=te['y'].astype(int)
    xs=sy['x_raw'].astype(np.float32); ys=sy['y'].astype(int); xc,yc=classical_augment(xtr,ytr,len(xs),args.temp_cap,rng)
    summary=json.loads((args.fold_dir/'summary.json').read_text(encoding='utf-8'))
    rows=[]
    methods={'real_only':(xtr,ytr),'classical':(np.concatenate([xtr,xc]),np.concatenate([ytr,yc])),'racdiff':(np.concatenate([xtr,xs]),np.concatenate([ytr,ys]))}
    for method,(xa,ya) in methods.items():
        for mode in ('temperature','force','fusion'):
            m=fit_eval(xa,ya,xte,yte,mode)
            rows.append({'heldout':summary['heldout'],'train_fraction':summary['train_fraction'],'method':method,'modality':mode,'n_real_train':len(xtr),'n_aug':0 if method=='real_only' else (len(xc) if method=='classical' else len(xs)),'n_test':len(yte),'test_positive':int(yte.sum()),**m})
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out,index=False)
    report={'heldout':summary['heldout'],'train_fraction':summary['train_fraction'],'real_train':len(xtr),'test_real':len(xte),'racdiff_aug':len(xs),'classical_aug':len(xc),'classical_positive':int(yc.sum()),'racdiff_positive':int(ys.sum())}
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
