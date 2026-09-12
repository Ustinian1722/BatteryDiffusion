#!/usr/bin/env python3
"""Test whether geometry-normalized mechanical descriptors transfer across cell/module TR tests.

The task definition is frozen (256 s context, 900 s first-vent horizon,
max temperature <= 120 C). This script changes only the mechanical
representation. The hypothesis is that absolute force level is apparatus- and
geometry-dependent, whereas within-window force shape is more transferable.

This is a representation diagnostic, not task-definition tuning.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_early_warning_fold import build, EXPS
from evaluate_early_warning_baseline import load_pouch, load_module, VENT, window_descriptors


def shape_descriptor_1ch(x: np.ndarray) -> np.ndarray:
    """Affine-invariant descriptors for N x 1 x L mechanics.

    Each window is baseline-relative and normalized by its own robust dynamic
    scale. This intentionally removes apparatus-dependent offset and gain.
    """
    out=[]
    for sample in x:
        s=sample[0].astype(float)
        r=s-s[0]
        # Robust dynamic scale; fall back to std then 1 for near-constant windows.
        q10,q90=np.percentile(r,[10,90]); scale=max(float(q90-q10),float(np.std(r)),1e-8)
        z=r/scale
        d=np.diff(z); t=np.arange(len(z),dtype=float); tc=t-t.mean()
        slope=float(np.dot(tc,z-z.mean())/max(np.dot(tc,tc),1e-12))
        # Coarse temporal occupancy/shape descriptors remain causal.
        q1=max(1,len(z)//4); q3=max(q1+1,3*len(z)//4)
        out.append([
            float(z[-1]), float(z.mean()), float(z.std()), float(z.min()), float(z.max()),
            float(z[-1]-z[0]), slope,
            float(np.mean(np.abs(d))) if len(d) else 0.0,
            float(np.max(np.abs(d))) if len(d) else 0.0,
            float(np.mean(z[:q1])), float(np.mean(z[q1:q3])), float(np.mean(z[q3:])),
            float(np.mean(d>0)) if len(d) else 0.0,
        ])
    return np.asarray(out,dtype=float)


def feats(x: np.ndarray, representation: str) -> np.ndarray:
    temp=x[:,0:1,:]; mech=x[:,1:2,:]
    if representation=='force_absolute':
        return window_descriptors(mech)
    if representation=='force_shape':
        return shape_descriptor_1ch(mech)
    if representation=='temperature':
        return window_descriptors(temp)
    if representation=='fusion_absolute':
        return window_descriptors(x)
    if representation=='fusion_shape':
        ft=window_descriptors(temp)
        fm=shape_descriptor_1ch(mech)
        return np.concatenate([ft,fm],axis=1)
    raise ValueError(representation)


def fit_eval(xtr,ytr,xte,yte,representation):
    model=Pipeline([
        ('scale',StandardScaler()),
        ('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026)),
    ])
    model.fit(feats(xtr,representation),ytr)
    p=model.predict_proba(feats(xte,representation))[:,1]; pred=(p>=0.5).astype(int)
    return {
        'auroc':float(roc_auc_score(yte,p)),
        'auprc':float(average_precision_score(yte,p)),
        'f1':float(f1_score(yte,pred,zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(yte,pred)),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data/raw/osf_c2hnq/Dataset_TR/Dataset_TR')); ap.add_argument('--out',type=Path,default=Path('reports/geometry_normalized_mechanics'))
    ap.add_argument('--window',type=int,default=256); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--horizon',type=int,default=900); ap.add_argument('--temp-cap',type=float,default=120.0)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    data={}
    for exp in EXPS:
        df=load_pouch(args.root,exp) if exp in ('A2','D1') else load_module(args.root,exp)
        x,m=build(df,exp,args.window,args.stride,args.horizon,args.temp_cap)
        data[exp]=(x,m.label.to_numpy(dtype=int))

    reps=('temperature','force_absolute','force_shape','fusion_absolute','fusion_shape')
    rows=[]
    for held in EXPS:
        train=[e for e in EXPS if e!=held]
        xtr=np.concatenate([data[e][0] for e in train]); ytr=np.concatenate([data[e][1] for e in train])
        xte,yte=data[held]
        for rep in reps:
            rows.append({'heldout':held,'representation':rep,'n_train':len(ytr),'n_test':len(yte),'test_positive':int(yte.sum()),**fit_eval(xtr,ytr,xte,yte,rep)})
    df=pd.DataFrame(rows); df.to_csv(args.out/'fold_metrics.csv',index=False)
    macro=df.groupby('representation').agg(auroc_mean=('auroc','mean'),auroc_worst=('auroc','min'),auprc_mean=('auprc','mean'),auprc_worst=('auprc','min'),f1_mean=('f1','mean'),balanced_acc_mean=('balanced_accuracy','mean')).reset_index()
    macro.to_csv(args.out/'macro_metrics.csv',index=False)

    lines=['# Geometry-normalized mechanical representation diagnostic','',
           f'Frozen task: {args.window} s context, {args.horizon} s first-vent horizon, max temperature <= {args.temp_cap:g} C, stride {args.stride} s.','',
           'Hypothesis: absolute expansion-force level is not comparable between pouch-cell and module fixtures. `force_shape` subtracts each window baseline and divides by a robust within-window dynamic scale before extracting causal shape descriptors. `fusion_shape` combines ordinary temperature descriptors with these geometry-normalized mechanical descriptors.','',
           '| representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |',
           '|---|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows():
        lines.append(f"| {r.representation} | {r.auroc_mean:.3f} | {r.auroc_worst:.3f} | {r.auprc_mean:.3f} | {r.auprc_worst:.3f} | {r.f1_mean:.3f} |")
    lines += ['', '## Guardrail','',
              'This is a post-freeze representation diagnostic on the same four experiments. It does not change the task horizon/temperature ceiling and is not independent external validation. A useful representation must later survive external real-experiment testing.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
