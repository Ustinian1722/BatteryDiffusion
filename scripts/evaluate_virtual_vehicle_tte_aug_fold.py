#!/usr/bin/env python3
"""Compare real-only, same-anchor classical and RA-CDiff for TTE regression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_virtual_vehicle_baseline import feature_matrix

REPS=('temperature','pressure_absolute','pressure_shape','fusion_absolute','fusion_shape')


def classical_same_anchor(xreal, anchor_idx, tau, rng, temp_cap=150.0):
    std=np.maximum(np.std(xreal,axis=(0,2)),1e-6)
    xs=[]
    for idx in anchor_idx:
        idx=int(idx); accepted=False
        for _ in range(80):
            s=xreal[idx].astype(np.float64).copy()
            for c in range(s.shape[0]):
                baseline=s[c,0]
                amp=rng.uniform(0.96,1.04)
                s[c]=baseline+amp*(s[c]-baseline)
                s[c]+=rng.normal(0,std[c]*0.003,size=s.shape[1])
            if np.isfinite(s).all() and np.max(s[0])<=temp_cap:
                xs.append(s.astype(np.float32)); accepted=True; break
        if not accepted:
            raise RuntimeError(f'failed same-anchor classical proposal for anchor {idx}')
    return np.stack(xs) if xs else np.empty((0,*xreal.shape[1:]),np.float32), tau.copy()


def make_model(kind):
    if kind=='huber':
        return Pipeline([('scale',StandardScaler()),('reg',HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=3000))])
    if kind=='rf':
        return RandomForestRegressor(n_estimators=300,max_depth=5,min_samples_leaf=3,random_state=2026,n_jobs=-1)
    raise ValueError(kind)


def fit_eval(xtr,ytr,xte,yte,rep,kind):
    m=make_model(kind); m.fit(feature_matrix(xtr,rep),ytr); p=np.asarray(m.predict(feature_matrix(xte,rep)),float)
    return {'mae_s':float(mean_absolute_error(yte,p)),
            'rmse_s':float(mean_squared_error(yte,p)**0.5),
            'medae_s':float(median_absolute_error(yte,p)),
            'r2':float(r2_score(yte,p)),
            'bias_s':float(np.mean(p-yte))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold-dir',type=Path,required=True); ap.add_argument('--synthetic',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(args.seed)
    tr=np.load(args.fold_dir/'train_real.npz'); te=np.load(args.fold_dir/'test_real.npz'); sy=np.load(args.synthetic)
    xtr=tr['x_raw'].astype(np.float32); ytr=tr['tau'].astype(float); xte=te['x_raw'].astype(np.float32); yte=te['tau'].astype(float)
    xs=sy['x_raw'].astype(np.float32); ys=sy['tau'].astype(float); anchors=sy['anchor_index'].astype(np.int64)
    xc,yc=classical_same_anchor(xtr,anchors,ys,rng)
    if len(yc)!=len(ys) or not np.allclose(yc,ys): raise RuntimeError('continuous label matching failed')
    summary=json.loads((args.fold_dir/'summary.json').read_text(encoding='utf-8'))
    methods={'real_only':(xtr,ytr),'classical_anchor_matched':(np.concatenate([xtr,xc]),np.concatenate([ytr,yc])),'racdiff':(np.concatenate([xtr,xs]),np.concatenate([ytr,ys]))}
    rows=[]
    for method,(xa,ya) in methods.items():
        for kind in ('huber','rf'):
            for rep in REPS:
                rows.append({'heldout':summary['heldout'],'train_experiments':'+'.join(summary['train_experiments']),
                             'method':method,'model':kind,'representation':rep,'n_real_train_windows':len(ytr),
                             'n_aug':0 if method=='real_only' else len(ys),'n_test_windows':len(yte),**fit_eval(xa,ya,xte,yte,rep,kind)})
    df=pd.DataFrame(rows); df.to_csv(args.out,index=False)
    audit={'heldout':summary['heldout'],'train_experiments':summary['train_experiments'],'real_train_windows':len(ytr),
           'test_windows':len(yte),'racdiff_aug':len(ys),'classical_aug':len(yc),
           'anchor_indices_exactly_shared':True,'continuous_labels_exactly_shared':bool(np.allclose(yc,ys)),
           'synthetic_tau_min_s':float(ys.min()) if len(ys) else None,'synthetic_tau_max_s':float(ys.max()) if len(ys) else None,'seed':args.seed}
    args.out.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(df.to_string(index=False)); print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
