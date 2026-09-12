#!/usr/bin/env python3
"""Exact 6/1/1 experiment-fold Random-Forest baseline for the method paper.

This is a classical sanity baseline under the same input/target protocol as the
deep architecture benchmark. The validation experiment is retained for fold
symmetry but RF hyperparameters are pre-fixed and never selected on test data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from remotezip import RemoteZip
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_virtual_vehicle_time_to_vent_baseline import build_windows
from evaluate_virtual_vehicle_baseline import feature_matrix


def metrics(y,p):
    return {'mae_s':float(mean_absolute_error(y,p)),
            'rmse_s':float(mean_squared_error(y,p)**0.5),
            'medae_s':float(median_absolute_error(y,p)),
            'r2':float(r2_score(y,p)),
            'bias_s':float(np.mean(p-y))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/method_paper_rf_baseline'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    data={}
    with RemoteZip(record_url(),headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df,event,_=load_one(rz,exp); x,y,m=build_windows(df,event,120,10,600,150.0); data[exp]=(x,y,m)
    rows=[]; preds=[]
    for hi,held in enumerate(EXPS):
        val=EXPS[(hi+1)%len(EXPS)]
        trains=[e for e in EXPS if e not in (held,val)]
        xtr=np.concatenate([data[e][0] for e in trains]); ytr=np.concatenate([data[e][1] for e in trains])
        xte,yte,meta=data[held]
        # Hyperparameters are inherited from the previously predeclared nonlinear diagnostic.
        model=RandomForestRegressor(n_estimators=300,max_depth=5,min_samples_leaf=3,random_state=2026,n_jobs=-1)
        model.fit(feature_matrix(xtr,'fusion_absolute'),ytr)
        p=np.clip(model.predict(feature_matrix(xte,'fusion_absolute')),0,600)
        rows.append({'heldout':held,'validation':val,'train_experiments':'+'.join(trains),'model':'rf','representation':'fusion_absolute','n_train':len(ytr),'n_test':len(yte),**metrics(yte,p)})
        for (_,mr),yy,pp in zip(meta.iterrows(),yte,p):
            preds.append({'heldout':held,'t_end_s':int(mr.t_end_s),'tau_true_s':float(yy),'tau_pred_s':float(pp),'abs_error_s':float(abs(pp-yy))})
    df=pd.DataFrame(rows); df.to_csv(args.out/'fold_metrics.csv',index=False); pd.DataFrame(preds).to_csv(args.out/'predictions.csv',index=False)
    summary={'mae_mean_s':df.mae_s.mean(),'mae_worst_s':df.mae_s.max(),'rmse_mean_s':df.rmse_s.mean(),'medae_mean_s':df.medae_s.mean(),'r2_mean':df.r2.mean()}
    lines=['# Exact-fold RF baseline','',
           'Same frozen primary task as the method-paper benchmark: 120 s causal temperature+pressure input, 0<tau<=600 s to first package `vent_gas`, max input T<=150 C, stride 10 s. Each fold uses six training experiments, one validation experiment (unused by fixed RF hyperparameters), and one held-out test experiment.','',
           f"- Mean MAE: **{summary['mae_mean_s']:.2f} s**",f"- Worst MAE: **{summary['mae_worst_s']:.2f} s**",f"- Mean RMSE: **{summary['rmse_mean_s']:.2f} s**",f"- Mean R2: **{summary['r2_mean']:.4f}**",'',
           '| heldout | MAE | RMSE | MedAE | R2 |','|---|---:|---:|---:|---:|']
    for _,r in df.iterrows(): lines.append(f"| {r.heldout} | {r.mae_s:.2f} | {r.rmse_s:.2f} | {r.medae_s:.2f} | {r.r2:.4f} |")
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
