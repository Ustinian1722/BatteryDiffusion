#!/usr/bin/env python3
"""Classical sanity diagnostic for the v3 internal precursor representation.

This does not tune LATH-Net. It asks whether baseline-relative pressure dynamics
carry cross-experiment TTV information under the exact same 6/1/1 folds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from remotezip import RemoteZip
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_virtual_vehicle_time_to_vent_baseline import build_windows
from evaluate_early_warning_baseline import window_descriptors


def diff_pad(x):
    d=np.zeros_like(x,dtype=float); d[...,1:]=x[...,1:]-x[...,:-1]; return d


def rep_features(x,rep):
    t=x[:,0:1,:].astype(float); p=x[:,1:2,:].astype(float)
    pr=p-p[...,:1]; dp=diff_pad(p)
    rms=np.sqrt(np.mean(pr*pr,axis=2,keepdims=True)+1e-8); ps=pr/rms
    if rep=='temperature': return window_descriptors(t)
    if rep=='fusion_absolute': return window_descriptors(x)
    if rep=='temp_relP': return np.concatenate([window_descriptors(t),window_descriptors(pr)],axis=1)
    if rep=='temp_relP_dP': return np.concatenate([window_descriptors(t),window_descriptors(pr),window_descriptors(dp)],axis=1)
    if rep=='temp_relP_dP_shape': return np.concatenate([window_descriptors(t),window_descriptors(pr),window_descriptors(dp),window_descriptors(ps)],axis=1)
    if rep=='temp_absP_relP_dP': return np.concatenate([window_descriptors(t),window_descriptors(p),window_descriptors(pr),window_descriptors(dp)],axis=1)
    raise ValueError(rep)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/v3_precursor_representation_audit'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    data={}
    with RemoteZip(record_url(),headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df,event,_=load_one(rz,exp); data[exp]=build_windows(df,event,120,10,600,150.0)
    reps=('temperature','fusion_absolute','temp_relP','temp_relP_dP','temp_relP_dP_shape','temp_absP_relP_dP')
    rows=[]
    for hi,held in enumerate(EXPS):
        val=EXPS[(hi+1)%len(EXPS)]; trains=[e for e in EXPS if e not in (held,val)]
        xtr=np.concatenate([data[e][0] for e in trains]); ytr=np.concatenate([data[e][1] for e in trains])
        xte,yte,_=data[held]
        for rep in reps:
            m=RandomForestRegressor(n_estimators=300,max_depth=5,min_samples_leaf=3,random_state=2026,n_jobs=-1)
            m.fit(rep_features(xtr,rep),ytr); p=np.clip(m.predict(rep_features(xte,rep)),0,600)
            rows.append({'heldout':held,'validation':val,'representation':rep,'mae_s':mean_absolute_error(yte,p),'rmse_s':mean_squared_error(yte,p)**0.5,'r2':r2_score(yte,p),'bias_s':float(np.mean(p-yte))})
    df=pd.DataFrame(rows); df.to_csv(args.out/'fold_metrics.csv',index=False)
    macro=df.groupby('representation').agg(mae_mean_s=('mae_s','mean'),mae_worst_s=('mae_s','max'),rmse_mean_s=('rmse_s','mean'),r2_mean=('r2','mean')).reset_index().sort_values('mae_mean_s')
    macro.to_csv(args.out/'macro_metrics.csv',index=False)
    lines=['# V3 precursor representation sanity audit','',
           'Fixed Random Forest under the same 6-train/1-validation/1-test TTV protocol. This is a representation diagnostic, not a model-selection result. Validation experiments are excluded from RF fitting even though RF hyperparameters are fixed.','',
           '| representation | mean MAE | worst MAE | mean RMSE | mean R2 |','|---|---:|---:|---:|---:|']
    for _,r in macro.iterrows(): lines.append(f"| {r.representation} | {r.mae_mean_s:.2f} | {r.mae_worst_s:.2f} | {r.rmse_mean_s:.2f} | {r.r2_mean:.4f} |")
    lines += ['', 'Interpretation: if relative/differential pressure variants outperform or stabilize `fusion_absolute`, the precursor-invariant representation has independent support. If they do not, v3 can still succeed through learned lag fusion, but removal of absolute pressure must not be claimed intrinsically beneficial.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
