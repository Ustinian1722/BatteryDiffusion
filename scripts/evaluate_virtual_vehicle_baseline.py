#!/usr/bin/env python3
"""Real-only LOEO baseline for the frozen Virtual Vehicle TS0330 task."""
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
from remotezip import RemoteZip

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_early_warning_baseline import window_descriptors
from evaluate_geometry_normalized_mechanics import shape_descriptor_1ch


def build_windows(df: pd.DataFrame,event: float,window=120,stride=10,horizon=300,cap=150.0):
    t=df.time_s.to_numpy(dtype=int); vals=df[['temperature_c','pressure_bar_abs']].to_numpy(dtype=float)
    xs=[]; ys=[]; meta=[]
    for s in range(0,len(df)-window+1,stride):
        e=s+window; tw=t[s:e]
        if len(tw)!=window or np.any(np.diff(tw)!=1): continue
        x=vals[s:e].T.astype(np.float32); end=int(tw[-1]); tau=float(event-end); mx=float(np.max(x[0]))
        if tau<=0 or mx>cap: continue
        y=int(tau<=horizon); xs.append(x); ys.append(y); meta.append({'t_end_s':end,'tau_to_event_s':tau,'label':y,'max_temperature_c':mx})
    if not xs: return np.empty((0,2,window),np.float32),np.empty(0,int),pd.DataFrame(meta)
    return np.stack(xs),np.asarray(ys,dtype=int),pd.DataFrame(meta)


def feature_matrix(x,rep):
    if rep=='temperature': return window_descriptors(x[:,0:1,:])
    if rep=='pressure_absolute': return window_descriptors(x[:,1:2,:])
    if rep=='pressure_shape': return shape_descriptor_1ch(x[:,1:2,:])
    if rep=='fusion_absolute': return window_descriptors(x)
    if rep=='fusion_shape': return np.concatenate([window_descriptors(x[:,0:1,:]),shape_descriptor_1ch(x[:,1:2,:])],axis=1)
    raise ValueError(rep)


def fit_eval(xtr,ytr,xte,yte,rep):
    m=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026))])
    m.fit(feature_matrix(xtr,rep),ytr); p=m.predict_proba(feature_matrix(xte,rep))[:,1]; pred=(p>=0.5).astype(int)
    return {'auroc':float(roc_auc_score(yte,p)),'auprc':float(average_precision_score(yte,p)),
            'f1':float(f1_score(yte,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred))},p


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/virtual_vehicle_baseline'))
    ap.add_argument('--window',type=int,default=120); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--horizon',type=int,default=300); ap.add_argument('--temp-cap',type=float,default=150.0)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    data={}; support=[]
    with RemoteZip(record_url(),headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df,event,a=load_one(rz,exp); x,y,m=build_windows(df,event,args.window,args.stride,args.horizon,args.temp_cap)
            data[exp]=(x,y,m); support.append({'experiment':exp,'event_time_s':event,'windows':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'positive_fraction':float(y.mean())})
    pd.DataFrame(support).to_csv(args.out/'support.csv',index=False)

    reps=('temperature','pressure_absolute','pressure_shape','fusion_absolute','fusion_shape')
    rows=[]; pred_rows=[]
    for held in EXPS:
        train=[e for e in EXPS if e!=held]
        xtr=np.concatenate([data[e][0] for e in train]); ytr=np.concatenate([data[e][1] for e in train])
        xte,yte,meta=data[held]
        for rep in reps:
            metrics,p=fit_eval(xtr,ytr,xte,yte,rep); rows.append({'heldout':held,'representation':rep,'n_train':len(ytr),'n_test':len(yte),'test_positive':int(yte.sum()),**metrics})
            for (_,mr),yy,pp in zip(meta.iterrows(),yte,p):
                pred_rows.append({'heldout':held,'representation':rep,'t_end_s':int(mr.t_end_s),'tau_to_event_s':float(mr.tau_to_event_s),'label':int(yy),'probability':float(pp)})
    detail=pd.DataFrame(rows); detail.to_csv(args.out/'loeo_metrics.csv',index=False); pd.DataFrame(pred_rows).to_csv(args.out/'loeo_predictions.csv',index=False)
    macro=detail.groupby('representation').agg(auroc_mean=('auroc','mean'),auroc_worst=('auroc','min'),auprc_mean=('auprc','mean'),auprc_worst=('auprc','min'),f1_mean=('f1','mean'),balanced_acc_mean=('balanced_accuracy','mean')).reset_index()
    macro.to_csv(args.out/'macro_metrics.csv',index=False)
    lines=['# Virtual Vehicle TS0330 real-only LOEO baseline','',
           f'Frozen before scoring: {args.window} s causal context, {args.horizon} s horizon to the first package `vent_gas` marker, max input cell-case temperature <= {args.temp_cap:g} C, stride {args.stride} s.','',
           'All eight BAK N21700CG-50 experiments are evaluated leave-one-real-experiment-out. No synthetic windows are used.','',
           '| representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 | mean balanced acc |','|---|---:|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows():
        lines.append(f"| {r.representation} | {r.auroc_mean:.3f} | {r.auroc_worst:.3f} | {r.auprc_mean:.3f} | {r.auprc_worst:.3f} | {r.f1_mean:.3f} | {r.balanced_acc_mean:.3f} |")
    lines += ['', '## Guardrail','',
              'This baseline was run only after `reports/virtual_vehicle_validation_protocol_freeze.md` was committed. Experiment count is n=8; overlapping causal windows are repeated decision points, not independent destructive tests.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
