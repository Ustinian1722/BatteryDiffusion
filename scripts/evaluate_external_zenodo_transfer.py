#!/usr/bin/env python3
"""Frozen cross-domain external validation on Zenodo 13981390.

Protocol was frozen in reports/external_validation_protocol_freeze.md before
this script was executed. Development: A2/D1/M1/M2. External tests: NMC111 and
NMC811. No synthetic data and no external calibration are used.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_early_warning_baseline import load_pouch, load_module, window_descriptors
from prepare_early_warning_fold import build, EXPS
from evaluate_geometry_normalized_mechanics import shape_descriptor_1ch
from analyze_external_zenodo_13981390_events import FILES, fetch, align_nmc111, align_nmc811

EXT_VENT={"NMC111":10260,"NMC811":1350}


def build_external(df: pd.DataFrame, vent: int, window: int, stride: int, horizon: int, cap: float):
    d=pd.DataFrame({
        'time':np.rint(df.time_s.to_numpy(dtype=float)).astype(int),
        'temperature':df.temperature_c.to_numpy(dtype=float),
        'mechanical':df.pressure.to_numpy(dtype=float),
    }).dropna().sort_values('time').drop_duplicates('time')
    d=d[d.time < vent]
    times=d.time.to_numpy(dtype=int); vals=d[['temperature','mechanical']].to_numpy(dtype=float)
    xs=[]; y=[]; ends=[]; taus=[]; max_ts=[]
    for s in range(0,len(d)-window+1,stride):
        e=s+window; tw=times[s:e]
        if len(tw)!=window or np.any(np.diff(tw)!=1):
            continue
        x=vals[s:e].T.astype(np.float32); t_end=int(tw[-1]); tau=vent-t_end
        if tau<=0 or float(np.max(x[0]))>cap:
            continue
        xs.append(x); y.append(int(tau<=horizon)); ends.append(t_end); taus.append(tau); max_ts.append(float(np.max(x[0])))
    if not xs:
        return np.empty((0,2,window),dtype=np.float32),np.empty(0,dtype=int),pd.DataFrame()
    meta=pd.DataFrame({'t_end_s':ends,'tau_to_vent_s':taus,'label':y,'max_temperature_c':max_ts})
    return np.stack(xs),np.asarray(y,dtype=int),meta


def feature_matrix(x: np.ndarray, rep: str)->np.ndarray:
    if rep=='temperature':
        return window_descriptors(x[:,0:1,:])
    if rep=='mechanical_shape':
        return shape_descriptor_1ch(x[:,1:2,:])
    if rep=='fusion_shape':
        return np.concatenate([window_descriptors(x[:,0:1,:]),shape_descriptor_1ch(x[:,1:2,:])],axis=1)
    raise ValueError(rep)


def fit_model(x: np.ndarray,y: np.ndarray,rep: str):
    m=Pipeline([
        ('scale',StandardScaler()),
        ('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026)),
    ])
    m.fit(feature_matrix(x,rep),y)
    return m


def evaluate(m,x,y,rep):
    p=m.predict_proba(feature_matrix(x,rep))[:,1]; pred=(p>=0.5).astype(int)
    out={'auroc':np.nan,'auprc':np.nan,'f1':float(f1_score(y,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred))}
    if len(np.unique(y))==2:
        out['auroc']=float(roc_auc_score(y,p)); out['auprc']=float(average_precision_score(y,p))
    return out,p


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data/raw/osf_c2hnq/Dataset_TR/Dataset_TR')); ap.add_argument('--out',type=Path,default=Path('reports/external_zenodo_transfer'))
    ap.add_argument('--window',type=int,default=256); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--horizon',type=int,default=900); ap.add_argument('--temp-cap',type=float,default=120.0)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    # Build development set from the four source experiments only.
    dx=[]; dy=[]; dev_support=[]
    for exp in EXPS:
        raw=load_pouch(args.root,exp) if exp in ('A2','D1') else load_module(args.root,exp)
        x,m=build(raw,exp,args.window,args.stride,args.horizon,args.temp_cap)
        dx.append(x); dy.append(m.label.to_numpy(dtype=int)); dev_support.append({'experiment':exp,'windows':len(m),'positive':int(m.label.sum())})
    xdev=np.concatenate(dx); ydev=np.concatenate(dy)
    pd.DataFrame(dev_support).to_csv(args.out/'development_support.csv',index=False)

    reps=('temperature','mechanical_shape','fusion_shape')
    models={rep:fit_model(xdev,ydev,rep) for rep in reps}
    rows=[]; preds=[]; support=[]

    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for label,url in FILES.items():
            p=td/f'{label}.mat'; fetch(p,url); mat=loadmat(p,squeeze_me=True,struct_as_record=False)
            ext=align_nmc111(mat) if label=='NMC111' else align_nmc811(mat)
            x,y,meta=build_external(ext,EXT_VENT[label],args.window,args.stride,args.horizon,args.temp_cap)
            support.append({'experiment':label,'vent_s':EXT_VENT[label],'windows':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'positive_fraction':float(y.mean()) if len(y) else np.nan,'max_retained_temperature_c':float(meta.max_temperature_c.max()) if len(meta) else np.nan})
            for rep in reps:
                if len(y)==0:
                    rows.append({'experiment':label,'representation':rep,'n_test':0,'positive':0,'auroc':np.nan,'auprc':np.nan,'f1':np.nan,'balanced_accuracy':np.nan}); continue
                metrics,prob=evaluate(models[rep],x,y,rep)
                rows.append({'experiment':label,'representation':rep,'n_test':len(y),'positive':int(y.sum()),**metrics})
                for (_,mr),yy,pp in zip(meta.iterrows(),y,prob):
                    preds.append({'experiment':label,'representation':rep,'t_end_s':int(mr.t_end_s),'tau_to_vent_s':int(mr.tau_to_vent_s),'label':int(yy),'probability':float(pp),'max_temperature_c':float(mr.max_temperature_c)})

    support_df=pd.DataFrame(support); support_df.to_csv(args.out/'external_support.csv',index=False)
    res=pd.DataFrame(rows); res.to_csv(args.out/'external_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(args.out/'external_predictions.csv',index=False)
    macro=res.groupby('representation',dropna=False).agg(auroc_mean=('auroc','mean'),auroc_worst=('auroc','min'),auprc_mean=('auprc','mean'),auprc_worst=('auprc','min'),f1_mean=('f1','mean'),balanced_acc_mean=('balanced_accuracy','mean')).reset_index()
    macro.to_csv(args.out/'macro_descriptive.csv',index=False)

    lines=['# Frozen external transfer validation — Zenodo 13981390','',
           f'Protocol: {args.window} s causal context, {args.horizon} s first-vent horizon, max input temperature <= {args.temp_cap:g} C, stride {args.stride} s. Development data are A2/D1/M1/M2 only; NMC111/NMC811 are untouched external tests.','',
           '## External support','',
           '| experiment | published vent | windows | positive | negative | max retained T |','|---|---:|---:|---:|---:|---:|']
    for _,r in support_df.iterrows():
        lines.append(f"| {r.experiment} | {int(r.vent_s)} s | {int(r.windows)} | {int(r.positive)} | {int(r.negative)} | {r.max_retained_temperature_c:.1f} C |")
    lines += ['', '## Per-experiment results','', '| experiment | representation | AUROC | AUPRC | F1@0.5 | balanced acc |','|---|---|---:|---:|---:|---:|']
    for _,r in res.iterrows():
        lines.append(f"| {r.experiment} | {r.representation} | {r.auroc:.3f} | {r.auprc:.3f} | {r.f1:.3f} | {r.balanced_accuracy:.3f} |")
    lines += ['', '## Two-experiment descriptive macro','', '| representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |','|---|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows():
        lines.append(f"| {r.representation} | {r.auroc_mean:.3f} | {r.auroc_worst:.3f} | {r.auprc_mean:.3f} | {r.auprc_worst:.3f} | {r.f1_mean:.3f} |")
    lines += ['', '## Interpretation guardrail','',
              'These are two independent external experiments under severe domain shift: pouch-cell/module expansion force in development versus prismatic-cell internal pressure externally, with different chemistry, SOC, heating profile and apparatus. The result is retained without retuning. Window counts are not independent experimental replicates.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
