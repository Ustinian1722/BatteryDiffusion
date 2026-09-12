#!/usr/bin/env python3
"""Support-only audit of Zenodo 13981390 under the frozen TS0330 task definitions.

No predictor is fit. Published first-vent times from the source paper define the
landmarks. The purpose is only to determine whether the already frozen Virtual
Vehicle classification/TTE interfaces can be evaluated out of domain without
retuning on external predictive scores.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from analyze_external_zenodo_13981390_events import FILES, PUBLISHED, fetch, align_nmc111, align_nmc811


def count_classification(df: pd.DataFrame, event: float, window=120, stride=10, horizon=300, cap=150.0):
    t=df.time_s.to_numpy(dtype=float)
    temp=df.temperature_c.to_numpy(dtype=float)
    pos=neg=0; taus=[]; max_t=[]
    for s in range(0,len(df)-window+1,stride):
        e=s+window; tw=t[s:e]
        if len(tw)!=window or np.any(np.diff(tw)!=1): continue
        tau=float(event-tw[-1]); mx=float(np.max(temp[s:e]))
        if tau<=0 or mx>cap: continue
        taus.append(tau); max_t.append(mx)
        if tau<=horizon: pos+=1
        else: neg+=1
    return {'positive':pos,'negative':neg,'total':pos+neg,
            'tau_min_s':min(taus) if taus else np.nan,'tau_max_s':max(taus) if taus else np.nan,
            'max_retained_temperature_c':max(max_t) if max_t else np.nan}


def count_tte(df: pd.DataFrame, event: float, window=120, stride=10, max_tau=600, cap=150.0):
    t=df.time_s.to_numpy(dtype=float); temp=df.temperature_c.to_numpy(dtype=float); taus=[]
    for s in range(0,len(df)-window+1,stride):
        e=s+window; tw=t[s:e]
        if len(tw)!=window or np.any(np.diff(tw)!=1): continue
        tau=float(event-tw[-1]); mx=float(np.max(temp[s:e]))
        if 0<tau<=max_tau and mx<=cap: taus.append(tau)
    a=np.asarray(taus,float)
    return {'tte_windows':int(len(a)),'tte_tau_min_s':float(a.min()) if len(a) else np.nan,
            'tte_tau_median_s':float(np.median(a)) if len(a) else np.nan,'tte_tau_max_s':float(a.max()) if len(a) else np.nan}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/external_zenodo_13981390/virtual_vehicle_task_support'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); rows=[]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for label,url in FILES.items():
            p=td/f'{label}.mat'; fetch(p,url); mat=loadmat(p,squeeze_me=True,struct_as_record=False)
            df=align_nmc111(mat) if label=='NMC111' else align_nmc811(mat)
            event=float(PUBLISHED[label]['vent_time_s'])
            c=count_classification(df,event); r=count_tte(df,event)
            rows.append({'experiment':label,'published_vent_s':event,'rows_1hz':len(df),**c,**r})
    out=pd.DataFrame(rows); out.to_csv(args.out/'support.csv',index=False)
    fully_cls=bool(((out.positive>0)&(out.negative>0)).all())
    fully_tte=bool((out.tte_windows>0).all())
    lines=['# External Zenodo 13981390 support under frozen Virtual Vehicle tasks','',
           'This is a support-only audit. No predictor is fit and no task parameter is selected from external predictive performance. Published first-vent times are used as event landmarks.','',
           'Frozen interfaces checked: classification = 120 s context / 300 s horizon / max input T <=150 C; TTE = 120 s context / 0<tau<=600 s / max input T <=150 C; stride = 10 s.','',
           '| experiment | cls positive | cls negative | cls tau range | TTE windows | TTE tau range |','|---|---:|---:|---:|---:|---:|']
    for _,r in out.iterrows():
        lines.append(f"| {r.experiment} | {int(r.positive)} | {int(r.negative)} | {r.tau_min_s:.0f}-{r.tau_max_s:.0f} s | {int(r.tte_windows)} | {r.tte_tau_min_s:.0f}-{r.tte_tau_max_s:.0f} s |")
    lines += ['',f'- Classification fully evaluable on both experiments: **{fully_cls}**',f'- TTE support present on both experiments: **{fully_tte}**','',
              '## Guardrail','',
              'These experiments differ in chemistry, form factor, SOC, heating program and apparatus from the Virtual Vehicle cohort. If predictive transfer is run, they remain untouched external/domain-shift tests and contribute nothing to scaling, generator fitting, threshold selection or task definition.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
