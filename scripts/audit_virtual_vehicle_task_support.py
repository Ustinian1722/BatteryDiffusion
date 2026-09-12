#!/usr/bin/env python3
"""Audit event-window support in the 8-test Virtual Vehicle TS0330 package.

This script deliberately computes *no predictive scores*. It uses the package's
own `order_TS0330X.xlsx` vent_gas event markers and compact `export_TS0330X.xlsx`
trajectories to determine which context/horizon/temperature-cap definitions have
both positive and negative pre-event windows across all eight independent tests.

No raw external workbook is committed.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip

RECORD_ID='18849418'
API=f'https://zenodo.org/api/records/{RECORD_ID}'
HEADERS={'User-Agent':'BatteryDiffusion-research-audit/1.0'}
EXPS=[f'TS0330{x}' for x in 'ABCDEFGH']


def record_url():
    r=requests.get(API,timeout=60,headers=HEADERS); r.raise_for_status(); rec=r.json()
    z=next(f for f in rec.get('files',[]) if str(f.get('key','')).lower().endswith('.zip'))
    links=z.get('links') or {}; url=links.get('content') or links.get('self')
    if not url: raise RuntimeError('Zenodo ZIP URL unavailable')
    return url


def read_xlsx_bytes(rz: RemoteZip,path: str)->pd.DataFrame:
    return pd.read_excel(io.BytesIO(rz.read(path)),header=None,engine='openpyxl')


def load_one(rz: RemoteZip,exp: str):
    export=f'Plots_and_raw_data/{exp}/export_{exp}.xlsx'
    order=f'Plots_and_raw_data/{exp}/order_{exp}.xlsx'
    edf=read_xlsx_bytes(rz,export)
    odf=read_xlsx_bytes(rz,order)
    tech=[str(v).strip() if pd.notna(v) else '' for v in edf.iloc[0].tolist()]
    sem=[str(v).strip() if pd.notna(v) else '' for v in edf.iloc[1].tolist()]
    num=edf.iloc[2:].apply(pd.to_numeric,errors='coerce')
    # Dataset compact export conventions verified from package metadata:
    # time is milliseconds, pressure is Pa, cell-case temperature is degC.
    t=num.iloc[:,0].to_numpy(dtype=float)/1000.0
    p=num.iloc[:,1].to_numpy(dtype=float)/1e5  # absolute bar; shape later is offset-invariant
    temp=num.iloc[:,5].to_numpy(dtype=float)
    ok=np.isfinite(t)&np.isfinite(p)&np.isfinite(temp)
    t,p,temp=t[ok],p[ok],temp[ok]
    order_types=odf.iloc[:,2].astype(str).str.strip().str.lower()
    vent_rows=odf[order_types.eq('vent_gas')]
    if vent_rows.empty: raise RuntimeError(f'{exp}: no vent_gas event marker')
    event=float(pd.to_numeric(vent_rows.iloc[:,0],errors='coerce').dropna().min())
    # 1-Hz interpolation on measured support only.
    lo=int(np.ceil(t.min())); hi=int(np.floor(min(t.max(),event-1e-6)))
    grid=np.arange(lo,hi+1,dtype=float)
    return pd.DataFrame({'time_s':grid,'temperature_c':np.interp(grid,t,temp),'pressure_bar_abs':np.interp(grid,t,p)}),event,{
        'experiment':exp,'export_rows':int(len(t)),'raw_t_min_s':float(t.min()),'raw_t_max_s':float(t.max()),
        'median_raw_dt_s':float(np.median(np.diff(t))) if len(t)>1 else np.nan,'event_time_s':event,
        'event_type':'vent_gas','pressure_min_bar_abs':float(np.min(p)),'pressure_max_bar_abs':float(np.max(p)),
        'temperature_min_c':float(np.min(temp)),'temperature_max_c':float(np.max(temp)),
        'pressure_technical_name':tech[1] if len(tech)>1 else '', 'pressure_semantic_name':sem[1] if len(sem)>1 else '',
        'temperature_technical_name':tech[5] if len(tech)>5 else '', 'temperature_semantic_name':sem[5] if len(sem)>5 else ''}


def count_windows(df,event,window,stride,horizon,cap):
    t=df.time_s.to_numpy(dtype=int); temp=df.temperature_c.to_numpy(dtype=float)
    pos=neg=0; max_ret=-np.inf; first_end=None; last_end=None
    for s in range(0,len(df)-window+1,stride):
        e=s+window; tw=t[s:e]
        if len(tw)!=window or np.any(np.diff(tw)!=1): continue
        mx=float(np.max(temp[s:e])); end=int(tw[-1]); tau=event-end
        if tau<=0 or mx>cap: continue
        if first_end is None: first_end=end
        last_end=end; max_ret=max(max_ret,mx)
        if tau<=horizon: pos+=1
        else: neg+=1
    return {'positive':pos,'negative':neg,'total':pos+neg,'max_retained_temperature_c':max_ret if np.isfinite(max_ret) else np.nan,
            'first_window_end_s':first_end,'last_window_end_s':last_end}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/external_virtual_vehicle_21700/task_support'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    url=record_url(); data={}; audits=[]
    with RemoteZip(url,headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df,event,a=load_one(rz,exp); data[exp]=(df,event); audits.append(a)
    pd.DataFrame(audits).to_csv(args.out/'signal_event_audit.csv',index=False)

    rows=[]
    for window in (60,120,256):
        for horizon in (60,120,180,300):
            for cap in (80,100,120,150):
                for exp,(df,event) in data.items():
                    c=count_windows(df,event,window,10,horizon,cap)
                    rows.append({'window_s':window,'horizon_s':horizon,'temp_cap_c':cap,'experiment':exp,**c})
    detail=pd.DataFrame(rows); detail.to_csv(args.out/'support_detail.csv',index=False)
    grouped=[]
    for (w,h,c),g in detail.groupby(['window_s','horizon_s','temp_cap_c']):
        grouped.append({'window_s':w,'horizon_s':h,'temp_cap_c':c,'experiments_with_both_classes':int(((g.positive>0)&(g.negative>0)).sum()),
                        'experiments_with_positive':int((g.positive>0).sum()),'experiments_with_negative':int((g.negative>0).sum()),
                        'min_positive':int(g.positive.min()),'min_negative':int(g.negative.min()),'total_windows':int(g.total.sum()),
                        'total_positive':int(g.positive.sum()),'total_negative':int(g.negative.sum())})
    summary=pd.DataFrame(grouped).sort_values(['experiments_with_both_classes','window_s','horizon_s','temp_cap_c'],ascending=[False,False,True,True])
    summary.to_csv(args.out/'support_summary.csv',index=False)
    top=summary[summary.experiments_with_both_classes==len(EXPS)].copy()
    top.to_csv(args.out/'fully_evaluable_definitions.csv',index=False)

    lines=['# Virtual Vehicle TS0330 task-support audit','',
           'This audit uses all eight independent BAK N21700CG-50 tests. The package-provided first `vent_gas` marker is treated only as the candidate event landmark for support counting. No prediction model is fit here.','',
           'Compact export conventions were audited directly: time is represented in milliseconds, pressure in Pa (converted to absolute bar for readability), and cell-case temperature in degC. Trajectories are interpolated to a common 1 Hz grid before causal-window counting.','',
           f'- Experiments: **{len(EXPS)}**',f'- Fully evaluable task definitions (both classes in all 8 tests): **{len(top)}**','',
           '## Best-supported definitions','',
           '| window | horizon | T cap | min positive/test | min negative/test | total windows |','|---:|---:|---:|---:|---:|---:|']
    for _,r in top.head(12).iterrows():
        lines.append(f"| {int(r.window_s)} s | {int(r.horizon_s)} s | {int(r.temp_cap_c)} C | {int(r.min_positive)} | {int(r.min_negative)} | {int(r.total_windows)} |")
    lines += ['', '## Guardrail','',
              'This file is label-support analysis only. Any secondary external predictive protocol must be frozen using these counts before model scores are computed. The earlier 900 s / 120 C frozen test is not retroactively changed.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
