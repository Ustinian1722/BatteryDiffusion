#!/usr/bin/env python3
"""Analyze time alignment and candidate TR landmarks in Zenodo 13981390.

Raw external MAT files are downloaded to a temporary runner directory and are
not committed. This script converts their time axes to seconds, aligns
1-second temperature/pressure trajectories, and reports descriptive event
landmarks useful for deciding cross-domain validation protocols.

Candidate event landmarks are algorithmic diagnostics, not source-provided
venting ground truth.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

FILES={
    'NMC111':'https://zenodo.org/records/13981390/files/NMC111.mat?download=1',
    'NMC811':'https://zenodo.org/records/13981390/files/NMC811.mat?download=1',
}


def fetch(path: Path,url: str):
    r=requests.get(url,timeout=120); r.raise_for_status(); path.write_bytes(r.content)


def moving_median(y: np.ndarray,width: int)->np.ndarray:
    return pd.Series(y).rolling(width,center=True,min_periods=1).median().to_numpy(dtype=float)


def align_nmc111(mat: dict)->pd.DataFrame:
    tt=np.asarray(mat['Time_T_nmc111_hour'],dtype=float).ravel()*3600.0
    tv=np.asarray(mat['T_average_nmc111'],dtype=float).ravel()
    pt=np.asarray(mat['Time_p_nmc111_hour'],dtype=float).ravel()*3600.0
    pv=np.asarray(mat['pressure_nmc111'],dtype=float).ravel()
    validt=np.isfinite(tt)&np.isfinite(tv); validp=np.isfinite(pt)&np.isfinite(pv)
    tt,tv=tt[validt],tv[validt]; pt,pv=pt[validp],pv[validp]
    start=max(tt.min(),pt.min()); end=min(tt.max(),pt.max())
    grid=np.arange(np.ceil(start),np.floor(end)+1,1.0)
    return pd.DataFrame({'time_s':grid,'temperature_c':np.interp(grid,tt,tv),'pressure':np.interp(grid,pt,pv)})


def align_nmc811(mat: dict)->pd.DataFrame:
    idx=np.asarray(mat['Time_nmc811_100Hz'],dtype=float).ravel()
    t=idx/100.0
    tv=np.asarray(mat['T_average_nmc811'],dtype=float).ravel()
    pv=np.asarray(mat['pressure_nmc811'],dtype=float).ravel()
    valid=np.isfinite(t)&np.isfinite(tv)&np.isfinite(pv)
    t,tv,pv=t[valid],tv[valid],pv[valid]
    grid=np.arange(np.ceil(t.min()),np.floor(t.max())+1,1.0)
    return pd.DataFrame({'time_s':grid,'temperature_c':np.interp(grid,t,tv),'pressure':np.interp(grid,t,pv)})


def landmarks(df: pd.DataFrame)->dict:
    t=df.time_s.to_numpy(dtype=float); temp=df.temperature_c.to_numpy(dtype=float); p=df.pressure.to_numpy(dtype=float)
    temp_s=moving_median(temp,5); p_s=moving_median(p,5)
    dT=np.gradient(temp_s,t); dP=np.gradient(p_s,t)
    peakT=int(np.argmax(temp_s)); peakP=int(np.argmax(p_s)); rapidT=int(np.argmax(dT[:peakT+1])) if peakT>2 else peakT
    # Vent-like candidate: strongest pressure drop after pressure has reached at
    # least 20% of its pre-peak excursion and before thermal peak.
    base=float(np.median(p_s[:min(120,len(p_s))])); excursion=max(float(np.max(p_s[:peakT+1])-base),1e-9)
    eligible=(np.arange(len(t))<peakT)&(p_s>=base+0.20*excursion)
    ids=np.flatnonzero(eligible)
    vent=int(ids[np.argmin(dP[ids])]) if len(ids) else int(np.argmin(dP[:peakT+1]))
    return {
        'duration_s':float(t[-1]-t[0]),'rows_1hz':int(len(df)),
        'temperature_start_c':float(temp_s[0]),'temperature_peak_c':float(temp_s[peakT]),'temperature_peak_time_s':float(t[peakT]),
        'max_dTdt_c_per_s':float(dT[rapidT]),'rapid_temperature_time_s':float(t[rapidT]),
        'pressure_baseline':base,'pressure_peak':float(p_s[peakP]),'pressure_peak_time_s':float(t[peakP]),
        'vent_like_drop_time_s':float(t[vent]),'vent_like_pressure':float(p_s[vent]),'vent_like_dpdt':float(dP[vent]),
        'vent_like_to_temp_peak_s':float(t[peakT]-t[vent]),
        'pre_peak_below_120_s':int(np.sum((t<t[peakT])&(temp_s<=120.0))),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/external_zenodo_13981390/event_analysis'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    results=[]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for label,url in FILES.items():
            p=td/f'{label}.mat'; fetch(p,url); mat=loadmat(p,squeeze_me=True,struct_as_record=False)
            df=align_nmc111(mat) if label=='NMC111' else align_nmc811(mat)
            rec={'experiment_label':label,**landmarks(df)}; results.append(rec)
            # Commit only a strongly downsampled, non-raw diagnostic profile so
            # event timing can be visually/reproducibly inspected without redistributing source data.
            step=max(1,len(df)//200)
            diag=df.iloc[::step].copy(); diag['experiment_label']=label
            diag.to_csv(args.out/f'{label}_diagnostic_downsample.csv',index=False)
    pd.DataFrame(results).to_csv(args.out/'event_landmarks.csv',index=False)
    lines=['# External Zenodo 13981390 — event-structure audit','',
           'Temperature and pressure were aligned to a common 1 Hz time base for compatibility analysis. The source MAT files are downloaded only transiently in CI.','',
           '**Important:** the pressure-drop times below are algorithmic vent-like candidates, not published vent annotations. They must not be used as ground truth without source-paper verification.','',
           '| experiment | duration | T peak | T-peak time | P peak | P-peak time | vent-like drop | drop→Tpeak |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['experiment_label']} | {r['duration_s']:.0f}s | {r['temperature_peak_c']:.1f}°C | {r['temperature_peak_time_s']:.0f}s | {r['pressure_peak']:.3f} | {r['pressure_peak_time_s']:.0f}s | {r['vent_like_drop_time_s']:.0f}s | {r['vent_like_to_temp_peak_s']:.0f}s |")
    lines += ['', '## Compatibility decision','',
              'Both records are long enough for the current 256-s causal-window machinery and expose synchronized temperature + pressure after alignment. Because these are prismatic cells with different chemistry/apparatus from the Hanyang 2170 cohort, they are best used as cross-domain thermo-pressure validation rather than pooled as homogeneous training replicas.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
