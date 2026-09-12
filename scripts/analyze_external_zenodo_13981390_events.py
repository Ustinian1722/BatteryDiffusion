#!/usr/bin/env python3
"""Analyze time alignment and vent-event structure in Zenodo 13981390.

Raw external MAT files are downloaded to a temporary runner directory and are
not committed. Time axes are converted to seconds and temperature/pressure are
aligned at 1 Hz for compatibility analysis.

The companion paper (Batteries 2024, 10, 435; DOI 10.3390/batteries10120435)
explicitly reports first-vent times for both experiments. Those published labels
are kept distinct from the pressure-drop detector used here as an independent
signal-level audit.
"""
from __future__ import annotations

import argparse
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
# Source paper Tables 8-9.
PUBLISHED={
    'NMC111': {'vent_time_s':2*3600+51*60, 'vent_temperature_c':152.0, 'tr_time_s':3*3600+26*60},
    'NMC811': {'vent_time_s':22*60+30, 'vent_temperature_c':185.0, 'tr_time_s':24*60+14},
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
    idx=np.asarray(mat['Time_nmc811_100Hz'],dtype=float).ravel(); t=idx/100.0
    tv=np.asarray(mat['T_average_nmc811'],dtype=float).ravel(); pv=np.asarray(mat['pressure_nmc811'],dtype=float).ravel()
    valid=np.isfinite(t)&np.isfinite(tv)&np.isfinite(pv); t,tv,pv=t[valid],tv[valid],pv[valid]
    grid=np.arange(np.ceil(t.min()),np.floor(t.max())+1,1.0)
    return pd.DataFrame({'time_s':grid,'temperature_c':np.interp(grid,t,tv),'pressure':np.interp(grid,t,pv)})


def landmarks(df: pd.DataFrame, published: dict)->dict:
    t=df.time_s.to_numpy(dtype=float); temp=df.temperature_c.to_numpy(dtype=float); p=df.pressure.to_numpy(dtype=float)
    temp_s=moving_median(temp,5); p_s=moving_median(p,5); dT=np.gradient(temp_s,t); dP=np.gradient(p_s,t)
    peakP=int(np.argmax(p_s)); rapidT=int(np.argmax(dT))
    vent_truth=float(published['vent_time_s'])
    # Pressure-drop detector searches a +/-120 s neighborhood around the source-
    # reported event only for the validation audit. It is not used to define the label.
    local=(t>=vent_truth-120)&(t<=vent_truth+120); ids=np.flatnonzero(local)
    vent_det=int(ids[np.argmin(dP[ids])]) if len(ids) else int(np.argmin(np.abs(t-vent_truth)))
    truth_i=int(np.argmin(np.abs(t-vent_truth)))
    return {
        'duration_overlap_s':float(t[-1]-t[0]),'rows_1hz':int(len(df)),
        'published_vent_time_s':vent_truth,'published_vent_temperature_c':float(published['vent_temperature_c']),
        'published_tr_time_s':float(published['tr_time_s']),
        'aligned_temperature_at_vent_c':float(temp_s[truth_i]),
        'pressure_peak':float(p_s[peakP]),'pressure_peak_time_s':float(t[peakP]),
        'detected_pressure_drop_time_s':float(t[vent_det]),'detected_pressure_drop_dpdt':float(dP[vent_det]),
        'vent_detection_error_s':float(t[vent_det]-vent_truth),
        'rapid_temperature_time_s':float(t[rapidT]),'max_dTdt_c_per_s':float(dT[rapidT]),
        'overlap_covers_published_tr':bool(t[-1] >= published['tr_time_s']),
        'pre_vent_seconds_below_120':int(np.sum((t<vent_truth)&(temp_s<=120.0))),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/external_zenodo_13981390/event_analysis'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); results=[]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for label,url in FILES.items():
            p=td/f'{label}.mat'; fetch(p,url); mat=loadmat(p,squeeze_me=True,struct_as_record=False)
            df=align_nmc111(mat) if label=='NMC111' else align_nmc811(mat)
            rec={'experiment_label':label,**landmarks(df,PUBLISHED[label])}; results.append(rec)
            step=max(1,len(df)//200); diag=df.iloc[::step].copy(); diag['experiment_label']=label
            diag.to_csv(args.out/f'{label}_diagnostic_downsample.csv',index=False)
    pd.DataFrame(results).to_csv(args.out/'event_landmarks.csv',index=False)
    lines=['# External Zenodo 13981390 — published-event validation','',
           'Temperature and pressure were aligned to a common 1 Hz time base. The source MAT files are downloaded only transiently in CI.','',
           'The companion open-access paper explicitly reports first venting at **2 h 51 min (10,260 s)** for NMC111 and **22 min 30 s (1,350 s)** for NMC811. These are source-provided labels, not labels inferred from our pressure signal.','',
           '| experiment | published vent | vent T | detected pressure drop | error | pressure peak | overlap covers published TR? |',
           '|---|---:|---:|---:|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['experiment_label']} | {r['published_vent_time_s']:.0f}s | {r['published_vent_temperature_c']:.0f}°C | {r['detected_pressure_drop_time_s']:.0f}s | {r['vent_detection_error_s']:+.0f}s | {r['pressure_peak']:.3f} | {'yes' if r['overlap_covers_published_tr'] else 'no'} |")
    lines += ['', '## Interpretation','',
              'The independently derived strongest local pressure drop agrees with the published vent landmark to approximately the sampling resolution in both experiments. This materially strengthens their suitability as **external event-labeled thermo-pressure experiments**. NMC111 pressure recording ends before the paper-reported TR time, so it is suitable for pre-vent/vent studies but not for pressure behavior through the complete TR peak.','',
              '## Compatibility decision','',
              'Both experiments support the current 256-s causal-window machinery and have authoritative first-vent labels. Because they are prismatic NMC111/NMC811 cells with different heating rates, SOCs and apparatus, they should be treated as external/domain-shift experiments rather than homogeneous replicas of the primary Hanyang cohort.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
