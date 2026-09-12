#!/usr/bin/env python3
"""Prepare one leakage-safe LOEO fold for the frozen early-warning task.

Frozen task definition (research_freeze_v2):
- context 256 s, stride 10 s;
- first-vent horizon 900 s;
- retain windows with max temperature <= 120 degC;
- held-out unit is one complete real experiment.

The scaler is fit on selected real training windows only. The held-out real
experiment is transformed with that scaler but never contributes to generator or
predictor training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_early_warning_baseline import load_pouch, load_module, VENT

AGE = {'A2':0,'M1':0,'D1':1,'M2':1}
LEVEL = {'A2':'cell','D1':'cell','M1':'module','M2':'module'}
EXPS = ('A2','D1','M1','M2')


def build(df: pd.DataFrame, exp: str, window: int, stride: int, horizon: int, cap: float):
    vent=VENT[exp]
    df=df[df.time < vent].sort_values('time').drop_duplicates('time')
    times=df.time.to_numpy(dtype=int); vals=df[['temperature','force']].to_numpy(dtype=float)
    xs=[]; meta=[]
    for s in range(0,len(df)-window+1,stride):
        e=s+window; tw=times[s:e]
        if len(tw)!=window or np.any(np.diff(tw)!=1): continue
        x=vals[s:e].T.astype(np.float32); t_end=int(tw[-1]); tau=vent-t_end
        if tau<=0 or float(np.max(x[0])) > cap: continue
        y=int(tau<=horizon)
        xs.append(x); meta.append({'experiment_id':exp,'age':AGE[exp],'level':LEVEL[exp],'label':y,'t_end_s':t_end,'tau_to_vent_s':tau,'max_temperature_c':float(np.max(x[0]))})
    return np.stack(xs),pd.DataFrame(meta)


def deterministic_stratified_subset(meta: pd.DataFrame, fraction: float) -> np.ndarray:
    if fraction >= 0.999:
        return np.arange(len(meta),dtype=int)
    keep=[]
    for (_,label),g in meta.groupby(['experiment_id','label'],sort=True):
        ids=g.index.to_numpy(dtype=int); n=max(2,int(round(len(ids)*fraction))) if len(ids)>=2 else len(ids); n=min(n,len(ids))
        pos=np.linspace(0,len(ids)-1,n).round().astype(int); keep.extend(ids[pos].tolist())
    return np.asarray(sorted(set(keep)),dtype=int)


def quantile_scale(x: np.ndarray, eps: float=1e-6):
    flat=np.transpose(x,(1,0,2)).reshape(x.shape[1],-1)
    q01=np.percentile(flat,1,axis=1); q99=np.percentile(flat,99,axis=1)
    center=0.5*(q01+q99); scale=np.maximum(0.5*(q99-q01),eps)
    return center.astype(np.float32),scale.astype(np.float32)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data/raw/osf_c2hnq/Dataset_TR/Dataset_TR'))
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--heldout',choices=EXPS,required=True); ap.add_argument('--fraction',type=float,default=1.0)
    ap.add_argument('--window',type=int,default=256); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--horizon',type=int,default=900); ap.add_argument('--temp-cap',type=float,default=120.0)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    all_x=[]; all_meta=[]
    for exp in EXPS:
        df=load_pouch(args.root,exp) if exp in ('A2','D1') else load_module(args.root,exp)
        x,m=build(df,exp,args.window,args.stride,args.horizon,args.temp_cap)
        all_x.append(x); all_meta.append(m)
    x_all=np.concatenate(all_x); meta_all=pd.concat(all_meta,ignore_index=True)
    train_mask=(meta_all.experiment_id != args.heldout).to_numpy(); test_mask=~train_mask
    x_train_full=x_all[train_mask]; m_train_full=meta_all.loc[train_mask].reset_index(drop=True)
    x_test=x_all[test_mask]; m_test=meta_all.loc[test_mask].reset_index(drop=True)

    ids=deterministic_stratified_subset(m_train_full,args.fraction)
    x_train=x_train_full[ids]; m_train=m_train_full.iloc[ids].reset_index(drop=True)
    center,scale=quantile_scale(x_train)
    xtr=(x_train-center[None,:,None])/scale[None,:,None]; xte=(x_test-center[None,:,None])/scale[None,:,None]

    # train_diffusion_v3_anchored expects `stage`; here it is deliberately the
    # frozen downstream class label, not the earlier temperature-shape stage.
    np.savez_compressed(args.out/'train_generator.npz',x=xtr.astype(np.float32),age=m_train.age.to_numpy(np.int64),stage=m_train.label.to_numpy(np.int64),experiment_id=m_train.experiment_id.to_numpy(dtype='U8'),center=center,scale=scale,channel_names=np.array(['temperature','force'],dtype='U32'))
    np.savez_compressed(args.out/'train_real.npz',x_raw=x_train.astype(np.float32),x_normalized=xtr.astype(np.float32),y=m_train.label.to_numpy(np.int64),age=m_train.age.to_numpy(np.int64),experiment_id=m_train.experiment_id.to_numpy(dtype='U8'),center=center,scale=scale)
    np.savez_compressed(args.out/'test_real.npz',x_raw=x_test.astype(np.float32),x_normalized=xte.astype(np.float32),y=m_test.label.to_numpy(np.int64),age=m_test.age.to_numpy(np.int64),experiment_id=m_test.experiment_id.to_numpy(dtype='U8'),center=center,scale=scale)
    m_train.to_csv(args.out/'train_windows.csv',index=False); m_test.to_csv(args.out/'test_windows.csv',index=False)
    summary={'heldout':args.heldout,'train_fraction':args.fraction,'train_experiments':sorted(m_train.experiment_id.unique().tolist()),'train_windows':len(m_train),'train_positive':int(m_train.label.sum()),'test_windows':len(m_test),'test_positive':int(m_test.label.sum()),'window_s':args.window,'stride_s':args.stride,'horizon_s':args.horizon,'temp_cap_c':args.temp_cap,'scaler_fit':'selected real training windows only'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
