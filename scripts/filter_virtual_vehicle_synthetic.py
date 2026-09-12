#!/usr/bin/env python3
"""Apply frozen TS0330 task/budget guards to accepted RA-CDiff windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def balanced_cap(indices: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator)->np.ndarray:
    if len(indices)<=cap: return indices
    chosen=[]; classes=np.unique(y[indices]); per=cap//len(classes)
    for c in classes:
        pool=indices[y[indices]==c]; take=min(per,len(pool));
        if take: chosen.extend(rng.choice(pool,size=take,replace=False).tolist())
    remain=cap-len(chosen)
    if remain>0:
        pool=np.setdiff1d(indices,np.asarray(chosen,dtype=int),assume_unique=False)
        if len(pool): chosen.extend(rng.choice(pool,size=min(remain,len(pool)),replace=False).tolist())
    out=np.asarray(chosen,dtype=int); rng.shuffle(out); return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--synthetic',type=Path,required=True); ap.add_argument('--train-real',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--temp-cap',type=float,default=150.0)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(args.seed)
    s=np.load(args.synthetic); tr=np.load(args.train_real)
    x=s['x_raw'].astype(np.float32); y=s['stage'].astype(np.int64)  # frozen implementation reuse: stage == binary warning class
    valid=np.isfinite(x).all(axis=(1,2)) & (np.max(x[:,0,:],axis=1)<=args.temp_cap)
    ids=np.flatnonzero(valid); ids=balanced_cap(ids,y,len(tr['y']),rng)
    xo=x[ids]; yo=y[ids]
    np.savez_compressed(args.out,x_raw=xo,y=yo,source_index=ids)
    rec={'input_accepted_quality_gate':int(len(x)),'pass_task_guard':int(valid.sum()),'retained_budgeted':int(len(ids)),'real_train_windows':int(len(tr['y'])),'retained_positive':int(yo.sum()),'retained_negative':int(len(yo)-yo.sum()),'temp_cap_c':args.temp_cap}
    args.out.with_suffix('.json').write_text(json.dumps(rec,indent=2),encoding='utf-8'); print(json.dumps(rec,indent=2))

if __name__=='__main__': main()
