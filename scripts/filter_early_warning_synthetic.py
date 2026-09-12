#!/usr/bin/env python3
"""Apply task-specific physical/label-preserving guards to fold synthetic data.

The generic RA-CDiff quality gate is training-manifold based. For the frozen
early-warning task we additionally require generated raw temperature to remain
within the task definition (max <= 120 degC) and within a conservative physical
range. Labels are inherited from the real anchor used for local perturbation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--synthetic',type=Path,required=True); ap.add_argument('--train-meta',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--temp-cap',type=float,default=120.0); ap.add_argument('--temp-min',type=float,default=-20.0)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    syn=np.load(args.synthetic); meta=pd.read_csv(args.train_meta)
    raw=syn['x_raw'].astype(np.float32); norm=syn['x_normalized'].astype(np.float32); anchor=syn['anchor_index'].astype(int)
    if np.any(anchor<0) or np.any(anchor>=len(meta)): raise RuntimeError('anchor index outside training metadata')
    labels=meta.label.to_numpy(dtype=np.int64)[anchor]; ages=meta.age.to_numpy(dtype=np.int64)[anchor]; exp=meta.experiment_id.astype(str).to_numpy()[anchor]
    max_t=raw[:,0,:].max(axis=1); min_t=raw[:,0,:].min(axis=1)
    finite=np.isfinite(raw).all(axis=(1,2)); task=(max_t<=args.temp_cap)&(min_t>=args.temp_min)&finite
    np.savez_compressed(args.out,x_raw=raw[task],x_normalized=norm[task],y=labels[task],age=ages[task],anchor_experiment_id=exp[task].astype('U8'),anchor_index=anchor[task],noise_start_t=syn['noise_start_t'][task])
    report={'input_accepted_generic':int(len(raw)),'accepted_task_guard':int(task.sum()),'rejected_task_guard':int((~task).sum()),'temp_cap_c':args.temp_cap,'temp_min_c':args.temp_min,'positive':int(labels[task].sum()),'negative':int(task.sum()-labels[task].sum()),'anchor_experiments':sorted(set(exp[task].tolist()))}
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
