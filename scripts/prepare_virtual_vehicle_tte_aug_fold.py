#!/usr/bin/env python3
"""Prepare one leakage-safe k=2 TS0330 time-to-event augmentation fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, HEADERS
from evaluate_virtual_vehicle_time_to_vent_baseline import build_windows


def quantile_scale(x: np.ndarray):
    flat = np.transpose(x, (1,0,2)).reshape(x.shape[1], -1)
    q01 = np.percentile(flat, 1, axis=1)
    q99 = np.percentile(flat, 99, axis=1)
    center = 0.5 * (q01 + q99)
    scale = np.maximum(0.5 * (q99 - q01), 1e-6)
    return center.astype(np.float32), scale.astype(np.float32)


def tertile_stage(tau: np.ndarray):
    q1, q2 = np.quantile(tau, [1/3, 2/3])
    stage = np.digitize(tau, [q1, q2], right=True).astype(np.int64)
    return stage, float(q1), float(q2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout', required=True)
    ap.add_argument('--train', nargs=2, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    held = args.heldout.upper(); train = [x.upper() for x in args.train]
    args.out.mkdir(parents=True, exist_ok=True)
    if held in train or len(set(train)) != 2:
        raise ValueError('heldout must differ from two unique train experiments')

    needed = train + [held]; data = {}; events = {}; audits = []
    with RemoteZip(record_url(), headers=HEADERS, initial_buffer_size=1024*1024) as rz:
        for exp in needed:
            df, event, audit = load_one(rz, exp)
            x, tau, meta = build_windows(df, event, window=120, stride=10, max_tau=600, cap=150.0)
            data[exp] = (x, tau, meta); events[exp] = event; audits.append(audit)

    xtr = np.concatenate([data[e][0] for e in train])
    ttr = np.concatenate([data[e][1] for e in train]).astype(np.float32)
    exptr = np.concatenate([np.full(len(data[e][1]), e, dtype='U16') for e in train])
    xte, tte, mte = data[held]
    if len(ttr) < 12 or len(tte) < 12:
        raise RuntimeError('unexpectedly weak support for frozen TTE fold')

    stage, q1, q2 = tertile_stage(ttr)
    if len(np.unique(stage)) != 3:
        raise RuntimeError('training-fold tertile conditioning failed')

    center, scale = quantile_scale(xtr)
    xnorm = (xtr - center[None,:,None]) / scale[None,:,None]
    np.savez_compressed(
        args.out/'train_generator.npz',
        x=xnorm.astype(np.float32), age=np.zeros(len(ttr), dtype=np.int64), stage=stage,
        tau=ttr, experiment_id=exptr, center=center, scale=scale,
        channel_names=np.array(['temperature_c','pressure_bar_abs'], dtype='U32'))
    np.savez_compressed(args.out/'train_real.npz', x_raw=xtr.astype(np.float32), tau=ttr,
                        stage=stage, experiment_id=exptr)
    np.savez_compressed(args.out/'test_real.npz', x_raw=xte.astype(np.float32), tau=tte.astype(np.float32),
                        experiment_id=np.full(len(tte), held, dtype='U16'))
    mte.to_csv(args.out/'test_meta.csv', index=False)
    pd.DataFrame(audits).to_csv(args.out/'source_signal_audit.csv', index=False)
    summary = {
        'heldout': held,
        'train_experiments': train,
        'n_train_windows': int(len(ttr)),
        'n_test_windows': int(len(tte)),
        'tau_train_min_s': float(ttr.min()),
        'tau_train_max_s': float(ttr.max()),
        'tau_test_min_s': float(tte.min()),
        'tau_test_max_s': float(tte.max()),
        'tau_tertile_edges_s': [q1, q2],
        'tau_tertile_counts': {str(i): int((stage==i).sum()) for i in range(3)},
        'center': center.tolist(), 'scale': scale.tolist(), 'events_s': events,
        'task': {'window_s':120, 'stride_s':10, 'max_tau_s':600, 'temp_cap_c':150},
    }
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
