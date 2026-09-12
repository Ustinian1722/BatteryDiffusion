#!/usr/bin/env python3
"""Prepare a leakage-safe two-experiment Virtual Vehicle source-training fold.

This is used for frozen cross-domain transfer. It creates source training arrays
only; external experiments are never loaded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, HEADERS
from evaluate_virtual_vehicle_baseline import build_windows


def quantile_scale(x: np.ndarray):
    flat = np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)
    q01 = np.percentile(flat, 1, axis=1)
    q99 = np.percentile(flat, 99, axis=1)
    center = 0.5 * (q01 + q99)
    scale = np.maximum(0.5 * (q99 - q01), 1e-6)
    return center.astype(np.float32), scale.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs=2, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    train = [x.upper() for x in args.train]
    if len(set(train)) != 2:
        raise ValueError('two unique source experiments are required')
    args.out.mkdir(parents=True, exist_ok=True)

    data = {}
    events = {}
    with RemoteZip(record_url(), headers=HEADERS, initial_buffer_size=1024*1024) as rz:
        for exp in train:
            df, event, _ = load_one(rz, exp)
            x, y, _ = build_windows(df, event, window=120, stride=10, horizon=300, cap=150.0)
            data[exp] = (x, y)
            events[exp] = float(event)

    xtr = np.concatenate([data[e][0] for e in train])
    ytr = np.concatenate([data[e][1] for e in train]).astype(np.int64)
    exptr = np.concatenate([np.full(len(data[e][1]), e, dtype='U16') for e in train])
    if len(np.unique(ytr)) < 2:
        raise RuntimeError('source pair unexpectedly lacks both classes under frozen task')

    center, scale = quantile_scale(xtr)
    xnorm = (xtr - center[None, :, None]) / scale[None, :, None]
    np.savez_compressed(
        args.out / 'train_generator.npz',
        x=xnorm.astype(np.float32),
        age=np.zeros(len(ytr), dtype=np.int64),
        stage=ytr,
        experiment_id=exptr,
        center=center,
        scale=scale,
        channel_names=np.array(['temperature_c', 'pressure_bar_abs'], dtype='U32'),
    )
    np.savez_compressed(
        args.out / 'train_real.npz',
        x_raw=xtr.astype(np.float32),
        y=ytr,
        experiment_id=exptr,
    )
    summary = {
        'train_experiments': train,
        'n_train_windows': int(len(ytr)),
        'n_train_positive': int(ytr.sum()),
        'n_train_negative': int(len(ytr) - ytr.sum()),
        'events_s': events,
        'center': center.tolist(),
        'scale': scale.tolist(),
        'task': {'window_s': 120, 'stride_s': 10, 'horizon_s': 300, 'temp_cap_c': 150.0},
        'external_data_used': False,
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
