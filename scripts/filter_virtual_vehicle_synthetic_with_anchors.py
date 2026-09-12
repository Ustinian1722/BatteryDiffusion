#!/usr/bin/env python3
"""Apply frozen TS0330 task/budget guards while preserving real anchor indices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def balanced_cap(indices: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= cap:
        out = indices.copy()
        rng.shuffle(out)
        return out
    chosen = []
    classes = np.unique(y[indices])
    base = cap // len(classes)
    for c in classes:
        pool = indices[y[indices] == c]
        take = min(base, len(pool))
        if take:
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    remain = cap - len(chosen)
    if remain > 0:
        pool = np.setdiff1d(indices, np.asarray(chosen, dtype=int), assume_unique=False)
        if len(pool):
            chosen.extend(rng.choice(pool, size=min(remain, len(pool)), replace=False).tolist())
    out = np.asarray(chosen, dtype=int)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', type=Path, required=True)
    ap.add_argument('--train-real', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--temp-cap', type=float, default=150.0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    s = np.load(args.synthetic)
    tr = np.load(args.train_real)
    x = s['x_raw'].astype(np.float32)
    y = s['stage'].astype(np.int64)
    anchors = s['anchor_index'].astype(np.int64)
    if len(x) != len(y) or len(x) != len(anchors):
        raise RuntimeError('synthetic arrays have inconsistent lengths')
    if len(anchors) and (anchors.min() < 0 or anchors.max() >= len(tr['y'])):
        raise RuntimeError('synthetic anchor index is outside training-real window range')

    valid = np.isfinite(x).all(axis=(1, 2)) & (np.max(x[:, 0, :], axis=1) <= args.temp_cap)
    ids = np.flatnonzero(valid)
    ids = balanced_cap(ids, y, len(tr['y']), rng)
    xo, yo, ao = x[ids], y[ids], anchors[ids]

    np.savez_compressed(args.out, x_raw=xo, y=yo, anchor_index=ao, source_index=ids)
    rec = {
        'input_accepted_quality_gate': int(len(x)),
        'pass_task_guard': int(valid.sum()),
        'retained_budgeted': int(len(ids)),
        'real_train_windows': int(len(tr['y'])),
        'retained_positive': int(yo.sum()),
        'retained_negative': int(len(yo) - yo.sum()),
        'unique_real_anchors': int(len(np.unique(ao))) if len(ao) else 0,
        'temp_cap_c': float(args.temp_cap),
        'guardrail': 'anchor_index points only to the training-real fold; no test experiment contributes to augmentation',
    }
    args.out.with_suffix('.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
