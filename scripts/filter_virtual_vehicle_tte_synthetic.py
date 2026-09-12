#!/usr/bin/env python3
"""Apply frozen TTE task/budget guards and inherit anchor tau labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def balanced_stage_cap(ids, stage, cap, rng):
    if len(ids) <= cap:
        out = ids.copy(); rng.shuffle(out); return out
    chosen = []
    for s in (0,1,2):
        pool = ids[stage[ids] == s]
        target = cap // 3
        take = min(target, len(pool))
        if take:
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    remain = cap - len(chosen)
    if remain > 0:
        pool = np.setdiff1d(ids, np.asarray(chosen, dtype=int), assume_unique=False)
        if len(pool):
            chosen.extend(rng.choice(pool, size=min(remain, len(pool)), replace=False).tolist())
    out = np.asarray(chosen, dtype=int); rng.shuffle(out); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', type=Path, required=True)
    ap.add_argument('--train-real', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--temp-cap', type=float, default=150.0)
    args = ap.parse_args(); args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    s = np.load(args.synthetic); tr = np.load(args.train_real)
    x = s['x_raw'].astype(np.float32)
    anchor = s['anchor_index'].astype(np.int64)
    stage = s['stage'].astype(np.int64)
    real_tau = tr['tau'].astype(np.float32)
    if np.any(anchor < 0) or np.any(anchor >= len(real_tau)):
        raise RuntimeError('synthetic anchor index outside training set')
    tau = real_tau[anchor]
    valid = (np.isfinite(x).all(axis=(1,2)) & np.isfinite(tau) &
             (tau > 0) & (tau <= 600.0) & (np.max(x[:,0,:], axis=1) <= args.temp_cap))
    ids = np.flatnonzero(valid)
    ids = balanced_stage_cap(ids, stage, len(real_tau), rng)

    xo, ao, so, to = x[ids], anchor[ids], stage[ids], tau[ids]
    np.savez_compressed(args.out, x_raw=xo, anchor_index=ao, stage=so, tau=to, source_index=ids)
    rec = {
        'input_accepted_quality_gate': int(len(x)),
        'pass_task_guard': int(valid.sum()),
        'retained_budgeted': int(len(ids)),
        'real_train_windows': int(len(real_tau)),
        'retained_stage_counts': {str(i): int((so==i).sum()) for i in range(3)},
        'tau_min_s': float(to.min()) if len(to) else None,
        'tau_max_s': float(to.max()) if len(to) else None,
        'tau_median_s': float(np.median(to)) if len(to) else None,
        'temp_cap_c': args.temp_cap,
        'label_rule': 'synthetic tau equals unchanged real-anchor tau; no synthetic event time is invented',
    }
    args.out.with_suffix('.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
