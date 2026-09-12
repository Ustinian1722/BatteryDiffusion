#!/usr/bin/env python3
"""Fair RA-CDiff vs classical comparison with matched synthetic class counts.

The first k=2 study matched total augmentation count but the RA-CDiff quality gate
sometimes retained a skewed warning-class composition. This corrective evaluator
keeps the RA-CDiff set unchanged and generates the classical comparator with the
*same number of synthetic samples in each class*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_virtual_vehicle_baseline import feature_matrix

REPS = ('temperature','pressure_absolute','pressure_shape','fusion_absolute','fusion_shape')


def classical_augment_matched(x, y, target_counts, rng, temp_cap=150.0):
    std = np.maximum(np.std(x, axis=(0,2)), 1e-6)
    xs, ys = [], []
    for cls in (0,1):
        need = int(target_counts.get(cls, 0))
        pool = np.flatnonzero(y == cls)
        if need and not len(pool):
            raise RuntimeError(f'classical comparator cannot match class {cls}: no real anchors')
        made = 0; attempts = 0
        while made < need and attempts < max(200, need * 80):
            attempts += 1
            idx = int(rng.choice(pool))
            s = x[idx].astype(np.float64).copy()
            for c in range(s.shape[0]):
                baseline = s[c,0]
                amp = rng.uniform(0.96, 1.04)
                s[c] = baseline + amp * (s[c] - baseline)
                s[c] += rng.normal(0, std[c] * 0.003, size=s.shape[1])
            if not np.isfinite(s).all() or np.max(s[0]) > temp_cap:
                continue
            xs.append(s.astype(np.float32)); ys.append(cls); made += 1
        if made != need:
            raise RuntimeError(f'failed to generate matched classical class {cls}: {made}/{need}')
    if not xs:
        return np.empty((0,*x.shape[1:]), np.float32), np.empty(0, dtype=int)
    order = rng.permutation(len(xs))
    return np.stack(xs)[order], np.asarray(ys, dtype=int)[order]


def fit_eval(xtr, ytr, xte, yte, rep):
    m = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=3000, class_weight='balanced', C=0.5,
                                   solver='liblinear', random_state=2026))])
    m.fit(feature_matrix(xtr, rep), ytr)
    p = m.predict_proba(feature_matrix(xte, rep))[:,1]
    pred = (p >= 0.5).astype(int)
    return {
        'auroc': float(roc_auc_score(yte, p)),
        'auprc': float(average_precision_score(yte, p)),
        'f1': float(f1_score(yte, pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(yte, pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold-dir', type=Path, required=True)
    ap.add_argument('--synthetic', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args(); args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tr = np.load(args.fold_dir/'train_real.npz')
    te = np.load(args.fold_dir/'test_real.npz')
    sy = np.load(args.synthetic)
    xtr, ytr = tr['x_raw'].astype(np.float32), tr['y'].astype(int)
    xte, yte = te['x_raw'].astype(np.float32), te['y'].astype(int)
    xs, ys = sy['x_raw'].astype(np.float32), sy['y'].astype(int)

    target_counts = {0: int((ys==0).sum()), 1: int((ys==1).sum())}
    xc, yc = classical_augment_matched(xtr, ytr, target_counts, rng)
    if len(yc) != len(ys) or int((yc==1).sum()) != int((ys==1).sum()):
        raise RuntimeError('class matching failed')

    summary = json.loads((args.fold_dir/'summary.json').read_text(encoding='utf-8'))
    methods = {
        'real_only': (xtr, ytr),
        'classical_classmatched': (np.concatenate([xtr,xc]), np.concatenate([ytr,yc])),
        'racdiff': (np.concatenate([xtr,xs]), np.concatenate([ytr,ys])),
    }
    rows = []
    for method, (xa, ya) in methods.items():
        for rep in REPS:
            rows.append({
                'heldout': summary['heldout'],
                'train_experiments': '+'.join(summary['train_experiments']),
                'method': method,
                'representation': rep,
                'n_real_train_windows': len(ytr),
                'n_aug': 0 if method == 'real_only' else len(ys),
                'n_aug_positive': 0 if method == 'real_only' else int(ys.sum()),
                'n_aug_negative': 0 if method == 'real_only' else int(len(ys)-ys.sum()),
                'n_test_windows': len(yte),
                'test_positive': int(yte.sum()),
                **fit_eval(xa, ya, xte, yte, rep),
            })
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    audit = {
        'heldout': summary['heldout'],
        'train_experiments': summary['train_experiments'],
        'racdiff_aug_total': int(len(ys)),
        'racdiff_aug_positive': int(ys.sum()),
        'racdiff_aug_negative': int(len(ys)-ys.sum()),
        'classical_aug_total': int(len(yc)),
        'classical_aug_positive': int(yc.sum()),
        'classical_aug_negative': int(len(yc)-yc.sum()),
        'class_composition_exact_match': bool(len(yc)==len(ys) and int(yc.sum())==int(ys.sum())),
        'seed': args.seed,
    }
    args.out.with_suffix('.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(df.to_string(index=False)); print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
