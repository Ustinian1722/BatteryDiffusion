#!/usr/bin/env python3
"""Evaluate real-only/classical/RA-CDiff on one frozen learning-curve fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_virtual_vehicle_baseline import feature_matrix
from evaluate_virtual_vehicle_sequence_fold import classical_from_same_anchors

REPS = ('temperature', 'fusion_absolute')


def fit_eval(xtr, ytr, xte, yte, rep):
    m = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=3000, class_weight='balanced', C=0.5,
                                   solver='liblinear', random_state=2026)),
    ])
    m.fit(feature_matrix(xtr, rep), ytr)
    p = m.predict_proba(feature_matrix(xte, rep))[:, 1]
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
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tr = np.load(args.fold_dir / 'train_real.npz')
    te = np.load(args.fold_dir / 'test_real.npz')
    sy = np.load(args.synthetic)
    xreal, yreal = tr['x_raw'].astype(np.float32), tr['y'].astype(np.int64)
    xte, yte = te['x_raw'].astype(np.float32), te['y'].astype(np.int64)
    xsyn, ysyn = sy['x_raw'].astype(np.float32), sy['y'].astype(np.int64)
    anchors = sy['anchor_index'].astype(np.int64)
    if len(xsyn) != len(anchors) or not np.array_equal(ysyn, yreal[anchors]):
        raise RuntimeError('RA-CDiff retained samples no longer match source anchors/labels')
    xclass, yclass = classical_from_same_anchors(xreal, yreal, anchors, args.seed, temp_cap=150.0)
    if not np.array_equal(yclass, yreal[anchors]):
        raise RuntimeError('classical augmentation is not exact-anchor matched')

    summary = json.loads((args.fold_dir / 'summary.json').read_text(encoding='utf-8'))
    methods = {
        'real_only': (xreal, yreal),
        'classical_anchor_matched': (np.concatenate([xreal, xclass]), np.concatenate([yreal, yclass])),
        'racdiff': (np.concatenate([xreal, xsyn]), np.concatenate([yreal, ysyn])),
    }
    rows = []
    for method, (xa, ya) in methods.items():
        for rep in REPS:
            rows.append({
                'heldout': summary['heldout'],
                'k_real_train_experiments': summary['k_real_train_experiments'],
                'train_experiments': '+'.join(summary['train_experiments']),
                'method': method,
                'representation': rep,
                'n_real_train_windows': int(len(yreal)),
                'n_aug': 0 if method == 'real_only' else int(len(xsyn)),
                'n_test_windows': int(len(yte)),
                'test_positive': int(yte.sum()),
                **fit_eval(xa, ya, xte, yte, rep),
            })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    audit = {
        'heldout': summary['heldout'],
        'k': summary['k_real_train_experiments'],
        'train_experiments': summary['train_experiments'],
        'real_train_windows': int(len(yreal)),
        'racdiff_aug': int(len(xsyn)),
        'unique_retained_anchors': int(len(np.unique(anchors))) if len(anchors) else 0,
        'seed': int(args.seed),
    }
    args.out.with_suffix('.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
