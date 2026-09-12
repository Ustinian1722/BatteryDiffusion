#!/usr/bin/env python3
"""Evaluate one frozen two-experiment source pair on two untouched Zenodo TR tests.

The external experiments are used only for final scoring. No external sample
contributes to source normalization, augmentation, representation selection or
classifier fitting.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_external_zenodo_13981390_events import FILES, PUBLISHED, fetch, align_nmc111, align_nmc811
from evaluate_virtual_vehicle_baseline import build_windows, feature_matrix
from evaluate_virtual_vehicle_sequence_fold import classical_from_same_anchors

REPS = ('temperature', 'pressure_absolute', 'pressure_shape', 'fusion_absolute', 'fusion_shape')


def external_windows(df: pd.DataFrame, event: float):
    # Reuse the frozen Virtual Vehicle task semantics exactly.
    x, y, meta = build_windows(df.rename(columns={'pressure': 'pressure_bar_abs'}), event,
                               window=120, stride=10, horizon=300, cap=150.0)
    return x.astype(np.float32), y.astype(np.int64), meta


def fit_model(xtr: np.ndarray, ytr: np.ndarray, rep: str):
    model = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=3000, class_weight='balanced', C=0.5,
                                   solver='liblinear', random_state=2026)),
    ])
    model.fit(feature_matrix(xtr, rep), ytr)
    return model


def score(model, x: np.ndarray, y: np.ndarray, rep: str):
    p = model.predict_proba(feature_matrix(x, rep))[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        'auroc': float(roc_auc_score(y, p)),
        'auprc': float(average_precision_score(y, p)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
    }, p


def load_external():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for label, url in FILES.items():
            path = td / f'{label}.mat'
            fetch(path, url)
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
            df = align_nmc111(mat) if label == 'NMC111' else align_nmc811(mat)
            x, y, meta = external_windows(df, float(PUBLISHED[label]['vent_time_s']))
            if len(np.unique(y)) < 2:
                raise RuntimeError(f'{label}: frozen external task lacks both classes')
            out[label] = (x, y, meta)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold-dir', type=Path, required=True)
    ap.add_argument('--synthetic', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tr = np.load(args.fold_dir / 'train_real.npz')
    sy = np.load(args.synthetic)
    xreal = tr['x_raw'].astype(np.float32)
    yreal = tr['y'].astype(np.int64)
    xsyn = sy['x_raw'].astype(np.float32)
    ysyn = sy['y'].astype(np.int64)
    anchors = sy['anchor_index'].astype(np.int64)
    if len(xsyn) != len(anchors):
        raise RuntimeError('synthetic/anchor length mismatch')
    if len(anchors) and (anchors.min() < 0 or anchors.max() >= len(yreal)):
        raise RuntimeError('retained synthetic anchor outside source training fold')
    if not np.array_equal(ysyn, yreal[anchors]):
        raise RuntimeError('RA-CDiff retained labels no longer match their source anchors')

    xclass, yclass = classical_from_same_anchors(xreal, yreal, anchors, args.seed, temp_cap=150.0)
    if not np.array_equal(yclass, yreal[anchors]):
        raise RuntimeError('classical labels do not match exact RA-CDiff anchors')

    methods = {
        'real_only': (xreal, yreal),
        'classical_anchor_matched': (np.concatenate([xreal, xclass]), np.concatenate([yreal, yclass])),
        'racdiff': (np.concatenate([xreal, xsyn]), np.concatenate([yreal, ysyn])),
    }
    ext = load_external()
    summary = json.loads((args.fold_dir / 'summary.json').read_text(encoding='utf-8'))

    rows = []
    pred_rows = []
    for method, (xa, ya) in methods.items():
        for rep in REPS:
            model = fit_model(xa, ya, rep)
            for external, (xte, yte, meta) in ext.items():
                met, p = score(model, xte, yte, rep)
                rows.append({
                    'source_pair': '+'.join(summary['train_experiments']),
                    'external': external,
                    'method': method,
                    'representation': rep,
                    'n_source_real_windows': int(len(yreal)),
                    'n_aug': 0 if method == 'real_only' else int(len(xsyn)),
                    'n_external_windows': int(len(yte)),
                    'external_positive': int(yte.sum()),
                    **met,
                })
                for (_, mr), yy, pp in zip(meta.iterrows(), yte, p):
                    pred_rows.append({
                        'source_pair': '+'.join(summary['train_experiments']),
                        'external': external,
                        'method': method,
                        'representation': rep,
                        't_end_s': float(mr.t_end_s),
                        'tau_to_event_s': float(mr.tau_to_event_s),
                        'label': int(yy),
                        'probability': float(pp),
                    })

    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(pred_rows).to_csv(args.out.with_name(args.out.stem + '_predictions.csv'), index=False)
    audit = {
        'source_experiments': summary['train_experiments'],
        'source_real_windows': int(len(yreal)),
        'source_positive': int(yreal.sum()),
        'racdiff_windows': int(len(xsyn)),
        'unique_retained_source_anchors': int(len(np.unique(anchors))) if len(anchors) else 0,
        'external_labels': {k: float(PUBLISHED[k]['vent_time_s']) for k in ext},
        'external_used_for_fit': False,
        'primary_cross_domain_representation': 'fusion_shape',
        'seed': int(args.seed),
    }
    args.out.with_suffix('.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
