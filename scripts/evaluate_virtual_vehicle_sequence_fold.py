#!/usr/bin/env python3
"""Evaluate fixed small CNN/GRU warning models on one frozen TS0330 scarcity fold.

No model selection is performed here. Architectures, optimizer, epochs and two
ensemble seeds are fixed before any sequence-model score is inspected.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallCNN(nn.Module):
    def __init__(self, channels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return self.fc(z).squeeze(-1)


class SmallGRU(nn.Module):
    def __init__(self, channels: int = 2, hidden: int = 24):
        super().__init__()
        self.gru = nn.GRU(input_size=channels, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        seq = x.transpose(1, 2)
        _, h = self.gru(seq)
        return self.fc(h[-1]).squeeze(-1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def channel_scale_from_real(x: np.ndarray):
    mu = x.mean(axis=(0, 2)).astype(np.float32)
    sd = x.std(axis=(0, 2)).astype(np.float32)
    sd = np.maximum(sd, 1e-6)
    return mu, sd


def normalize(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return ((x - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)


def classical_from_same_anchors(x_real: np.ndarray, y_real: np.ndarray, anchors: np.ndarray, seed: int, temp_cap: float = 150.0):
    rng = np.random.default_rng(seed)
    std = np.maximum(np.std(x_real, axis=(0, 2)), 1e-6)
    xs, ys = [], []
    for a in anchors:
        a = int(a)
        base = x_real[a].astype(np.float64)
        accepted = None
        for _ in range(100):
            s = base.copy()
            for c in range(s.shape[0]):
                b0 = s[c, 0]
                amp = rng.uniform(0.96, 1.04)
                s[c] = b0 + amp * (s[c] - b0)
                s[c] += rng.normal(0.0, std[c] * 0.003, size=s.shape[1])
            if np.isfinite(s).all() and np.max(s[0]) <= temp_cap:
                accepted = s.astype(np.float32)
                break
        if accepted is None:
            accepted = base.astype(np.float32)
        xs.append(accepted)
        ys.append(int(y_real[a]))
    return np.stack(xs) if xs else np.empty((0, *x_real.shape[1:]), np.float32), np.asarray(ys, dtype=np.int64)


def train_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, model_name: str, seed: int, pos_weight: float):
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model = SmallCNN(xtr.shape[1]) if model_name == 'cnn' else SmallGRU(xtr.shape[1])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device))
    ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr.astype(np.float32)))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True, generator=gen, drop_last=False)
    model.train()
    for _ in range(100):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb)
            loss = loss_fn(logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(xte), 64):
            logits = model(torch.from_numpy(xte[i:i+64]).to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def metrics(y: np.ndarray, p: np.ndarray):
    pred = (p >= 0.5).astype(int)
    return {
        'auroc': float(roc_auc_score(y, p)),
        'auprc': float(average_precision_score(y, p)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
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
    x_real = tr['x_raw'].astype(np.float32)
    y_real = tr['y'].astype(np.int64)
    x_test = te['x_raw'].astype(np.float32)
    y_test = te['y'].astype(np.int64)
    x_syn = sy['x_raw'].astype(np.float32)
    y_syn = sy['y'].astype(np.int64)
    anchors = sy['anchor_index'].astype(np.int64)

    if len(x_syn) != len(anchors):
        raise RuntimeError('synthetic/anchor length mismatch')
    x_class, y_class = classical_from_same_anchors(x_real, y_real, anchors, args.seed)
    if not np.array_equal(y_class, y_real[anchors]):
        raise RuntimeError('classical labels do not match exact real anchors')
    if not np.array_equal(y_syn, y_real[anchors]):
        raise RuntimeError('RA-CDiff labels do not match retained real anchors')

    mu, sd = channel_scale_from_real(x_real)
    xte_n = normalize(x_test, mu, sd)
    methods = {
        'real_only': (x_real, y_real),
        'classical_anchor_matched': (np.concatenate([x_real, x_class]), np.concatenate([y_real, y_class])),
        'racdiff': (np.concatenate([x_real, x_syn]), np.concatenate([y_real, y_syn])),
    }
    npos = max(int(y_real.sum()), 1)
    nneg = max(int(len(y_real) - y_real.sum()), 1)
    pos_weight = nneg / npos
    summary = json.loads((args.fold_dir / 'summary.json').read_text(encoding='utf-8'))

    rows = []
    pred_rows = []
    for method, (xa, ya) in methods.items():
        xa_n = normalize(xa, mu, sd)
        for arch in ('cnn', 'gru'):
            p1 = train_predict(xa_n, ya, xte_n, arch, args.seed + 1000, pos_weight)
            p2 = train_predict(xa_n, ya, xte_n, arch, args.seed + 2000, pos_weight)
            p = 0.5 * (p1 + p2)
            rec = {
                'heldout': summary['heldout'],
                'train_experiments': '+'.join(summary['train_experiments']),
                'method': method,
                'architecture': arch,
                'n_real_train_windows': int(len(y_real)),
                'n_aug': 0 if method == 'real_only' else int(len(x_syn)),
                'n_test_windows': int(len(y_test)),
                'test_positive': int(y_test.sum()),
                **metrics(y_test, p),
            }
            rows.append(rec)
            for j, (yy, pp) in enumerate(zip(y_test, p)):
                pred_rows.append({'heldout': summary['heldout'], 'train_experiments': '+'.join(summary['train_experiments']), 'method': method, 'architecture': arch, 'index': j, 'label': int(yy), 'probability': float(pp)})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(pred_rows).to_csv(args.out.with_name(args.out.stem + '_predictions.csv'), index=False)
    audit = {
        'heldout': summary['heldout'],
        'train_experiments': summary['train_experiments'],
        'real_train_windows': int(len(y_real)),
        'synthetic_windows': int(len(x_syn)),
        'unique_retained_anchors': int(len(np.unique(anchors))) if len(anchors) else 0,
        'test_windows': int(len(y_test)),
        'optimizer': 'AdamW(lr=1e-3, weight_decay=1e-4)',
        'epochs': 100,
        'ensemble_seeds': [args.seed + 1000, args.seed + 2000],
        'scaler_fit': 'training-real windows only',
        'positive_weight_fit': 'training-real labels only',
    }
    args.out.with_suffix('.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
