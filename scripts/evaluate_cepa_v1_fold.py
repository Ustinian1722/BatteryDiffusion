#!/usr/bin/env python3
"""Evaluate frozen CEPA-Net v1 on one held-out destructive experiment.

Protocol and loss coefficients are frozen in reports/cepa_v1_protocol_freeze.md
before predictive scores are inspected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from remotezip import RemoteZip
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_virtual_vehicle_time_to_vent_baseline import build_windows
from evaluate_method_paper_benchmark import ResidualTCNBlock, set_seed, channel_stats, norm_x

MAX_TAU_S = 600.0
HORIZONS_S = (60, 120, 180, 300, 450)


def causal_diff(x: torch.Tensor) -> torch.Tensor:
    d = torch.zeros_like(x)
    d[..., 1:] = x[..., 1:] - x[..., :-1]
    return d


def dynamic_channels(x: torch.Tensor) -> torch.Tensor:
    t = x[:, 0:1]
    p = x[:, 1:2]
    return torch.cat([
        t - t[..., :1],
        causal_diff(t),
        p - p[..., :1],
        causal_diff(p),
    ], dim=1)


class MultiScaleEncoder(nn.Module):
    def __init__(self, c: int, h: int = 24):
        super().__init__()
        self.stem = nn.Conv1d(c, h, 1)
        self.short = nn.Sequential(
            nn.Conv1d(h, h, 3, padding=1), nn.GELU(),
            nn.Conv1d(h, h, 3, padding=1), nn.GELU(),
        )
        self.long = nn.Sequential(
            nn.Conv1d(h, h, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(h, h, 3, padding=8, dilation=8), nn.GELU(),
        )
        self.mix = nn.Conv1d(2 * h, h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.stem(x)
        s = self.short(b)
        l = self.long(b)[..., :x.shape[-1]]
        return F.gelu(self.mix(torch.cat([s, l], dim=1)))


class CEPANetV1(nn.Module):
    def __init__(self, h: int = 24, progress_dim: int = 16):
        super().__init__()
        self.state_enc = MultiScaleEncoder(2, h)
        self.dynamic_enc = MultiScaleEncoder(4, h)
        self.gate = nn.Conv1d(2 * h, h, 1)
        self.fuse = nn.Conv1d(4 * h, 2 * h, 1)
        self.post = nn.Sequential(
            ResidualTCNBlock(2 * h, 1),
            ResidualTCNBlock(2 * h, 2),
            ResidualTCNBlock(2 * h, 4),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * h, 2 * h), nn.GELU(), nn.Dropout(0.1)
        )
        self.progress = nn.Linear(2 * h, progress_dim)
        self.tau = nn.Linear(2 * h, 1)
        self.risk = nn.Linear(2 * h, len(HORIZONS_S))

    def forward(self, x: torch.Tensor):
        s = self.state_enc(x)
        d = self.dynamic_enc(dynamic_channels(x))
        g = torch.sigmoid(self.gate(torch.cat([s, d], dim=1)))
        gd = g * d
        z = F.gelu(self.fuse(torch.cat([s, gd, s - gd, s * gd], dim=1)))
        z = self.post(z)
        pooled = torch.cat([z.mean(-1), z.amax(-1)], dim=1)
        h = self.head(pooled)
        progress = F.normalize(self.progress(h), p=2, dim=1, eps=1e-6)
        tau_n = self.tau(h).squeeze(-1)
        raw = self.risk(h)
        mono_logits = torch.cat([
            raw[:, :1],
            raw[:, :1] + torch.cumsum(F.softplus(raw[:, 1:]), dim=1),
        ], dim=1)
        return tau_n, mono_logits, progress, g.mean(dim=(1, 2))


def ranking_loss(pred_n: torch.Tensor, true_n: torch.Tensor, expid: torch.Tensor, margin: float = 0.05):
    pi = pred_n[:, None]
    pj = pred_n[None, :]
    ti = true_n[:, None]
    tj = true_n[None, :]
    same = expid[:, None].eq(expid[None, :])
    ordered = (tj - ti) > 0.10  # >60 s
    mask = same & ordered
    if not torch.any(mask):
        return pred_n.sum() * 0.0
    return torch.relu(margin - (pj - pi))[mask].mean()


def cepa_alignment_loss(progress: torch.Tensor, y_n: torch.Tensor, expid: torch.Tensor):
    cos = torch.clamp(progress @ progress.T, -1.0, 1.0)
    dz = torch.sqrt(torch.clamp(2.0 - 2.0 * cos, min=1e-8)) / 2.0
    dy = torch.abs(y_n[:, None] - y_n[None, :])
    target = torch.clamp(dy / 0.50, 0.0, 1.0)
    weight = 1.0 + 2.0 * torch.exp(-dy / 0.10)
    cross = ~expid[:, None].eq(expid[None, :])
    upper = torch.triu(torch.ones_like(cross, dtype=torch.bool), diagonal=1)
    mask = cross & upper
    if not torch.any(mask):
        return progress.sum() * 0.0
    return (weight * (dz - target).pow(2))[mask].mean()


def loss_fn(pred_n, risk_logits, progress, y_n, expid):
    ttv = F.smooth_l1_loss(pred_n, y_n, beta=0.05)
    align = cepa_alignment_loss(progress, y_n, expid)
    rank = ranking_loss(pred_n, y_n, expid)
    h = torch.tensor(HORIZONS_S, dtype=y_n.dtype, device=y_n.device) / MAX_TAU_S
    labels = (y_n[:, None] <= h[None, :]).float()
    warn = F.binary_cross_entropy_with_logits(risk_logits, labels)
    return ttv + 0.15 * align + 0.05 * rank + 0.05 * warn


def predict(model, x: np.ndarray, device: torch.device):
    model.eval()
    ps, risks, zs, gates = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            p, r, z, g = model(torch.from_numpy(x[i:i+128]).to(device))
            ps.append(p.cpu().numpy())
            risks.append(torch.sigmoid(r).cpu().numpy())
            zs.append(z.cpu().numpy())
            gates.append(g.cpu().numpy())
    return np.concatenate(ps), np.concatenate(risks), np.concatenate(zs), np.concatenate(gates)


def train_one(xtr, ytr_n, eidtr, xv, yv_n, seed, max_epochs=260, patience=45):
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model = CEPANetV1().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    ds = TensorDataset(
        torch.from_numpy(xtr),
        torch.from_numpy(ytr_n.astype(np.float32)),
        torch.from_numpy(eidtr.astype(np.int64)),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, generator=gen)
    best, best_mae, stale, epochs = None, float('inf'), 0, 0
    for ep in range(max_epochs):
        model.train()
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            p, r, z, _ = model(xb)
            loss = loss_fn(p, r, z, yb, eb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        pv, *_ = predict(model, xv, device)
        val_mae = float(np.mean(np.abs(pv - yv_n)))
        epochs = ep + 1
        if val_mae < best_mae - 1e-5:
            best_mae = val_mae
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best is None:
        raise RuntimeError('no best checkpoint captured')
    model.load_state_dict(best)
    return model, best_mae * MAX_TAU_S, epochs, sum(p.numel() for p in model.parameters())


def reg_metrics(y, p):
    return {
        'mae_s': float(mean_absolute_error(y, p)),
        'rmse_s': float(mean_squared_error(y, p) ** 0.5),
        'medae_s': float(median_absolute_error(y, p)),
        'r2': float(r2_score(y, p)),
        'bias_s': float(np.mean(p - y)),
    }


def warning_metrics(y, risk):
    out = {}
    for j, h in enumerate(HORIZONS_S):
        prob = np.clip(risk[:, j].astype(float), 0.0, 1.0)
        lab = (y <= h).astype(int)
        out[f'brier_{h}'] = float(brier_score_loss(lab, prob))
        if len(np.unique(lab)) == 2:
            out[f'auroc_{h}'] = float(roc_auc_score(lab, prob))
            out[f'auprc_{h}'] = float(average_precision_score(lab, prob))
        else:
            out[f'auroc_{h}'] = np.nan
            out[f'auprc_{h}'] = np.nan
    return out


def embedding_knn_diagnostics(ztr, ytr, zte, yte, k=5):
    sim = np.clip(zte @ ztr.T, -1.0, 1.0)
    order = np.argsort(-sim, axis=1)
    topk = order[:, :min(k, len(ztr))]
    pred = ytr[topk].mean(axis=1)
    top1 = order[:, 0]
    return {
        'progress_knn5_mae_s': float(np.mean(np.abs(pred - yte))),
        'progress_nn_tau_gap_s': float(np.mean(np.abs(ytr[top1] - yte))),
        'progress_nn_cosine': float(np.mean(sim[np.arange(len(zte)), top1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout', required=True, choices=EXPS)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = {}
    with RemoteZip(record_url(), headers=HEADERS, initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df, event, _ = load_one(rz, exp)
            x, y, meta = build_windows(df, event, 120, 10, 600, 150.0)
            data[exp] = (x, y, meta)

    hi = EXPS.index(args.heldout)
    val = EXPS[(hi + 1) % len(EXPS)]
    trains = [e for e in EXPS if e not in (args.heldout, val)]
    xtr = np.concatenate([data[e][0] for e in trains])
    ytr = np.concatenate([data[e][1] for e in trains])
    eidtr = np.concatenate([
        np.full(len(data[e][1]), EXPS.index(e), dtype=np.int64) for e in trains
    ])
    xv, yv, _ = data[val]
    xte, yte, meta = data[args.heldout]

    mu, sd = channel_stats(xtr)
    xtr = norm_x(xtr, mu, sd)
    xv = norm_x(xv, mu, sd)
    xte = norm_x(xte, mu, sd)
    ytr_n = (ytr / MAX_TAU_S).astype(np.float32)
    yv_n = (yv / MAX_TAU_S).astype(np.float32)

    rows, pred_rows = [], []
    for seed in args.seeds:
        model, val_mae, epochs, params = train_one(xtr, ytr_n, eidtr, xv, yv_n, seed)
        pn, risk, zte, gate = predict(model, xte, torch.device('cpu'))
        _, _, ztr, _ = predict(model, xtr, torch.device('cpu'))
        pred = np.clip(pn * MAX_TAU_S, 0.0, MAX_TAU_S)
        diag = embedding_knn_diagnostics(ztr, ytr, zte, yte, k=5)
        rows.append({
            'heldout': args.heldout,
            'validation': val,
            'train_experiments': '+'.join(trains),
            'model': 'cepa_net_v1',
            'seed': seed,
            'n_train': len(ytr),
            'n_val': len(yv),
            'n_test': len(yte),
            'params': params,
            'epochs': epochs,
            'val_mae_s': val_mae,
            'dynamic_gate_mean': float(gate.mean()),
            **diag,
            **reg_metrics(yte, pred),
            **warning_metrics(yte, risk),
        })
        for i in range(len(yte)):
            rec = {
                'heldout': args.heldout,
                'seed': seed,
                'tau_true_s': float(yte[i]),
                'tau_pred_s': float(pred[i]),
                'dynamic_gate': float(gate[i]),
            }
            if isinstance(meta, pd.DataFrame):
                for c in meta.columns:
                    v = meta.iloc[i][c]
                    rec[c] = v.item() if hasattr(v, 'item') else v
            pred_rows.append(rec)

    pd.DataFrame(rows).to_csv(args.out / 'metrics.csv', index=False)
    pd.DataFrame(pred_rows).to_csv(args.out / 'predictions.csv', index=False)
    audit = {
        'heldout': args.heldout,
        'validation': val,
        'train_experiments': trains,
        'seeds': args.seeds,
        'window_s': 120,
        'stride_s': 10,
        'max_tau_s': 600,
        'temp_cap_c': 150.0,
        'scaler_fit': 'training experiments only',
        'selection': 'validation experiment MAE only',
        'augmentation': 'none',
        'loss': 'SmoothL1 + 0.15 CEPA + 0.05 rank + 0.05 warning',
    }
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
