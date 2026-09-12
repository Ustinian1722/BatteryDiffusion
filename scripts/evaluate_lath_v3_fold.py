#!/usr/bin/env python3
"""Evaluate frozen precursor-invariant LATH-Net v3 on one experiment fold.

Protocol/design frozen in reports/method_paper_target_freeze_v3.md before any
v3 predictive score is computed.
"""
from __future__ import annotations

import argparse
import json
import math
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
LAGS = (0, 2, 5, 10, 20, 30)


def causal_diff(x: torch.Tensor) -> torch.Tensor:
    d = torch.zeros_like(x)
    d[..., 1:] = x[..., 1:] - x[..., :-1]
    return d


def precursor_channels(x: torch.Tensor):
    """Create causal internal features from standardized T/P only."""
    t = x[:, 0:1]
    p = x[:, 1:2]
    dt0 = t - t[..., :1]
    dtemp = causal_diff(t)
    prel = p - p[..., :1]
    dp = causal_diff(p)
    rms = torch.sqrt(torch.mean(prel * prel, dim=-1, keepdim=True) + 1e-6)
    pshape = prel / rms
    tfeat = torch.cat([t, dt0, dtemp], dim=1)
    pfeat = torch.cat([prel, dp, pshape], dim=1)
    return tfeat, pfeat


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

    def forward(self, x):
        b = self.stem(x)
        s = self.short(b)
        l = self.long(b)[..., :x.shape[-1]]
        return F.gelu(self.mix(torch.cat([s, l], dim=1)))


def shift_past(z: torch.Tensor, lag: int) -> torch.Tensor:
    if lag == 0:
        return z
    out = torch.zeros_like(z)
    out[:, :, lag:] = z[:, :, :-lag]
    return out


class NormalizedLagAlign(nn.Module):
    """Window-level lag distribution from normalized latent cross-correlation."""
    def __init__(self, h: int):
        super().__init__()
        self.q = nn.Conv1d(h, h, 1, bias=False)
        self.k = nn.Conv1d(h, h, 1, bias=False)
        # 1 + softplus(2) ~= 3.13 initial sharpness; learned from train data only.
        self.raw_sharpness = nn.Parameter(torch.tensor(2.0))

    def forward(self, t: torch.Tensor, p: torch.Tensor):
        q = F.normalize(self.q(t), p=2, dim=1, eps=1e-6)
        k = F.normalize(self.k(p), p=2, dim=1, eps=1e-6)
        shifted = [shift_past(p, lag) for lag in LAGS]
        scores = []
        for lag in LAGS:
            if lag == 0:
                qv, kv = q, k
            else:
                qv, kv = q[:, :, lag:], k[:, :, :-lag]
            # cosine agreement over feature channels, averaged over valid time.
            s = torch.sum(qv * kv, dim=1).mean(dim=1)
            scores.append(s)
        corr = torch.stack(scores, dim=1)
        sharpness = 1.0 + F.softplus(self.raw_sharpness)
        w = torch.softmax(sharpness * corr, dim=1)
        aligned = sum(w[:, i, None, None] * shifted[i] for i in range(len(LAGS)))
        return aligned, w, corr, sharpness


class LATHNetV3(nn.Module):
    def __init__(self, h: int = 24):
        super().__init__()
        self.tenc = MultiScaleEncoder(3, h)
        self.penc = MultiScaleEncoder(3, h)
        self.align = NormalizedLagAlign(h)
        self.rel = nn.Sequential(
            nn.Linear(4 * h, 2 * h), nn.GELU(), nn.Linear(2 * h, 1)
        )
        self.fuse = nn.Conv1d(4 * h, 2 * h, 1)
        self.post = nn.Sequential(
            ResidualTCNBlock(2 * h, 1),
            ResidualTCNBlock(2 * h, 2),
            ResidualTCNBlock(2 * h, 4),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * h, 2 * h), nn.GELU(), nn.Dropout(0.1)
        )
        self.tau = nn.Linear(2 * h, 1)
        self.risk = nn.Linear(2 * h, len(HORIZONS_S))

    def forward(self, x):
        tf, pf = precursor_channels(x)
        t = self.tenc(tf)
        p = self.penc(pf)
        pa, lag_w, corr, sharpness = self.align(t, p)
        pooled_rel = torch.cat([
            t.mean(-1), pa.mean(-1), torch.abs(t - pa).mean(-1), (t * pa).mean(-1)
        ], dim=1)
        reliability = torch.sigmoid(self.rel(pooled_rel)).squeeze(-1)
        pg = reliability[:, None, None] * pa
        z = F.gelu(self.fuse(torch.cat([t, pg, t - pg, t * pg], dim=1)))
        z = self.post(z)
        pooled = torch.cat([z.mean(-1), z.amax(-1)], dim=1)
        h = self.head(pooled)
        tau_n = self.tau(h).squeeze(-1)
        raw = self.risk(h)
        mono_logits = torch.cat([
            raw[:, :1],
            raw[:, :1] + torch.cumsum(F.softplus(raw[:, 1:]), dim=1),
        ], dim=1)
        return tau_n, mono_logits, lag_w, corr, reliability, sharpness


def ranking_loss(pred_n, true_n, expid, margin=0.05):
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


def loss_fn(pred_n, risk_logits, y_n, expid):
    ttv = F.smooth_l1_loss(pred_n, y_n, beta=0.05)
    h = torch.tensor(HORIZONS_S, dtype=y_n.dtype, device=y_n.device) / MAX_TAU_S
    labels = (y_n[:, None] <= h[None, :]).float()
    warn = F.binary_cross_entropy_with_logits(risk_logits, labels)
    rank = ranking_loss(pred_n, y_n, expid)
    return ttv + 0.10 * warn + 0.10 * rank


def predict(model, x, device):
    model.eval()
    ps, risks, ws, corrs, rels, sharps = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            p, r, w, c, rel, sharp = model(torch.from_numpy(x[i:i+128]).to(device))
            ps.append(p.cpu().numpy())
            risks.append(torch.sigmoid(r).cpu().numpy())
            ws.append(w.cpu().numpy())
            corrs.append(c.cpu().numpy())
            rels.append(rel.cpu().numpy())
            sharps.append(float(sharp.cpu()))
    return (
        np.concatenate(ps), np.concatenate(risks), np.concatenate(ws),
        np.concatenate(corrs), np.concatenate(rels), float(np.mean(sharps)),
    )


def train_one(xtr, ytr_n, eidtr, xv, yv_n, seed, max_epochs=240, patience=40):
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model = LATHNetV3().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    ds = TensorDataset(
        torch.from_numpy(xtr), torch.from_numpy(ytr_n.astype(np.float32)),
        torch.from_numpy(eidtr.astype(np.int64)),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, generator=gen)
    best, best_mae, stale, epochs = None, float('inf'), 0, 0
    for ep in range(max_epochs):
        model.train()
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            p, risk, *_ = model(xb)
            loss = loss_fn(p, risk, yb, eb)
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
        raise RuntimeError('no best model captured')
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


def lag_entropy(w):
    ent = -np.sum(w * np.log(np.clip(w, 1e-12, 1.0)), axis=1)
    return ent / np.log(w.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout', required=True, choices=EXPS)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

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
    eidtr = np.concatenate([np.full(len(data[e][1]), EXPS.index(e), dtype=np.int64) for e in trains])
    xv, yv, _ = data[val]
    xte, yte, meta = data[args.heldout]

    mu, sd = channel_stats(xtr)
    xtr = norm_x(xtr, mu, sd)
    xv = norm_x(xv, mu, sd)
    xte = norm_x(xte, mu, sd)
    ytr_n = (ytr / MAX_TAU_S).astype(np.float32)
    yv_n = (yv / MAX_TAU_S).astype(np.float32)

    rows, pred_rows, mech_rows = [], [], []
    for seed in args.seeds:
        model, val_mae, epochs, params = train_one(xtr, ytr_n, eidtr, xv, yv_n, seed)
        pn, risk, w, corr, rel, sharp = predict(model, xte, torch.device('cpu'))
        pred = np.clip(pn * MAX_TAU_S, 0.0, MAX_TAU_S)
        entropy = lag_entropy(w)
        rows.append({
            'heldout': args.heldout, 'validation': val, 'train_experiments': '+'.join(trains),
            'model': 'lath_net_v3', 'seed': seed, 'n_train': len(ytr), 'n_val': len(yv),
            'n_test': len(yte), 'params': params, 'epochs': epochs, 'val_mae_s': val_mae,
            'lag_sharpness': sharp, 'lag_max_weight_mean': float(w.max(axis=1).mean()),
            'lag_entropy_norm_mean': float(entropy.mean()), 'reliability_mean': float(rel.mean()),
            **reg_metrics(yte, pred), **warning_metrics(yte, risk),
        })
        bands = pd.cut(yte, bins=[0, 120, 300, 600], labels=['0-120', '120-300', '300-600'], include_lowest=True)
        for band in bands.categories:
            idx = np.where(np.asarray(bands == band))[0]
            if len(idx) == 0:
                continue
            d = {
                'heldout': args.heldout, 'seed': seed, 'tau_band_s': str(band), 'n': len(idx),
                'reliability_mean': float(rel[idx].mean()),
                'lag_max_weight_mean': float(w[idx].max(axis=1).mean()),
                'lag_entropy_norm_mean': float(entropy[idx].mean()), 'lag_sharpness': sharp,
            }
            for j, lag in enumerate(LAGS):
                d[f'lag_weight_{lag}s'] = float(w[idx, j].mean())
                d[f'corr_{lag}s'] = float(corr[idx, j].mean())
            mech_rows.append(d)
        for j, ((_, mr), yy, pp) in enumerate(zip(meta.iterrows(), yte, pred)):
            pr = {
                'heldout': args.heldout, 'seed': seed, 'index': j, 't_end_s': int(mr.t_end_s),
                'tau_true_s': float(yy), 'tau_pred_s': float(pp), 'abs_error_s': float(abs(pp - yy)),
                'reliability': float(rel[j]), 'lag_max_weight': float(w[j].max()),
                'lag_entropy_norm': float(entropy[j]),
            }
            for k, lag in enumerate(LAGS):
                pr[f'lag_weight_{lag}s'] = float(w[j, k])
            pred_rows.append(pr)

    pd.DataFrame(rows).to_csv(args.out / 'metrics.csv', index=False)
    pd.DataFrame(pred_rows).to_csv(args.out / 'predictions.csv', index=False)
    pd.DataFrame(mech_rows).to_csv(args.out / 'mechanism_diagnostics.csv', index=False)
    audit = {
        'heldout': args.heldout, 'validation': val, 'train_experiments': trains,
        'seeds': args.seeds, 'window_s': 120, 'stride_s': 10, 'max_tau_s': 600,
        'temp_cap_c': 150, 'lags_s': list(LAGS),
        'input': 'same measured T/P; precursor transforms internal',
        'scaler_fit': 'training experiments only',
        'selection': 'early stopping on validation experiment only',
    }
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
