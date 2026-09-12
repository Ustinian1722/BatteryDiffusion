#!/usr/bin/env python3
"""Evaluate frozen LATH-Net v2 on one experiment-level fold.

Protocol is frozen in reports/method_paper_target_freeze_v2.md before v2 scores.
The full model derives both continuous TTV and monotone multi-horizon risk from
one discrete-time event-hazard distribution.
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
from remotezip import RemoteZip
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    brier_score_loss,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_virtual_vehicle_time_to_vent_baseline import build_windows
from evaluate_method_paper_benchmark import MultiScale1D, ResidualTCNBlock, set_seed, channel_stats, norm_x

BIN_WIDTH_S = 15.0
MAX_TAU_S = 600.0
N_BINS = int(MAX_TAU_S / BIN_WIDTH_S)
LAGS = (0, 2, 5, 10, 20, 30)
HORIZONS_S = (60, 120, 180, 300, 450)


def shift_past(z: torch.Tensor, lag: int) -> torch.Tensor:
    if lag == 0:
        return z
    out = torch.zeros_like(z)
    out[:, :, lag:] = z[:, :, :-lag]
    return out


class DynamicLagAlign(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(2 * h, h, 1),
            nn.GELU(),
            nn.Conv1d(h, 1, 1),
        )

    def forward(self, t: torch.Tensor, p: torch.Tensor):
        shifted = [shift_past(p, lag) for lag in LAGS]
        scores = torch.stack(
            [self.score(torch.cat([t, s], dim=1)).squeeze(1) for s in shifted],
            dim=1,
        )  # B,K,L
        weights = torch.softmax(scores / math.sqrt(t.shape[1]), dim=1)
        aligned = sum(weights[:, i:i+1] * shifted[i] for i in range(len(shifted)))
        return aligned, weights


class LATHNetV2(nn.Module):
    def __init__(self, h: int = 24):
        super().__init__()
        self.tenc = MultiScale1D(h)
        self.penc = MultiScale1D(h)
        self.align = DynamicLagAlign(h)
        self.reliability = nn.Conv1d(4 * h, h, 1)
        self.fuse = nn.Conv1d(4 * h, 2 * h, 1)
        self.post = nn.Sequential(
            ResidualTCNBlock(2 * h, 1),
            ResidualTCNBlock(2 * h, 2),
            ResidualTCNBlock(2 * h, 4),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * h, 2 * h),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Last hazard is forced to one; learn the first N_BINS-1 hazards only.
        self.hazard = nn.Linear(2 * h, N_BINS - 1)

    @staticmethod
    def distribution_from_logits(logits: torch.Tensor):
        learned = torch.sigmoid(logits)
        forced = torch.ones((logits.shape[0], 1), dtype=logits.dtype, device=logits.device)
        hazards = torch.cat([learned, forced], dim=1)
        one = torch.ones((logits.shape[0], 1), dtype=logits.dtype, device=logits.device)
        survival_before = torch.cat([one, torch.cumprod(1.0 - hazards[:, :-1], dim=1)], dim=1)
        pmf = survival_before * hazards
        pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mid_s = (torch.arange(N_BINS, device=logits.device, dtype=logits.dtype) + 0.5) * BIN_WIDTH_S
        expected_s = (pmf * mid_s[None, :]).sum(dim=1)
        cdf = torch.cumsum(pmf, dim=1)
        return expected_s, pmf, cdf, hazards

    def forward(self, x: torch.Tensor):
        t = self.tenc(x[:, 0:1])
        p = self.penc(x[:, 1:2])
        pa, lag_w = self.align(t, p)
        joint = torch.cat([t, pa, t - pa, t * pa], dim=1)
        gate = torch.sigmoid(self.reliability(joint))
        pg = gate * pa
        z = torch.nn.functional.gelu(self.fuse(torch.cat([t, pg, t - pg, t * pg], dim=1)))
        z = self.post(z)
        pooled = torch.cat([z.mean(-1), z.amax(-1)], dim=1)
        h = self.head(pooled)
        logits = self.hazard(h)
        expected_s, pmf, cdf, hazards = self.distribution_from_logits(logits)
        return expected_s, pmf, cdf, logits, lag_w, gate


def ranking_loss(pred_s: torch.Tensor, true_s: torch.Tensor, expid: torch.Tensor, margin_s: float = 30.0):
    pi = pred_s[:, None]
    pj = pred_s[None, :]
    ti = true_s[:, None]
    tj = true_s[None, :]
    same = expid[:, None].eq(expid[None, :])
    ordered = (tj - ti) >= 60.0
    mask = same & ordered
    if not torch.any(mask):
        return pred_s.sum() * 0.0
    return torch.relu(margin_s - (pj - pi))[mask].mean() / MAX_TAU_S


def event_bin(true_s: torch.Tensor):
    idx = torch.ceil(true_s / BIN_WIDTH_S).long() - 1
    return torch.clamp(idx, 0, N_BINS - 1)


def total_loss(pred_s, pmf, true_s, expid, use_rank: bool = True):
    pred_n = pred_s / MAX_TAU_S
    true_n = true_s / MAX_TAU_S
    ttv = torch.nn.functional.smooth_l1_loss(pred_n, true_n, beta=0.05)
    idx = event_bin(true_s)
    nll = -torch.log(pmf.gather(1, idx[:, None]).squeeze(1).clamp_min(1e-8)).mean()
    rank = ranking_loss(pred_s, true_s, expid) if use_rank else pred_s.sum() * 0.0
    return ttv + 0.15 * nll + 0.10 * rank, ttv.detach(), nll.detach(), rank.detach()


def predict(model, x: np.ndarray, device: torch.device):
    model.eval()
    pred, cdfs, lags, gates = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            out = model(torch.from_numpy(x[i:i+128]).to(device))
            pred.append(out[0].cpu().numpy())
            cdfs.append(out[2].cpu().numpy())
            lags.append(out[4].mean(dim=2).cpu().numpy())
            gates.append(out[5].mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(pred), np.concatenate(cdfs), np.concatenate(lags), np.concatenate(gates)


def train_one(xtr, ytr, eidtr, xv, yv, seed: int, max_epochs: int = 260, patience: int = 45):
    set_seed(seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model = LATHNetV2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    ds = TensorDataset(
        torch.from_numpy(xtr),
        torch.from_numpy(ytr.astype(np.float32)),
        torch.from_numpy(eidtr.astype(np.int64)),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, generator=gen, drop_last=False)
    best_state = None
    best_val = float('inf')
    stale = 0
    epochs = 0
    for ep in range(max_epochs):
        model.train()
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            pred_s, pmf, *_ = model(xb)
            loss, *_ = total_loss(pred_s, pmf, yb, eb, use_rank=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        pv, _, _, _ = predict(model, xv, device)
        vmae = float(np.mean(np.abs(pv - yv)))
        epochs = ep + 1
        if vmae < best_val - 1e-4:
            best_val = vmae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError('No best state captured')
    model.load_state_dict(best_state)
    return model, best_val, epochs, sum(p.numel() for p in model.parameters())


def regression_metrics(y, p):
    return {
        'mae_s': float(mean_absolute_error(y, p)),
        'rmse_s': float(mean_squared_error(y, p) ** 0.5),
        'medae_s': float(median_absolute_error(y, p)),
        'r2': float(r2_score(y, p)),
        'bias_s': float(np.mean(p - y)),
    }


def horizon_metrics(y, cdf):
    out = {}
    for h in HORIZONS_S:
        n = int(math.ceil(h / BIN_WIDTH_S))
        prob = cdf[:, n - 1]
        lab = (y <= h).astype(int)
        out[f'brier_{h}'] = float(brier_score_loss(lab, prob))
        if len(np.unique(lab)) == 2:
            out[f'auroc_{h}'] = float(roc_auc_score(lab, prob))
            out[f'auprc_{h}'] = float(average_precision_score(lab, prob))
        else:
            out[f'auroc_{h}'] = np.nan
            out[f'auprc_{h}'] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout', required=True, choices=EXPS)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--window', type=int, default=120)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--max-tau', type=float, default=600.0)
    ap.add_argument('--temp-cap', type=float, default=150.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.window != 120 or args.max_tau != 600:
        raise ValueError('v2 protocol is frozen at window=120, max_tau=600')

    data = {}
    with RemoteZip(record_url(), headers=HEADERS, initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df, event, _ = load_one(rz, exp)
            x, y, meta = build_windows(df, event, args.window, args.stride, args.max_tau, args.temp_cap)
            data[exp] = (x, y, meta)

    hi = EXPS.index(args.heldout)
    val = EXPS[(hi + 1) % len(EXPS)]
    trains = [e for e in EXPS if e not in (args.heldout, val)]
    xtr = np.concatenate([data[e][0] for e in trains])
    ytr = np.concatenate([data[e][1] for e in trains]).astype(np.float32)
    eidtr = np.concatenate([np.full(len(data[e][1]), EXPS.index(e), dtype=np.int64) for e in trains])
    xv, yv, _ = data[val]
    xte, yte, meta = data[args.heldout]

    mu, sd = channel_stats(xtr)
    xtr = norm_x(xtr, mu, sd)
    xv = norm_x(xv, mu, sd)
    xte = norm_x(xte, mu, sd)

    rows = []
    pred_rows = []
    diag_rows = []
    for seed in args.seeds:
        model, val_mae, epochs, params = train_one(xtr, ytr, eidtr, xv, yv.astype(np.float32), seed)
        pred, cdf, lag_w, gate = predict(model, xte, torch.device('cpu'))
        pred = np.clip(pred, 0.0, MAX_TAU_S)
        rec = {
            'heldout': args.heldout,
            'validation': val,
            'train_experiments': '+'.join(trains),
            'model': 'lath_net_v2',
            'seed': seed,
            'n_train': len(ytr),
            'n_val': len(yv),
            'n_test': len(yte),
            'params': params,
            'epochs': epochs,
            'val_mae_s': val_mae,
            **regression_metrics(yte, pred),
            **horizon_metrics(yte, cdf),
        }
        rows.append(rec)
        for j, ((_, mr), yy, pp) in enumerate(zip(meta.iterrows(), yte, pred)):
            pr = {
                'heldout': args.heldout,
                'seed': seed,
                'index': j,
                't_end_s': int(mr.t_end_s),
                'tau_true_s': float(yy),
                'tau_pred_s': float(pp),
                'abs_error_s': float(abs(pp - yy)),
                'gate_mean': float(gate[j]),
            }
            for li, lag in enumerate(LAGS):
                pr[f'lag_weight_{lag}s'] = float(lag_w[j, li])
            for h in HORIZONS_S:
                n = int(math.ceil(h / BIN_WIDTH_S))
                pr[f'risk_{h}s'] = float(cdf[j, n - 1])
            pred_rows.append(pr)
        # Summarize mechanism diagnostics by broad true-TTV band, without using them for model selection.
        bands = pd.cut(yte, bins=[0, 120, 300, 600], labels=['0-120', '120-300', '300-600'], include_lowest=True)
        for band in bands.categories:
            idx = np.where(np.asarray(bands == band))[0]
            if len(idx) == 0:
                continue
            d = {'heldout': args.heldout, 'seed': seed, 'tau_band_s': str(band), 'n': len(idx), 'gate_mean': float(gate[idx].mean())}
            for li, lag in enumerate(LAGS):
                d[f'lag_weight_{lag}s'] = float(lag_w[idx, li].mean())
            diag_rows.append(d)

    pd.DataFrame(rows).to_csv(args.out / 'metrics.csv', index=False)
    pd.DataFrame(pred_rows).to_csv(args.out / 'predictions.csv', index=False)
    pd.DataFrame(diag_rows).to_csv(args.out / 'mechanism_diagnostics.csv', index=False)
    audit = {
        'heldout': args.heldout,
        'validation': val,
        'train_experiments': trains,
        'seeds': args.seeds,
        'window_s': args.window,
        'stride_s': args.stride,
        'max_tau_s': args.max_tau,
        'temp_cap_c': args.temp_cap,
        'hazard_bins': N_BINS,
        'bin_width_s': BIN_WIDTH_S,
        'lags_s': list(LAGS),
        'scaler_fit': 'training experiments only',
        'selection': 'early stopping on one validation experiment only',
    }
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
