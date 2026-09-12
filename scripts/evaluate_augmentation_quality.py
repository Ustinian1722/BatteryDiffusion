#!/usr/bin/env python3
"""Evaluate accepted synthetic windows against the real training manifold.

This is a descriptive augmentation audit. It does not convert windows into
independent samples and it is not a downstream predictive validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def features(x: np.ndarray) -> np.ndarray:
    out = []
    for sample in x:
        row = []
        for c in range(sample.shape[0]):
            s = sample[c]
            d = np.diff(s)
            ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if len(s) > 2 and np.std(s[:-1]) > 0 and np.std(s[1:]) > 0 else 0.0
            row.extend([
                float(np.mean(s)), float(np.std(s)), float(np.min(s)), float(np.max(s)),
                float(np.mean(np.abs(d))), float(np.max(np.abs(d))) if len(d) else 0.0, ac1,
            ])
        cross = float(np.corrcoef(sample[0], sample[1])[0, 1]) if sample.shape[0] >= 2 and np.std(sample[0]) > 0 and np.std(sample[1]) > 0 else 0.0
        row.append(cross)
        out.append(row)
    return np.asarray(out, dtype=np.float64)


def downsample_flat(x: np.ndarray, points: int = 32) -> np.ndarray:
    idx = np.linspace(0, x.shape[-1] - 1, points).round().astype(int)
    return x[:, :, idx].reshape(len(x), -1).astype(np.float64)


def nn_distance(a: np.ndarray, b: np.ndarray, self_compare: bool = False) -> np.ndarray:
    vals = np.full(len(a), np.inf, dtype=np.float64)
    for i in range(0, len(a), 128):
        aa = a[i:i + 128]
        d = np.sqrt(((aa[:, None, :] - b[None, :, :]) ** 2).mean(axis=2))
        if self_compare:
            for local, global_i in enumerate(range(i, min(i + len(aa), len(a)))):
                if global_i < d.shape[1]:
                    d[local, global_i] = np.inf
        vals[i:i + len(aa)] = d.min(axis=1)
    return vals


def pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = np.sum(x * x, axis=1)[:, None]
    yy = np.sum(y * y, axis=1)[None, :]
    return np.maximum(xx + yy - 2 * x @ y.T, 0.0)


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float('nan'), float('nan')
    # Median heuristic from the real group, excluding diagonal zeros.
    dxx = pairwise_sq_dist(x, x)
    positive = dxx[dxx > 0]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    sigma2 = max(sigma2, 1e-8)
    kxx = np.exp(-dxx / (2 * sigma2))
    kyy = np.exp(-pairwise_sq_dist(y, y) / (2 * sigma2))
    kxy = np.exp(-pairwise_sq_dist(x, y) / (2 * sigma2))
    # Biased MMD is stable for very small groups and used only descriptively.
    mmd2 = float(kxx.mean() + kyy.mean() - 2 * kxy.mean())
    return mmd2, float(np.sqrt(sigma2))


def group_metrics(real_x: np.ndarray, syn_x: np.ndarray) -> dict:
    if len(real_x) == 0 or len(syn_x) == 0:
        return {
            'n_real': int(len(real_x)), 'n_synthetic': int(len(syn_x)),
            'feature_smd_mean': np.nan, 'feature_smd_max': np.nan,
            'feature_range_coverage': np.nan, 'real_real_nn_median': np.nan,
            'synthetic_real_nn_median': np.nan, 'nn_ratio': np.nan,
            'mmd2': np.nan, 'mmd_sigma': np.nan,
        }
    fr, fs = features(real_x), features(syn_x)
    mu = fr.mean(axis=0)
    sd = np.maximum(fr.std(axis=0), 0.05)
    smd = np.abs((fs.mean(axis=0) - mu) / sd)
    lo, hi = fr.min(axis=0), fr.max(axis=0)
    coverage = ((fs >= lo[None, :]) & (fs <= hi[None, :])).mean()
    rr = nn_distance(downsample_flat(real_x), downsample_flat(real_x), True)
    sr = nn_distance(downsample_flat(syn_x), downsample_flat(real_x), False)
    rr = rr[np.isfinite(rr)]
    rr_med = float(np.median(rr)) if len(rr) else np.nan
    sr_med = float(np.median(sr)) if len(sr) else np.nan
    mmd2, sigma = rbf_mmd2(downsample_flat(real_x), downsample_flat(syn_x))
    return {
        'n_real': int(len(real_x)), 'n_synthetic': int(len(syn_x)),
        'feature_smd_mean': float(np.mean(smd)), 'feature_smd_max': float(np.max(smd)),
        'feature_range_coverage': float(coverage),
        'real_real_nn_median': rr_med, 'synthetic_real_nn_median': sr_med,
        'nn_ratio': float(sr_med / rr_med) if rr_med and np.isfinite(rr_med) else np.nan,
        'mmd2': mmd2, 'mmd_sigma': sigma,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', type=Path, required=True)
    ap.add_argument('--synthetic', type=Path, required=True)
    ap.add_argument('--quality-csv', type=Path, required=True)
    ap.add_argument('--pilot-json', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('reports/augmentation_quality'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    real = np.load(args.real)
    syn = np.load(args.synthetic)
    real_x = real['x'].astype(np.float64)
    real_age = real['age'].astype(int)
    real_stage = real['stage'].astype(int)
    syn_x = syn['x_normalized'].astype(np.float64)
    syn_age = syn['age'].astype(int)
    syn_stage = syn['stage'].astype(int)

    rows = []
    rows.append({'group': 'overall', **group_metrics(real_x, syn_x)})
    for age in (0, 1):
        for stage in (0, 1, 2):
            r = real_x[(real_age == age) & (real_stage == stage)]
            s = syn_x[(syn_age == age) & (syn_stage == stage)]
            rows.append({'group': f'age{age}_stage{stage}', **group_metrics(r, s)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / 'distribution_metrics.csv', index=False)

    quality = pd.read_csv(args.quality_csv)
    pilot = json.loads(args.pilot_json.read_text(encoding='utf-8'))
    rejects = []
    threshold_map = pilot.get('quality_thresholds', {})
    for (age, stage), g in quality.groupby(['age', 'stage']):
        key = f'age{int(age)}_stage{int(stage)}'
        t = threshold_map[key]
        pass_feature = g['feature_z_max'] <= t['feature_z_max']
        pass_low = g['nearest_real_distance'] >= t['nearest_real_min']
        pass_high = g['nearest_real_distance'] <= t['nearest_real_max']
        pass_abs = g['max_abs_normalized'] <= t['max_abs_normalized']
        rejects.append({
            'group': key,
            'candidates': int(len(g)),
            'accepted': int(g['accepted'].sum()),
            'pass_feature': int(pass_feature.sum()),
            'pass_memorization_floor': int(pass_low.sum()),
            'pass_ood_ceiling': int(pass_high.sum()),
            'pass_amplitude': int(pass_abs.sum()),
            'fail_feature_only_or_combined': int((~pass_feature).sum()),
            'fail_memorization_floor': int((~pass_low).sum()),
            'fail_ood_ceiling': int((~pass_high).sum()),
            'fail_amplitude': int((~pass_abs).sum()),
            'feature_z_median': float(g['feature_z_max'].median()),
            'nearest_real_median': float(g['nearest_real_distance'].median()),
        })
    reject_df = pd.DataFrame(rejects)
    reject_df.to_csv(args.out / 'gate_diagnostics.csv', index=False)

    missing_groups = metrics[(metrics.group != 'overall') & (metrics.n_synthetic == 0)].group.tolist()
    overall = metrics.iloc[0].to_dict()
    notes = []
    if missing_groups:
        notes.append('Synthetic condition coverage is incomplete: ' + ', '.join(missing_groups) + '.')
    if np.isfinite(overall['feature_smd_mean']) and overall['feature_smd_mean'] > 1.0:
        notes.append('Overall feature mean shift remains large (>1 real-group standard deviation on average).')
    if np.isfinite(overall['nn_ratio']) and overall['nn_ratio'] < 0.2:
        notes.append('Synthetic windows may be too close to the real training windows; inspect memorization.')
    if np.isfinite(overall['nn_ratio']) and overall['nn_ratio'] > 10:
        notes.append('Synthetic windows may be too far from the real manifold; inspect OOD behavior.')
    if not notes:
        notes.append('No automatic red flag was triggered by the descriptive thresholds, but downstream usefulness is not yet established.')

    report = [
        '# Augmentation quality audit', '',
        f"- Real windows: **{len(real_x)}**",
        f"- Accepted synthetic windows: **{len(syn_x)}**",
        f"- Overall feature SMD mean/max: **{overall['feature_smd_mean']:.3f} / {overall['feature_smd_max']:.3f}**",
        f"- Overall feature range coverage: **{100*overall['feature_range_coverage']:.1f}%**",
        f"- Overall real-real / synthetic-real NN median: **{overall['real_real_nn_median']:.4f} / {overall['synthetic_real_nn_median']:.4f}**",
        f"- Overall NN ratio: **{overall['nn_ratio']:.2f}**",
        f"- Overall RBF-MMD²: **{overall['mmd2']:.4f}**",
        '', '## Automatic diagnostic', '',
    ] + [f'- {n}' for n in notes] + [
        '', '## Interpretation guardrail', '',
        'All metrics here are descriptive at window level. They do not change the number of independent TR experiments and must not be reported as experiment-level validation.',
    ]
    (args.out / 'augmentation_quality.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))
    print('\nGate diagnostics:\n', reject_df.to_string(index=False))


if __name__ == '__main__':
    main()
