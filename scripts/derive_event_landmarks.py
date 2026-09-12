#!/usr/bin/env python3
"""Build a cautious event-landmark table for downstream task feasibility.

Published vent times are used only where the source paper explicitly reports
them (A2, D1, M1, M2). For 2170 experiments, this script produces *algorithmic
candidates* from the pressure trace and does not silently promote them to ground
truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_two_col(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path, header=None)
    else:
        df = pd.read_excel(path, header=None)
    df = df.iloc[:, :2].copy()
    df.columns = ['time', 'value']
    df['time'] = pd.to_numeric(df.time, errors='coerce')
    df['value'] = pd.to_numeric(df.value, errors='coerce')
    return df.dropna().sort_values('time').drop_duplicates('time').reset_index(drop=True)


def smooth_and_derivative(df: pd.DataFrame, seconds: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = df.time.to_numpy(dtype=float)
    y = df.value.to_numpy(dtype=float)
    if len(t) < 5:
        return t, y, np.gradient(y, t)
    dt = float(np.median(np.diff(t)))
    width = max(3, int(round(seconds / max(dt, 1e-6))))
    if width % 2 == 0:
        width += 1
    ys = pd.Series(y).rolling(width, center=True, min_periods=1).median().to_numpy(dtype=float)
    dydt = np.gradient(ys, t)
    return t, ys, dydt


def pressure_drop_candidates(pressure: pd.DataFrame, temp: pd.DataFrame, topk: int = 8) -> list[dict]:
    tt = temp.time.to_numpy(dtype=float)
    tv = temp.value.to_numpy(dtype=float)
    peak_t = float(tt[int(np.nanargmax(tv))])
    t, p, dp = smooth_and_derivative(pressure, seconds=2.0)
    search = (t >= t.min() + 30.0) & (t <= peak_t - 5.0)
    if not search.any():
        return []
    base_mask = t <= min(t.min() + 120.0, peak_t)
    baseline = float(np.nanmedian(p[base_mask])) if base_mask.any() else float(np.nanmedian(p))
    pre = p[search]
    excursion = max(float(np.nanpercentile(pre, 95) - baseline), 1e-6)
    elevated = p >= baseline + 0.03 * excursion
    eligible = search & elevated & np.isfinite(dp)
    ids = np.flatnonzero(eligible)
    if len(ids) == 0:
        ids = np.flatnonzero(search & np.isfinite(dp))
    order = ids[np.argsort(dp[ids])]  # most negative derivative first
    out = []
    for i in order:
        # Keep candidates separated by at least 10 s to avoid reporting the same drop repeatedly.
        if any(abs(float(t[i]) - r['time_s']) < 10.0 for r in out):
            continue
        out.append({
            'time_s': float(t[i]),
            'pressure_smoothed': float(p[i]),
            'dpdt': float(dp[i]),
            'temperature_peak_time_s': peak_t,
            'baseline_pressure': baseline,
        })
        if len(out) >= topk:
            break
    return out


def local_mechanical_audit(path: Path, event_time: float) -> dict:
    df = read_two_col(path)
    t, y, dy = smooth_and_derivative(df, seconds=2.0)
    i = int(np.argmin(np.abs(t - event_time)))
    lo = max(0, i - 10); hi = min(len(t), i + 11)
    local_min_i = lo + int(np.argmin(dy[lo:hi]))
    return {
        'requested_event_time_s': float(event_time),
        'nearest_sample_time_s': float(t[i]),
        'mechanical_value_at_event': float(y[i]),
        'derivative_at_event': float(dy[i]),
        'strongest_local_drop_time_s': float(t[local_min_i]),
        'strongest_local_drop_dpdt': float(dy[local_min_i]),
    }


def module_force_path(root: Path, exp: str) -> Path:
    state = 'Fresh' if exp == 'M1' else 'Aged'
    return root / f'Module/NMC pouch/{state}/{exp}_Force.csv'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/raw/osf_c2hnq/Dataset_TR/Dataset_TR'))
    ap.add_argument('--out', type=Path, default=Path('reports/event_landmarks'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    # Explicit values reported in the source paper (Scientific Data 2026,
    # DOI 10.1038/s41597-026-07857-1).
    published = [
        {'experiment_id':'A2','level':'cell','form_factor':'pouch','aging':'fresh','vent_time_s':1812.0,'vtp_s':614.0,'source':'published'},
        {'experiment_id':'D1','level':'cell','form_factor':'pouch','aging':'aged','vent_time_s':1703.0,'vtp_s':1037.0,'source':'published'},
        {'experiment_id':'M1','level':'module','form_factor':'pouch','aging':'fresh','vent_time_s':2137.0,'vtp_s':478.0,'source':'published_cell1'},
        {'experiment_id':'M2','level':'module','form_factor':'pouch','aging':'aged','vent_time_s':1569.0,'vtp_s':1039.0,'source':'published_cell1'},
    ]
    pub_df = pd.DataFrame(published)
    pub_df['implied_peak_time_s'] = pub_df.vent_time_s + pub_df.vtp_s
    pub_df.to_csv(args.out / 'published_event_landmarks.csv', index=False)

    audits = []
    audits.append({'experiment_id':'A2', **local_mechanical_audit(args.root/'Cell/NMC pouch/Fresh/A2_Force.xlsx', 1812.0)})
    # D1 filename says Pressure in the archive; the paper identifies the pouch mechanical signal as force.
    audits.append({'experiment_id':'D1', **local_mechanical_audit(args.root/'Cell/NMC pouch/Aged/D1_Pressure.xlsx', 1703.0)})
    audits.append({'experiment_id':'M1', **local_mechanical_audit(module_force_path(args.root,'M1'), 2137.0)})
    audits.append({'experiment_id':'M2', **local_mechanical_audit(module_force_path(args.root,'M2'), 1569.0)})
    pd.DataFrame(audits).to_csv(args.out / 'published_landmark_signal_audit.csv', index=False)

    candidates = {}
    for exp, aging in [('B2','Fresh'),('C1','Aged'),('C2','Aged')]:
        p = args.root / f'Cell/NMC 2170/{aging}/{exp}_Pressure.xlsx'
        t = args.root / f'Cell/NMC 2170/{aging}/{exp}_Temperature.xlsx'
        candidates[exp] = pressure_drop_candidates(read_two_col(p), read_two_col(t))
    (args.out / '2170_vent_candidates.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')

    lines = [
        '# Event-landmark audit', '',
        'The source paper explicitly reports venting times for A2, D1, M1 and M2. Those values are stored as published labels. For B2/C1/C2, the repository currently stores only algorithmic pressure-drop candidates; they are **not yet frozen as ground truth**.', '',
        '## Published labels', '',
        '| ID | vent time (s) | vent-to-peak (s) | implied peak (s) |',
        '|---|---:|---:|---:|',
    ]
    for r in published:
        lines.append(f"| {r['experiment_id']} | {r['vent_time_s']:.0f} | {r['vtp_s']:.0f} | {r['vent_time_s']+r['vtp_s']:.0f} |")
    lines += ['', '## 2170 pressure-drop candidates', '']
    for exp, vals in candidates.items():
        lines.append(f'### {exp}')
        for rank, r in enumerate(vals[:5], start=1):
            lines.append(f"- candidate {rank}: t={r['time_s']:.1f} s, dP/dt={r['dpdt']:.6g}, P_smooth={r['pressure_smoothed']:.6g}, T-peak={r['temperature_peak_time_s']:.1f} s")
        lines.append('')
    lines += [
        '## Downstream implication', '',
        'Time-to-vent / impending-vent prediction is label-supported for the pouch cell and module experiments. A 2170 version is feasible only after the pressure-drop candidates are visually/algorithmically validated and frozen before any predictive experiment.',
    ]
    (args.out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
