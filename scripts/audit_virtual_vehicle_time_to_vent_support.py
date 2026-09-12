#!/usr/bin/env python3
"""Support-only audit for Virtual Vehicle time-to-vent regression.

No predictive model is fit. Candidate regression supports are compared only by
coverage of the eight independent destructive experiments and by the available
range of package-provided time-to-first-`vent_gas` labels.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip

RECORD_ID = '18849418'
API = f'https://zenodo.org/api/records/{RECORD_ID}'
HEADERS = {'User-Agent': 'BatteryDiffusion-time-to-vent-support/1.0'}
EXPS = [f'TS0330{x}' for x in 'ABCDEFGH']


def record_url() -> str:
    r = requests.get(API, timeout=60, headers=HEADERS)
    r.raise_for_status()
    rec = r.json()
    z = next(f for f in rec.get('files', []) if str(f.get('key', '')).lower().endswith('.zip'))
    links = z.get('links') or {}
    url = links.get('content') or links.get('self')
    if not url:
        raise RuntimeError('Zenodo ZIP URL unavailable')
    return url


def read_xlsx_bytes(rz: RemoteZip, path: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(rz.read(path)), header=None, engine='openpyxl')


def load_one(rz: RemoteZip, exp: str):
    edf = read_xlsx_bytes(rz, f'Plots_and_raw_data/{exp}/export_{exp}.xlsx')
    odf = read_xlsx_bytes(rz, f'Plots_and_raw_data/{exp}/order_{exp}.xlsx')
    num = edf.iloc[2:].apply(pd.to_numeric, errors='coerce')
    t = num.iloc[:, 0].to_numpy(float) / 1000.0
    p = num.iloc[:, 1].to_numpy(float) / 1e5
    temp = num.iloc[:, 5].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(p) & np.isfinite(temp)
    t, p, temp = t[ok], p[ok], temp[ok]

    types = odf.iloc[:, 2].astype(str).str.strip().str.lower()
    vent_rows = odf[types.eq('vent_gas')]
    if vent_rows.empty:
        raise RuntimeError(f'{exp}: no vent_gas event marker')
    event = float(pd.to_numeric(vent_rows.iloc[:, 0], errors='coerce').dropna().min())

    lo = int(np.ceil(t.min()))
    hi = int(np.floor(min(t.max(), event - 1e-6)))
    grid = np.arange(lo, hi + 1, dtype=float)
    df = pd.DataFrame({
        'time_s': grid,
        'temperature_c': np.interp(grid, t, temp),
        'pressure_bar_abs': np.interp(grid, t, p),
    })
    return df, event


def collect(df: pd.DataFrame, event: float, window: int, stride: int, cap: float, max_tau: float):
    t = df.time_s.to_numpy(dtype=int)
    temp = df.temperature_c.to_numpy(float)
    rows = []
    for s in range(0, len(df) - window + 1, stride):
        e = s + window
        tw = t[s:e]
        if len(tw) != window or np.any(np.diff(tw) != 1):
            continue
        end = int(tw[-1])
        tau = float(event - end)
        if tau <= 0 or tau > max_tau:
            continue
        mx = float(np.max(temp[s:e]))
        if mx > cap:
            continue
        rows.append((end, tau, mx))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('reports/external_virtual_vehicle_21700/time_to_vent_support'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = {}
    with RemoteZip(record_url(), headers=HEADERS, initial_buffer_size=1024 * 1024) as rz:
        for exp in EXPS:
            data[exp] = load_one(rz, exp)

    detail_rows = []
    for window in (60, 120, 256):
        for cap in (100, 120, 150):
            for max_tau in (300, 600, 900, 1200):
                for exp, (df, event) in data.items():
                    rows = collect(df, event, window, 10, cap, max_tau)
                    taus = np.asarray([r[1] for r in rows], dtype=float)
                    detail_rows.append({
                        'window_s': window,
                        'temp_cap_c': cap,
                        'max_tau_s': max_tau,
                        'experiment': exp,
                        'n_windows': int(len(taus)),
                        'tau_min_s': float(np.min(taus)) if len(taus) else np.nan,
                        'tau_q25_s': float(np.quantile(taus, .25)) if len(taus) else np.nan,
                        'tau_median_s': float(np.median(taus)) if len(taus) else np.nan,
                        'tau_q75_s': float(np.quantile(taus, .75)) if len(taus) else np.nan,
                        'tau_max_s': float(np.max(taus)) if len(taus) else np.nan,
                    })
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(args.out / 'support_detail.csv', index=False)

    summary_rows = []
    for (w, cap, mt), g in detail.groupby(['window_s', 'temp_cap_c', 'max_tau_s']):
        valid = g.n_windows > 0
        summary_rows.append({
            'window_s': int(w),
            'temp_cap_c': int(cap),
            'max_tau_s': int(mt),
            'experiments_supported': int(valid.sum()),
            'min_windows_per_experiment': int(g.n_windows.min()),
            'median_windows_per_experiment': float(g.n_windows.median()),
            'total_windows': int(g.n_windows.sum()),
            'worst_experiment_tau_span_s': float((g.tau_max_s - g.tau_min_s).min(skipna=True)),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ['experiments_supported', 'min_windows_per_experiment', 'window_s', 'max_tau_s'],
        ascending=[False, False, False, True])
    summary.to_csv(args.out / 'support_summary.csv', index=False)

    full = summary[summary.experiments_supported == len(EXPS)].copy()
    full.to_csv(args.out / 'fully_supported_candidates.csv', index=False)

    lines = [
        '# Virtual Vehicle time-to-vent support audit', '',
        'No predictor is fit in this audit. Labels use the package-provided first `vent_gas` event. Candidate definitions are compared only by real-experiment support and label-range coverage.', '',
        f'- Independent experiments: **{len(EXPS)}**',
        f'- Fully supported candidate definitions: **{len(full)}**', '',
        '## Highest-support candidates', '',
        '| context | T cap | max target tau | min windows/test | median windows/test | worst tau span |',
        '|---:|---:|---:|---:|---:|---:|'
    ]
    for _, r in full.head(15).iterrows():
        lines.append(
            f"| {int(r.window_s)} s | {int(r.temp_cap_c)} C | {int(r.max_tau_s)} s | "
            f"{int(r.min_windows_per_experiment)} | {r.median_windows_per_experiment:.1f} | {r.worst_experiment_tau_span_s:.1f} s |")
    lines += ['', '## Guardrail', '',
              'This audit is support-only. A regression protocol must be frozen from these support statistics before any time-to-vent model is scored. Window counts remain repeated decision points, not independent destructive experiments.']
    (args.out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
