#!/usr/bin/env python3
"""Focused audit for mechanically suspicious/heterogeneous channels.

This script does not modify raw data. It reports quantiles, missing locations,
possible time-column contamination, and extreme rows for the pouch-cell and
module mechanical recordings that should be understood before augmentation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('data/raw/osf_c2hnq')
OUT = Path('reports/problem_channel_audit')


def read_two_col(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path, header=None)
    else:
        df = pd.read_excel(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(path)
    df = df.iloc[:, :2].copy()
    df.columns = ['time', 'value']
    df['time'] = pd.to_numeric(df.time, errors='coerce')
    df['value'] = pd.to_numeric(df.value, errors='coerce')
    return df


def describe(path: Path) -> dict:
    df = read_two_col(path)
    v = df.value
    finite = v[np.isfinite(v)]
    t = df.time
    qs = finite.quantile([0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]).to_dict()
    equal_time = np.isfinite(v) & np.isfinite(t) & np.isclose(v, t, rtol=0, atol=1e-8)
    corr = float(np.corrcoef(t[np.isfinite(t) & np.isfinite(v)], v[np.isfinite(t) & np.isfinite(v)])[0,1]) if finite.size > 2 else np.nan
    extreme_idx = finite.abs().nlargest(min(20, len(finite))).index
    extremes = df.loc[extreme_idx, ['time','value']].to_dict(orient='records')
    return {
        'path': str(path),
        'rows': int(len(df)),
        'missing_time': int(t.isna().sum()),
        'missing_value': int(v.isna().sum()),
        'quantiles': {str(k): float(val) for k,val in qs.items()},
        'count_abs_gt_1000': int((finite.abs() > 1000).sum()),
        'count_abs_gt_2000': int((finite.abs() > 2000).sum()),
        'count_value_equals_time': int(equal_time.sum()),
        'time_value_corr': corr,
        'extreme_rows': extremes,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = []
    patterns = ['D1_Pressure.xlsx', 'A2_Force.xlsx', 'M1_Force.csv', 'M2_Force.csv']
    for pat in patterns:
        matches = list(ROOT.rglob(pat))
        targets.extend(matches)
    reports = [describe(p) for p in targets]
    (OUT / 'problem_channels.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')

    lines = ['# Focused mechanical-channel audit', '']
    for r in reports:
        lines += [
            f"## `{Path(r['path']).name}`", '',
            f"- rows: **{r['rows']}**; missing value: **{r['missing_value']}**",
            f"- |value| > 1000: **{r['count_abs_gt_1000']}**; |value| > 2000: **{r['count_abs_gt_2000']}**",
            f"- value exactly equals time: **{r['count_value_equals_time']}** rows",
            f"- time/value Pearson correlation: **{r['time_value_corr']:.4f}**",
            f"- q01 / median / q99: **{r['quantiles'].get('0.01', float('nan')):.4f} / {r['quantiles'].get('0.5', float('nan')):.4f} / {r['quantiles'].get('0.99', float('nan')):.4f}**",
            '', 'Largest absolute-value rows:', '',
            '```json', json.dumps(r['extreme_rows'][:10], indent=2), '```', ''
        ]
    (OUT / 'problem_channels.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
