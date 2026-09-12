#!/usr/bin/env python3
"""Audit whether the published pouch/module vent labels support an early-warning task.

No predictive model is trained here. We only count leakage-safe causal windows
that end before the published vent event. A positive window means the first vent
will occur within the chosen forecast horizon; a negative window ends earlier.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd


VENT = {
    'A2': 1812,
    'D1': 1703,
    'M1': 2137,
    'M2': 1569,
}
# All four records start near test-time zero/one and have valid paired signals
# through the first vent according to the repository cleaning/alignment audits.
START = {'A2': 1, 'D1': 1, 'M1': 1, 'M2': 1}
META = {
    'A2': ('cell','fresh'),
    'D1': ('cell','aged'),
    'M1': ('module','fresh'),
    'M2': ('module','aged'),
}


def count(exp: str, window: int, stride: int, horizon: int) -> dict:
    vent = VENT[exp]
    start0 = START[exp]
    pos = neg = 0
    ends = []
    s = start0
    while s + window - 1 < vent:
        end = s + window - 1
        tau = vent - end
        label = int(0 < tau <= horizon)
        pos += label
        neg += 1 - label
        ends.append((end, tau, label))
        s += stride
    level, aging = META[exp]
    return {
        'experiment_id': exp, 'level': level, 'aging': aging,
        'window_s': window, 'stride_s': stride, 'horizon_s': horizon,
        'vent_time_s': vent, 'total_prevent_windows': pos + neg,
        'positive_impending_vent': pos, 'negative_earlier': neg,
        'positive_fraction': pos / (pos + neg) if pos + neg else 0.0,
        'earliest_window_end_s': ends[0][0] if ends else None,
        'latest_window_end_s': ends[-1][0] if ends else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('reports/downstream_feasibility'))
    ap.add_argument('--stride', type=int, default=10)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for w, h, exp in itertools.product((60,120,256), (60,120,300,600), VENT):
        rows.append(count(exp,w,args.stride,h))
    df = pd.DataFrame(rows)
    df.to_csv(args.out/'early_warning_window_counts.csv', index=False)

    configs = []
    for (w,h), g in df.groupby(['window_s','horizon_s']):
        configs.append({
            'window_s': w, 'horizon_s': h,
            'total_windows': int(g.total_prevent_windows.sum()),
            'total_positive': int(g.positive_impending_vent.sum()),
            'total_negative': int(g.negative_earlier.sum()),
            'min_positive_per_experiment': int(g.positive_impending_vent.min()),
            'max_positive_fraction': float(g.positive_fraction.max()),
            'min_positive_fraction': float(g.positive_fraction.min()),
        })
    cfg = pd.DataFrame(configs).sort_values(['window_s','horizon_s'])
    cfg.to_csv(args.out/'early_warning_config_summary.csv', index=False)

    # 256 s matches the current diffusion window length. 300 s gives roughly
    # 30 positive windows per experiment at 10 s stride without letting the
    # positive class dominate the pre-vent history.
    chosen = df[(df.window_s==256)&(df.horizon_s==300)]
    lines = [
        '# Downstream application feasibility audit', '',
        '## Candidate selected for first predictive pilot', '',
        '**Impending-vent early warning** using a causal temperature + mechanical window.', '',
        'Label definition for a real window ending at time `t`:', '',
        '`y=1` if `0 < t_vent - t <= H`; otherwise `y=0`, with all post-vent windows excluded.', '',
        'The source paper explicitly reports `t_vent` for A2, D1, M1 and M2, so these labels do not need to be inferred from the same input signal.', '',
        '### Recommended first protocol', '',
        '- context window: **256 s** (matches the current RA-CDiff window length);',
        '- stride for predictive dataset construction: **10 s**;',
        '- warning horizon: **300 s**;',
        '- evaluation unit: **held-out real experiment**, never random windows;',
        '- headline metrics: AUROC, AUPRC, F1, sensitivity at fixed false-alarm rate, and warning lead time;',
        '- comparisons: temperature-only, mechanical-only, temperature+mechanical; each with real-only vs classical augmentation vs RA-CDiff.', '',
        '### Real-window support at 256 s / 300 s', '',
        '| ID | level | aging | pre-vent windows | positive | negative | positive fraction |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for _,r in chosen.iterrows():
        lines.append(f"| {r.experiment_id} | {r.level} | {r.aging} | {int(r.total_prevent_windows)} | {int(r.positive_impending_vent)} | {int(r.negative_earlier)} | {100*r.positive_fraction:.1f}% |")
    lines += [
        '', '## Why this is preferable to the other obvious tasks', '',
        '- SOH regression is not supported because SOH has only a few discrete experiment-level values.',
        '- Peak-temperature/peak-force regression has effectively one target per experiment, so generative windows do not create independent target observations.',
        '- Shape-stage classification would be partly circular because the current stage labels are derived from the temperature trajectory itself.',
        '- Time-to-vent / impending-vent prediction uses an externally reported physical event landmark and creates many causal pre-event decision points while preserving experiment-blocked evaluation.', '',
        '## Important limitation', '',
        'There are still only four experiments with explicitly published vent times in the force-based pouch/module subset. A publishable predictive claim should therefore either (a) add compatible external TR data, or (b) frame this as a small-sample proof-of-concept with experiment-level resampling and very conservative claims.',
    ]
    (args.out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
