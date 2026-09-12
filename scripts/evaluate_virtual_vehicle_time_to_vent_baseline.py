#!/usr/bin/env python3
"""Real-only LOEO baseline for frozen Virtual Vehicle time-to-vent regression."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_virtual_vehicle_baseline import feature_matrix

REPS = ('temperature','pressure_absolute','pressure_shape','fusion_absolute','fusion_shape')


def build_windows(df, event, window=120, stride=10, max_tau=600, cap=150.0):
    t = df.time_s.to_numpy(dtype=int)
    vals = df[['temperature_c','pressure_bar_abs']].to_numpy(dtype=float)
    xs, ys, meta = [], [], []
    for s in range(0, len(df)-window+1, stride):
        e = s + window; tw = t[s:e]
        if len(tw) != window or np.any(np.diff(tw) != 1):
            continue
        x = vals[s:e].T.astype(np.float32)
        end = int(tw[-1]); tau = float(event-end); mx = float(np.max(x[0]))
        if tau <= 0 or tau > max_tau or mx > cap:
            continue
        xs.append(x); ys.append(tau)
        meta.append({'t_end_s':end,'tau_to_event_s':tau,'max_temperature_c':mx})
    if not xs:
        return np.empty((0,2,window),np.float32), np.empty(0,float), pd.DataFrame(meta)
    return np.stack(xs), np.asarray(ys,dtype=float), pd.DataFrame(meta)


def make_model(kind):
    if kind == 'huber':
        return Pipeline([('scale', StandardScaler()),
                         ('reg', HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=3000))])
    if kind == 'rf':
        return RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                     random_state=2026, n_jobs=-1)
    raise ValueError(kind)


def fit_eval(xtr, ytr, xte, yte, rep, kind):
    m = make_model(kind)
    m.fit(feature_matrix(xtr, rep), ytr)
    p = np.asarray(m.predict(feature_matrix(xte, rep)), dtype=float)
    return {
        'mae_s': float(mean_absolute_error(yte,p)),
        'rmse_s': float(mean_squared_error(yte,p)**0.5),
        'medae_s': float(median_absolute_error(yte,p)),
        'r2': float(r2_score(yte,p)),
        'bias_s': float(np.mean(p-yte)),
    }, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('reports/virtual_vehicle_time_to_vent_baseline'))
    ap.add_argument('--window', type=int, default=120)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--max-tau', type=float, default=600.0)
    ap.add_argument('--temp-cap', type=float, default=150.0)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    data = {}; support = []
    with RemoteZip(record_url(), headers=HEADERS, initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df,event,_ = load_one(rz,exp)
            x,y,m = build_windows(df,event,args.window,args.stride,args.max_tau,args.temp_cap)
            data[exp] = (x,y,m)
            support.append({'experiment':exp,'event_time_s':event,'windows':len(y),
                            'tau_min_s':float(np.min(y)),'tau_max_s':float(np.max(y)),
                            'tau_median_s':float(np.median(y))})
    pd.DataFrame(support).to_csv(args.out/'support.csv',index=False)

    rows=[]; preds=[]
    for held in EXPS:
        train=[e for e in EXPS if e!=held]
        xtr=np.concatenate([data[e][0] for e in train]); ytr=np.concatenate([data[e][1] for e in train])
        xte,yte,meta=data[held]
        for kind in ('huber','rf'):
            for rep in REPS:
                metrics,p=fit_eval(xtr,ytr,xte,yte,rep,kind)
                rows.append({'heldout':held,'model':kind,'representation':rep,
                             'n_train':len(ytr),'n_test':len(yte),**metrics})
                for (_,mr),yy,pp in zip(meta.iterrows(),yte,p):
                    preds.append({'heldout':held,'model':kind,'representation':rep,
                                  't_end_s':int(mr.t_end_s),'tau_true_s':float(yy),'tau_pred_s':float(pp),
                                  'abs_error_s':float(abs(pp-yy))})
    detail=pd.DataFrame(rows); detail.to_csv(args.out/'loeo_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(args.out/'loeo_predictions.csv',index=False)
    macro=detail.groupby(['model','representation']).agg(
        mae_mean_s=('mae_s','mean'),mae_worst_s=('mae_s','max'),
        rmse_mean_s=('rmse_s','mean'),medae_mean_s=('medae_s','mean'),
        r2_mean=('r2','mean'),bias_mean_s=('bias_s','mean')).reset_index()
    macro.to_csv(args.out/'macro_metrics.csv',index=False)

    lines=['# Virtual Vehicle time-to-vent real-only LOEO baseline','',
           f'Frozen before scoring: {args.window} s causal context, target 0 < tau <= {args.max_tau:g} s to first package `vent_gas`, max input temperature <= {args.temp_cap:g} C, stride {args.stride} s.','',
           'All eight experiments are held out one at a time. Huber is the predeclared headline model; random forest is a secondary nonlinear diagnostic.','',
           '| model | representation | mean MAE | worst MAE | mean RMSE | mean MedAE | mean R2 |','|---|---|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows():
        lines.append(f"| {r.model} | {r.representation} | {r.mae_mean_s:.1f} s | {r.mae_worst_s:.1f} s | {r.rmse_mean_s:.1f} s | {r.medae_mean_s:.1f} s | {r.r2_mean:.3f} |")
    lines += ['', '## Guardrail','',
              'The regression definition was frozen in `reports/virtual_vehicle_time_to_vent_protocol_freeze.md` after support-only auditing and before these scores. Statistical n remains 8 independent destructive experiments.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
