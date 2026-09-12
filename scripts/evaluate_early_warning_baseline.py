#!/usr/bin/env python3
"""Real-only, experiment-blocked feasibility baseline for impending venting.

This is deliberately a small classical baseline, not a paper headline model. It
asks a narrow question before any downstream architecture work: do causal
pre-vent temperature/force windows contain enough transferable information to
separate an impending-vent horizon from earlier heating?

Protocol
--------
* Experiments: A2, D1, M1, M2 (published vent labels only).
* Context: 256 s at 1 Hz, stride 10 s; no post-vent samples.
* Positive: 0 < t_vent - t_end <= 300 s.
* Evaluation: leave-one-experiment-out. The held-out experiment never
  contributes windows to model fitting or feature scaling.
* Inputs compared: temperature only, force only, and multimodal fusion.
* Predictor: class-weighted logistic regression on causal window descriptors.

The module temperature input is the trigger cell (cell 1) hot-spot temperature,
constructed as the per-second max of TC1-TC6. Pouch cells use their recorded
single temperature channel. This gives a common [trigger-temperature, force]
interface across cell and module experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the already audited preprocessing implementations.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pouch_clean_windows import read_xlsx_two_col, conservative_valid_prefix, fill_small_temperature_gaps, common_integer_grid
from prepare_module_windows import aggregate_module

VENT = {'A2':1812, 'D1':1703, 'M1':2137, 'M2':1569}


def load_pouch(root: Path, exp: str) -> pd.DataFrame:
    if exp == 'A2':
        temp_path = root/'Cell/NMC pouch/Fresh/A2_Temperature.xlsx'
        force_path = root/'Cell/NMC pouch/Fresh/A2_Force.xlsx'
        detect_ramp = False
    elif exp == 'D1':
        temp_path = root/'Cell/NMC pouch/Aged/D1_Temperature.xlsx'
        force_path = root/'Cell/NMC pouch/Aged/D1_Pressure.xlsx'
        detect_ramp = True
    else:
        raise ValueError(exp)
    temp = fill_small_temperature_gaps(read_xlsx_two_col(temp_path))
    temp = temp.loc[np.isfinite(temp.value)].reset_index(drop=True)
    mech, _ = conservative_valid_prefix(read_xlsx_two_col(force_path), detect_ramp=detect_ramp)
    grid, tv, fv = common_integer_grid(temp, mech)
    return pd.DataFrame({'time':grid.astype(int), 'temperature':tv, 'force':fv})


def load_module(root: Path, exp: str) -> pd.DataFrame:
    if exp == 'M1':
        state = 'Fresh'
    elif exp == 'M2':
        state = 'Aged'
    else:
        raise ValueError(exp)
    base = root/f'Module/NMC pouch/{state}'
    df, _ = aggregate_module(base/f'{exp}_Temperature.csv', base/f'{exp}_Force.csv')
    return pd.DataFrame({
        'time': df.second.astype(int),
        'temperature': df.T_cell1_max.astype(float),
        'force': df.force.astype(float),
    })


def window_descriptors(x: np.ndarray) -> np.ndarray:
    """Causal descriptors for N x C x L input."""
    feats = []
    for sample in x:
        row = []
        for s in sample:
            d = np.diff(s)
            tt = np.arange(len(s), dtype=float)
            # Stable closed-form slope around centered time.
            tc = tt - tt.mean()
            slope = float(np.dot(tc, s - s.mean()) / max(np.dot(tc, tc), 1e-12))
            row.extend([
                float(s[-1]), float(s.mean()), float(s.std()), float(s.min()), float(s.max()),
                float(s[-1]-s[0]), slope,
                float(np.mean(np.abs(d))) if len(d) else 0.0,
                float(np.max(np.abs(d))) if len(d) else 0.0,
            ])
        if sample.shape[0] == 2 and np.std(sample[0]) > 0 and np.std(sample[1]) > 0:
            row.append(float(np.corrcoef(sample[0], sample[1])[0,1]))
        elif sample.shape[0] == 2:
            row.append(0.0)
        feats.append(row)
    return np.asarray(feats, dtype=np.float64)


def build_windows(df: pd.DataFrame, vent: int, window: int, stride: int, horizon: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    df = df[(df.time < vent)].sort_values('time').drop_duplicates('time').copy()
    # Exact one-second continuity is required; no window crosses a missing second.
    times = df.time.to_numpy(dtype=int)
    vals = df[['temperature','force']].to_numpy(dtype=float)
    out=[]; labels=[]; ends=[]
    for start in range(0, len(df)-window+1, stride):
        end = start+window
        tw = times[start:end]
        if len(tw) != window or np.any(np.diff(tw) != 1):
            continue
        t_end = int(tw[-1]); tau = vent - t_end
        if tau <= 0:
            continue
        out.append(vals[start:end].T)
        labels.append(int(tau <= horizon)); ends.append(t_end)
    return np.stack(out).astype(np.float32), np.asarray(labels,dtype=int), np.asarray(ends,dtype=int)


def select_modality(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'temperature': return x[:,0:1,:]
    if mode == 'force': return x[:,1:2,:]
    if mode == 'fusion': return x
    raise ValueError(mode)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('data/raw/osf_c2hnq/Dataset_TR/Dataset_TR'))
    ap.add_argument('--out',type=Path,default=Path('reports/early_warning_baseline'))
    ap.add_argument('--window',type=int,default=256); ap.add_argument('--stride',type=int,default=10); ap.add_argument('--horizon',type=int,default=300)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    datasets={}
    for exp in ('A2','D1'):
        datasets[exp]=build_windows(load_pouch(args.root,exp),VENT[exp],args.window,args.stride,args.horizon)
    for exp in ('M1','M2'):
        datasets[exp]=build_windows(load_module(args.root,exp),VENT[exp],args.window,args.stride,args.horizon)

    detail=[]; prediction_rows=[]
    for mode in ('temperature','force','fusion'):
        for test_exp in datasets:
            train_exps=[e for e in datasets if e!=test_exp]
            xtr=np.concatenate([select_modality(datasets[e][0],mode) for e in train_exps],axis=0)
            ytr=np.concatenate([datasets[e][1] for e in train_exps],axis=0)
            xte=select_modality(datasets[test_exp][0],mode); yte=datasets[test_exp][1]; ends=datasets[test_exp][2]
            ftr=window_descriptors(xtr); fte=window_descriptors(xte)
            model=Pipeline([
                ('scale',StandardScaler()),
                ('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026)),
            ])
            model.fit(ftr,ytr); prob=model.predict_proba(fte)[:,1]; pred=(prob>=0.5).astype(int)
            row={
                'modality':mode,'test_experiment':test_exp,'n_train_windows':len(ytr),'n_test_windows':len(yte),
                'test_positive':int(yte.sum()),'auroc':float(roc_auc_score(yte,prob)),
                'auprc':float(average_precision_score(yte,prob)),'f1_at_0p5':float(f1_score(yte,pred,zero_division=0)),
                'balanced_accuracy_at_0p5':float(balanced_accuracy_score(yte,pred)),
            }
            detail.append(row)
            for t,y,p in zip(ends,yte,prob):
                prediction_rows.append({'modality':mode,'test_experiment':test_exp,'window_end_s':int(t),'label':int(y),'probability':float(p)})

    detail_df=pd.DataFrame(detail)
    detail_df.to_csv(args.out/'loso_metrics.csv',index=False)
    pd.DataFrame(prediction_rows).to_csv(args.out/'loso_predictions.csv',index=False)
    macro=detail_df.groupby('modality').agg(
        auroc_mean=('auroc','mean'),auroc_min=('auroc','min'),
        auprc_mean=('auprc','mean'),auprc_min=('auprc','min'),
        f1_mean=('f1_at_0p5','mean'),balanced_acc_mean=('balanced_accuracy_at_0p5','mean')
    ).reset_index()
    macro.to_csv(args.out/'macro_metrics.csv',index=False)

    lines=['# Real-only impending-vent baseline','',
           f'- Context/horizon/stride: **{args.window}s / {args.horizon}s / {args.stride}s**',
           '- Evaluation: **leave-one-real-experiment-out** across A2, D1, M1, M2.',
           '- No synthetic windows are used in this baseline.',
           '- Predictor: class-weighted logistic regression on causal signal descriptors.', '',
           '## Macro results','',
           '| modality | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1@0.5 | mean balanced acc |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows():
        lines.append(f"| {r.modality} | {r.auroc_mean:.3f} | {r.auroc_min:.3f} | {r.auprc_mean:.3f} | {r.auprc_min:.3f} | {r.f1_mean:.3f} | {r.balanced_acc_mean:.3f} |")
    lines += ['', '## Fold detail','',
              '| modality | held-out ID | AUROC | AUPRC | F1@0.5 |', '|---|---|---:|---:|---:|']
    for _,r in detail_df.iterrows():
        lines.append(f"| {r.modality} | {r.test_experiment} | {r.auroc:.3f} | {r.auprc:.3f} | {r.f1_at_0p5:.3f} |")
    lines += ['', '## Interpretation guardrail','',
              'These are feasibility results from only four independent real experiments. Window-level metric counts do not increase experimental n. The purpose is to decide whether the impending-vent task is worth developing, not to make a publishable generalization claim.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__': main()
