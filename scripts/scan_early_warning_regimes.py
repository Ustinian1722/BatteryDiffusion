#!/usr/bin/env python3
"""Scan early-warning definitions that suppress trivial thermal separation.

The first 256/300-s feasibility baseline is intentionally easy: temperature
alone orders all four held-out experiments almost perfectly. That is useful as a
sanity check but not a compelling thermo-mechanical research problem.

This script therefore asks a stricter question: while the *entire causal input
window* remains below a temperature ceiling, can the mechanical signal improve
impending-vent discrimination? We scan temperature ceilings and warning
horizons, always using leave-one-real-experiment-out evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_early_warning_baseline import load_pouch, load_module, window_descriptors, select_modality, VENT


def build(df: pd.DataFrame, vent: int, window: int, stride: int, horizon: int):
    df=df[df.time < vent].sort_values('time').drop_duplicates('time')
    times=df.time.to_numpy(dtype=int); vals=df[['temperature','force']].to_numpy(dtype=float)
    xs=[]; ys=[]; ends=[]; max_t=[]; last_t=[]
    for s in range(0,len(df)-window+1,stride):
        e=s+window; tw=times[s:e]
        if len(tw)!=window or np.any(np.diff(tw)!=1): continue
        end=int(tw[-1]); tau=vent-end
        if tau<=0: continue
        x=vals[s:e].T
        xs.append(x); ys.append(int(tau<=horizon)); ends.append(end)
        max_t.append(float(np.max(x[0]))); last_t.append(float(x[0,-1]))
    return np.stack(xs).astype(np.float32),np.asarray(ys),np.asarray(ends),np.asarray(max_t),np.asarray(last_t)


def fit_score(train_parts,test_part,mode,cap):
    xtr=np.concatenate([select_modality(p[0][p[3] <= cap],mode) for p in train_parts])
    ytr=np.concatenate([p[1][p[3] <= cap] for p in train_parts])
    mask=test_part[3] <= cap; xte=select_modality(test_part[0][mask],mode); yte=test_part[1][mask]
    if len(np.unique(ytr))<2 or len(np.unique(yte))<2 or len(yte)<10:
        return None
    ftr=window_descriptors(xtr); fte=window_descriptors(xte)
    model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026))])
    model.fit(ftr,ytr); p=model.predict_proba(fte)[:,1]
    return {
        'n_train':int(len(ytr)),'n_test':int(len(yte)),'test_positive':int(yte.sum()),
        'auroc':float(roc_auc_score(yte,p)),'auprc':float(average_precision_score(yte,p)),
        'test_temp_max_observed':float(test_part[3][mask].max()),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data/raw/osf_c2hnq/Dataset_TR/Dataset_TR'))
    ap.add_argument('--out',type=Path,default=Path('reports/early_warning_regime_scan')); ap.add_argument('--window',type=int,default=256); ap.add_argument('--stride',type=int,default=10)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    raw={}
    for e in ('A2','D1'): raw[e]=load_pouch(args.root,e)
    for e in ('M1','M2'): raw[e]=load_module(args.root,e)

    counts=[]; scores=[]
    for horizon in (300,600,900):
        dat={e:build(raw[e],VENT[e],args.window,args.stride,horizon) for e in raw}
        for cap in (45,50,60,70,80,100,120,150,200):
            for e,p in dat.items():
                mask=p[3] <= cap; y=p[1][mask]
                counts.append({'horizon_s':horizon,'temp_cap_c':cap,'experiment_id':e,'windows':int(mask.sum()),'positive':int(y.sum()),'negative':int(len(y)-y.sum())})
            for mode in ('temperature','force','fusion'):
                for test_e in dat:
                    train=[dat[e] for e in dat if e!=test_e]
                    res=fit_score(train,dat[test_e],mode,cap)
                    if res is not None:
                        scores.append({'horizon_s':horizon,'temp_cap_c':cap,'modality':mode,'test_experiment':test_e,**res})
    cdf=pd.DataFrame(counts); sdf=pd.DataFrame(scores)
    cdf.to_csv(args.out/'class_counts.csv',index=False); sdf.to_csv(args.out/'loso_scores.csv',index=False)

    summary=[]
    if len(sdf):
        for (h,cap,mode),g in sdf.groupby(['horizon_s','temp_cap_c','modality']):
            summary.append({'horizon_s':h,'temp_cap_c':cap,'modality':mode,'valid_folds':int(g.test_experiment.nunique()),
                            'auroc_mean':float(g.auroc.mean()),'auroc_min':float(g.auroc.min()),
                            'auprc_mean':float(g.auprc.mean()),'auprc_min':float(g.auprc.min())})
    sm=pd.DataFrame(summary).sort_values(['valid_folds','horizon_s','temp_cap_c','modality'],ascending=[False,True,True,True]) if summary else pd.DataFrame()
    sm.to_csv(args.out/'summary.csv',index=False)

    # Rank definitions with all four folds valid, favoring a difficult temperature baseline and fusion/force gain.
    candidates=[]
    if len(sm):
        for (h,cap),g in sm.groupby(['horizon_s','temp_cap_c']):
            by={r.modality:r for _,r in g.iterrows()}
            if all(m in by and by[m].valid_folds==4 for m in ('temperature','force','fusion')):
                t=by['temperature']; f=by['force']; u=by['fusion']
                candidates.append({'horizon_s':h,'temp_cap_c':cap,'temperature_auroc':t.auroc_mean,'force_auroc':f.auroc_mean,'fusion_auroc':u.auroc_mean,
                                   'force_gain_vs_temp':f.auroc_mean-t.auroc_mean,'fusion_gain_vs_temp':u.auroc_mean-t.auroc_mean,
                                   'worst_fusion_auroc':u.auroc_min})
    rank=pd.DataFrame(candidates)
    if len(rank):
        rank=rank.sort_values(['fusion_gain_vs_temp','force_gain_vs_temp','worst_fusion_auroc'],ascending=False)
    rank.to_csv(args.out/'candidate_definitions.csv',index=False)

    lines=['# Thermally subtle impending-vent regime scan','',
           'A sample is retained only if the **maximum temperature anywhere in its 256-s causal input window** is at or below the specified ceiling. This prevents an apparently strong early-warning result from being driven solely by obvious high-temperature escalation.','',
           'All reported predictive scores are leave-one-real-experiment-out; configurations without both classes in every held-out experiment are excluded from the top candidate table.','']
    if len(rank):
        lines += ['## Best fully evaluable definitions','',
                  '| horizon | T cap | T AUROC | force AUROC | fusion AUROC | force gain | fusion gain | worst fusion |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for _,r in rank.head(12).iterrows():
            lines.append(f"| {int(r.horizon_s)}s | {int(r.temp_cap_c)}°C | {r.temperature_auroc:.3f} | {r.force_auroc:.3f} | {r.fusion_auroc:.3f} | {r.force_gain_vs_temp:+.3f} | {r.fusion_gain_vs_temp:+.3f} | {r.worst_fusion_auroc:.3f} |")
    else:
        lines += ['## Result','','No scanned temperature ceiling/horizon retained both classes in all four held-out experiments.']
    lines += ['', '## Guardrail','',
              'This scan is exploratory task-definition work on four experiments. A chosen threshold/horizon must be frozen before any final model comparison, and later final claims require either external data or explicitly small-sample scope.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
