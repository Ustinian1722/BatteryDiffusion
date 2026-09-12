#!/usr/bin/env python3
"""Exhaustive experiment-level scarcity audit for frozen TS0330 task.

No synthetic data. For each held-out real experiment, evaluate every k=2 and
k=4 training-experiment combination plus k=7 full LOEO support.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from remotezip import RemoteZip

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_virtual_vehicle_task_support import record_url, load_one, EXPS, HEADERS
from evaluate_virtual_vehicle_baseline import build_windows, feature_matrix

REPS=('temperature','pressure_absolute','pressure_shape','fusion_absolute','fusion_shape')


def fit_eval(ftr,ytr,fte,yte):
    m=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,solver='liblinear',random_state=2026))])
    m.fit(ftr,ytr); p=m.predict_proba(fte)[:,1]; pred=(p>=0.5).astype(int)
    return {'auroc':float(roc_auc_score(yte,p)),'auprc':float(average_precision_score(yte,p)),
            'f1':float(f1_score(yte,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('reports/virtual_vehicle_scarcity'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    data={}
    with RemoteZip(record_url(),headers=HEADERS,initial_buffer_size=1024*1024) as rz:
        for exp in EXPS:
            df,event,_=load_one(rz,exp); x,y,m=build_windows(df,event,120,10,300,150.0); data[exp]=(x,y)

    # Precompute representation-specific descriptors once per experiment.
    feats={(e,r):feature_matrix(data[e][0],r) for e in EXPS for r in REPS}
    rows=[]
    for held in EXPS:
        pool=[e for e in EXPS if e!=held]
        for k in (2,4,7):
            combos=[tuple(pool)] if k==7 else list(itertools.combinations(pool,k))
            for combo_id,combo in enumerate(combos):
                ytr=np.concatenate([data[e][1] for e in combo]); yte=data[held][1]
                # All source experiments under this frozen task contain both classes,
                # but retain explicit audit fields for each subset.
                for rep in REPS:
                    ftr=np.concatenate([feats[(e,rep)] for e in combo]); fte=feats[(held,rep)]
                    met=fit_eval(ftr,ytr,fte,yte)
                    rows.append({'heldout':held,'k_real_train_experiments':k,'combo_id':combo_id,'train_experiments':'+'.join(combo),
                                 'representation':rep,'n_train_windows':len(ytr),'n_test_windows':len(yte),
                                 'train_positive':int(ytr.sum()),'test_positive':int(yte.sum()),**met})
    detail=pd.DataFrame(rows); detail.to_csv(args.out/'all_subset_metrics.csv',index=False)

    # First average across all train subsets for each held-out experiment; then
    # macro-average across the eight independent held-out tests.
    held=detail.groupby(['k_real_train_experiments','representation','heldout']).agg(
        subset_count=('combo_id','size'),auroc_mean_over_subsets=('auroc','mean'),auroc_worst_subset=('auroc','min'),
        auprc_mean_over_subsets=('auprc','mean'),auprc_worst_subset=('auprc','min'),
        f1_mean_over_subsets=('f1','mean'),balanced_acc_mean_over_subsets=('balanced_accuracy','mean')).reset_index()
    held.to_csv(args.out/'heldout_macro_over_subsets.csv',index=False)
    macro=held.groupby(['k_real_train_experiments','representation']).agg(
        auroc_macro=('auroc_mean_over_subsets','mean'),auroc_worst_heldout=('auroc_mean_over_subsets','min'),
        auprc_macro=('auprc_mean_over_subsets','mean'),auprc_worst_heldout=('auprc_mean_over_subsets','min'),
        f1_macro=('f1_mean_over_subsets','mean'),balanced_acc_macro=('balanced_acc_mean_over_subsets','mean')).reset_index()
    macro.to_csv(args.out/'macro_metrics.csv',index=False)

    lines=['# Virtual Vehicle experiment-level scarcity audit','',
           'Frozen task: 120 s context, 300 s horizon to package `vent_gas`, max input cell-case temperature <=150 C. No synthetic data are used.','',
           'For k=2 and k=4, every training-experiment combination is evaluated after holding out the complete test experiment. Metrics are first averaged across training subsets within each held-out test and then macro-averaged over the eight held-out experiments.','',
           '| k real train experiments | representation | macro AUROC | worst held-out AUROC | macro AUPRC | worst held-out AUPRC | macro F1 |','|---:|---|---:|---:|---:|---:|---:|']
    for _,r in macro.iterrows():
        lines.append(f"| {int(r.k_real_train_experiments)} | {r.representation} | {r.auroc_macro:.3f} | {r.auroc_worst_heldout:.3f} | {r.auprc_macro:.3f} | {r.auprc_worst_heldout:.3f} | {r.f1_macro:.3f} |")
    lines += ['', '## Guardrail','',
              'The scarcity unit is the independent destructive experiment. Exhaustive subset averaging prevents choosing a favorable pair or quartet after seeing predictive performance. The k=2 result is the predefined primary opportunity for the later augmentation study.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
