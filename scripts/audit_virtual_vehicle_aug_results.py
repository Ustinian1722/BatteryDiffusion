#!/usr/bin/env python3
"""Audit completed k=2 augmentation results before any corrective rerun.

This script does not refit models. It diagnoses synthetic class coverage and
computes held-out-experiment-level paired uncertainty for the already archived
24-fold study.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def exact_signflip_p(d: np.ndarray) -> float:
    """Two-sided exact randomization p-value for mean paired difference."""
    d=np.asarray(d,dtype=float); d=d[np.isfinite(d)]
    if len(d)==0: return np.nan
    obs=abs(float(d.mean())); vals=[]
    for signs in itertools.product((-1.0,1.0),repeat=len(d)):
        vals.append(abs(float(np.mean(d*np.asarray(signs)))))
    vals=np.asarray(vals); return float((np.sum(vals>=obs-1e-15))/len(vals))


def bootstrap_ci(d: np.ndarray,seed=20260912,b=50000):
    d=np.asarray(d,dtype=float); d=d[np.isfinite(d)]
    if not len(d): return (np.nan,np.nan)
    rng=np.random.default_rng(seed); ids=rng.integers(0,len(d),size=(b,len(d)))
    means=d[ids].mean(axis=1); return tuple(np.quantile(means,[0.025,0.975]).tolist())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('reports/virtual_vehicle_aug_utility')); args=ap.parse_args()
    root=args.root
    held=pd.read_csv(root/'heldout_macro_over_pairs.csv')
    audits=[]
    for fp in sorted((root/'fold_audits').glob('*_filter.json')):
        tag=fp.stem.replace('_filter',''); rec=json.loads(fp.read_text(encoding='utf-8'))
        genp=root/'fold_audits'/f'{tag}_generator.json'; foldp=root/'fold_audits'/f'{tag}_fold.json'
        gen=json.loads(genp.read_text(encoding='utf-8')); fold=json.loads(foldp.read_text(encoding='utf-8'))
        groups={(int(g['stage'])):(int(g['count']),int(g['sum'])) for g in gen.get('condition_group_counts',[]) if int(g.get('age',0))==0}
        audits.append({'tag':tag,'heldout':fold['heldout'],'train_experiments':'+'.join(fold['train_experiments']),
                       'real_train_windows':rec['real_train_windows'],'real_train_positive':fold['n_train_positive'],
                       'candidate_negative':groups.get(0,(0,0))[0],'accepted_negative_gate':groups.get(0,(0,0))[1],
                       'candidate_positive':groups.get(1,(0,0))[0],'accepted_positive_gate':groups.get(1,(0,0))[1],
                       'retained_negative':rec['retained_negative'],'retained_positive':rec['retained_positive'],
                       'retained_total':rec['retained_budgeted'],'acceptance_rate':gen['acceptance_rate']})
    ad=pd.DataFrame(audits); ad.to_csv(root/'synthetic_class_coverage_audit.csv',index=False)

    coverage={
        'folds':int(len(ad)),
        'folds_zero_positive_synthetic':int((ad.retained_positive==0).sum()),
        'folds_zero_negative_synthetic':int((ad.retained_negative==0).sum()),
        'folds_both_classes':int(((ad.retained_positive>0)&(ad.retained_negative>0)).sum()),
        'median_retained_positive':float(ad.retained_positive.median()),
        'median_retained_negative':float(ad.retained_negative.median()),
        'mean_positive_fraction':float((ad.retained_positive/ad.retained_total.replace(0,np.nan)).mean()),
        'mean_acceptance_rate':float(ad.acceptance_rate.mean()),
    }
    (root/'synthetic_class_coverage_summary.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')

    # Paired inference over the correct independent unit: 8 held-out experiments.
    rows=[]
    for rep in sorted(held.representation.unique()):
        h=held[held.representation==rep].set_index(['heldout','method'])
        for a,bname in [('racdiff','real_only'),('classical','real_only'),('racdiff','classical')]:
            for metric in ('auroc','auprc','f1','balanced_accuracy'):
                vals=[]
                for exp in sorted(held.heldout.unique()):
                    vals.append(float(h.loc[(exp,a),metric]-h.loc[(exp,bname),metric]))
                d=np.asarray(vals); lo,hi=bootstrap_ci(d,seed=20260912+len(rows))
                rows.append({'representation':rep,'contrast':f'{a}-{bname}','metric':metric,'n_experiments':len(d),
                             'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'ci95_lo':lo,'ci95_hi':hi,
                             'improved_experiments':int((d>0).sum()),'tied_experiments':int((np.isclose(d,0)).sum()),
                             'exact_signflip_p':exact_signflip_p(d)})
    stat=pd.DataFrame(rows); stat.to_csv(root/'paired_experiment_statistics.csv',index=False)

    focus=stat[(stat.representation=='fusion_absolute') & (stat.metric.isin(['auroc','auprc']))]
    lines=['# Post-run audit of the first k=2 augmentation study','',
           'This audit was added **after** the 24-fold result was archived. It does not alter any predictive score. Its purpose is to check whether the synthetic class composition makes the RA-CDiff/classical comparison scientifically fair.','',
           '## Synthetic class coverage','',
           f"- folds audited: **{coverage['folds']}**",
           f"- folds with both synthetic warning classes retained: **{coverage['folds_both_classes']}/{coverage['folds']}**",
           f"- folds with zero retained positive synthetic windows: **{coverage['folds_zero_positive_synthetic']}/{coverage['folds']}**",
           f"- median retained positives / negatives: **{coverage['median_retained_positive']:.1f} / {coverage['median_retained_negative']:.1f}**",
           f"- mean retained positive fraction: **{coverage['mean_positive_fraction']:.3f}**",
           '', 'The generator proposes equal candidate counts per warning class, but the existing condition-wise quality gate can reject nearly all scarce positive-class candidates. Therefore an equal-total-count classical comparator is not necessarily an equal-class-composition comparator. The first study remains a valid audit of the current RA-CDiff pipeline, but it is **not yet the final fair augmentation comparison**.','',
           '## Held-out-experiment paired statistics for fusion_absolute','',
           '| contrast | metric | mean delta | 95% experiment-bootstrap CI | improved | exact sign-flip p |','|---|---|---:|---:|---:|---:|']
    for _,r in focus.iterrows():
        lines.append(f"| {r.contrast} | {r.metric} | {r.mean_delta:+.3f} | [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] | {int(r.improved_experiments)}/8 | {r.exact_signflip_p:.4f} |")
    lines += ['', '## Decision','',
              'Preserve the first study unchanged. Before making a final augmentation claim, run one protocol-corrective experiment that enforces matched synthetic class composition for RA-CDiff and classical augmentation. This correction addresses a fairness defect revealed by the audit; it is not a task/horizon/model retuning step.']
    (root/'POSTRUN_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
