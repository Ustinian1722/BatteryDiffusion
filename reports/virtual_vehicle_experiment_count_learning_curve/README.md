# Experiment-count learning curve for RA-CDiff

The prediction task, logistic learner, fusion representation, RA-CDiff configuration and exact-anchor classical comparator were frozen before scoring. The same eight held-out real experiments are paired across k=2,4,6; only the number of independent real source experiments changes.

## Primary fusion results

| real train experiments k | method | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC |
|---:|---|---:|---:|---:|---:|
| 2 | classical_anchor_matched | 0.976 | 0.897 | 0.962 | 0.847 |
| 2 | racdiff | 0.987 | 0.893 | 0.980 | 0.840 |
| 2 | real_only | 0.938 | 0.750 | 0.903 | 0.660 |
| 4 | classical_anchor_matched | 0.965 | 0.750 | 0.967 | 0.782 |
| 4 | racdiff | 0.992 | 0.976 | 0.980 | 0.933 |
| 4 | real_only | 0.973 | 0.815 | 0.971 | 0.800 |
| 6 | classical_anchor_matched | 0.958 | 0.714 | 0.961 | 0.775 |
| 6 | racdiff | 0.995 | 0.985 | 0.988 | 0.959 |
| 6 | real_only | 0.957 | 0.714 | 0.959 | 0.775 |

## RA-CDiff paired gain versus real-only

| k | metric | mean delta | 95% bootstrap CI | improved experiments | exact sign-flip p |
|---:|---|---:|---:|---:|---:|
| 2 | auprc | +0.077 | [+0.006, +0.168] | 4/8 | 0.1250 |
| 2 | auroc | +0.049 | [+0.003, +0.111] | 4/8 | 0.1250 |
| 4 | auprc | +0.009 | [-0.022, +0.049] | 2/8 | 0.8750 |
| 4 | auroc | +0.019 | [-0.006, +0.061] | 2/8 | 0.7500 |
| 6 | auprc | +0.030 | [+0.001, +0.081] | 3/8 | 0.2500 |
| 6 | auroc | +0.038 | [+0.000, +0.108] | 3/8 | 0.2500 |

## Scarcity-hypothesis diagnostic

- AUROC gain decreases monotonically from k=2 to k=6: **False**.
- This is a predeclared descriptive mechanism check, not a license to retune k-specific augmentation ratios or generator settings.

## Guardrail

Independent statistical n is eight held-out destructive experiments at every k. Overlapping windows and synthetic variants do not increase n.
