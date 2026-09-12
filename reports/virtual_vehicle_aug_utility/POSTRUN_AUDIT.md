# Post-run audit of the first k=2 augmentation study

This audit was added **after** the 24-fold result was archived. It does not alter any predictive score. Its purpose is to check whether the synthetic class composition makes the RA-CDiff/classical comparison scientifically fair.

## Synthetic class coverage

- folds audited: **24**
- folds with both synthetic warning classes retained: **18/24**
- folds with zero retained positive synthetic windows: **6/24**
- median retained positives / negatives: **5.0 / 22.0**
- mean retained positive fraction: **0.266**

The generator proposes equal candidate counts per warning class, but the existing condition-wise quality gate can reject nearly all scarce positive-class candidates. Therefore an equal-total-count classical comparator is not necessarily an equal-class-composition comparator. The first study remains a valid audit of the current RA-CDiff pipeline, but it is **not yet the final fair augmentation comparison**.

## Held-out-experiment paired statistics for fusion_absolute

| contrast | metric | mean delta | 95% experiment-bootstrap CI | improved | exact sign-flip p |
|---|---|---:|---:|---:|---:|
| racdiff-real_only | auroc | +0.041 | [+0.005, +0.104] | 6/8 | 0.0469 |
| racdiff-real_only | auprc | +0.053 | [+0.010, +0.109] | 7/8 | 0.0156 |
| classical-real_only | auroc | +0.024 | [+0.004, +0.052] | 6/8 | 0.0625 |
| classical-real_only | auprc | +0.034 | [+0.006, +0.067] | 6/8 | 0.0469 |
| racdiff-classical | auroc | +0.017 | [-0.008, +0.057] | 3/8 | 0.6250 |
| racdiff-classical | auprc | +0.019 | [-0.009, +0.063] | 3/8 | 0.6875 |

## Decision

Preserve the first study unchanged. Before making a final augmentation claim, run one protocol-corrective experiment that enforces matched synthetic class composition for RA-CDiff and classical augmentation. This correction addresses a fairness defect revealed by the audit; it is not a task/horizon/model retuning step.
