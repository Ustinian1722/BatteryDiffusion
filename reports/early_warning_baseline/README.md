# Real-only impending-vent baseline

- Context/horizon/stride: **256s / 300s / 10s**
- Evaluation: **leave-one-real-experiment-out** across A2, D1, M1, M2.
- No synthetic windows are used in this baseline.
- Predictor: class-weighted logistic regression on causal signal descriptors.

## Macro results

| modality | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1@0.5 | mean balanced acc |
|---|---:|---:|---:|---:|---:|---:|
| force | 0.975 | 0.900 | 0.979 | 0.915 | 0.925 | 0.957 |
| fusion | 0.975 | 0.900 | 0.979 | 0.915 | 0.888 | 0.939 |
| temperature | 1.000 | 1.000 | 1.000 | 0.999 | 0.800 | 0.895 |

## Fold detail

| modality | held-out ID | AUROC | AUPRC | F1@0.5 |
|---|---|---:|---:|---:|
| temperature | A2 | 1.000 | 0.999 | 0.983 |
| temperature | D1 | 1.000 | 1.000 | 0.822 |
| temperature | M1 | 1.000 | 1.000 | 0.857 |
| temperature | M2 | 1.000 | 1.000 | 0.537 |
| force | A2 | 1.000 | 1.000 | 0.909 |
| force | D1 | 1.000 | 1.000 | 0.984 |
| force | M1 | 0.900 | 0.915 | 0.900 |
| force | M2 | 1.000 | 1.000 | 0.909 |
| fusion | A2 | 1.000 | 1.000 | 0.909 |
| fusion | D1 | 1.000 | 1.000 | 0.952 |
| fusion | M1 | 0.900 | 0.915 | 0.844 |
| fusion | M2 | 1.000 | 1.000 | 0.846 |

## Interpretation guardrail

These are feasibility results from only four independent real experiments. Window-level metric counts do not increase experimental n. The purpose is to decide whether the impending-vent task is worth developing, not to make a publishable generalization claim.
