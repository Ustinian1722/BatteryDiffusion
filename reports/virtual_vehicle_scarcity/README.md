# Virtual Vehicle experiment-level scarcity audit

Frozen task: 120 s context, 300 s horizon to package `vent_gas`, max input cell-case temperature <=150 C. No synthetic data are used.

For k=2 and k=4, every training-experiment combination is evaluated after holding out the complete test experiment. Metrics are first averaged across training subsets within each held-out test and then macro-averaged over the eight held-out experiments.

| k real train experiments | representation | macro AUROC | worst held-out AUROC | macro AUPRC | worst held-out AUPRC | macro F1 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | fusion_absolute | 0.942 | 0.719 | 0.906 | 0.712 | 0.565 |
| 2 | fusion_shape | 0.910 | 0.731 | 0.818 | 0.684 | 0.674 |
| 2 | pressure_absolute | 0.847 | 0.728 | 0.722 | 0.441 | 0.430 |
| 2 | pressure_shape | 0.635 | 0.390 | 0.477 | 0.214 | 0.409 |
| 2 | temperature | 0.971 | 0.773 | 0.976 | 0.812 | 0.721 |
| 4 | fusion_absolute | 0.957 | 0.720 | 0.952 | 0.766 | 0.666 |
| 4 | fusion_shape | 0.924 | 0.719 | 0.842 | 0.683 | 0.701 |
| 4 | pressure_absolute | 0.882 | 0.756 | 0.750 | 0.495 | 0.444 |
| 4 | pressure_shape | 0.671 | 0.319 | 0.519 | 0.164 | 0.445 |
| 4 | temperature | 0.976 | 0.815 | 0.978 | 0.835 | 0.727 |
| 7 | fusion_absolute | 0.961 | 0.714 | 0.965 | 0.775 | 0.618 |
| 7 | fusion_shape | 0.943 | 0.714 | 0.874 | 0.639 | 0.727 |
| 7 | pressure_absolute | 0.902 | 0.746 | 0.781 | 0.350 | 0.378 |
| 7 | pressure_shape | 0.714 | 0.246 | 0.553 | 0.142 | 0.474 |
| 7 | temperature | 0.982 | 0.857 | 0.977 | 0.815 | 0.729 |

## Guardrail

The scarcity unit is the independent destructive experiment. Exhaustive subset averaging prevents choosing a favorable pair or quartet after seeing predictive performance. The k=2 result is the predefined primary opportunity for the later augmentation study.
