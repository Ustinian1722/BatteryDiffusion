# Fold-safe RA-CDiff augmentation utility

Frozen task: 256 s causal context, 900 s first-vent horizon, max temperature <= 120 C, stride 10 s.

Every row below is based on leave-one-real-experiment-out evaluation. The scaler and RA-CDiff generator are refit using training experiments only. Synthetic samples never enter the held-out real experiment.

Two training-data regimes are audited: 25% of eligible real training windows (scarcity stress test) and 100%.

| real fraction | method | modality | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.25 | classical | force | 0.543 | 0.155 | 0.533 | 0.186 | 0.269 |
| 0.25 | classical | fusion | 0.987 | 0.946 | 0.974 | 0.897 | 0.726 |
| 0.25 | classical | temperature | 0.955 | 0.821 | 0.930 | 0.718 | 0.821 |
| 0.25 | racdiff | force | 0.691 | 0.176 | 0.660 | 0.324 | 0.483 |
| 0.25 | racdiff | fusion | 0.972 | 0.900 | 0.922 | 0.702 | 0.717 |
| 0.25 | racdiff | temperature | 0.949 | 0.795 | 0.910 | 0.640 | 0.723 |
| 0.25 | real_only | force | 0.533 | 0.000 | 0.585 | 0.167 | 0.248 |
| 0.25 | real_only | fusion | 1.000 | 1.000 | 1.000 | 1.000 | 0.761 |
| 0.25 | real_only | temperature | 0.972 | 0.889 | 0.952 | 0.809 | 0.826 |
| 1.00 | classical | force | 0.477 | 0.000 | 0.498 | 0.167 | 0.232 |
| 1.00 | classical | fusion | 0.996 | 0.990 | 0.994 | 0.988 | 0.709 |
| 1.00 | classical | temperature | 0.967 | 0.868 | 0.949 | 0.796 | 0.829 |
| 1.00 | racdiff | force | 0.541 | 0.022 | 0.612 | 0.168 | 0.270 |
| 1.00 | racdiff | fusion | 0.994 | 0.987 | 0.989 | 0.975 | 0.665 |
| 1.00 | racdiff | temperature | 0.951 | 0.804 | 0.896 | 0.584 | 0.730 |
| 1.00 | real_only | force | 0.507 | 0.000 | 0.551 | 0.167 | 0.264 |
| 1.00 | real_only | fusion | 0.999 | 0.998 | 0.999 | 0.998 | 0.709 |
| 1.00 | real_only | temperature | 0.969 | 0.877 | 0.947 | 0.789 | 0.831 |

## Interpretation guardrail

This is still an n=4 destructive-experiment proof-of-concept. The utility question is whether training-only synthetic variants improve held-out real-experiment performance, especially in the 25% real-data regime. Window counts are not experimental replicates.
