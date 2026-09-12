# LATH-Net v2 frozen benchmark

Protocol frozen before scores: 120 s causal temperature+pressure context, 0<time-to-first-`vent_gas`<=600 s, max input temperature <=150 C. One experiment is test, next cyclic experiment validation, six training. Three fixed seeds are averaged within held-out experiment. The only post-launch numerical fix clips float32 CDF values such as 1.000000119 to [0,1] before sklearn metrics; model training and predictions are unchanged.

## TTV headline metrics

- Experiment-macro MAE: **47.42 s**
- Worst held-out-experiment MAE: **83.10 s**
- Experiment-macro RMSE: **50.88 s**
- Experiment-macro MedAE: **47.37 s**
- Experiment-macro R2: **0.6883**

| held-out | MAE | RMSE | MedAE | R2 | bias |
|---|---:|---:|---:|---:|---:|
| TS0330A | 70.03 | 76.10 | 78.92 | 0.5849 | 46.61 |
| TS0330B | 21.38 | 24.93 | 22.05 | 0.9440 | -15.56 |
| TS0330C | 37.69 | 39.56 | 35.45 | 0.8274 | -37.69 |
| TS0330D | 25.41 | 28.69 | 26.18 | 0.9008 | -4.72 |
| TS0330E | 32.75 | 39.77 | 27.42 | 0.8743 | 30.44 |
| TS0330F | 40.12 | 44.15 | 38.11 | 0.7751 | 40.00 |
| TS0330G | 68.89 | 69.95 | 68.61 | 0.3867 | -68.89 |
| TS0330H | 83.10 | 83.93 | 82.26 | 0.2134 | 83.10 |

## Multi-horizon risk

| horizon | AUROC | AUPRC | Brier |
|---:|---:|---:|---:|
| 60 s | nan | nan | 0.0019 |
| 120 s | nan | nan | 0.0073 |
| 180 s | 1.000 | 1.000 | 0.0371 |
| 300 s | 0.985 | 0.983 | 0.1158 |
| 450 s | 0.979 | 0.993 | 0.1160 |

## Guardrail

Statistical unit is the independent destructive experiment (n=8). No augmentation is used. The original failed run is retained in Actions as an auditable numerical-metric failure, not deleted or overwritten.
