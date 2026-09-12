# Exact-fold RF baseline

Same frozen primary task as the method-paper benchmark: 120 s causal temperature+pressure input, 0<tau<=600 s to first package `vent_gas`, max input T<=150 C, stride 10 s. Each fold uses six training experiments, one validation experiment (unused by fixed RF hyperparameters), and one held-out test experiment.

- Mean MAE: **40.42 s**
- Worst MAE: **62.90 s**
- Mean RMSE: **44.59 s**
- Mean R2: **0.7826**

| heldout | MAE | RMSE | MedAE | R2 |
|---|---:|---:|---:|---:|
| TS0330A | 41.32 | 47.86 | 42.64 | 0.8364 |
| TS0330B | 56.57 | 58.97 | 58.26 | 0.7108 |
| TS0330C | 54.28 | 57.97 | 53.49 | 0.6293 |
| TS0330D | 17.21 | 23.28 | 12.61 | 0.9364 |
| TS0330E | 49.03 | 56.71 | 50.08 | 0.7461 |
| TS0330F | 15.15 | 18.48 | 12.22 | 0.9623 |
| TS0330G | 26.94 | 29.37 | 23.85 | 0.8921 |
| TS0330H | 62.90 | 64.09 | 62.15 | 0.5470 |
