# LATH-Net v3 frozen benchmark

V3 was frozen before scoring after v2 exposed uniform lag weights and cross-experiment calibration error. Task/splits are unchanged; no augmentation is used.

- Experiment-macro MAE: **43.76 s**
- Worst held-out MAE: **82.19 s**
- Experiment-macro RMSE: **48.68 s**
- Experiment-macro MedAE: **42.37 s**
- Experiment-macro R2: **0.7063**
- Mean max lag weight: **0.184** (uniform reference = 0.167)
- Mean normalized lag entropy: **0.995** (uniform reference = 1.000)
- Mean pressure reliability gate: **0.479**

| heldout | MAE | RMSE | R2 | bias | max lag w | lag entropy | reliability |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS0330A | 34.15 | 40.25 | 0.8765 | 25.55 | 0.201 | 0.990 | 0.486 |
| TS0330B | 30.53 | 36.12 | 0.8866 | -12.75 | 0.173 | 1.000 | 0.479 |
| TS0330C | 42.22 | 43.42 | 0.7892 | -42.22 | 0.188 | 0.993 | 0.474 |
| TS0330D | 54.40 | 60.95 | 0.5608 | -54.40 | 0.191 | 0.995 | 0.485 |
| TS0330E | 22.75 | 29.71 | 0.9294 | 2.63 | 0.177 | 0.998 | 0.470 |
| TS0330F | 33.09 | 35.73 | 0.8574 | 32.80 | 0.180 | 0.997 | 0.483 |
| TS0330G | 50.71 | 58.36 | 0.5594 | -46.72 | 0.173 | 1.000 | 0.484 |
| TS0330H | 82.19 | 84.89 | 0.1913 | 82.19 | 0.191 | 0.992 | 0.473 |

## Multi-horizon auxiliary warning

| horizon | AUROC | AUPRC | Brier |
|---:|---:|---:|---:|
| 60 s | nan | nan | 0.0122 |
| 120 s | nan | nan | 0.0269 |
| 180 s | 1.000 | 1.000 | 0.0641 |
| 300 s | 1.000 | 0.999 | 0.1670 |
| 450 s | 0.998 | 1.000 | 0.1676 |

## Predeclared development gate

RF exact-fold reference: mean MAE 40.42 s, worst 62.90 s. V1 LATH reference: mean MAE 37.4 s. V3 proceeds to full ablation only if predictive performance and lag-mechanism diagnostics jointly justify it. Statistical n remains eight real destructive experiments.
