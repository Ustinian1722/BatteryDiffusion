# CEPA-Net v1 frozen benchmark

Task/splits/loss were frozen in `reports/cepa_v1_protocol_freeze.md` before scoring. No augmentation is used.

- Experiment-macro MAE: **39.74 s**
- Worst held-out MAE: **69.62 s**
- Experiment-macro RMSE: **43.51 s**
- Experiment-macro MedAE: **39.44 s**
- Experiment-macro R2: **0.7420**
- Mean dynamic gate: **0.500**
- Mean progress-embedding 5-NN MAE: **53.79 s**
- Mean nearest-embedding TTV gap: **56.17 s**

| heldout | MAE | RMSE | R2 | bias | dyn gate | embed 5NN MAE | NN TTV gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS0330A | 64.68 | 70.44 | 0.6170 | 63.39 | 0.501 | 97.93 | 102.80 |
| TS0330B | 10.25 | 12.25 | 0.9871 | -1.15 | 0.500 | 38.40 | 40.45 |
| TS0330C | 27.22 | 27.92 | 0.9049 | -20.36 | 0.499 | 35.45 | 39.43 |
| TS0330D | 11.72 | 15.03 | 0.9731 | -7.19 | 0.499 | 32.44 | 35.55 |
| TS0330E | 31.23 | 39.62 | 0.8754 | 30.99 | 0.499 | 57.34 | 56.09 |
| TS0330F | 39.01 | 45.65 | 0.7466 | 37.99 | 0.501 | 21.26 | 30.44 |
| TS0330G | 64.20 | 65.20 | 0.4685 | -64.20 | 0.500 | 60.62 | 56.85 |
| TS0330H | 69.62 | 71.96 | 0.3634 | 68.77 | 0.498 | 86.86 | 87.76 |

## Frozen-reference comparison

- CEPA beats LATH-Net v1 on **4/8** held-out experiments.
- LATH-Net v1 reference: mean MAE **37.39 s**, worst **70.10 s**.
- Exact-fold RF reference: mean MAE **40.42 s**, worst **62.90 s**.

| reference | CEPA mean MAE advantage | 95% bootstrap CI | CEPA wins | exact sign-flip p |
|---|---:|---:|---:|---:|
| lath_v1 | -2.36 s | [-8.68, +3.49] | 4/8 | 0.5234 |
| attention_gru | -0.27 s | [-13.54, +10.37] | 5/8 | 0.9844 |

## Predeclared development gate

**FAIL**: mean MAE <37.4 s, worst MAE <70.1 s, and CEPA beats LATH-Net v1 on at least 5/8 held-out experiments.

Statistical n is eight independent destructive experiments; overlapping windows are never treated as independent samples.
