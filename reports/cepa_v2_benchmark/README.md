# Predictive-CEPA (CEPA-Net v2) frozen benchmark

V2 was frozen after v1 exposed an inactive dynamic gate and a progress projection decoupled from the predictor. No augmentation is used.

- Experiment-macro MAE: **40.16 s**
- Worst held-out MAE: **66.80 s**
- Experiment-macro RMSE: **44.53 s**
- Experiment-macro MedAE: **38.78 s**
- Experiment-macro R2: **0.7494**
- Mean predictive-bottleneck 5-NN MAE: **50.55 s**
- Mean nearest-embedding TTV gap: **52.38 s**

| heldout | MAE | RMSE | R2 | bias | embed 5NN MAE | NN TTV gap | NN cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS0330A | 66.54 | 73.37 | 0.5990 | 60.04 | 90.57 | 97.06 | 0.9999 |
| TS0330B | 27.12 | 29.84 | 0.9250 | -20.27 | 41.00 | 43.99 | 0.9999 |
| TS0330C | 27.65 | 29.06 | 0.9015 | -27.62 | 42.73 | 36.98 | 0.9999 |
| TS0330D | 16.39 | 19.55 | 0.9416 | -13.86 | 22.22 | 27.41 | 1.0000 |
| TS0330E | 27.72 | 35.43 | 0.8999 | 26.56 | 50.80 | 47.92 | 0.9999 |
| TS0330F | 29.01 | 35.07 | 0.8417 | 27.23 | 20.30 | 25.24 | 0.9999 |
| TS0330G | 66.80 | 70.13 | 0.3841 | -66.80 | 59.48 | 58.32 | 0.9999 |
| TS0330H | 60.09 | 63.77 | 0.5027 | 58.98 | 77.27 | 82.13 | 0.9999 |

## Frozen-reference comparison

- V2 beats LATH-Net v1 on **4/8** held-out experiments.
- LATH-Net v1 mean/worst MAE: **37.39/70.10 s**.
- RF exact-fold mean/worst MAE: **40.42/62.90 s**.
- CEPA v1 mean/worst MAE: **39.74/69.62 s**; mean embedding 5-NN MAE **53.79 s**.

| reference | V2 mean MAE advantage | 95% bootstrap CI | V2 wins | sign-flip p |
|---|---:|---:|---:|---:|
| lath_v1 | -2.78 s | [-9.52, +3.82] | 4/8 | 0.4609 |
| attention_gru | -0.69 s | [-10.43, +6.55] | 4/8 | 0.9062 |
| cepa_v1 | -0.42 s | [-6.32, +4.80] | 3/8 | 0.9141 |

## Predeclared v2 gate

**FAIL**: macro MAE <37.39 s; worst MAE <69.62 s; >=5/8 wins vs LATH-Net v1; predictive-bottleneck 5-NN MAE <53.79 s.

If this gate fails, CEPA is not to be tuned further by hidden-size/loss-weight/attention search. Statistical n remains eight independent destructive experiments.
