# Method-paper benchmark v1

Primary architecture benchmark is **real-only** for every model. Each of the eight TS0330 destructive tests is held out once; the next test cyclically is validation and the remaining six train the model. Input is the same 120 s causal temperature+pressure history; target is 0<time-to-first-`vent_gas`<=600 s, with max input temperature <=150 C. Three frozen seeds are averaged within each held-out experiment.

| model | mean MAE | worst MAE | mean RMSE | mean MedAE | mean R2 | params |
|---|---:|---:|---:|---:|---:|---:|
| lath_net | 37.4s | 70.1s | 40.5s | 36.7s | 0.778 | 71,886 |
| attention_gru | 39.5s | 65.7s | 45.0s | 36.1s | 0.781 | 3,522 |
| tcn_transformer | 42.2s | 77.7s | 46.4s | 39.2s | 0.722 | 27,681 |
| transformer | 45.6s | 84.5s | 50.4s | 42.6s | 0.687 | 21,057 |
| tcn | 46.7s | 80.5s | 50.3s | 46.0s | 0.658 | 25,473 |
| gru | 47.7s | 68.3s | 54.4s | 44.3s | 0.686 | 9,825 |
| lstm | 54.7s | 81.2s | 61.9s | 48.6s | 0.593 | 13,089 |
| cnn_lstm | 54.7s | 80.4s | 60.3s | 54.0s | 0.583 | 11,937 |
| cnn | 56.6s | 107.0s | 61.1s | 55.7s | 0.574 | 5,537 |

## Proposed-vs-baseline paired MAE

| contrast | mean MAE advantage | 95% bootstrap CI | better experiments | sign-flip p |
|---|---:|---:|---:|---:|
| lath_net_vs_attention_gru | +2.1s | [-8.1, +11.7] | 5/8 | 0.7109 |
| lath_net_vs_cnn | +19.2s | [+7.5, +31.8] | 7/8 | 0.0234 |
| lath_net_vs_cnn_lstm | +17.4s | [+7.8, +28.5] | 7/8 | 0.0234 |
| lath_net_vs_gru | +10.3s | [-3.2, +24.6] | 5/8 | 0.2031 |
| lath_net_vs_lstm | +17.3s | [+8.5, +25.3] | 6/8 | 0.0312 |
| lath_net_vs_tcn | +9.3s | [-0.1, +20.1] | 5/8 | 0.1562 |
| lath_net_vs_tcn_transformer | +4.9s | [-3.3, +13.6] | 5/8 | 0.3203 |
| lath_net_vs_transformer | +8.2s | [-0.1, +17.3] | 5/8 | 0.1328 |

## Guardrail

RA-CDiff is excluded from this table by design. This table tests the **model contribution** only. Synthetic augmentation will be attached later as a secondary experiment after architecture/ablation are frozen. Statistical n is eight independent experiments, never the number of windows.
