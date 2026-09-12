# Fold-safe RA-CDiff utility for time-to-vent regression

Frozen task: 120 s context, 0 < tau <=600 s to first package `vent_gas`, max input temperature <=150 C. Primary scarcity regime: two independent real training experiments. Synthetic continuous labels are inherited from unchanged real anchor times. Classical augmentation uses exactly the same anchors and tau labels as RA-CDiff.

| method | model | representation | mean MAE | worst MAE | mean RMSE | mean MedAE | mean R2 |
|---|---|---|---:|---:|---:|---:|---:|
| classical_anchor_matched | huber | fusion_absolute | 186.5 s | 375.3 s | 188.6 s | 184.6 s | -5.818 |
| classical_anchor_matched | huber | fusion_shape | 58.3 s | 84.0 s | 69.0 s | 55.3 s | 0.294 |
| classical_anchor_matched | huber | pressure_absolute | 205.9 s | 290.8 s | 237.0 s | 187.2 s | -6.419 |
| classical_anchor_matched | huber | pressure_shape | 123.4 s | 153.7 s | 150.8 s | 107.3 s | -1.525 |
| classical_anchor_matched | huber | temperature | 49.6 s | 80.7 s | 54.9 s | 47.2 s | 0.599 |
| classical_anchor_matched | rf | fusion_absolute | 50.7 s | 79.0 s | 57.1 s | 48.2 s | 0.617 |
| classical_anchor_matched | rf | fusion_shape | 53.0 s | 83.4 s | 59.3 s | 49.7 s | 0.584 |
| classical_anchor_matched | rf | pressure_absolute | 101.0 s | 126.9 s | 121.2 s | 86.5 s | -0.549 |
| classical_anchor_matched | rf | pressure_shape | 100.3 s | 109.7 s | 120.2 s | 92.4 s | -0.471 |
| classical_anchor_matched | rf | temperature | 52.6 s | 84.6 s | 58.5 s | 51.5 s | 0.586 |
| racdiff | huber | fusion_absolute | 179.4 s | 346.3 s | 181.4 s | 178.5 s | -5.481 |
| racdiff | huber | fusion_shape | 54.2 s | 78.1 s | 59.0 s | 52.8 s | 0.591 |
| racdiff | huber | pressure_absolute | 148.4 s | 186.2 s | 174.5 s | 134.7 s | -2.284 |
| racdiff | huber | pressure_shape | 112.6 s | 140.5 s | 135.9 s | 100.5 s | -0.982 |
| racdiff | huber | temperature | 49.7 s | 81.1 s | 52.8 s | 48.2 s | 0.630 |
| racdiff | rf | fusion_absolute | 52.1 s | 81.8 s | 57.9 s | 51.1 s | 0.606 |
| racdiff | rf | fusion_shape | 52.3 s | 83.3 s | 57.8 s | 51.1 s | 0.598 |
| racdiff | rf | pressure_absolute | 91.6 s | 113.9 s | 110.3 s | 79.2 s | -0.292 |
| racdiff | rf | pressure_shape | 100.1 s | 115.4 s | 118.5 s | 93.3 s | -0.442 |
| racdiff | rf | temperature | 52.5 s | 84.2 s | 57.5 s | 52.0 s | 0.594 |
| real_only | huber | fusion_absolute | 211.2 s | 437.6 s | 215.3 s | 208.0 s | -8.569 |
| real_only | huber | fusion_shape | 63.2 s | 87.3 s | 78.0 s | 58.4 s | 0.117 |
| real_only | huber | pressure_absolute | 235.2 s | 559.1 s | 260.2 s | 217.8 s | -13.114 |
| real_only | huber | pressure_shape | 111.1 s | 146.1 s | 132.3 s | 99.7 s | -0.855 |
| real_only | huber | temperature | 55.9 s | 81.9 s | 70.3 s | 48.9 s | 0.197 |
| real_only | rf | fusion_absolute | 52.3 s | 81.2 s | 58.1 s | 50.0 s | 0.598 |
| real_only | rf | fusion_shape | 53.2 s | 82.6 s | 59.3 s | 50.9 s | 0.579 |
| real_only | rf | pressure_absolute | 90.4 s | 121.5 s | 108.9 s | 79.5 s | -0.249 |
| real_only | rf | pressure_shape | 99.5 s | 110.7 s | 118.4 s | 89.7 s | -0.458 |
| real_only | rf | temperature | 52.7 s | 83.2 s | 57.4 s | 50.5 s | 0.596 |

## Headline paired statistics: Huber + fusion_absolute (n=8 real held-out experiments)

| contrast | metric | mean delta | 95% bootstrap CI | improved | exact sign-flip p |
|---|---|---:|---:|---:|---:|
| racdiff-real_only | mae_s | -31.8 s | [-53.4, -12.9] s | 8/8 | 0.0078 |
| racdiff-real_only | rmse_s | -33.9 s | [-56.1, -13.4] s | 8/8 | 0.0078 |
| classical-real_only | mae_s | -24.7 s | [-42.3, -8.4] s | 6/8 | 0.0469 |
| classical-real_only | rmse_s | -26.8 s | [-44.3, -9.9] s | 6/8 | 0.0469 |
| racdiff-classical_anchor_matched | mae_s | -7.1 s | [-16.5, +1.7] s | 6/8 | 0.2500 |
| racdiff-classical_anchor_matched | rmse_s | -7.1 s | [-17.3, +1.9] s | 6/8 | 0.2734 |

## Guardrail

The generator never invents an event time: each local synthetic variant inherits its unchanged real anchor tau. Statistical n remains 8 independent destructive experiments.
