# Virtual Vehicle time-to-vent real-only LOEO baseline

Frozen before scoring: 120 s causal context, target 0 < tau <= 600 s to first package `vent_gas`, max input temperature <= 150 C, stride 10 s.

All eight experiments are held out one at a time. Huber is the predeclared headline model; random forest is a secondary nonlinear diagnostic.

| model | representation | mean MAE | worst MAE | mean RMSE | mean MedAE | mean R2 |
|---|---|---:|---:|---:|---:|---:|
| huber | fusion_absolute | 50.2 s | 89.2 s | 53.7 s | 51.1 s | 0.674 |
| huber | fusion_shape | 52.5 s | 74.3 s | 68.8 s | 47.8 s | 0.310 |
| huber | pressure_absolute | 78.0 s | 123.6 s | 89.5 s | 74.8 s | 0.140 |
| huber | pressure_shape | 88.7 s | 125.9 s | 105.9 s | 84.2 s | -0.135 |
| huber | temperature | 43.3 s | 71.8 s | 51.1 s | 42.2 s | 0.685 |
| rf | fusion_absolute | 37.4 s | 65.6 s | 41.2 s | 36.7 s | 0.812 |
| rf | fusion_shape | 42.4 s | 80.7 s | 47.3 s | 39.7 s | 0.743 |
| rf | pressure_absolute | 89.9 s | 164.9 s | 108.9 s | 80.0 s | -0.275 |
| rf | pressure_shape | 83.5 s | 100.7 s | 100.8 s | 78.2 s | -0.023 |
| rf | temperature | 43.3 s | 76.7 s | 46.0 s | 44.5 s | 0.742 |

## Guardrail

The regression definition was frozen in `reports/virtual_vehicle_time_to_vent_protocol_freeze.md` after support-only auditing and before these scores. Statistical n remains 8 independent destructive experiments.
