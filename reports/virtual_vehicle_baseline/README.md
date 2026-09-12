# Virtual Vehicle TS0330 real-only LOEO baseline

Frozen before scoring: 120 s causal context, 300 s horizon to the first package `vent_gas` marker, max input cell-case temperature <= 150 C, stride 10 s.

All eight BAK N21700CG-50 experiments are evaluated leave-one-real-experiment-out. No synthetic windows are used.

| representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 | mean balanced acc |
|---|---:|---:|---:|---:|---:|---:|
| fusion_absolute | 0.961 | 0.714 | 0.965 | 0.775 | 0.618 | 0.820 |
| fusion_shape | 0.943 | 0.714 | 0.874 | 0.639 | 0.727 | 0.872 |
| pressure_absolute | 0.902 | 0.746 | 0.781 | 0.350 | 0.378 | 0.667 |
| pressure_shape | 0.714 | 0.246 | 0.553 | 0.142 | 0.474 | 0.654 |
| temperature | 0.982 | 0.857 | 0.977 | 0.815 | 0.729 | 0.876 |

## Guardrail

This baseline was run only after `reports/virtual_vehicle_validation_protocol_freeze.md` was committed. Experiment count is n=8; overlapping causal windows are repeated decision points, not independent destructive tests.
