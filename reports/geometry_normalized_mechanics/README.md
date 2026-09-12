# Geometry-normalized mechanical representation diagnostic

Frozen task: 256 s context, 900 s first-vent horizon, max temperature <= 120 C, stride 10 s.

Hypothesis: absolute expansion-force level is not comparable between pouch-cell and module fixtures. `force_shape` subtracts each window baseline and divides by a robust within-window dynamic scale before extracting causal shape descriptors. `fusion_shape` combines ordinary temperature descriptors with these geometry-normalized mechanical descriptors.

| representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |
|---|---:|---:|---:|---:|---:|
| force_absolute | 0.507 | 0.000 | 0.551 | 0.167 | 0.264 |
| force_shape | 0.704 | 0.645 | 0.419 | 0.306 | 0.397 |
| fusion_absolute | 0.999 | 0.998 | 0.999 | 0.998 | 0.709 |
| fusion_shape | 0.995 | 0.986 | 0.989 | 0.962 | 0.820 |
| temperature | 0.969 | 0.877 | 0.947 | 0.789 | 0.831 |

## Guardrail

This is a post-freeze representation diagnostic on the same four experiments. It does not change the task horizon/temperature ceiling and is not independent external validation. A useful representation must later survive external real-experiment testing.
