# Fold-safe RA-CDiff utility on 8-test Virtual Vehicle cohort

Frozen task: 120 s context, 300 s horizon to the first package `vent_gas` marker, max input cell-case temperature <=150 C. Primary scarcity regime: two independent real training experiments.

Each of the eight held-out real experiments is evaluated with three predeclared training pairs. RA-CDiff is refit inside every training pair. Classical augmentation receives the same retained augmentation count. Metrics below are first averaged over the three pairs for each held-out experiment and then macro-averaged over the eight independent tests.

| method | representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |
|---|---|---:|---:|---:|---:|---:|
| classical | fusion_absolute | 0.967 | 0.853 | 0.954 | 0.833 | 0.415 |
| classical | fusion_shape | 0.924 | 0.865 | 0.811 | 0.575 | 0.655 |
| classical | pressure_absolute | 0.465 | 0.169 | 0.378 | 0.171 | 0.274 |
| classical | pressure_shape | 0.517 | 0.218 | 0.390 | 0.141 | 0.316 |
| classical | temperature | 0.995 | 0.960 | 0.989 | 0.917 | 0.738 |
| racdiff | fusion_absolute | 0.984 | 0.931 | 0.973 | 0.896 | 0.456 |
| racdiff | fusion_shape | 0.947 | 0.881 | 0.866 | 0.700 | 0.666 |
| racdiff | pressure_absolute | 0.514 | 0.380 | 0.380 | 0.122 | 0.311 |
| racdiff | pressure_shape | 0.540 | 0.221 | 0.397 | 0.148 | 0.378 |
| racdiff | temperature | 0.998 | 0.988 | 0.995 | 0.970 | 0.729 |
| real_only | fusion_absolute | 0.943 | 0.746 | 0.920 | 0.781 | 0.431 |
| real_only | fusion_shape | 0.913 | 0.758 | 0.804 | 0.548 | 0.672 |
| real_only | pressure_absolute | 0.814 | 0.596 | 0.701 | 0.322 | 0.434 |
| real_only | pressure_shape | 0.600 | 0.340 | 0.435 | 0.163 | 0.380 |
| real_only | temperature | 0.972 | 0.778 | 0.975 | 0.797 | 0.718 |

## Paired change versus real-only (n=8 held-out experiments)

| method | representation | ΔAUROC mean | AUROC improved | ΔAUPRC mean | AUPRC improved | ΔF1 mean |
|---|---|---:|---:|---:|---:|---:|
| classical | fusion_absolute | +0.024 | 6/8 | +0.034 | 6/8 | -0.016 |
| classical | fusion_shape | +0.012 | 5/8 | +0.007 | 4/8 | -0.017 |
| classical | pressure_absolute | -0.349 | 1/8 | -0.322 | 1/8 | -0.160 |
| classical | pressure_shape | -0.083 | 3/8 | -0.045 | 3/8 | -0.064 |
| classical | temperature | +0.023 | 1/8 | +0.015 | 1/8 | +0.020 |
| racdiff | fusion_absolute | +0.041 | 6/8 | +0.053 | 7/8 | +0.025 |
| racdiff | fusion_shape | +0.035 | 5/8 | +0.063 | 4/8 | -0.006 |
| racdiff | pressure_absolute | -0.301 | 0/8 | -0.320 | 0/8 | -0.123 |
| racdiff | pressure_shape | -0.060 | 2/8 | -0.039 | 2/8 | -0.002 |
| racdiff | temperature | +0.026 | 1/8 | +0.020 | 1/8 | +0.011 |

## Guardrail

The statistical unit is the held-out real experiment (n=8), not 24 training-pair folds and not overlapping windows. Synthetic windows remain local training variants and do not increase destructive-experiment sample size. Results are retained whether positive, mixed or negative.
