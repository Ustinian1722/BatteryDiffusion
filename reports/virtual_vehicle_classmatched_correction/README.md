# Class-composition-matched Virtual Vehicle augmentation correction

The predictive task, 24 predeclared folds, RA-CDiff generator, task guard, feature representations and classifier are unchanged. The only correction is that the classical comparator receives exactly the same retained synthetic negative/positive counts as RA-CDiff in each fold.

| method | representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |
|---|---|---:|---:|---:|---:|---:|
| classical_classmatched | fusion_absolute | 0.955 | 0.776 | 0.948 | 0.796 | 0.488 |
| classical_classmatched | fusion_shape | 0.914 | 0.796 | 0.798 | 0.448 | 0.651 |
| classical_classmatched | pressure_absolute | 0.571 | 0.355 | 0.452 | 0.173 | 0.329 |
| classical_classmatched | pressure_shape | 0.519 | 0.218 | 0.382 | 0.141 | 0.348 |
| classical_classmatched | temperature | 0.964 | 0.714 | 0.972 | 0.775 | 0.723 |
| racdiff | fusion_absolute | 0.984 | 0.931 | 0.973 | 0.896 | 0.456 |
| racdiff | fusion_shape | 0.948 | 0.881 | 0.867 | 0.700 | 0.666 |
| racdiff | pressure_absolute | 0.514 | 0.380 | 0.380 | 0.122 | 0.311 |
| racdiff | pressure_shape | 0.541 | 0.221 | 0.397 | 0.148 | 0.379 |
| racdiff | temperature | 0.998 | 0.988 | 0.995 | 0.970 | 0.729 |
| real_only | fusion_absolute | 0.943 | 0.746 | 0.920 | 0.781 | 0.431 |
| real_only | fusion_shape | 0.913 | 0.758 | 0.804 | 0.548 | 0.672 |
| real_only | pressure_absolute | 0.814 | 0.596 | 0.701 | 0.322 | 0.434 |
| real_only | pressure_shape | 0.600 | 0.340 | 0.435 | 0.163 | 0.380 |
| real_only | temperature | 0.972 | 0.778 | 0.975 | 0.797 | 0.718 |

## Headline paired statistics: fusion_absolute (n=8 real held-out experiments)

| contrast | metric | mean delta | 95% bootstrap CI | improved | exact sign-flip p |
|---|---|---:|---:|---:|---:|
| racdiff-real_only | auroc | +0.041 | [+0.005, +0.104] | 6/8 | 0.0469 |
| racdiff-real_only | auprc | +0.053 | [+0.010, +0.109] | 7/8 | 0.0156 |
| classical-real_only | auroc | +0.013 | [+0.002, +0.024] | 6/8 | 0.1094 |
| classical-real_only | auprc | +0.028 | [+0.003, +0.062] | 6/8 | 0.0781 |
| racdiff-classical_classmatched | auroc | +0.028 | [-0.003, +0.086] | 3/8 | 0.4375 |
| racdiff-classical_classmatched | auprc | +0.025 | [-0.006, +0.077] | 3/8 | 0.6250 |

## Guardrail

This is a comparator-fairness correction, not a retuned task. Statistical n remains 8 independent destructive experiments.
