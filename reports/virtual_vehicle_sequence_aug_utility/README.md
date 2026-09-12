# Domain-standard sequence-model validation of RA-CDiff

Frozen task and 24 k=2 scarcity folds are unchanged: 120 s context, 300 s to first package `vent_gas`, max input temperature <=150 C. Training normalization is fit on real training windows only. Classical augmentation uses the exact retained real anchors and labels used by RA-CDiff. CNN/GRU architecture, optimizer and 100-epoch schedule were frozen before scores were inspected.

| method | architecture | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |
|---|---|---:|---:|---:|---:|---:|
| classical_anchor_matched | cnn | 0.875 | 0.667 | 0.825 | 0.556 | 0.071 |
| racdiff | cnn | 0.875 | 0.667 | 0.825 | 0.556 | 0.071 |
| real_only | cnn | 0.938 | 0.833 | 0.911 | 0.732 | 0.071 |
| classical_anchor_matched | gru | 1.000 | 0.996 | 0.997 | 0.972 | 0.363 |
| racdiff | gru | 1.000 | 0.996 | 0.997 | 0.972 | 0.386 |
| real_only | gru | 1.000 | 0.996 | 0.997 | 0.972 | 0.366 |

## Paired experiment-level contrasts

| architecture | contrast | metric | mean delta | 95% bootstrap CI | improved | exact sign-flip p |
|---|---|---|---:|---:|---:|---:|
| cnn | racdiff-real_only | auroc | -0.062 | [-0.146, +0.000] | 0/8 | 0.5000 |
| cnn | racdiff-real_only | auprc | -0.086 | [-0.203, +0.000] | 0/8 | 0.5000 |
| cnn | classical-real_only | auroc | -0.062 | [-0.146, +0.000] | 0/8 | 0.5000 |
| cnn | classical-real_only | auprc | -0.086 | [-0.203, +0.000] | 0/8 | 0.5000 |
| cnn | racdiff-classical | auroc | +0.000 | [+0.000, +0.000] | 0/8 | 1.0000 |
| cnn | racdiff-classical | auprc | +0.000 | [+0.000, +0.000] | 0/8 | 1.0000 |
| gru | racdiff-real_only | auroc | +0.000 | [+0.000, +0.000] | 0/8 | 1.0000 |
| gru | racdiff-real_only | auprc | +0.000 | [+0.000, +0.000] | 0/8 | 1.0000 |
| gru | classical-real_only | auroc | +0.000 | [+0.000, +0.000] | 0/8 | 1.0000 |
| gru | classical-real_only | auprc | +0.000 | [+0.000, +0.000] | 1/8 | 1.0000 |
| gru | racdiff-classical | auroc | +0.000 | [+0.000, +0.000] | 0/8 | 1.0000 |
| gru | racdiff-classical | auprc | -0.000 | [-0.000, +0.000] | 0/8 | 1.0000 |

## Guardrail

The statistical unit is the held-out real experiment (n=8). These standard sequence learners are an architecture-robustness check, not a new model-search stage. Synthetic windows never increase experimental n.
