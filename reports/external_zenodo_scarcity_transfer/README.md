# Frozen scarcity transfer to Zenodo 13981390

Two real prismatic NMC experiments (NMC111 and NMC811) are untouched external/domain-shift tests. Eight predeclared two-experiment Virtual Vehicle source pairs are used only to quantify sensitivity to which scarce source experiments are available. External data do not enter scaling, RA-CDiff fitting, augmentation screening, classifier fitting, task definition or threshold selection.

The primary cross-domain representation (`fusion_shape`) was frozen before scoring because pressure offset/gain can differ across apparatus.

## External-experiment macro summary

| method | representation | mean AUROC | worst external AUROC | mean AUPRC | worst external AUPRC |
|---|---|---:|---:|---:|---:|
| classical_anchor_matched | fusion_absolute | 0.447 | 0.428 | 0.029 | 0.006 |
| racdiff | fusion_absolute | 0.472 | 0.454 | 0.039 | 0.010 |
| real_only | fusion_absolute | 0.390 | 0.345 | 0.067 | 0.005 |
| classical_anchor_matched | fusion_shape | 0.654 | 0.628 | 0.367 | 0.333 |
| racdiff | fusion_shape | 0.706 | 0.653 | 0.386 | 0.368 |
| real_only | fusion_shape | 0.663 | 0.640 | 0.317 | 0.179 |
| classical_anchor_matched | pressure_absolute | 0.518 | 0.510 | 0.061 | 0.011 |
| racdiff | pressure_absolute | 0.398 | 0.383 | 0.099 | 0.068 |
| real_only | pressure_absolute | 0.673 | 0.605 | 0.150 | 0.019 |
| classical_anchor_matched | pressure_shape | 0.636 | 0.625 | 0.222 | 0.155 |
| racdiff | pressure_shape | 0.703 | 0.689 | 0.218 | 0.177 |
| real_only | pressure_shape | 0.702 | 0.643 | 0.227 | 0.114 |
| classical_anchor_matched | temperature | 0.805 | 0.735 | 0.559 | 0.524 |
| racdiff | temperature | 0.870 | 0.855 | 0.742 | 0.712 |
| real_only | temperature | 0.861 | 0.840 | 0.529 | 0.480 |

## Primary representation by external experiment

| external | method | mean AUROC over 8 source pairs | worst source-pair AUROC | mean AUPRC | worst source-pair AUPRC |
|---|---|---:|---:|---:|---:|
| NMC111 | classical_anchor_matched | 0.680 | 0.017 | 0.333 | 0.003 |
| NMC111 | racdiff | 0.758 | 0.029 | 0.368 | 0.003 |
| NMC111 | real_only | 0.685 | 0.212 | 0.179 | 0.004 |
| NMC811 | classical_anchor_matched | 0.628 | 0.034 | 0.401 | 0.032 |
| NMC811 | racdiff | 0.653 | 0.034 | 0.405 | 0.032 |
| NMC811 | real_only | 0.640 | 0.130 | 0.454 | 0.035 |

## Guardrail

The independent external sample size is n=2. Eight source-pair fits are nuisance/sensitivity replications, not additional external experiments. No inferential significance test is reported and no external result is used to retune the frozen task or representation.
