# Frozen external transfer validation — Zenodo 13981390

Protocol: 256 s causal context, 900 s first-vent horizon, max input temperature <= 120 C, stride 10 s. Development data are A2/D1/M1/M2 only; NMC111/NMC811 are untouched external tests.

## External support

| experiment | published vent | windows | positive | negative | max retained T |
|---|---:|---:|---:|---:|---:|
| NMC111 | 10260 s | 624 | 0 | 624 | 119.9 C |
| NMC811 | 1350 s | 59 | 39 | 20 | 119.5 C |

## Per-experiment results

| experiment | representation | AUROC | AUPRC | F1@0.5 | balanced acc |
|---|---|---:|---:|---:|---:|
| NMC111 | temperature | nan | nan | 0.000 | 0.000 |
| NMC111 | mechanical_shape | nan | nan | 0.000 | 0.000 |
| NMC111 | fusion_shape | nan | nan | 0.000 | 0.000 |
| NMC811 | temperature | 0.997 | 0.999 | 0.886 | 0.897 |
| NMC811 | mechanical_shape | 0.764 | 0.839 | 0.625 | 0.631 |
| NMC811 | fusion_shape | 0.997 | 0.999 | 0.946 | 0.949 |

## Two-experiment descriptive macro

| representation | mean AUROC | worst AUROC | mean AUPRC | worst AUPRC | mean F1 |
|---|---:|---:|---:|---:|---:|
| fusion_shape | 0.997 | 0.997 | 0.999 | 0.999 | 0.473 |
| mechanical_shape | 0.764 | 0.764 | 0.839 | 0.839 | 0.312 |
| temperature | 0.997 | 0.997 | 0.999 | 0.999 | 0.443 |

## Interpretation guardrail

These are two independent external experiments under severe domain shift: pouch-cell/module expansion force in development versus prismatic-cell internal pressure externally, with different chemistry, SOC, heating profile and apparatus. The result is retained without retuning. Window counts are not independent experimental replicates.
