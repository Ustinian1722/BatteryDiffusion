# Frozen external transfer validation — Zenodo 13981390

Protocol: 256 s causal context, 900 s first-vent horizon, max input temperature <= 120 C, stride 10 s. Development data are A2/D1/M1/M2 only; NMC111/NMC811 are untouched external tests.

## External support

| experiment | published vent | windows | positive | negative | max retained T |
|---|---:|---:|---:|---:|---:|
| NMC111 | 10260 s | 624 | 0 | 624 | 119.9 C |
| NMC811 | 1350 s | 59 | 39 | 20 | 119.5 C |

The frozen binary definition is **not evaluable on NMC111**: under the <=120 C guard, all retained NMC111 windows remain more than 900 s before the published vent event, so the external test contains no positive class. This is a support mismatch caused by the much slower thermal-abuse trajectory, not a predictive failure. NMC111 is retained as a negative result of the preregistered task definition and is not used to retune the horizon or temperature ceiling.

## Evaluable external result: NMC811

| representation | AUROC | AUPRC | F1@0.5 | balanced acc |
|---|---:|---:|---:|---:|
| temperature | 0.997 | 0.999 | 0.886 | 0.897 |
| mechanical_shape | 0.764 | 0.839 | 0.625 | 0.631 |
| fusion_shape | 0.997 | 0.999 | **0.946** | **0.949** |

Despite a severe sensor/domain shift (development expansion force -> external internal pressure), the frozen geometry-normalized mechanical representation remains informative on NMC811 (AUROC 0.764). Fusion does not materially improve ranking over temperature alone, but it improves the fixed-threshold F1 from 0.886 to 0.946 and balanced accuracy from 0.897 to 0.949.

## NMC111 retained output

The classifier outputs on NMC111 are archived in `external_predictions.csv`, but AUROC/AUPRC are undefined because only the negative class is present. F1/balanced-accuracy values from this one-class subset must not be interpreted as ordinary binary-test performance.

## Interpretation guardrail

There are **two independent external experiments, but only one is evaluable for the frozen binary AUROC/AUPRC task**. Therefore no two-experiment macro performance claim is made. The NMC811 result is a single-experiment transfer-feasibility result; NMC111 demonstrates that a fixed clock-time horizon can lose label support under large heating-rate shifts. Window counts are not independent experimental replicates, and the frozen task is not retroactively changed.
