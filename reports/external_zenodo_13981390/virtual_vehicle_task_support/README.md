# External Zenodo 13981390 support under frozen Virtual Vehicle tasks

This is a support-only audit. No predictor is fit and no task parameter is selected from external predictive performance. Published first-vent times are used as event landmarks.

Frozen interfaces checked: classification = 120 s context / 300 s horizon / max input T <=150 C; TTE = 120 s context / 0<tau<=600 s / max input T <=150 C; stride = 10 s.

| experiment | cls positive | cls negative | cls tau range | TTE windows | TTE tau range |
|---|---:|---:|---:|---:|---:|
| NMC111 | 5 | 984 | 260-10140 s | 35 | 260-600 s |
| NMC811 | 5 | 94 | 251-1231 s | 35 | 251-591 s |

- Classification fully evaluable on both experiments: **True**
- TTE support present on both experiments: **True**

## Guardrail

These experiments differ in chemistry, form factor, SOC, heating program and apparatus from the Virtual Vehicle cohort. If predictive transfer is run, they remain untouched external/domain-shift tests and contribute nothing to scaling, generator fitting, threshold selection or task definition.
