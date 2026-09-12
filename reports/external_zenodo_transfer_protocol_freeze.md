# Frozen cross-domain external validation protocol — Zenodo 13981390

Date: 2026-09-12

This protocol is committed **before any predictive score on NMC111/NMC811 is computed**.

## Purpose

Test whether the few-shot warning pipeline developed on repeated 21700 experiments transfers to genuinely different real thermal-runaway experiments. The external tests differ in chemistry, form factor, SOC, heating profile and apparatus, so this is a domain-shift case study rather than homogeneous replication.

## External tests

- NMC111 prismatic experiment, published first vent = 10,260 s;
- NMC811 prismatic experiment, published first vent = 1,350 s.

The published vent landmarks are source-provided. A prior signal audit found independent pressure-drop landmarks within about 1–2 s, but those detected landmarks are **not** used as labels.

Support-only auditing was performed before this freeze. Under the already frozen Virtual Vehicle task, both external experiments contain both classes:

- 120 s causal context;
- stride 10 s;
- positive if 0 < tau_to_vent <=300 s;
- negative if tau_to_vent >300 s;
- max input temperature <=150 C;
- post-vent windows excluded.

NMC111 has 5 positive / 984 negative retained windows; NMC811 has 5 positive / 94 negative retained windows.

## Source training scarcity protocol

Use only two independent Virtual Vehicle BAK N21700CG-50 experiments for each source model. Eight predeclared adjacent/cyclic source pairs are evaluated:

- AB, BC, CD, DE, EF, FG, GH, HA.

For each source pair independently:

1. construct the frozen 120 s / 300 s / 150 C causal warning windows;
2. fit all feature scaling on those two source experiments only;
3. fit RA-CDiff on those two source experiments only;
4. generate/screen local synthetic variants from source-training anchors only;
5. create an anchor-matched classical augmentation comparator from exactly the same retained real anchors and labels;
6. fit the frozen class-weighted logistic classifier (`C=0.5`, liblinear, random_state=2026);
7. evaluate without adaptation on both external real experiments.

The external NMC111/NMC811 data may not affect normalization, generator fitting, augmentation screening, classifier fitting, threshold selection or representation selection.

## Frozen representations

All representations were defined before external scoring:

- `temperature`;
- `pressure_absolute`;
- `pressure_shape`;
- `fusion_absolute`;
- `fusion_shape`.

Because pressure zero/baseline and sensor calibration can shift across apparatus, **`fusion_shape` is the primary cross-domain representation**. `fusion_absolute` is a secondary consistency check and remains the main within-cohort representation from the earlier study. Temperature-only is the key modality comparator.

## Methods compared

- real-only;
- exact-anchor-matched classical local augmentation;
- RA-CDiff.

Synthetic data never enter external test sets.

## Metrics and statistical unit

Primary metrics: AUROC and AUPRC per external experiment. F1/balanced accuracy at threshold 0.5 are diagnostic only because no external calibration is permitted.

Results are first averaged across the eight predeclared source-pair choices for each external experiment. The number of independent external experiments is **n=2**; eight source-pair fits do not turn this into n=16. No significance claim will be made from the two external experiments.

## Decision rule

This external test is supportive, not required to rescue the main within-cohort result. A positive transfer result strengthens the generalization story. A negative result is retained and interpreted as domain shift, not followed by retuning the external task or feature representation.
