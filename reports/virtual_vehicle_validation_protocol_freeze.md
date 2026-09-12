# Virtual Vehicle TS0330 predictive protocol freeze

Date: 2026-09-12

This file freezes the first predictive protocol on the eight-test Virtual Vehicle BAK N21700CG-50 package **before any predictor score on this cohort is computed**.

## Why a separate protocol is necessary

The earlier Hanyang-to-Zenodo external test used a 256 s context, 900 s horizon and <=120 C temperature guard. That protocol is intentionally retained unchanged, but it does not provide two-class support across faster thermal-runaway experiments: NMC111 already demonstrated the opposite extreme (slow heating, no positives under the frozen definition), while the Virtual Vehicle repeated tests are much faster.

Therefore this is a new, separately frozen **within-cohort repeated-experiment validation**, not a retroactive change to the previous confirmatory test.

## Data and independent unit

Dataset: Zenodo **18849418**, BAK N21700CG-50, 60% SOC.

Independent real experiments:

- TS0330A
- TS0330B
- TS0330C
- TS0330D
- TS0330E
- TS0330F
- TS0330G
- TS0330H

The package supplies one compact `export_TS0330X.xlsx` trajectory and one `order_TS0330X.xlsx` event table for each experiment. The first package-provided `vent_gas` marker is the event landmark. It is not inferred from the same pressure/temperature signal by our model.

## Frozen signal interface

From each compact export:

- time: column 0, milliseconds -> seconds;
- pressure: semantic channel `Pressure`, Pa -> bar for physical readability;
- temperature: semantic channel `Cell case ...`, degC.

Signals are interpolated only within measured support to a common 1 Hz grid. No post-event values enter an input window.

## Frozen task definition

The support audit was performed without fitting any predictor. Among the tested definitions, only three configurations had both classes in all eight experiments. We freeze the middle-context option:

- causal context: **120 s**;
- stride: **10 s**;
- warning horizon: **300 s** before the first package `vent_gas` marker;
- positive: `0 < t_event - t_end <= 300 s`;
- negative: `t_event - t_end > 300 s`;
- temperature guard: maximum cell-case temperature anywhere in the input window **<=150 C**;
- post-event windows excluded.

Support-only audit before scoring: all 8 experiments contain both classes; minimum per experiment is 3 positive and 18 negative windows. The 120 s context is selected over 60 s because it preserves more precursor history, and over 256 s because it retains materially more eligible windows while keeping the same minimum positive support.

## Frozen first baseline

Evaluation is **leave-one-real-experiment-out (LOEO)**. All feature scaling and model fitting use the seven training experiments only.

Predeclared representations:

1. `temperature` — causal cell-case-temperature descriptors;
2. `pressure_absolute` — causal pressure descriptors in physical scale;
3. `pressure_shape` — baseline-relative, robust within-window scale-normalized pressure-shape descriptors;
4. `fusion_absolute` — temperature + absolute pressure descriptors;
5. `fusion_shape` — temperature + normalized pressure-shape descriptors.

Predictor: StandardScaler + class-weighted logistic regression (`C=0.5`, `liblinear`, `random_state=2026`, `max_iter=3000`).

Headline feasibility metrics: AUROC and AUPRC. F1 and balanced accuracy at fixed threshold 0.5 are secondary because probability calibration is not tuned on held-out experiments.

## Augmentation study rule

RA-CDiff is evaluated only after this real-only baseline is archived. For every augmentation fold:

1. hold out the complete real test experiment first;
2. fit normalization and the generator on training experiments only;
3. generate/screen synthetic windows from training experiments only;
4. train the downstream predictor with or without augmentation;
5. evaluate only on the held-out real experiment.

The main scarcity study must reduce the number of **real training experiments**, not merely randomly discard overlapping windows. Synthetic windows never count as additional independent experiments.

## Guardrails

- No task horizon/cap/context changes after seeing predictive scores in this cohort.
- Event markers from `order_*.xlsx` are retained with their package terminology (`vent_gas`); they are not renamed as a different physical event without source evidence.
- Window-level metrics do not increase experimental n=8.
- Negative augmentation results remain part of the audit trail.
