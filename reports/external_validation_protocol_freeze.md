# External validation protocol freeze

Date: 2026-09-12

This file freezes the first external predictive validation **before external predictive scores are computed**.

## Development data

Train on all four source-paper event-labeled Hanyang experiments:

- A2 — pouch cell, fresh;
- D1 — pouch cell, aged;
- M1 — pouch module, fresh;
- M2 — pouch module, aged.

The task definition and mechanical representation were selected using these development experiments only.

## External test data

Two independent experiments from Zenodo record **13981390** are reserved as untouched external tests:

- NMC111 prismatic cell; published first vent = **10,260 s**;
- NMC811 prismatic cell; published first vent = **1,350 s**.

The companion source paper provides these first-vent landmarks. They are used only to construct external test labels and are not used for predictor fitting or representation selection.

## Frozen prediction task

- causal context: **256 s**;
- stride: **10 s**;
- positive label: `0 < t_vent - t_end <= 900 s`;
- exclude post-vent windows;
- thermally subtle guard: maximum temperature anywhere in the causal input window must be **<= 120 C**.

This is identical to the frozen development task.

## Frozen representations

1. `temperature`: ordinary causal descriptors of temperature;
2. `mechanical_shape`: per-window baseline-relative, robust-scale-normalized mechanical shape descriptors;
3. `fusion_shape`: temperature descriptors concatenated with `mechanical_shape`.

The shape representation is used because the development-only diagnostic showed that absolute mechanical level does not transfer between cell and module fixtures, whereas baseline-relative shape substantially improved worst-fold mechanical transfer. No external result may be used to change this representation in the current test.

The mechanical sensor changes from expansion force in the Hanyang pouch/module development set to internal pressure in the external prismatic set. Therefore the external experiment is explicitly a **cross-form-factor, cross-apparatus, cross-mechanical-sensor domain-shift test**, not an in-distribution replication.

## Frozen predictor

- StandardScaler fit on development feature rows only;
- class-weighted logistic regression;
- `C=0.5`, `solver=liblinear`, `random_state=2026`, `max_iter=3000`;
- no external calibration or threshold selection.

## Metrics and interpretation

Headline descriptive metrics per external experiment:

- AUROC;
- AUPRC;
- F1 at fixed probability threshold 0.5;
- balanced accuracy at fixed threshold 0.5.

AUROC/AUPRC are primary because probability calibration can shift across apparatus/domain. There are only **two independent external experiments**, so the result is evidence of transfer feasibility, not a population-level generalization estimate.

## Leakage guardrails

1. External signal values never contribute to training-feature scaling.
2. External labels never contribute to model fitting, model selection, augmentation or representation selection.
3. No synthetic data are used in this first external test.
4. The two external experiments remain separate; window counts do not increase independent experimental n.
5. Results are retained whether positive or negative; no task horizon, temperature ceiling or representation is retuned after viewing them.
