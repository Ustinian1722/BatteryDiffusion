# Virtual Vehicle k=2 augmentation protocol freeze

Date: 2026-09-12

This file freezes the RA-CDiff augmentation experiment on the eight-test Virtual Vehicle cohort before any synthetic-augmentation predictive score is computed.

## Frozen task

Identical to the already frozen TS0330 protocol:

- 120 s causal context;
- stride 10 s;
- 300 s horizon to the first package `vent_gas` marker;
- max cell-case temperature in the input <=150 C;
- complete held-out real experiment for testing.

## Primary scarcity regime

The augmentation study uses **k=2 independent real training experiments**. This is the predeclared severe destructive-data-scarcity regime from `virtual_vehicle_scarcity_protocol_freeze.md`.

For each held-out experiment, three training pairs are fixed by a cyclic schedule that depends only on experiment labels, not on model performance:

| held-out | pair 1 | pair 2 | pair 3 |
|---|---|---|---|
| A | B+C | D+E | F+G |
| B | C+D | E+F | G+H |
| C | D+E | F+G | H+A |
| D | E+F | G+H | A+B |
| E | F+G | H+A | B+C |
| F | G+H | A+B | C+D |
| G | H+A | B+C | D+E |
| H | A+B | C+D | E+F |

This produces 24 training-pair/test folds. The same exact folds are used for real-only, classical augmentation and RA-CDiff.

## Fold-safe generator protocol

For each of the 24 folds:

1. load only the two real training experiments;
2. construct frozen-task windows;
3. fit per-channel robust 1st/99th-percentile midpoint/half-range normalization on those training windows only;
4. fit RA-CDiff using only normalized training windows;
5. condition local real-anchored generation on the binary warning class inherited from the anchor;
6. apply the existing feature/OOD/amplitude/memorization gate using training windows only;
7. reject any generated raw window whose max temperature exceeds 150 C;
8. never use the held-out experiment in normalization, generator fitting, quality thresholds or synthetic selection.

Generator hyperparameters are carried over from the prior fold-safe Hanyang study rather than tuned on TS0330 predictive scores:

- training steps: **600**;
- diffusion steps: **100**;
- hidden channels: **24**;
- batch size: **64**;
- candidate windows: **160**;
- real-anchored noise-start range: **0.08-0.25** of the diffusion schedule;
- deterministic fold seed derived from held-out/pair identity.

The existing age/stage conditional implementation is reused with `age=0` for all TS0330 windows and the binary warning class stored in the `stage` slot (`0=negative`, `1=positive`). This is an implementation reuse only; no aging interpretation is assigned to TS0330.

## Fair augmentation budget

After RA-CDiff quality/task screening, let `n_syn` be the number of retained synthetic windows in that fold. Classical augmentation generates exactly `n_syn` local perturbations from the same two real training experiments, balanced across warning classes as far as source support permits. Thus classical and RA-CDiff have the same added-sample budget.

RA-CDiff retained samples are capped at the number of real training windows to prevent synthetic rows from dominating the predictor.

## Downstream learner and representations

The same frozen StandardScaler + class-weighted logistic regression is used. Predeclared representations:

- temperature;
- pressure_absolute;
- pressure_shape;
- fusion_absolute;
- fusion_shape.

## Statistical unit and decision rule

The 24 pair/test folds are not treated as 24 independent destructive experiments. For each held-out experiment, metrics are first averaged over its three fixed training pairs. Paired augmentation differences are then summarized over the **eight held-out real experiments**.

Primary metrics: AUROC and AUPRC. Secondary: F1 and balanced accuracy at threshold 0.5.

The result is considered evidence that RA-CDiff helps destructive-data scarcity only if the held-out-experiment macro gain over real-only is positive and the comparison against equal-budget classical augmentation is favorable. Negative or mixed results remain archived and will not trigger post-hoc task retuning.
