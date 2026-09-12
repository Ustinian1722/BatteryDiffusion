# Virtual Vehicle experiment-scarcity protocol freeze

Date: 2026-09-12

The real-only 7-train/1-test LOEO baseline has been archived. Before any scarcity or augmentation score is computed, the following experiment-level scarcity protocol is frozen.

## Frozen task

Unchanged from `virtual_vehicle_validation_protocol_freeze.md`:

- 120 s causal context;
- stride 10 s;
- 300 s horizon to the first package-provided `vent_gas` marker;
- max cell-case temperature in the input <=150 C;
- eight independent TS0330A-H experiments.

## Real-data scarcity audit

For each held-out test experiment, use training-set sizes:

- **k=2 real experiments** — severe destructive-data scarcity;
- **k=4 real experiments** — intermediate scarcity;
- **k=7 real experiments** — full LOEO training support.

For k=2 and k=4, evaluate **every combination** of real training experiments available after removing the held-out test. No subset is selected by validation performance. For k=7 there is one training set per held-out experiment.

Predeclared representations are the same five real-only baselines already used:

- temperature;
- pressure_absolute;
- pressure_shape;
- fusion_absolute;
- fusion_shape.

The downstream learner remains StandardScaler + class-weighted logistic regression with the already frozen hyperparameters.

## Purpose

This audit asks whether reducing the number of independent destructive training tests—not merely the number of overlapping windows—creates a reproducible generalization gap. The result determines where augmentation has a scientifically meaningful opportunity.

## Augmentation comparison rule

If RA-CDiff is tested after this audit, the primary augmentation stress test will be **k=2**. To avoid cherry-picking train pairs while keeping diffusion compute tractable, the augmentation stage must use a deterministic predeclared pair schedule or a fixed random subset schedule independent of predictive scores, and the same training pairs must be used by real-only, classical augmentation, and RA-CDiff.

No synthetic sample is counted as an additional independent experiment.
