# Virtual Vehicle class-composition correction protocol

Date: 2026-09-12

This protocol was frozen **after** the first k=2 augmentation study and its post-run audit, but **before** the corrective scores are computed.

## Why this correction exists

The first study matched the total number of classical and RA-CDiff synthetic windows. The later audit showed that RA-CDiff quality screening can strongly skew the retained warning-class composition; 6/24 folds retained zero positive synthetic windows. Therefore equal total synthetic count was not always equal class composition.

The correction changes **only comparator fairness**. It does not change the predictive task, data split, generator, task guard, feature representations, classifier, metric definitions, or held-out experiments.

## Frozen task

Exactly the existing Virtual Vehicle protocol is retained:

- independent experiments: TS0330A–H;
- causal context: 120 s;
- stride: 10 s;
- event: first package-provided `vent_gas` marker;
- warning horizon: 300 s;
- max temperature within input window: <=150 C;
- primary scarcity regime: 2 real training experiments;
- three predeclared training pairs per held-out experiment;
- held-out real experiment is untouched by generator/scaler/predictor fitting.

## Frozen correction

For every fold:

1. prepare the same strict train/test fold;
2. refit the same RA-CDiff generator on the two training experiments only;
3. apply the same quality and task guards to obtain the retained RA-CDiff set;
4. count retained RA-CDiff negatives and positives separately: `(n0, n1)`;
5. generate classical augmentation from real training anchors using the same local amplitude/noise transform as the first study, but require **exactly `(n0, n1)`** retained classical samples;
6. train the same downstream models;
7. evaluate on the same held-out real experiment only.

The RA-CDiff set itself is not artificially rebalanced in this correction. The purpose is to isolate augmentation-method differences from synthetic class-composition differences.

## Predeclared outputs

Headline representation: `fusion_absolute`.

Report:

- mean and worst held-out AUROC;
- mean and worst held-out AUPRC;
- experiment-level paired delta versus real-only;
- experiment-level paired delta RA-CDiff versus class-matched classical;
- 95% bootstrap CI over the 8 held-out real experiments;
- exact sign-flip p-value where applicable;
- number of held-out experiments improved.

Temperature-only, pressure-only and `fusion_shape` remain secondary diagnostics.

## Decision rule

A final claim that **RA-CDiff outperforms classical augmentation** requires a positive experiment-level mean delta and a 95% experiment-bootstrap interval that does not include zero for at least one predeclared headline ranking metric (AUROC or AUPRC). Otherwise the paper may still claim that augmentation helps relative to real-only, but not that the diffusion method is demonstrably superior to the classical comparator.

The independent statistical unit remains the real destructive experiment (`n=8`). Synthetic windows and overlapping causal windows do not increase n.
