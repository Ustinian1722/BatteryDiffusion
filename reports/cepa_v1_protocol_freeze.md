# CEPA-Net v1 protocol freeze

Frozen **before any CEPA-Net v1 predictive score is inspected**.

## Scientific question

The previous LATH-Net development series showed that temperature-pressure fusion is useful for time-to-first-vent prediction, but an explicit pressure-to-temperature lag mechanism did not receive empirical support: learned lag distributions remained near-uniform and experiment-specific TTV bias dominated several held-out folds. A separate representation audit also showed that absolute pressure state carries important predictive information and should not be removed in favor of relative/differential pressure alone.

CEPA-Net therefore changes the model hypothesis rather than continuing to tune lag attention:

> Can cross-experiment event-progress alignment reduce experiment-specific TTV calibration drift while retaining both absolute thermo-pressure state and causal precursor dynamics?

CEPA = **Cross-Experiment Progress Alignment**.

## Frozen primary task

Unchanged from the method-paper benchmark:

- data: eight independent TS0330 destructive experiments;
- outer evaluation: leave one entire experiment out as test;
- validation: next experiment cyclically;
- training: remaining six experiments only;
- input: 120 s causal temperature + pressure history;
- stride: 10 s;
- target: remaining time to package-provided first `vent_gas`;
- support: `0 < tau <= 600 s`;
- thermal guard: maximum temperature in the causal input window <= 150 C;
- scaler: fit on the six training experiments only;
- no synthetic augmentation in the primary architecture experiment;
- three fixed seeds: 2026, 2027, 2028;
- statistical unit: independent destructive experiment (`n=8`), never the number of overlapping windows.

## Frozen CEPA-Net v1 architecture

### 1. Absolute-state branch

A multiscale 1-D encoder receives the standardized raw channels `[T, P]`. Absolute pressure is deliberately retained because the pre-model representation audit showed that `T + absolute P` is stronger than temperature plus relative/differential pressure alone.

### 2. Dynamic-precursor branch

A second multiscale encoder receives only causal within-window changes:

- `T - T_start`,
- first difference of T,
- `P - P_start`,
- first difference of P.

This branch is residual evidence; it does **not** replace the absolute-state branch.

### 3. State-dynamics fusion

The two latent sequences are fused using a learned dynamic gate and the interaction tensor `[state, gated_dynamic, state-gated_dynamic, state*gated_dynamic]`, followed by residual dilated temporal blocks and mean/max temporal pooling.

### 4. Event-progress embedding

The pooled representation is projected into a normalized progress embedding `z`. This embedding is explicitly regularized across different real experiments.

For a cross-experiment pair `(i,j)`, define normalized true TTV `y=tau/600`. The target progress distance is

`d_target = clip(|y_i-y_j| / 0.5, 0, 1)`.

The embedding distance is normalized cosine distance

`d_z = sqrt(max(2-2*cos(z_i,z_j), eps)) / 2`.

The cross-experiment alignment loss minimizes weighted squared error between `d_z` and `d_target` for pairs from **different training experiments only**. Pairs with similar TTV receive larger weight so equal-progress states from different destructive experiments are explicitly pulled together, while distant progress states remain separated.

### 5. Prediction heads

- primary continuous TTV regression head;
- weak monotone multi-horizon warning auxiliary head for 60/120/180/300/450 s horizons.

The warning head is auxiliary only. Headline model comparison remains continuous TTV regression.

## Frozen objective

`L = L_TTV + 0.15 L_CEPA + 0.05 L_rank + 0.05 L_warn`

where:

- `L_TTV`: Smooth-L1 on normalized TTV;
- `L_CEPA`: cross-experiment continuous progress-distance alignment;
- `L_rank`: within-experiment progress ranking for pairs separated by >60 s;
- `L_warn`: monotone multi-horizon BCE.

No loss coefficient will be changed after looking at CEPA v1 test scores.

## Frozen optimizer / selection

- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- cosine annealing to `1e-5`;
- maximum 260 epochs;
- early stopping patience 45;
- gradient clipping 1.0;
- checkpoint selected only by validation-experiment MAE;
- held-out experiment is never used for fitting, normalization, checkpoint selection, or hyperparameter choice.

## Mechanism diagnostics

For each held-out experiment, after the model is frozen, report:

1. mean dynamic-gate activation;
2. held-out progress-embedding 5-NN TTV MAE using embeddings from training experiments only;
3. mean absolute TTV gap to the nearest training embedding neighbor.

These diagnostics are post-hoc evaluation only and cannot affect checkpoint selection.

## Predeclared development gate

CEPA-Net v1 advances to the full ablation/augmentation phase only if it jointly satisfies:

1. experiment-macro MAE < **37.4 s** (current best real-only deep-model reference, LATH-Net v1);
2. worst held-out-experiment MAE < **70.1 s** (LATH-Net v1 worst-fold reference);
3. the gain is not produced by a single fold only: CEPA must beat LATH-Net v1 on at least 5 of 8 held-out experiments.

The exact-fold RF reference (mean MAE 40.42 s, worst 62.90 s) remains an additional robustness comparator, not a tuning target.

If CEPA v1 fails this gate, the next change must be motivated by the fold-level error and alignment diagnostics. We will not perform open-ended architecture or hyperparameter search.
