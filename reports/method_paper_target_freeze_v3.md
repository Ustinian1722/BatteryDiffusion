# Method-paper target freeze v3

Date: 2026-09-12

This freeze is written **after the fully frozen v2 result was inspected and before any v3 score is computed**. V2 is retained as a negative design result; v3 makes one hypothesis-driven correction rather than broad model search.

## Why v3 is justified

Frozen LATH-Net v2 produced experiment-macro MAE 47.42 s and worst-experiment MAE 83.10 s, worse than the exact-fold Random-Forest descriptor baseline (40.42 s / 62.90 s). However, v2 retained excellent within-trajectory multi-horizon ranking at supported horizons while exhibiting large experiment-specific TTV bias.

The mechanism audit also exposed a concrete failure of the intended lag mechanism: the learned lag weights were essentially uniform (`~1/6` for every lag) and the pressure reliability gate stayed near 0.50. Thus v2 did **not actually realize the proposed precursor-lag mechanism**, despite having the corresponding layers.

Two causes are fixed before v3 scoring:

1. the v1/v2 lag score was over-smoothed/scaled. In v1, a feature-and-time mean was additionally divided by `sqrt(h)`; in v2 a learned scalar score was again divided by `sqrt(h)`. This makes already-small lag differences nearly indistinguishable;
2. absolute package-pressure level can vary by experiment/apparatus, whereas the pre-vent information of interest is mainly the within-window pressure evolution and its lead/lag relation to thermal evolution.

V3 therefore changes **representation and lag scoring only in a physically motivated way** and returns the continuous TTV objective to a direct regression head because forcing TTV to be the expectation of a 40-bin hazard distribution harmed absolute cross-experiment calibration in v2.

## Frozen primary task

Unchanged from v1/v2:

- 120 s causal input;
- synchronized cell-case temperature + package pressure only;
- first package `vent_gas` is the event landmark;
- `0 < tau <= 600 s`;
- max input temperature <=150 C;
- stride 10 s;
- outer experiment split: one test, next cyclic experiment validation, six training;
- train-only normalization;
- three fixed seeds: 2026, 2027, 2028.

No RA-CDiff is used in architecture development.

## LATH-Net v3 — Precursor-Invariant Lag-Aware Thermo-pressure Network

### M1. Causal precursor representation

No additional sensor is introduced. Deterministic causal transformations are computed internally from the same two measured channels.

Temperature branch receives:

- standardized absolute temperature;
- baseline-relative temperature `Delta T = T_t - T_start`;
- causal first difference `dT`.

Pressure branch receives:

- baseline-relative pressure `Delta P = P_t - P_start`;
- causal first difference `dP`;
- robust within-window pressure shape `Delta P / RMS(Delta P)`.

Absolute pressure is deliberately excluded from the pressure precursor branch. Pressure dynamic magnitude is retained through unnormalized `Delta P` and `dP`, while the shape channel supplies apparatus-offset/scale robustness.

### M2. Normalized cross-lag precursor correlation

Temperature and pressure precursor latents are L2-normalized over feature channels. For each causal lag in

`{0, 2, 5, 10, 20, 30} s`,

the model computes the mean feature-wise cosine agreement over valid overlapping time positions. These normalized correlations, not an arbitrarily small learned scalar, form the lag logits. A single learnable positive sharpness parameter rescales the correlation vector before softmax.

This yields one lag distribution per input window. The aligned pressure latent is the weighted combination of causally shifted pressure states.

Diagnostics are frozen:

- mean lag weights by true TTV band;
- mean maximum lag weight;
- normalized lag entropy.

A useful lag mechanism should no longer remain indistinguishable from a uniform distribution.

### M3. Reliability-gated cross-modal fusion

A sample-level reliability gate is predicted from pooled thermal state, aligned pressure state, their difference and element-wise interaction. It scales the pressure precursor contribution before fusion. The gate is diagnostic and later receives its own ablation.

### M4. Direct TTV head + monotone auxiliary warning head

Continuous TTV is the headline output and is predicted directly from the fused state. Multi-horizon cumulative warning risk at 60/120/180/300/450 s is retained as an auxiliary output, constrained to be monotone with horizon by construction.

The auxiliary risk head is **not claimed as the main novelty**. Its role is to regularize event proximity and provide warning-oriented secondary metrics without constraining the continuous TTV value to a discrete hazard expectation.

### M5. Within-experiment progress ranking

Same as before: within a real training experiment, pairs separated by at least 60 s of true remaining time are ordered so the closer-to-vent window should not receive a longer predicted TTV.

## Frozen v3 loss

With TTV normalized by 600 s:

`L = L_SmoothL1 + 0.10 L_monotone-warning + 0.10 L_progress-ranking`.

No test experiment is used for model selection or calibration.

## Success criterion before ablation

V3 is considered worth taking into the full ablation stage only if it satisfies both practical criteria:

1. experiment-macro MAE is below the exact-fold RF baseline of 40.42 s and competitive with or better than v1 LATH-Net (37.4 s);
2. the lag diagnostic is materially non-uniform in aggregate, i.e. mean max lag weight is meaningfully above 1/6 and/or lag entropy is measurably below the uniform maximum.

Worst-experiment MAE <=62.90 s (RF) is a strong robustness target but not an absolute gate if mean performance and mechanism evidence are clearly better.

If v3 still fails both predictive and mechanism criteria, further unconstrained architecture search is not justified; the project should stop for strategic reconsideration.

## Comparison hierarchy if v3 succeeds

Main real-only table:

RF descriptor baseline + CNN + LSTM + GRU + TCN + Transformer + CNN-LSTM + Attention-GRU + TCN-Transformer + LATH-Net v3.

Predeclared v3 ablations:

- Full v3;
- w/o pressure;
- w/o precursor transform (absolute standardized T/P);
- w/o lag alignment (synchronous fusion);
- w/o reliability gate;
- w/o auxiliary monotone warning loss;
- w/o progress ranking.

Only after this architecture/ablation stage is frozen will RA-CDiff be attached as a secondary scarcity-training experiment.
