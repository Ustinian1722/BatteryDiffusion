# Method-direction decision after CEPA-Net v2

## Status

CEPA-Net v2 has completed the frozen 8-experiment evaluation and **failed its predeclared development gate**. Per the v2 freeze, CEPA is stopped here: no hidden-size sweep, loss-weight sweep, alternative attention block, or CEPA v3 is permitted without a new scientific rationale.

## Evidence accumulated so far

### 1. Temperature-pressure fusion is useful

The exact-fold Random Forest using absolute temperature+pressure achieves mean MAE 40.42 s, while the corresponding temperature-only representation is materially worse in the representation audit. Absolute pressure therefore carries real predictive information and should remain available to downstream models.

### 2. Explicit pressure-to-temperature lag is not the core mechanism

LATH-Net v2/v3 learned near-uniform lag weights despite explicit lag modules. The lag-aware story is therefore not supported strongly enough to be the method-paper centerpiece.

### 3. Absolute TTV calibration, not event ordering, is the dominant problem

Several deep models show large held-out experiment bias even when within-experiment warning/ranking metrics are strong. The failure pattern is experiment-specific offset/calibration drift rather than complete loss of event progression information.

### 4. Pairwise latent alignment did not solve calibration drift

CEPA v1: mean MAE 39.74 s, worst 69.62 s, 4/8 wins vs LATH v1. Its dynamic gate collapsed near 0.5 and the separate progress embedding was weakly coupled to prediction.

Predictive-CEPA v2 tied the aligned representation directly to the predictor. It still obtained mean MAE 40.16 s and only 4/8 wins vs LATH v1. Although worst MAE improved to 66.80 s and embedding 5-NN MAE improved from 53.79 s to 50.55 s, the primary accuracy criterion failed.

The v2 embedding diagnostic is especially important: nearest-training cosine similarity is approximately 0.9999 across all held-out experiments while nearest-neighbor TTV gaps remain tens of seconds (and ~97 s for TS0330A). This means cosine-direction geometry is nearly collapsed and is not a reliable scalar progress coordinate.

## Scientific implication

The next method should **not** attempt another generic latent metric-learning variant. The accumulated results support a different decomposition:

1. learn a robust, monotonic **event-progression / hazard representation** from thermo-pressure trajectories;
2. separately convert that progression state into calibrated remaining time using training experiments only.

This directly targets the observed asymmetry: event ordering/risk discrimination can be good while absolute TTV calibration drifts across destructive experiments.

## Recommended next route: Progress-to-Time Calibration (PTC)

Working name: **TPC-Hazard / Progress-Calibrated Hazard Network**.

### Stage A — shared causal progression model

From the same 120 s temperature+pressure context, predict an ordered discrete-time event distribution / cumulative hazard over the remaining 600 s. Use a monotone survival-style objective rather than direct scalar TTV regression as the primary representation-learning signal.

The output is a one-dimensional event-progress statistic such as expected normalized remaining-time rank or cumulative event probability. This statistic must be monotonic by construction and directly interpretable.

### Stage B — train-only progress-to-time calibration

Fit a low-capacity monotonic calibration map from the progression statistic to absolute TTV using **training experiments only**. The held-out experiment is never used for calibration. Candidate calibrators should be predeclared and minimal (for example isotonic regression or a monotone piecewise-linear map), with any selection performed on the validation experiment only.

This separates two tasks that the current results show behave differently:

- progression ordering / warning discrimination;
- absolute seconds calibration.

### Why this is scientifically distinct from CEPA

CEPA forces pairwise geometry in a high-dimensional latent space and hopes that a scalar TTV head inherits that geometry. PTC instead makes the progress variable explicit and one-dimensional, then calibrates it to physical time under a monotonic constraint. There is no hidden latent-distance claim.

## Alternative route

A second defensible route is a pure discrete-time survival/hazard model with TTV derived as the expected event time from the predicted survival curve. This is simpler, but it may not explicitly address cross-experiment calibration shift. Therefore it is recommended as a strong baseline/ablation inside the PTC study rather than the main new method.

## Decision point

Recommended route for the next method-development cycle: **PTC — explicit progression/hazard learning + train-only monotonic TTV calibration**.

Before implementation, freeze:

- the hazard binning;
- progress statistic;
- monotonic calibrator family;
- validation-only model/calibrator selection rule;
- primary MAE/worst-fold thresholds relative to LATH v1 and RF;
- no augmentation during architecture development.

RA-CDiff remains a secondary experiment only after the final predictor is frozen.
