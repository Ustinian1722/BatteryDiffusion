# Virtual Vehicle time-to-vent regression protocol freeze

Date: 2026-09-12

This protocol is frozen after a **support-only** audit and before any time-to-vent regression score is computed.

## Application

Secondary safety task:

> Given the preceding temperature + pressure history, estimate the remaining time until the package-provided first `vent_gas` event.

This complements the primary binary early-warning task: binary warning answers whether an intervention is needed soon, while time-to-event estimates the remaining mitigation time.

## Literature-guided target range

Recent TR warning studies report useful warning lead times on the order of several hundred seconds, including approximately 540 s in a 2026 transfer-learning study and >642–682 s in 2024 expansion-force studies. The support-only audit of the eight TS0330 experiments showed that a 600 s target range is fully supported while retaining at least 31 eligible windows in every real experiment under the same 120 s / 150 C interface as the existing binary benchmark.

Therefore the target range is frozen at **0 < time-to-vent <= 600 s**. The 900/1200 s alternatives are not used for the first regression benchmark even though they are support-feasible; this avoids adding a longer extrapolation range merely because it exists in some experiments.

## Frozen data interface

Dataset: Zenodo 18849418, BAK N21700CG-50, 60% SOC.

Independent experiments: TS0330A–H (`n=8`).

- causal context: **120 s**;
- stride: **10 s**;
- event: first package-provided `vent_gas` marker;
- regression target: `tau = t_event - t_window_end` in seconds;
- retain only **0 < tau <= 600 s**;
- maximum temperature anywhere in the input window: **<=150 C**;
- post-event samples excluded;
- signals interpolated to 1 Hz only within measured support.

The support audit gives at least **31 windows per real experiment** for this definition. Window count is not statistical n.

## Predeclared representations

The feature interface is inherited from the frozen binary benchmark:

1. `temperature`;
2. `pressure_absolute`;
3. `pressure_shape`;
4. `fusion_absolute`;
5. `fusion_shape`.

No future or post-event feature is permitted.

## Predeclared first predictor

The first feasibility model is deliberately simple and fixed:

- `StandardScaler`;
- `RandomForestRegressor` is **not** used as the headline because its nonlinear flexibility can obscure whether the task is intrinsically supported under n=8;
- headline predictor: **HuberRegressor** (`epsilon=1.35`, `alpha=1e-4`, `max_iter=3000`).

A secondary `RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=2026)` may be reported only as a nonlinear diagnostic, not used to redefine the task.

## Evaluation

Outer evaluation is leave-one-real-experiment-out (LOEO). All scalers and predictor fitting use only the seven real training experiments.

Headline metrics are computed per held-out experiment and then macro-averaged over `n=8`:

- MAE (s);
- RMSE (s);
- median absolute error (s);
- worst held-out-experiment MAE (s).

R2 is secondary because restricted time-to-event ranges and small per-experiment support can make it unstable.

## Augmentation rule

The first run is **real-only**. RA-CDiff is evaluated only after the real-only result is archived. Any augmentation comparison must refit the generator inside each training fold and evaluate only on the held-out real experiment.

No regression horizon, temperature cap, context length, feature representation, or model hyperparameter may be changed after viewing the first regression scores.
