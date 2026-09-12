# Downstream application feasibility audit

## Candidate selected for first predictive pilot

**Impending-vent early warning** using a causal temperature + mechanical window.

Label definition for a real window ending at time `t`:

`y=1` if `0 < t_vent - t <= H`; otherwise `y=0`, with all post-vent windows excluded.

The source paper explicitly reports `t_vent` for A2, D1, M1 and M2, so these labels do not need to be inferred from the same input signal.

### Recommended first protocol

- context window: **256 s** (matches the current RA-CDiff window length);
- stride for predictive dataset construction: **10 s**;
- warning horizon: **300 s**;
- evaluation unit: **held-out real experiment**, never random windows;
- headline metrics: AUROC, AUPRC, F1, sensitivity at fixed false-alarm rate, and warning lead time;
- comparisons: temperature-only, mechanical-only, temperature+mechanical; each with real-only vs classical augmentation vs RA-CDiff.

### Real-window support at 256 s / 300 s

| ID | level | aging | pre-vent windows | positive | negative | positive fraction |
|---|---|---|---:|---:|---:|---:|
| A2 | cell | fresh | 156 | 30 | 126 | 19.2% |
| D1 | cell | aged | 145 | 30 | 115 | 20.7% |
| M1 | module | fresh | 189 | 30 | 159 | 15.9% |
| M2 | module | aged | 132 | 30 | 102 | 22.7% |

## Why this is preferable to the other obvious tasks

- SOH regression is not supported because SOH has only a few discrete experiment-level values.
- Peak-temperature/peak-force regression has effectively one target per experiment, so generative windows do not create independent target observations.
- Shape-stage classification would be partly circular because the current stage labels are derived from the temperature trajectory itself.
- Time-to-vent / impending-vent prediction uses an externally reported physical event landmark and creates many causal pre-event decision points while preserving experiment-blocked evaluation.

## Important limitation

There are still only four experiments with explicitly published vent times in the force-based pouch/module subset. A publishable predictive claim should therefore either (a) add compatible external TR data, or (b) frame this as a small-sample proof-of-concept with experiment-level resampling and very conservative claims.
