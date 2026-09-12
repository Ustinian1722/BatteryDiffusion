# LATH-Net development diagnosis: v2 -> v3

Date: 2026-09-12

This note records the development decision **before v3 scores** so later model changes remain auditable rather than result-driven architecture search.

## Fixed references

Under the same 120 s / 600 s TTV task and 6-train/1-validation/1-test experiment protocol:

- exact-fold Random Forest on causal T/P descriptors: mean MAE **40.42 s**, worst **62.90 s**, mean R2 **0.7826**;
- exploratory LATH-Net v1 direct-TTV architecture: mean MAE **37.4 s**, worst **70.1 s**, mean R2 **0.778**;
- frozen LATH-Net v2 unified discrete-hazard architecture: mean MAE **47.42 s**, worst **83.10 s**, mean R2 **0.6883**.

Thus v1 shows that a specialized thermo-pressure architecture can be competitive with a strong descriptor baseline, but v2's stricter unified hazard formulation damages cross-experiment absolute TTV calibration.

## V2 error structure

V2 held-out bias is strongly experiment dependent:

- A: +46.61 s;
- B: -15.56 s;
- C: -37.69 s;
- D: -4.72 s;
- E: +30.44 s;
- F: +40.00 s;
- G: -68.89 s;
- H: +83.10 s.

Several folds have `|bias| ~= MAE`, indicating predominantly one-directional error rather than random local noise. In contrast, the multi-horizon ranking remains very strong at supported horizons (macro AUROC 1.000 at 180 s, 0.985 at 300 s, 0.979 at 450 s). This pattern is consistent with learning **event progression/order** while failing to transfer the absolute time scale across experiments.

## Mechanism failure

The intended v2 lag mechanism did not specialize:

- each mean lag weight is approximately 0.1666-0.1668 for the six-lag bank;
- normalized distribution is therefore effectively uniform;
- mean reliability gate remains ~0.50 across TTV bands.

This means the architecture contained a lag module but did not empirically use it. No novelty claim should be made from v2.

## Root-cause hypothesis

Two independent design issues are plausible and testable:

1. **lag-logit attenuation**: v1 averages latent products over feature/time and then divides by `sqrt(h)`; v2's scalar scoring network is also divided by `sqrt(h)`. Both operations compress lag-score differences and favor uniform softmax weights;
2. **pressure-domain offset**: package pressure absolute level is apparatus/test dependent. Absolute pressure can dominate global train normalization even though the transferable precursor information is pressure rise/shape/derivative within the causal window.

## Single frozen correction

V3 therefore uses:

- internally derived `T_abs`, `Delta T`, `dT`;
- internally derived `Delta P`, `dP`, and RMS-normalized pressure shape;
- normalized latent cross-correlation over the same causal lag bank `{0,2,5,10,20,30}s` with a learnable positive sharpness parameter, without the erroneous extra score attenuation;
- sample-level pressure reliability gating;
- direct continuous TTV as the primary output;
- monotone multi-horizon warning and event-progress ranking only as auxiliary regularization.

No extra measured sensor, target/test calibration, augmentation or task redefinition is introduced.

## Decision rule

V3 proceeds to full ablation only if it is both predictively competitive (target mean MAE <40.42 s, ideally <=37.4 s) and the lag mechanism is demonstrably non-uniform. If it fails both criteria, continuing to invent architectures on eight experiments would be unjustified and strategic reconsideration is required.
