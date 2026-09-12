# Strategic stop after LATH-Net v3

Date: 2026-09-12

This project reached the predeclared strategic stop condition in `method_paper_target_freeze_v3.md`. Further unconstrained architecture search is paused pending a deliberate method pivot decision.

## Evidence

Exact 6-train/1-validation/1-test experiment protocol, real-only, same 120 s temperature+pressure input and 0<time-to-first-vent<=600 s target:

| method | mean MAE (s) | worst MAE (s) | mean R2 | note |
|---|---:|---:|---:|---|
| RF causal descriptors | 40.42 | 62.90 | 0.7826 | strong classical baseline |
| Attention-GRU | 39.5 | 65.7 | 0.781 | strongest compact deep baseline |
| LATH-Net v1 | **37.4** | 70.1 | 0.778 | best mean deep result, but lag mechanism not independently validated |
| LATH-Net v2 | 47.42 | 83.10 | 0.6883 | unified discrete hazard harmed absolute TTV calibration |
| LATH-Net v3 | 43.76 | 82.19 | 0.7063 | precursor-invariant lag redesign failed the predeclared gate |

V3 mechanism diagnostics also fail to support a strong lag-centric claim:

- mean maximum lag weight = 0.184 versus uniform 0.167;
- normalized lag entropy = 0.995 versus uniform 1.000;
- reliability gate mean = 0.479.

Thus the lag distribution is only weakly non-uniform and the reliability gate is close to neutral.

Independent representation audit under the same experiment folds shows that **absolute pressure is important**:

- temperature + absolute pressure descriptors: 40.42 s mean MAE;
- temperature only: 47.59 s;
- temperature + relative pressure: 46.85 s;
- temperature + relative pressure + derivative: 47.62 s;
- temperature + relative pressure + derivative + normalized shape: 48.57 s.

Therefore the data support the value of pressure, but not the hypothesis that discarding absolute pressure and emphasizing explicit pressure-to-temperature lag is the best modeling principle for this cohort.

## What is already valid

1. The engineering task (pre-vent TTV / multi-horizon warning) is well supported and remains valid.
2. Temperature+pressure fusion is useful; absolute pressure carries genuine predictive information.
3. A specialized deep model can be competitive: exploratory LATH v1 obtains the lowest mean MAE among current deep baselines.
4. RA-CDiff remains a valid secondary scarcity contribution and is orthogonal to the main-model decision.
5. The current evidence is not sufficient to make **lag-aware fusion** the headline novelty.

## Recommended strategic pivot

Recommended next method direction: **Cross-Experiment Event-Progress Alignment**, not further lag-module tuning.

Rationale: the dominant failure mode across v1-v3 is experiment-specific absolute TTV bias. The new method should explicitly learn an experiment-invariant event-progress representation while preserving the absolute pressure information proven useful by the representation audit.

Candidate architecture:

1. **Absolute-state branch:** raw standardized temperature + absolute pressure;
2. **change/precursor branch:** Delta T, dT, Delta P, dP;
3. **state-change gated fusion:** change branch provides a residual correction to the absolute-state estimate rather than replacing absolute pressure;
4. **cross-experiment progress-alignment loss:** training windows from different experiments with similar TTV are pulled together in latent space; windows with clearly different TTV are separated. Experiment identity is used only for training pair construction, never as predictor input;
5. **within-experiment ranking:** preserve monotonic event progression;
6. **direct TTV + monotone auxiliary warning:** primary output remains continuous TTV.

Working concept name: **CEPA-Net — Cross-Experiment Progress Alignment Network**.

This is a genuine strategic change of the main scientific hypothesis: from “explicit thermo-pressure lag is the key” to “experiment-invariant event-progress representation is the key”. It should only be pursued with explicit project approval.

## Alternatives

- Conservative alternative: retain LATH v1 as Proposed and proceed to ablation. This is lower-risk computationally, but the margin over Attention-GRU is small and statistically weak, worst-experiment robustness is not best, and lag novelty lacks mechanism evidence.
- Augmentation-centric alternative: make RA-CDiff the main story. This conflicts with the project owner's stated objective and is not recommended.

No further model-search workflow is launched after this note until the strategic choice is made.
