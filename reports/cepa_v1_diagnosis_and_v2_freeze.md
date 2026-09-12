# CEPA-Net v1 diagnosis and v2 protocol freeze

Frozen **after CEPA-Net v1 completed and before any CEPA-Net v2 predictive score is inspected**.

## Why v1 does not advance

CEPA-Net v1 failed its predeclared development gate:

- experiment-macro MAE: 39.74 s vs 37.39 s for LATH-Net v1;
- worst held-out MAE: 69.62 s, slightly better than LATH-Net v1's 70.10 s;
- CEPA v1 beats LATH-Net v1 on 4/8 experiments, below the required 5/8.

The failure is mechanistically informative rather than a reason for open-ended model search.

Two diagnostics are decisive:

1. **The dynamic gate is inactive.** Mean gate activation is 0.500 and fold values stay approximately 0.498-0.501. The model did not learn meaningful state-dependent use of the dynamic branch.
2. **The aligned embedding is too weakly coupled to the predictor.** Mean held-out 5-NN TTV MAE in the progress embedding is 53.79 s and nearest-embedding TTV gap is 56.17 s. In individual v1 fold logs, cosine similarity to the nearest training embedding is almost one even when the TTV gap is large. The separate progress projection can therefore satisfy little of the intended cross-experiment geometry while the TTV head continues to operate through another latent path.

The next change is therefore architectural coupling, not hyperparameter tuning.

## CEPA-Net v2 hypothesis

> Cross-experiment progress regularization can only influence TTV calibration reliably if the aligned representation is the **same predictive bottleneck** used by the TTV head.

V2 is named **Predictive-CEPA** internally. The primary task, folds, causal support, normalization, optimizer, seeds, and headline metrics remain unchanged from v1.

## Frozen v2 changes

### 1. Remove the dead dynamic gate

The v1 sigmoid gate is removed. Absolute-state and dynamic-precursor branches are fused directly using

`[state, dynamics, state-dynamics, state*dynamics]`.

This is a simplification motivated by the observed v1 gate collapse around 0.5. No replacement attention/gating module is introduced.

### 2. Shared predictive bottleneck

After residual temporal fusion and mean/max pooling, the model produces one `32-D` bottleneck `h_p`.

- TTV regression is computed **only from `h_p`**.
- multi-horizon warning is computed **only from `h_p`**.
- CEPA alignment is computed on `normalize(h_p)` itself.

There is no separate progress projection. Therefore the model cannot route TTV prediction around the aligned representation.

### 3. Preserve absolute pressure and causal dynamics

The two branches remain:

- absolute branch: standardized `[T,P]`;
- dynamic branch: `[T-T_start, dT, P-P_start, dP]`.

This preserves the representation-audit result that absolute pressure is useful while retaining causal precursor changes as residual evidence.

## Frozen alignment objective

The v1 continuous cross-experiment geometry is retained unchanged so that v2 tests **coupling**, not a new distance formulation:

- `y=tau/600`;
- `d_target=clip(|y_i-y_j|/0.5,0,1)`;
- `d_z=sqrt(max(2-2*cos(z_i,z_j),eps))/2`;
- `w_ij=1+2*exp(-|y_i-y_j|/0.10)`;
- only cross-experiment upper-triangle pairs are counted.

The full frozen loss remains:

`L = L_TTV + 0.15 L_CEPA + 0.05 L_rank + 0.05 L_warn`.

No coefficient is changed from v1.

## Frozen training protocol

Unchanged:

- eight independent TS0330 destructive tests;
- one held-out test experiment;
- next cyclic experiment is validation;
- remaining six train;
- 120 s causal temperature+pressure input;
- `0<tau<=600 s` to package first `vent_gas`;
- max input temperature <=150 C;
- stride 10 s;
- training-only scaler;
- no augmentation;
- AdamW `1e-3`, weight decay `1e-4`;
- cosine annealing to `1e-5`;
- max 260 epochs, patience 45;
- seeds 2026/2027/2028.

## Frozen mechanism diagnostics

Report for each held-out experiment:

- predictive-bottleneck 5-NN TTV MAE using training embeddings only;
- nearest-training-embedding TTV gap;
- nearest-training-embedding cosine similarity;
- experiment-level bias.

A useful v2 representation should reduce the v1 embedding mismatch (53.79 s 5-NN MAE / 56.17 s nearest TTV gap) while improving predictive calibration.

## Predeclared v2 gate

V2 advances to full ablation and RA-CDiff augmentation only if all are met:

1. experiment-macro MAE < **37.39 s**;
2. worst held-out MAE < **69.62 s** (must improve the v1 worst fold, not merely its mean);
3. beats LATH-Net v1 on at least **5/8** experiments;
4. mean predictive-bottleneck 5-NN TTV MAE < **53.79 s** (must improve the v1 alignment diagnostic).

If v2 fails, CEPA is not to be rescued by repeated hidden-size/loss-weight/attention searches. At that point the method-paper direction must be reconsidered explicitly.