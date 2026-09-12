# Augmentation model selection — current freeze

## Scope

This report freezes the **data-augmentation stage only**. No downstream application is selected yet.

The independent experimental unit remains the TR experiment ID. Synthetic windows are local training augmentation samples and must never be counted as new independent experiments.

## Data state

The downloaded OSF archive currently exposes 9 experiment IDs and 16 signal files, of which 7 IDs have temperature + mechanical modalities. The 2170 paired pilot cohort contains 174 aligned windows from B2, C1 and C2. The pouch-cell paired cohort contains 215 windows from A2 and D1, but D1's mechanical file requires a separate anomaly/nomenclature audit before it is used for generative training. Module temperature streams also use a different multi-thermocouple time representation and are intentionally deferred to a dedicated alignment step.

## Generator iterations

| Version | Generation mode | Candidates | Accepted | Main diagnostic |
|---|---|---:|---:|---|
| Pilot v1 | pure-noise DDPM, aging-conditioned | 128 | 0 | heterogeneous stages caused severe feature mismatch |
| v2 | pure-noise cosine DDPM, aging + shape-stage, balanced batches | 300 | 4 | terminal noise issue fixed, but pure-noise sampling remained unstable/OOD in this few-experiment regime |
| **v3 (selected)** | **real-anchored conditional diffusion perturbation** | **360** | **197 (54.7%)** | **best real-manifold agreement; one rare condition remains unsupported** |

## v3 quality audit

For the accepted v3 set:

- feature SMD mean / max: 0.273 / 1.462;
- feature-range coverage: 93.7%;
- real-real / synthetic-real nearest-neighbor median: 0.0246 / 0.0349;
- nearest-neighbor ratio: 1.42;
- RBF-MMD²: 0.0107.

The remaining red flag is `aged + rapid_peak`: no candidate simultaneously passed the novelty/amplitude/feature gate. This group has only a handful of real windows. **We will not loosen the gate merely to manufacture coverage.** Until additional evidence is available, that rare condition remains real-only.

## Current method freeze

The augmentation method is therefore frozen provisionally as:

> **Real-Anchored Age-and-Stage Conditional Diffusion Augmentation (RA-CDiff)**

Pipeline:

1. split/organize by independent experiment ID;
2. align homogeneous multimodal cohorts;
3. create fixed-length windows;
4. derive categorical aging condition plus a coarse temperature-shape stage (`pre_rapid`, `rapid_peak`, `post_peak`);
5. train a conditional 1D diffusion denoiser;
6. perturb real training windows to an intermediate noise level;
7. condition-preserving reverse denoising creates local synthetic variants;
8. reject samples that fail feature-shape, amplitude, OOD-distance or memorization checks.

The stage labels are **shape labels**, not venting ground truth.

## What happens next

Before choosing a downstream task, the next research step is to finish augmentation coverage/auditing for the remaining cohorts:

- audit and clean the pouch-cell D1 mechanical channel before training a pouch cohort generator;
- align the module multi-thermocouple temperature stream with force;
- run the same RA-CDiff quality audit separately for each physically homogeneous cohort;
- only then compare candidate applications using what the augmented data can actually support.

Potential downstream tasks remain unfrozen: early warning/time-to-event, TR-stage recognition, vent-to-peak/response-time estimation, peak thermal/mechanical hazard prediction, or degradation-aware TR dynamics characterization.
