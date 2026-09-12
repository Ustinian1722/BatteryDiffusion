# Research freeze v2 — augmentation complete, downstream pilot selected

Date: 2026-09-12

## 1. What is frozen

The first-stage generative augmentation study is now sufficiently mature to stop changing the basic augmentation paradigm. The selected family is:

> **RA-CDiff — Real-Anchored Conditional Diffusion Augmentation**

RA-CDiff does not attempt to synthesize a destructive thermal-runaway (TR) trajectory from unconstrained Gaussian noise. It starts from a real training window, moves it to an intermediate diffusion time, then performs condition-preserving reverse denoising. Candidate samples are rejected if they violate feature-shape, amplitude, real-manifold-distance, or memorization-distance gates.

Synthetic windows remain **training augmentation samples only**. They never increase the number of independent TR experiments.

## 2. Cohort-specific augmentation freeze

The raw archive is physically heterogeneous, so a single generator is not forced across 2170 cells, pouch cells, and pouch modules.

| Cohort | Real experiment IDs | Channels used | Selected noise-start range (100 diffusion steps) | Accepted / candidates | Key quality summary |
|---|---|---|---:|---:|---|
| NMC 2170 cell | B2, C1, C2 | temperature + pressure | 20–50 | 197 / 360 | SMD mean 0.273; NN ratio 1.42; MMD² 0.0107 |
| NMC pouch cell | A2, D1 | temperature + force | 10–35 | 202 / 360 | SMD mean 0.573; NN ratio 2.32; MMD² 0.1767 |
| NMC pouch module | M1, M2 | four cell-level hot-spot temperatures + force | 3–15 | 274 / 360 | SMD mean 0.784; NN ratio 0.93; MMD² 0.2558 |

The increasingly local perturbation range from 2170 → pouch cell → module is deliberate. The module cohort has only two independent experiments and five generated channels; a more aggressive diffusion start moved pre-rapid windows away from the real manifold. The final module setting substantially improved acceptance in the pre-rapid groups while preserving a nonzero novelty floor.

### Data-cleaning guardrails

- `A2_Force.xlsx`: only the measured prefix before the long missing suffix is used.
- `D1_Pressure.xlsx`: the source paper describes pouch mechanics as force. The filename is preserved for provenance, but the physical modality is treated as expansion force. A long exact +1-per-sample corrupted ramp is quarantined rather than repaired.
- M1/M2 temperature: the repeated `ms` field is reconstructed as a 5 Hz time base. The reconstruction was independently checked against the source paper's reported module vent-to-peak timing. Module thermocouples are aggregated into per-cell hot-spot channels according to the source-paper TC placement.

## 3. Downstream task investigation

The source paper explicitly reports first-vent times for A2, D1, M1 and M2. This makes venting a preferable target to labels inferred from the same temperature trace.

A first naive impending-vent task was audited:

- context: 256 s;
- horizon: 300 s;
- stride: 10 s;
- label: `1` when first vent occurs within the next horizon;
- post-vent samples excluded;
- leave-one-real-experiment-out (LOEO) evaluation.

That nominal task is **too easy** for a meaningful multimodal study: temperature-only achieved mean AUROC 1.000 in the real-only classical baseline. We therefore do not freeze the 300 s nominal task as the paper problem.

## 4. Selected first research problem

The more informative problem is **thermally subtle impending-vent early warning**:

> Predict whether first venting will occur within a long warning horizon while the entire causal input window is still below a fixed temperature ceiling.

An exploratory grid over horizon and temperature ceiling found a useful, fully evaluable operating point:

- causal context: **256 s**;
- stride: **10 s**;
- first-vent horizon: **900 s**;
- retain only windows with **maximum temperature ≤ 120 °C**;
- held-out unit: **real experiment**.

With the same real-only logistic feasibility model:

| Input | Mean AUROC | Worst-fold AUROC | Mean AUPRC | Worst-fold AUPRC |
|---|---:|---:|---:|---:|
| Temperature only | 0.969 | 0.877 | 0.947 | 0.789 |
| Force only | 0.507 | 0.000 | 0.551 | 0.167 |
| **Temperature + force** | **0.999** | **0.998** | **0.999** | **0.998** |

The important observation is not that force alone is universally predictive—it is not, especially across cell and module geometries. Rather, the mechanical channel supplies complementary information that removes the worst temperature-only failure under this early/thermally constrained regime.

This operating point is now **frozen for the next pilot**. It was selected using the exploratory task-definition scan, so it must not be re-tuned after downstream augmentation/model comparisons begin.

## 5. Next mandatory experiment

The next experiment is not another generator search. It is a leakage-safe downstream augmentation-utility test:

1. LOEO split on A2/D1/M1/M2.
2. Construct only causal, pre-vent training windows satisfying the frozen 120 °C rule.
3. Fit all scalers and the RA-CDiff generator **inside the training fold only**.
4. Generate label-preserving local variants only from training anchors.
5. Compare:
   - real only;
   - simple classical augmentation;
   - RA-CDiff augmentation.
6. Evaluate only the held-out **real** experiment.
7. Compare temperature-only, mechanical-only, and multimodal predictors.
8. Report fold-level AUROC/AUPRC plus worst-experiment performance; do not treat windows as independent replicates.

The current all-data synthetic corpora are useful for augmentation-method development and visualization, but they are **not** permitted in this LOEO downstream comparison because they were generated after fitting on all experiments of their cohort.

## 6. Publication-scope limitation

Only four experiments currently have explicit published first-vent labels in the force-based pouch cell/module subset. A strong paper should therefore add at least one compatible external TR dataset if possible. If no compatible external data can be obtained, claims must remain explicitly small-sample/destructive-experiment proof-of-concept, with experiment-level uncertainty and no inflation of sample size through window counts.
