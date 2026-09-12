# Research freeze v5 — application results after protocol corrections

Date: 2026-09-12

## 1. Primary application remains data-scarce pre-vent early warning

The main paper application remains the source-grounded `vent_gas` early-warning task on the eight repeated Virtual Vehicle BAK N21700CG-50 experiments.

Frozen task:

- causal context: 120 s;
- stride: 10 s;
- event: first package-provided `vent_gas` marker;
- warning horizon: 300 s;
- max temperature in the causal input window: <=150 C;
- primary scarcity regime: two independent real training experiments;
- held-out evaluation unit: one complete real destructive experiment.

RA-CDiff is fit inside each training fold only. Synthetic windows do not increase experimental n.

## 2. Comparator-fairness correction is complete

The first augmentation study matched total synthetic count but a post-run audit showed that RA-CDiff quality screening sometimes retained an imbalanced warning-class composition. A corrective protocol was therefore frozen before rescoring. The correction changed only the classical comparator: in each fold, classical augmentation was required to match the exact retained RA-CDiff negative/positive synthetic counts.

All 24 predeclared folds completed successfully.

### Headline `fusion_absolute` results

| Method | Mean AUROC | Worst AUROC | Mean AUPRC | Worst AUPRC |
|---|---:|---:|---:|---:|
| Real-only | 0.943 | 0.746 | 0.920 | 0.781 |
| Class-matched classical | 0.955 | 0.776 | 0.948 | 0.796 |
| **RA-CDiff** | **0.984** | **0.931** | **0.973** | **0.896** |

Experiment-level paired statistics (`n=8` real held-out experiments):

- RA-CDiff vs real-only AUROC: mean delta **+0.041**, 95% experiment-bootstrap CI **[+0.005, +0.104]**, 6/8 improved, exact sign-flip p=0.0469.
- RA-CDiff vs real-only AUPRC: mean delta **+0.053**, CI **[+0.010, +0.109]**, 7/8 improved, p=0.0156.
- RA-CDiff vs class-matched classical AUROC: mean delta **+0.028**, CI **[-0.003, +0.086]**, 3/8 improved, p=0.4375.
- RA-CDiff vs class-matched classical AUPRC: mean delta **+0.025**, CI **[-0.006, +0.077]**, 3/8 improved, p=0.6250.

## 3. Frozen interpretation of Task 1

The final primary claim is:

> **RA-CDiff improves the fusion early-warning model relative to real-only training under severe destructive-experiment scarcity, including a large improvement in worst-experiment ranking performance.**

The project does **not** claim that RA-CDiff is statistically demonstrated to outperform the class-matched classical augmentation comparator at n=8. The point estimates favor RA-CDiff, but the experiment-level confidence intervals for the RA-CDiff-vs-classical contrasts include zero.

This distinction is frozen and will not be changed by searching for a more favorable downstream classifier.

## 4. Secondary application: time-to-vent is feasible

A support-only audit was completed before any regression score. The secondary task was then frozen as:

- context: 120 s;
- stride: 10 s;
- target: remaining time to first package `vent_gas` event;
- target range: `0 < tau <= 600 s`;
- max input temperature: <=150 C;
- outer validation: leave-one-real-experiment-out over the same eight independent experiments.

The frozen first real-only LOEO baseline is now complete.

### Predeclared headline Huber model

| Representation | Mean MAE | Worst MAE | Mean RMSE | Mean R2 |
|---|---:|---:|---:|---:|
| temperature | **43.3 s** | **71.8 s** | 51.1 s | 0.685 |
| fusion_absolute | 50.2 s | 89.2 s | 53.7 s | 0.674 |
| fusion_shape | 52.5 s | 74.3 s | 68.8 s | 0.310 |
| pressure_absolute | 78.0 s | 123.6 s | 89.5 s | 0.140 |
| pressure_shape | 88.7 s | 125.9 s | 105.9 s | -0.135 |

The predeclared nonlinear random-forest diagnostic shows a different pattern: `fusion_absolute` reaches mean MAE **37.4 s**, versus **43.3 s** for temperature-only, suggesting that pressure may contribute useful nonlinear information even though it does not improve the fixed Huber headline model.

## 5. Frozen interpretation of Task 2

Time-to-vent is sufficiently learnable to retain as a secondary application. However:

- the first headline Huber result does not establish a multimodal advantage;
- the nonlinear diagnostic suggests a possible fusion advantage but is secondary by protocol;
- therefore Task 2 should be presented first as **remaining-safety-time feasibility**, not as proof that pressure always improves regression.

Any augmentation study for Task 2 must preserve the event-relative regression label without inventing synthetic event times. If synthetic windows are used, they must be local variants anchored to a real window and inherit only that anchor's event-relative time label; the time axis itself may not be warped or shifted.

## 6. Current paper story

The defensible narrative is now:

> destructive TR experiments are expensive and scarce -> real-anchored conditional diffusion creates screened local multimodal variants -> under only two real training experiments, the synthetic variants improve pre-vent warning relative to real-only training on unseen real destructive tests -> the same sensor history can also estimate remaining time to vent -> propagation forecasting remains a module-level extension.

The strongest method claim is **scarcity mitigation vs real-only**, not universal superiority over every classical augmentation method.

## 7. Next work

1. Test whether RA-CDiff also improves **time-to-vent regression under k=2 experiment scarcity**, using anchor-preserved continuous labels and an exactly anchor-matched classical comparator.
2. Continue independent external-domain validation on NMC111/NMC811 and Warwick VTC6A datasets.
3. Retain TR propagation as an extension until module-level independent sample count is larger.
4. Do not change the existing Task 1 definitions or shop for a new classifier to manufacture a stronger RA-CDiff-vs-classical significance result.
