# Research freeze v5 — sequence-model check retained, scarcity mechanism becomes the next test

Date: 2026-09-12

## 1. Sequence-model robustness check is complete

The frozen 24-fold k=2 early-warning comparison was repeated on two conventional raw-sequence learners, a compact 1D-CNN and a one-layer GRU. Architecture, optimizer, normalization, epochs and two ensemble seeds were fixed before scores were inspected. Classical augmentation used the exact retained real anchors and labels used by RA-CDiff.

| Method | CNN mean/worst AUROC | CNN mean/worst AUPRC | GRU mean/worst AUROC | GRU mean/worst AUPRC |
|---|---:|---:|---:|---:|
| Real only | **0.938 / 0.833** | **0.911 / 0.732** | 1.000 / 0.996 | 0.997 / 0.972 |
| Classical anchor-matched | 0.875 / 0.667 | 0.825 / 0.556 | 1.000 / 0.996 | 0.997 / 0.972 |
| RA-CDiff | 0.875 / 0.667 | 0.825 / 0.556 | 1.000 / 0.996 | 0.997 / 0.972 |

RA-CDiff therefore does **not** show a generic architecture-independent gain. On the small CNN, both local augmentation methods degrade ranking relative to real-only. On the GRU, the frozen task is essentially saturated and neither augmentation method changes AUROC/AUPRC.

This negative/mixed result is retained and will be reported. We will not retune CNN/GRU architectures, learning rates, augmentation ratios or the task to manufacture a positive deep-learning result.

## 2. Consequence for the central claim

The defensible claim is narrowed to:

> Under severe destructive-experiment scarcity, RA-CDiff can improve specific low-capacity / feature-based warning learners whose decision boundary is sensitive to sparse coverage, but augmentation is not universally beneficial for every downstream architecture.

This remains consistent with the earlier leakage-safe logistic result, where `fusion_absolute` improved from mean/worst AUROC 0.943/0.746 (real only) to 0.984/0.931 (RA-CDiff), and with the TTE result where RA-CDiff regularized the unstable k=2 Huber fusion model but added almost nothing to an already robust random forest.

The paper should therefore present the downstream learners as application probes, not claim a novel universal augmentation theorem.

## 3. Next mechanism test: experiment-count learning curve

The next question is now causal/mechanistic rather than architectural:

> Does the augmentation benefit shrink as the number of **independent real training experiments** increases?

This directly tests the proposed role of RA-CDiff as a scarcity remedy.

The learning-curve task remains exactly the frozen Virtual Vehicle early-warning definition:

- 120 s causal context;
- 300 s horizon to first package `vent_gas`;
- stride 10 s;
- max input temperature <=150 C;
- held-out unit = one complete real experiment;
- predictor = frozen class-weighted logistic regression (`C=0.5`, liblinear);
- primary representation = `fusion_absolute`;
- temperature-only retained as a secondary comparator.

Real training experiment counts: **k = 2, 4, 6**.

For a clean paired learning curve, one deterministic cyclic source subset is predeclared for every held-out experiment and every k. Starting immediately after the held-out experiment in the ordered ring A→B→C→D→E→F→G→H→A, take the next k distinct experiments. Examples:

- held A: k2=B,C; k4=B,C,D,E; k6=B,C,D,E,F,G;
- held H: k2=A,B; k4=A,B,C,D; k6=A,B,C,D,E,F.

Thus each k has exactly eight held-out real experiments, and the same held-out units are paired across k.

For each fold compare:

- real-only;
- exact-anchor-matched classical local augmentation;
- RA-CDiff.

RA-CDiff is refit inside every fold. Classical augmentation uses the exact retained anchors and labels used by RA-CDiff. Synthetic count is capped by the number of real training windows, exactly as in the frozen scarcity pipeline.

## 4. Predeclared interpretation

The scarcity hypothesis is supported if the RA-CDiff minus real-only gain is largest at k=2 and attenuates toward zero as k rises to 4 and 6. A non-monotonic or negative curve will be retained and interpreted as evidence that augmentation utility depends on coverage structure rather than simply real experiment count.

No task, representation, generator hyperparameter or predictor hyperparameter will be changed after viewing this learning curve.

## 5. Parallel external test

A separately frozen scarcity-transfer experiment is already running on the two event-labeled Zenodo prismatic experiments. It uses eight predeclared k=2 Virtual Vehicle source pairs and keeps NMC111/NMC811 completely external. Its result is supportive only; it will not be used to retune the within-cohort learning curve.
