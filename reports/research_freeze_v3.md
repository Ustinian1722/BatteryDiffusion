# Research freeze v3 — downstream augmentation result and pivot

Date: 2026-09-12

## 1. Fold-safe augmentation utility experiment is complete

The mandatory leakage-safe downstream test from `research_freeze_v2.md` has now been executed. The frozen task remains:

- causal context: 256 s;
- stride: 10 s;
- first-vent horizon: 900 s;
- retain only windows whose maximum temperature is <= 120 C;
- outer evaluation unit: one complete held-out real experiment;
- experiments: A2, D1, M1, M2;
- scaler and RA-CDiff generator are fit inside each outer training fold only.

Two real-data regimes were tested: 25% of eligible training windows and 100%. Three training strategies were compared: real-only, classical augmentation, and RA-CDiff augmentation.

## 2. Main empirical finding

RA-CDiff does **not** improve the already-strong temperature+mechanical fusion predictor when synthetic windows are naively pooled with real training windows.

At 25% real-data availability:

| Method | Modality | Mean AUROC | Worst AUROC | Mean AUPRC | Worst AUPRC |
|---|---|---:|---:|---:|---:|
| real-only | temperature | 0.972 | 0.889 | 0.952 | 0.809 |
| real-only | force | 0.533 | 0.000 | 0.585 | 0.167 |
| real-only | fusion | **1.000** | **1.000** | **1.000** | **1.000** |
| classical augmentation | fusion | 0.987 | 0.946 | 0.974 | 0.897 |
| RA-CDiff | fusion | 0.972 | 0.900 | 0.922 | 0.702 |
| RA-CDiff | force | **0.691** | 0.176 | **0.660** | 0.324 |

At 100% real-data availability, the same qualitative pattern remains: the real-only fusion predictor is already near ceiling, while naive synthetic pooling does not improve it.

## 3. Interpretation

The correct conclusion is **not** that the generator failed. The augmentation-quality audits show that RA-CDiff can create local, non-identical, physically screened variants. The downstream result instead shows that the current fusion task is already saturated by a small real dataset and that treating synthetic multimodal windows as ordinary new observations can perturb a strong decision boundary.

The useful positive signal is more specific: under 25% real-data scarcity, RA-CDiff improves the weak force-only branch from mean AUROC 0.533 to 0.691. This suggests that synthetic data are more useful as a **modality-specific regularizer / representation augmenter** than as a replacement for independent multimodal experiments.

This distinction is now frozen. We will not claim that RA-CDiff increases the effective number of TR experiments, nor that naive synthetic pooling universally improves prediction.

## 4. Research direction after this result

The next stage is split into two tracks.

### Track A — external real-experiment expansion (priority)

The strongest limitation is still the number of independent destructive experiments. External datasets therefore take priority over additional generator tuning.

Already audited:

- Zenodo 13981390, NMC111 and NMC811 prismatic cells;
- synchronized temperature + pressure available;
- source paper provides authoritative first-vent labels at 10,260 s and 1,350 s respectively;
- independently detected strongest pressure-drop landmarks agree within approximately 1–2 s;
- these experiments are domain-shift external tests, not homogeneous replicas.

Next integration target:

- Warwick/Mendeley `10.17632/rgfhdhcd9k.1` / `.2`;
- three Sony VTC6A 21700 thermal-runaway tests;
- internal gas pressure, internal/surface/vent temperatures and voltage;
- source description explicitly separates pre-vent, soft-vent and flame/explosion stages.

### Track B — selective augmentation rather than naive pooling

A later predictive pilot may test a constrained strategy in which RA-CDiff augments only the weak mechanical branch, while the final temperature+mechanical decision layer is trained on real windows only. This is motivated directly by the 25% scarcity result and must be evaluated without changing the frozen outer test experiments.

Track B is secondary to adding more independent real experiments. It should not become another unconstrained architecture search.

## 5. Publication framing after the pivot

The defensible paper story is now:

> destructive TR experiments are scarce and heterogeneous -> real-anchored conditional diffusion produces physically screened local variants -> synthetic variants are audited as augmentation, not new experiments -> naive pooling has limited value on a saturated multimodal task -> augmentation is most useful for weak-signal/scarcity regimes -> independent external experiments are used to test transfer and robustness.

This is a stronger and more credible claim than forcing a universal downstream gain.

## 6. Guardrails

1. Independent sample size is always the number of real experiments, never the number of windows or synthetic trajectories.
2. External datasets retain their chemistry, form factor, trigger, SOC and apparatus metadata.
3. No external test experiment may contribute to scaler fitting, generator fitting, model selection or threshold selection.
4. Task-definition scans and final model comparisons must remain separated.
5. Negative augmentation results are retained in the repository and will not be hidden by subsequent tuning.
