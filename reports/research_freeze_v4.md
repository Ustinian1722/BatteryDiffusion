# Research freeze v4 — domain-standard downstream validation after augmentation

Date: 2026-09-12

## 1. Paper problem remains unchanged

The source thermal-runaway datasets contain very few independent destructive experiments. Generative augmentation is therefore treated as a data-scarcity method, not as a replacement for real experiments. The paper continues to follow the standard battery-safety research pattern after augmentation: use causal temperature/mechanical signals for pre-vent warning and time-to-vent estimation, evaluate on completely held-out real experiments, and report experiment-level robustness.

RA-CDiff synthetic windows remain local training variants only. The independent sample size is always the number of real destructive experiments.

## 2. Main downstream task: pre-vent early warning

Frozen Virtual Vehicle task:

- 120 s causal context;
- 300 s warning horizon to the first package-provided `vent_gas` marker;
- max input cell-case temperature <=150 C;
- primary scarcity regime: two independent real training experiments;
- 24 predeclared training-pair / held-out-experiment folds;
- test unit: one untouched real experiment.

After correcting the classical comparator so that it receives the same retained synthetic class composition as RA-CDiff, the headline `fusion_absolute` results are:

| Method | Mean AUROC | Worst AUROC | Mean AUPRC | Worst AUPRC |
|---|---:|---:|---:|---:|
| Real only | 0.943 | 0.746 | 0.920 | 0.781 |
| Classical augmentation | 0.955 | 0.776 | 0.948 | 0.796 |
| RA-CDiff | **0.984** | **0.931** | **0.973** | **0.896** |

Experiment-level paired statistics for RA-CDiff versus real-only:

- AUROC delta: +0.041, 95% experiment-bootstrap CI [+0.005, +0.104], improved 6/8 held-out experiments, exact sign-flip p=0.0469;
- AUPRC delta: +0.053, 95% CI [+0.010, +0.109], improved 7/8, exact sign-flip p=0.0156.

RA-CDiff is numerically better than the class-matched classical comparator, but that contrast is not statistically resolved at n=8. We therefore claim robust benefit versus real-only under severe experiment scarcity, not universal superiority over every classical augmentation.

## 3. Secondary downstream task: time-to-vent regression

Frozen task:

- 120 s causal context;
- target 0 < tau <=600 s to first `vent_gas`;
- max input temperature <=150 C;
- two real training experiments in the scarcity study;
- synthetic variants inherit the continuous tau of their unchanged real anchor; the generator never invents an event time.

The full 24-fold augmentation study is complete.

For the predeclared Huber + `fusion_absolute` comparison:

| Method | Mean MAE | Worst MAE | Mean RMSE |
|---|---:|---:|---:|
| Real only | 211.2 s | 437.6 s | 215.3 s |
| Classical anchor-matched | 186.5 s | 375.3 s | 188.6 s |
| RA-CDiff | **179.4 s** | **346.3 s** | **181.4 s** |

RA-CDiff versus real-only:

- MAE delta: -31.8 s, 95% CI [-53.4, -12.9] s, improved 8/8 held-out experiments, exact sign-flip p=0.0078;
- RMSE delta: -33.9 s, 95% CI [-56.1, -13.4] s, improved 8/8, exact sign-flip p=0.0078.

However, this does not mean the augmented Huber fusion model is the best absolute TTE predictor. The real-only nonlinear RF baseline with `fusion_absolute` already reaches mean MAE 52.3 s, and RA-CDiff gives 52.1 s; temperature-only RF is also about 52.7 s. Thus augmentation strongly regularizes an unstable low-data linear fusion model but provides little additional benefit once the nonlinear predictor is already robust.

This mixed result is retained as a secondary analysis, not promoted into a stronger claim than the data support.

## 4. Interpretation now frozen

The combined evidence supports the following narrower conclusion:

> Real-anchored diffusion augmentation is most useful when the number of independent destructive experiments is extremely small and the downstream learner is vulnerable to sparse coverage of the pre-event manifold. Its benefit is task- and model-dependent; it cannot substitute for additional real experiments and does not universally improve already-saturated predictors.

This is preferable to claiming that synthetic data simply increase the effective experimental sample size.

## 5. Next mandatory experiment: domain-standard sequence learners

The next validation step returns to the conventional modeling style used in battery thermal-runaway warning literature. Without changing the frozen task or outer folds, evaluate raw causal temperature + pressure sequences using two compact standard sequence learners:

1. a small 1D-CNN;
2. a one-layer GRU.

For each architecture compare:

- real-only training;
- anchor/class-matched local classical augmentation;
- RA-CDiff augmentation.

Guardrails:

- same 24 predeclared k=2 folds;
- training-only normalization;
- fixed architecture and optimization protocol before scores are inspected;
- no test-based early stopping or threshold tuning;
- AUROC/AUPRC are headline metrics;
- report held-out-experiment macro and worst-experiment performance;
- after this check, do not start an unconstrained architecture search.

If augmentation utility persists across at least one standard sequence learner, the paper can state that the gain is not an artifact of hand-engineered logistic features. If it disappears, retain the negative result and keep the classical/feature-based downstream model as the validated application layer.

## 6. Subsequent work after the sequence-model check

Provided no methodological defect is found, proceed in this order without requesting a strategic decision:

1. experiment-count learning curve (real-only versus RA-CDiff under increasing numbers of real training experiments);
2. generator ablation / quality evidence needed for the methods section;
3. external real-dataset validation on the already audited 21700/prismatic cohorts where signal and event semantics are compatible;
4. final paper tables, figures and statistical summary.

Stop for user decision only if the sequence-model validation contradicts the current central claim, external datasets prove semantically incompatible with the frozen task, or a choice would materially change the paper's scientific claim.
