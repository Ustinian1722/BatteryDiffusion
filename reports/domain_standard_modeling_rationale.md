# Domain-standard downstream modeling rationale

Date: 2026-09-12

This note fixes the downstream modeling philosophy after generative augmentation. The project will not turn into a generic time-series architecture search. After RA-CDiff addresses destructive-experiment scarcity, downstream tasks and predictors should remain recognizable to the battery thermal-runaway (TR) warning literature.

## Literature-supported pattern

Recent TR-warning studies repeatedly use the following structure:

1. causal sensor windows from temperature, voltage, pressure/force, gas/acoustic or related signals;
2. CNN/RNN/GRU/LSTM/TCN-style temporal learners or compact classical models;
3. an application target such as warning-stage classification, anomaly/residual warning, or time-to-failure/event prediction;
4. warning lead time, classification accuracy/AUROC-type discrimination, or regression error as practical metrics;
5. small-sample transfer/augmentation/domain adaptation when destructive TR data are scarce.

Examples supporting this pattern include:

- Applied Energy 365 (2024) 123248, *Multi-source domain transfer learning with small sample learning for thermal runaway diagnosis of lithium-ion battery*, DOI 10.1016/j.apenergy.2024.123248. The paper explicitly frames TR diagnosis as a small-sample/domain-shift problem.
- Journal of Energy Storage 148 (2026) 120305, *Transfer learning-enabled deep learning framework for early thermal runaway prediction in lithium-ion energy storage batteries*, DOI 10.1016/j.est.2025.120305. It uses CNN-LSTM-SE and compares CNN, LSTM and CNN-LSTM under small-sample transfer learning, reporting warning lead time.
- Green Energy and Intelligent Transportation (2025) 100368, *Multi-level early warning of thermal runaway based on internal pressure-temperature fusion for lithium-ion batteries*, DOI 10.1016/j.geits.2025.100368. It uses pressure + temperature with an Attention-GRU warning network.
- Scientific Reports (2025), *Early warning method for charging thermal runaway of electric vehicle lithium-ion battery based on charging network*. It uses an LSTM-TCN temperature prediction/residual-warning pipeline.
- Journal of Power Sources 595 (2024) 234065, *A combined multiphysics modeling and deep learning framework to predict thermal runaway in cylindrical Li-ion batteries*, DOI 10.1016/j.jpowsour.2024.234065. It treats TR evolution as a stage-classification problem with CNN-based deep learning.
- NIST / Fire Safety Journal 2024, *Development of a Robust Early-Stage Thermal Runaway Detection Model for Lithium-ion Batteries*, uses data augmentation plus a CNN detector on short causal acoustic segments.

## Consequence for this project

The paper's novelty should therefore remain concentrated in **scarcity-aware generative augmentation and leakage-safe validation**, while the downstream application layer uses conventional and interpretable domain baselines.

Frozen downstream hierarchy:

- Primary: pre-vent warning classification from causal temperature + pressure/force windows.
- Secondary: time-to-vent regression.
- Model robustness check: compact 1D-CNN and GRU in addition to the already-audited feature-based logistic/RF/Huber baselines.
- No broad Transformer/Mamba/PatchTST/FEDformer search unless a later scientific question specifically requires it.

The intended paper claim is not that a novel warning classifier beats every architecture. It is that **RA-CDiff can make standard warning learners more reliable when only a few independent destructive TR experiments are available**, while all headline testing remains on real held-out experiments.

## Reporting convention

Classification: AUROC, AUPRC, worst-experiment AUROC/AUPRC; thresholded F1/balanced accuracy secondary.

Regression: MAE, RMSE, worst-experiment MAE; R2 secondary because narrow per-experiment target ranges can make R2 unstable.

All uncertainty/statistical comparisons are performed at the independent experiment level, not the overlapping-window level.
