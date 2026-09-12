# Method-paper literature positioning

Date: 2026-09-12

Purpose: establish the final method-paper target against recent data-driven thermal-runaway literature. This document is a design rationale, not a result report.

## Representative field pattern

Recent method papers confirm a conventional method-paper structure rather than an augmentation-centric structure:

1. define an engineering warning/prognostic target;
2. propose a task-specific temporal or multimodal model;
3. compare against standard recurrent/convolutional/Transformer and hybrid baselines;
4. perform module ablations;
5. report accuracy plus warning/prognostic timeliness and robustness;
6. treat small-sample transfer/simulation/generation as a supporting training strategy when needed.

Representative examples used for positioning:

- Huang et al., Energy 334 (2025), 137676: ConvTransformer for remaining time before thermal runaway; compared against DNN, LSTM, GRU, Bi-LSTM, Bi-GRU and ablated CNN/attention/Transformer components. DOI: 10.1016/j.energy.2025.137676.
- Wang et al., Journal of Energy Storage 163 (2026), 122078: TCN-Transformer for remaining time-to-runaway; local causal temporal extraction + long-range Transformer dependencies; compared with recurrent and hybrid baselines and reported Diebold-Mariano significance. DOI: 10.1016/j.est.2026.122078.
- Multi-level early warning based on internal pressure-temperature fusion, Green Energy and Intelligent Transportation (2025), 100368: pressure + temperature are fed to an Attention-GRU for risk-level classification; reported 98.2% classification accuracy and 1.51 h mean warning time. DOI: 10.1016/j.geits.2025.100368.
- Li et al., Journal of Energy Storage 116 (2025), 116098: multimodal TR temperature prediction using bidirectional cross-attention and multi-scale gated fusion; explicit ablations validate individual fusion components. DOI: 10.1016/j.est.2025.116098.
- MSTA-Net, Sensors 26 (2026), 3083: multi-scale local/temporal/spatial branches plus adaptive fusion gate; compared with RF/LSTM/Transformer and verified components with ablation. DOI: 10.3390/s26103083.
- Transfer-learning CNN-LSTM-SE, Journal of Energy Storage 148 (2026), 120305: the prediction model remains the main method and transfer learning addresses small-sample TR data. DOI: 10.1016/j.est.2025.120305.
- Shi et al., Electronics 15 (2026), 2909: integrates multi-sensor health information extraction, early warning, and time-to-TR prognostics, showing that remaining-time prediction is already a legitimate TR-safety task rather than an artificial downstream task. DOI: 10.3390/electronics15132909.

## Crowded ideas that are not sufficient novelty by themselves

The following have already appeared in recent TR literature and should not be sold as the sole innovation:

- generic CNN + Transformer or TCN + Transformer hybrids;
- generic multi-scale temporal branches;
- generic attention or adaptive fusion gate;
- pressure-temperature fusion by direct GRU/Attention-GRU;
- simply combining warning classification with a remaining-time regression output;
- small-sample transfer/augmentation by itself.

## Distinct target for this project

The method claim is narrowed to two linked ideas not found as the central mechanism in the above representative works:

1. **explicit causal thermo-pressure lag alignment**: learn which past pressure precursor state best aligns with the current thermal latent state, using only a fixed causal lag bank and time-varying lag attention;
2. **unified event-time hazard distribution**: one discrete-time event distribution generates both continuous time-to-first-vent and monotone multi-horizon vent risk, eliminating logical inconsistency between independently trained warning and prognostic heads.

The supporting components are modality-specific dual-timescale encoders, reliability gating, and within-experiment event-progress ranking.

## Important overlap risk

A July 2026 arXiv preprint titled *Regime-Aware Physics-Guided Early Warning of Lithium-Ion Battery Thermal Runaway Using Thermo-Mechanical Signals* already combines causal temporal convolutions, physics-biased attention, regime-dependent gating, TR detection, and time-to-disaster estimation under mechanical abuse. Therefore this project must not claim that causal TCN + gating + joint detection/prognostics is new in itself. The paper must emphasize **lag-resolved pressure-to-thermal precursor alignment and one hazard-consistent event-time distribution**.

Recent work on multi-sensor prognostics also already estimates time-to-TR. Consequently, our contribution is the predictive architecture and consistency mechanism, not the mere use of a remaining-time target.

## Final comparison/ablation logic

Main real-only architecture table:

CNN, LSTM, GRU, TCN, Transformer, CNN-LSTM, Attention-GRU, TCN-Transformer, LATH-Net.

Frozen ablations:

- temperature only;
- single-scale encoder;
- synchronous fusion without lag alignment;
- lag alignment without reliability gate;
- direct scalar TTV regression instead of unified hazard distribution;
- no progress-ranking loss.

The main claim is defensible only if Full LATH-Net improves aggregate experiment-level TTV metrics over strong baselines and the lag/hazard modules survive ablation. RA-CDiff is evaluated only afterwards as a secondary scarcity-training contribution.
