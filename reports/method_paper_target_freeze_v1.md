# Method-paper target freeze v1

Date: 2026-09-12

## Positioning

RA-CDiff is **not the main contribution**. It is a supporting solution to destructive-test scarcity. The paper is method-centric: the headline contribution must be a new thermo-mechanical temporal model that outperforms standard and recent deep-learning baselines under leakage-safe, experiment-level validation.

## Literature-derived field pattern

Representative data-driven thermal-runaway papers generally follow the same structure:

1. define an engineering prediction target such as multi-level warning, remaining time to runaway/vent, future temperature/TR propagation, or anomaly warning;
2. propose a task-specific temporal model (e.g. Attention-GRU, CNN-LSTM-SE, ConvTransformer, TCN-Transformer, multi-task CNN-LSTM);
3. compare against standard neural baselines such as CNN, LSTM, GRU, Transformer and hybrid models;
4. perform ablation of the proposed architectural modules;
5. report task metrics plus warning lead time / robustness / generalization;
6. when data are scarce, use transfer learning, simulation/virtual data, or augmentation as a supporting training strategy rather than as the sole scientific contribution.

This project will follow that pattern.

## Primary scientific task

The current binary `future vent within 300 s` task is retained as a secondary diagnostic because strong GRU models can saturate it. It is not suitable as the only headline task for a method paper.

The primary task is therefore frozen as **causal remaining-time / multi-horizon pre-vent prognostics**:

- input: a causal history window of synchronized cell-case temperature and package pressure;
- event: package-provided first `vent_gas` landmark;
- outputs:
  1. remaining time to first vent (TTV), and
  2. monotone cumulative vent-risk over multiple future horizons.

This matches the field trend toward remaining-time-to-TR regression and multi-level early warning while respecting the authoritative event labels actually present in the Virtual Vehicle package.

## Proposed model target

Working name: **PMH-Net — Precursor-aware Multimodal Hazard Network**.

The architecture is deliberately task-specific rather than a generic model soup.

### M1. Dual-timescale causal encoders

Temperature and pressure are encoded in separate causal temporal branches. Each branch combines short-receptive-field and dilated long-receptive-field convolutions. The goal is to preserve the different temporal scales of rapid mechanical/gas-pressure changes and slower thermal accumulation.

### M2. Lag-aware cross-modal gated fusion

The two modality streams are not concatenated directly. A causal cross-modal fusion block allows each modality to condition the other using only present/past information and learns a reliability gate. This is intended to capture precursor lead-lag structure while suppressing apparatus-specific or noisy mechanical information.

### M3. Monotone multi-horizon hazard head

Instead of a single binary classifier, the model predicts discrete future-event hazards over ordered time bins. Cumulative event probability is mathematically monotone with forecast horizon. The same head yields a continuous expected time-to-vent estimate.

### M4. Event-progress regularization

Within the same real experiment, a window closer to first vent should not be assigned systematically lower cumulative risk than a clearly earlier window. A pairwise progress-ranking term is added only within training experiments. This supplies weak event-order supervision without using cycle/test identifiers as predictor inputs.

## Why this is the target

The design addresses three gaps visible in the literature and in the current data:

- single-backbone models often mix fast precursor transients with slow thermal accumulation;
- naive early fusion does not model thermo-pressure lead-lag/reliability explicitly;
- separate classification and remaining-time regressors can produce inconsistent warning outputs, whereas a hazard formulation unifies both.

RA-CDiff is attached only after the model protocol is fixed, as a scarcity-training option.

## Main comparison protocol

All architectures receive the same causal temperature+pressure inputs and identical experiment-level folds.

Baselines to implement:

- CNN / 1D-CNN;
- LSTM;
- GRU;
- TCN;
- Transformer encoder;
- CNN-LSTM;
- Attention-GRU;
- TCN-Transformer (strong recent hybrid baseline).

The main architecture table must compare models under the **same training-data regime**. Augmentation is not allowed to be a hidden advantage of Proposed.

## Ablations

Ablations are frozen before headline model scores:

- PMH-Net full;
- w/o pressure (temperature only);
- w/o dual-timescale encoder (single-scale causal encoder);
- w/o lag-aware cross-modal gate (plain concatenation);
- w/o hazard head (direct TTV regression + independent binary classifier);
- w/o progress-ranking loss;
- optionally pressure only as a signal-value diagnostic.

Expected interpretation is module-specific; no module will be retained solely because it improves one fold.

## Augmentation study — secondary contribution

After the architecture is frozen:

- Proposed + real only;
- Proposed + classical anchor-matched augmentation;
- Proposed + RA-CDiff.

A second robustness table may apply the same three regimes to selected baselines, but the augmentation contribution remains secondary.

## Metrics

Headline TTV metrics:

- MAE (s), RMSE (s), MedAE (s), R2;
- worst-held-out-experiment MAE.

Multi-horizon warning metrics:

- AUROC and AUPRC per horizon;
- experiment-macro AUROC/AUPRC;
- F1 / balanced accuracy at a threshold frozen on training/validation only;
- warning lead time and false-alarm behavior;
- Brier score / calibration as a secondary hazard-quality metric.

Statistical unit is always the independent destructive experiment, never the number of windows.

## Data roles

- Virtual Vehicle TS0330 A-H: primary method-development benchmark because it provides eight repeated 21700 destructive tests with package event annotations.
- Hanyang/OSF thermo-mechanical experiments: secondary mechanistic / cross-format validation where event labels are defensible.
- Zenodo NMC111/NMC811: untouched cross-chemistry/cross-form-factor external validation with published first-vent landmarks.
- Warwick 21700: desirable additional external cohort if raw access is resolved.

## Paper claim hierarchy

1. **Main method claim:** PMH-Net improves causal pre-vent prognostics over standard and strong hybrid temporal baselines.
2. **Mechanism claim:** modality-specific timescale encoding + causal cross-modal gating + unified hazard modeling each contribute measurably.
3. **Scarcity claim:** RA-CDiff is a supporting training strategy that can improve PMH-Net when the number of real destructive experiments is small.
4. **Generalization claim:** external results are reported as domain-shift evidence and never used for task/model retuning.

## Stop condition for strategic reconsideration

Do not change the paper direction because a single fold/model is weak. Reconsider only if either:

- the full PMH-Net cannot consistently outperform strong baselines on experiment-level TTV/multi-horizon metrics after reasonable, predeclared tuning; or
- the proposed modules fail ablation in aggregate, leaving no defensible architectural contribution.
