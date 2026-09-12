# BatteryDiffusion

Generative augmentation research for multimodal battery thermal-runaway data.

## Current status

- Raw OSF archive is recursively unpacked under `data/raw/osf_c2hnq/`.
- Dataset audit is under `reports/`; the archive currently exposes **9 experiment IDs**, **7 multimodal IDs**, and **16 signal files**.
- Leakage rule: the independent unit is the **TR experiment ID**, never a sliding window.
- Selected augmentation family: **RA-CDiff — Real-Anchored Conditional Diffusion Augmentation**.
- Cohort-specific augmentation is established for:
  - NMC 2170 cell: temperature + pressure;
  - NMC pouch cell: temperature + force, with conservative gap/corruption quarantine;
  - NMC pouch module: four per-cell hot-spot temperature channels + module force.
- Current accepted synthetic sets:
  - 2170: **197 / 360** candidates;
  - pouch cell: **202 / 360** candidates;
  - pouch module: **274 / 360** candidates.
- Synthetic windows are training augmentation only; they are **not** additional independent TR experiments.

## Downstream finding

The first leakage-safe application study uses **thermally subtle impending-vent early warning**:

- causal context: **256 s**;
- stride: **10 s**;
- warning horizon: **900 s**;
- retain only windows whose **maximum temperature is <= 120 C**;
- evaluation: **leave-one-real-experiment-out** on A2, D1, M1 and M2.

The mandatory fold-safe augmentation utility experiment is complete. All scalers and RA-CDiff generators were refit inside each training fold and the held-out real experiment was untouched.

The result is deliberately retained even though it is not a universal win: naive pooling of RA-CDiff windows does **not** improve the already-saturated temperature+force fusion predictor. At 25% real-data availability, real-only fusion reaches mean/worst AUROC **1.000 / 1.000**, while RA-CDiff pooled fusion gives **0.972 / 0.900**. In contrast, the weak force-only branch improves from mean AUROC **0.533** (real-only) to **0.691** with RA-CDiff. This motivates modality-selective augmentation rather than treating synthetic multimodal windows as new independent observations.

## External validation expansion

External real experiments now have higher priority than further unconstrained generator tuning.

- **Zenodo 13981390**: NMC111 and NMC811 prismatic thermo-pressure experiments are audited. The source paper provides first-vent labels at 10,260 s and 1,350 s; independently detected pressure drops agree within about 1–2 s. These are reserved as cross-domain external tests.
- **Warwick/Mendeley 21700 dataset** (`10.17632/rgfhdhcd9k.1` / `.2`): three Sony VTC6A experiments with internal gas pressure, internal/surface/vent temperatures and voltage are the next integration target.

## Key files

- `docs/augmentation_protocol.md` — leakage-safe augmentation protocol.
- `reports/detailed_dataset_audit.md` — experiment/signal audit.
- `reports/research_freeze_v2.md` — frozen early-warning task definition.
- `reports/research_freeze_v3.md` — completed augmentation-utility result and research pivot.
- `reports/early_warning_aug_utility/` — fold-safe real-only/classical/RA-CDiff comparison.
- `reports/external_zenodo_13981390/` — external thermo-pressure schema and event-label audit.
- `reports/external_dataset_shortlist.md` — independent external TR datasets ranked for integration.
- `scripts/train_diffusion_v3_anchored.py` — RA-CDiff generator.

## Important methodological guardrail

Any final downstream validation must split real data by `experiment_id` **before** scaler fitting, generator fitting, augmentation, predictive-model training, model selection, or threshold selection. Headline performance must be reported on held-out real experiments only. Synthetic windows cannot be used to inflate experimental sample size.
