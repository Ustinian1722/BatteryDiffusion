# BatteryDiffusion

Generative augmentation research for the 2026 *Scientific Data* multimodal battery thermal-runaway dataset.

## Current status

- Raw OSF archive is recursively unpacked under `data/raw/osf_c2hnq/`.
- Dataset audit is under `reports/`; the archive currently exposes **9 experiment IDs**, **7 multimodal IDs**, and **16 signal files**.
- Leakage rule: the independent unit is the **TR experiment ID**, never a sliding window.
- Selected augmentation family: **RA-CDiff — Real-Anchored Conditional Diffusion Augmentation**.
- Cohort-specific augmentation is now established for:
  - NMC 2170 cell: temperature + pressure;
  - NMC pouch cell: temperature + force, with conservative gap/corruption quarantine;
  - NMC pouch module: four per-cell hot-spot temperature channels + module force.
- Current accepted synthetic sets:
  - 2170: **197 / 360** candidates;
  - pouch cell: **202 / 360** candidates (selected lower-noise setting);
  - pouch module: **274 / 360** candidates (selected pre-vent-local setting).
- Synthetic windows are training augmentation only; they are **not** additional independent TR experiments.

## First downstream direction

The first application study is now provisionally frozen as **thermally subtle impending-vent early warning** using the source paper's explicitly reported first-vent labels for A2, D1, M1 and M2.

Frozen pilot definition:

- causal context: **256 s**;
- stride: **10 s**;
- warning horizon: **900 s**;
- retain only windows whose **maximum temperature is ≤ 120 °C**;
- evaluation: **leave-one-real-experiment-out**.

The unconstrained 300 s task was rejected as too easy because temperature alone reached near-ceiling performance. Under the stricter 900 s / 120 °C regime, the exploratory real-only baseline gives mean AUROC **0.969** for temperature alone versus **0.999** for temperature + force, with worst-fold AUROC improving from **0.877** to **0.998**. These numbers are task-definition evidence, not final paper results.

## Key files

- `docs/augmentation_protocol.md` — leakage-safe augmentation protocol.
- `reports/detailed_dataset_audit.md` — experiment/signal audit.
- `reports/remaining_cohort_audit/` — pouch gaps/corruption and module time-base audit.
- `reports/module_timebase_validation.md` — validation of module time reconstruction against source-paper landmarks.
- `reports/research_freeze_v2.md` — current augmentation and downstream-task freeze.
- `reports/event_landmarks/` — published vent labels and 2170 pressure-drop candidates.
- `reports/downstream_feasibility/` — causal early-warning label/window audit.
- `reports/early_warning_baseline/` — real-only LOEO feasibility baseline.
- `reports/early_warning_regime_scan/` — temperature-ceiling/horizon task-definition scan.
- `scripts/train_diffusion_v3_anchored.py` — RA-CDiff generator.
- `data/synthetic/` — candidate/accepted synthetic corpora and audits.

## Important methodological guardrail

Any final downstream validation must split real data by `experiment_id` **before** scaler fitting, generator fitting, augmentation, predictive-model training, or model selection. Headline performance must be reported on held-out real experiments only. The current all-data synthetic corpora are method-development artifacts and must not be reused as training data in the final LOEO downstream comparison.
