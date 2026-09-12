# BatteryDiffusion

Generative augmentation research for the 2026 *Scientific Data* multimodal battery thermal-runaway dataset.

## Current status

- Raw OSF archive is recursively unpacked under `data/raw/osf_c2hnq/`.
- Dataset audit is under `reports/`; the archive currently exposes **9 experiment IDs**, **7 multimodal IDs**, and **16 signal files**.
- Leakage rule: the independent unit is the **TR experiment ID**, never a sliding window.
- The first homogeneous augmentation cohort is NMC 2170 `temperature + pressure`.
- Current selected augmentation method: **RA-CDiff — Real-Anchored Age-and-Stage Conditional Diffusion Augmentation**.
- Current v3 run: 360 candidate windows → **197 accepted (54.7%)** after feature/OOD/amplitude/memorization screening.
- Synthetic windows are training augmentation only; they are **not** additional independent TR experiments.
- Downstream application selection is intentionally deferred until the remaining pouch/module augmentation audits are complete.

## Key files

- `docs/augmentation_protocol.md` — leakage-safe scientific protocol.
- `reports/detailed_dataset_audit.md` — experiment/signal audit.
- `reports/augmentation_model_selection.md` — generator iterations and current method freeze.
- `scripts/prepare_aug_windows.py` — alignment/windowing and coarse shape-stage labels.
- `scripts/train_diffusion_v3_anchored.py` — selected real-anchored conditional diffusion augmenter.
- `reports/augmentation_quality_v3/` — current synthetic-quality audit.
- `data/synthetic/v3_anchored_cell_2170/` — accepted and candidate v3 synthetic windows.

## Important methodological guardrail

Any later downstream validation must split real data by `experiment_id` **before** generator fitting, windowing for model evaluation, or predictive-model training. Headline performance must be reported on held-out real experiments only.
