# BatteryDiffusion augmentation protocol

## Goal

Build a **generative data-augmentation layer** for the destructive thermal-runaway (TR) dataset before deciding the final downstream application. The augmentation stage must preserve the distinction between *time samples/windows* and *independent TR experiments*.

## Scientific constraints

1. **Independent unit = experiment ID**, not a sliding window. Any final downstream validation must split by experiment ID before fitting the generator or predictor.
2. Synthetic trajectories are **training augmentation only**. They never count as additional independent real experiments and never enter a real-only test set.
3. Do not synthesize unsupported continuous SOH labels. For now, conditioning is categorical (`fresh` / `aged`) because the dataset exposes only a few representative SOH levels.
4. Preserve modalities and units. Do not mix cylindrical-cell pressure with pouch-cell/module force in a single raw-unit channel.
5. Prefer event-/stage-aware generation and quality screening over unconditional generation of entire long trajectories.
6. A synthetic sample is accepted only after distribution, derivative/event-shape, physical-range, and memorization checks.

## Phase A — repository/data audit

- Archive unpacked under `data/raw/osf_c2hnq/`.
- Experiment inventory is generated under `reports/`.
- Current archive exposes 9 experiment IDs and 7 multimodal IDs; the paper reports 8 TR experiments, so this discrepancy is explicitly recorded rather than silently reconciled.

## Phase B — homogeneous augmentation cohorts

To avoid unit/geometry leakage, generation is initially separated into three cohorts:

- `cell_2170_temp_pressure`: cylindrical NMC 2170 cell, temperature + pressure.
- `cell_pouch_temp_mechanical`: pouch cell, temperature + mechanical response (archive filenames are retained; a generic mechanical channel is used internally if nomenclature is inconsistent).
- `module_pouch_temp_force`: pouch module, temperature + expansion force.

Temperature-only repeats can be used for marginal-temperature regularization but not as paired multimodal targets.

## Phase C — conditional diffusion model

Primary model: **few-shot conditional 1D DDPM**.

Condition vector initially includes:

- cohort / geometry,
- aging category (`fresh`, `aged`),
- event stage (when reliable stage labels are available),
- modality mask.

The model generates fixed-length multi-channel windows. Long trajectories are reconstructed only after window-level generation is stable.

### Training objective

Base denoising objective:

`L_diff = ||eps - eps_theta(x_t, t, c)||^2`

Optional weak constraints are added only when supported by observed signals:

- physical range penalty,
- derivative-envelope penalty,
- cross-modal consistency penalty,
- event ordering/shape penalty.

No unvalidated electrochemical governing equation is imposed.

## Phase D — synthetic sample filtering

Generated candidates pass a quality gate based on:

1. finite values and modality-specific physical ranges;
2. level/slope/derivative statistics relative to real data;
3. spectral/autocorrelation similarity;
4. cross-modal dependence consistency;
5. lower nearest-neighbor distance (reject memorized copies);
6. upper distance/OOD threshold (reject implausible samples).

The thresholds are derived from real training experiments only.

## Phase E — validation of augmentation itself

Before choosing an application, report generator quality using experiment-blocked diagnostics:

- real-vs-synthetic feature distribution distance;
- derivative/spectral statistics;
- nearest-neighbor memorization audit;
- discriminator two-sample test;
- PCA/UMAP visualization for descriptive analysis only;
- event metric preservation where available (`t_vent`, peak timing, vent-to-peak interval).

## Phase F — downstream application (not frozen yet)

Only after augmentation quality is acceptable will the project choose the final task. Candidate tasks include:

- TR early warning / time-to-vent estimation;
- TR-stage classification;
- vent-to-peak / evacuation-time regression;
- peak temperature or mechanical-hazard prediction;
- degradation-aware TR kinetics characterization.

Final headline results must always be measured on **held-out real experiments only**. In each evaluation fold, the generator must be re-fit without the held-out experiment.
