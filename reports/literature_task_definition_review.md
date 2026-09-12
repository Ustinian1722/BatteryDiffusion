# Literature review — downstream task definitions for BatteryDiffusion

Date: 2026-09-12

## Purpose

This note fixes the application logic after the augmentation stage. The generator is not the end task. The question is which battery-safety problem the generated trajectories should support, and how that task should be defined without inventing labels that the source data do not provide.

## 1. Dominant application families in recent literature

Recent 2023–2026 work clusters into four application families.

| Family | Typical inputs | Typical target | Engineering meaning | Fit to BatteryDiffusion |
|---|---|---|---|---|
| Early warning / early detection | temperature, voltage, pressure, force, gas, fused signals | warning state / risk level / impending failure | trigger BMS or pack intervention before severe failure | **Primary** |
| Time-to-event / warning lead time | same causal sensor history | remaining time to vent/TR/failure | quantify remaining mitigation time | **Secondary** |
| Thermal-runaway propagation (TRP) forecasting | multi-cell temperatures and pack conditions | future cell temperatures / propagation state | forecast spread through a module or pack | Extension |
| Physics-informed surrogate / digital twin | operating conditions + physical states | future thermal / reaction state | accelerate high-fidelity simulation | Separate future branch |

The literature is increasingly explicit that single temperature or voltage signals can provide limited warning lead time and robustness, while mechanical/gas-pressure and multimodal sensing may expose earlier precursors.

## 2. Representative early-warning studies

### Expansion-force warning

- Li et al., *Journal of Cleaner Production* 457 (2024), 142422, DOI `10.1016/j.jclepro.2024.142422`.
  - Expansion force provided warning as early as **682 s before TR**.
  - Abnormal force was detectable at a reported minimum temperature of **35.4 °C**.
  - This directly supports studying mechanically assisted warning before strong thermal evidence appears.

- Li et al., *Applied Energy* 362 (2024), 122998, DOI `10.1016/j.apenergy.2024.122998`.
  - Expansion-force abnormality was reported **>642 s before TR** under thermal-abuse tests across SOC conditions.
  - The work explicitly treats expansion force as an early-warning signal and proposes hazard classification.

- Jin et al., *Energy* (2024), DOI `10.1016/j.energy.2024.133685`.
  - Expansion-force signal was used for rapid TR detection and compared against conventional terminal-voltage/surface-temperature indicators.

- *Journal of Energy Storage* 109 (2025), 115085, DOI `10.1016/j.est.2024.115085`.
  - Expansion-force warning was evaluated across NCM/LFP chemistries and external heating, penetration, and overcharge triggers.
  - Warning lead time was explicitly compared with temperature and voltage.

### Pressure-temperature fusion

- *Green Energy and Intelligent Transportation* (2025), 100368, DOI `10.1016/j.geits.2025.100368`.
  - Internal pressure + temperature are Attention-GRU inputs.
  - Output is a **multi-level TR risk warning state**.
  - Reported accuracy is 98.2%, with average early-warning time of 1.51 h in its micro-overcharge setting.
  - This supports multimodal warning as an application, but the exact risk labels are specific to that experimental protocol and should not be copied into our datasets.

### Small-sample warning

- *Journal of Energy Storage* 148 (2026), 120305, DOI `10.1016/j.est.2025.120305`.
  - Transfer learning is explicitly motivated by the difficulty of obtaining large pre/post-TR datasets.
  - Normal-operation source data are used for pretraining and a small target-domain TR dataset for fine-tuning.
  - The reported warning occurs **540 s before the failure trigger point**.
  - This is strong precedent for framing destructive-experiment scarcity itself as the methodological problem.

## 3. Why venting is a defensible temporal boundary

A 2026 sensing review, *Sensors and Actuators A: Physical*, DOI `10.1016/j.sna.2026.118395`, explicitly organizes sensing around a **venting-aligned pre-/post-venting framework**. It notes that pressure and expansion-force sensing provide information on gas accumulation and safety venting, and reviews multimodal fusion for early warning.

This matters for BatteryDiffusion because several available datasets provide an authoritative first-vent or `vent_gas` landmark, whereas they do not always provide a universally consistent TR-onset label. Therefore:

> **Primary labels should retain the source event terminology (`vent_gas` / first vent) rather than silently relabeling it as TR onset.**

A pre-vent warning task is both physically meaningful and source-grounded.

## 4. Why binary warning should be Task 1

The primary downstream task is:

`past causal T + P/F history -> probability that first vent occurs within a fixed future horizon`

This matches the dominant early-warning paradigm while allowing rigorous event-level validation.

For the Virtual Vehicle TS0330 cohort the already frozen benchmark remains:

- context: 120 s;
- stride: 10 s;
- horizon: 300 s;
- max input temperature: <=150 °C;
- event: first package-provided `vent_gas` marker;
- held-out unit: complete real experiment;
- scarcity setting: only two independent real training experiments for the primary augmentation study.

This task must not be retuned after the existing scores.

## 5. Why time-to-vent should be Task 2

Binary warning answers whether the system is entering a dangerous pre-vent interval. A deployment system also benefits from a continuous estimate of how much mitigation time remains.

Task 2 is therefore:

`past causal T + P/F history -> remaining seconds to first vent`

The literature lead times above cluster naturally in the several-hundred-second range. A support-only audit on all eight TS0330 experiments found that a 120 s context, <=150 °C input cap and target range `0 < tau <= 600 s` retain at least 31 eligible windows in every experiment. This definition was frozen before regression scoring.

Headline regression metrics:

- held-out-experiment MAE (s);
- RMSE (s);
- median absolute error (s);
- worst held-out-experiment MAE (s).

R2 is secondary. Statistical n remains the number of real experiments.

## 6. Why TR propagation is not the primary paper task

Ouyang et al., *Energy* 273 (2023), 127168, DOI `10.1016/j.energy.2023.127168`, constructed a TRP dataset from **25 experiments + 130 simulations** and trained a multi-task CNN-LSTM to predict multiple cell temperatures several steps ahead.

This demonstrates two things:

1. using non-experimental data to supplement scarce destructive TR experiments is an accepted research pattern;
2. credible TRP forecasting needs much richer module-level condition coverage than the current two Hanyang module experiments.

Therefore TRP remains an extension until more independent module/pack experiments or a validated multiphysics simulator are available.

## 7. Final task hierarchy for this repository

**Task 1 — Primary:** multimodal pre-vent early warning under destructive-experiment scarcity.

**Task 2 — Secondary:** time-to-vent / remaining safety time.

**Task 3 — Extension:** module-level TR propagation / future temperature forecasting.

The role of RA-CDiff is the same across Tasks 1–2: create screened local training variants **inside each training fold** and test whether they improve generalization to real unseen destructive experiments. It is never treated as creating additional independent experiments.

## 8. Paper-level question

The central scientific question is now:

> **Can real-anchored generative augmentation improve the generalization and robustness of battery thermal-runaway early-warning models when only a few destructive experiments are available?**

Secondary question:

> **Can the same multimodal representation estimate remaining time to the first venting event with useful experiment-level accuracy?**

This framing is preferred over an unconditional synthetic-data paper because it ties the generative method to a concrete battery-safety decision problem and permits evaluation entirely on real held-out experiments.
