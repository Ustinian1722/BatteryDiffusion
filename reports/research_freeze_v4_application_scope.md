# Research freeze v4 — application scope after literature review

Date: 2026-09-12

## 1. Why this freeze is needed

The project has already demonstrated that RA-CDiff can generate physically screened local variants of scarce thermal-runaway (TR) time series. The next question is therefore not "can the generator make more curves?" but "what safety application should those synthetic samples support?"

A targeted literature review of recent TR work (2023–2026) shows that the dominant data-driven application families are:

1. **Early warning / early detection** before venting or TR, using temperature, voltage, pressure, force, gas or fused signals.
2. **Thermal-runaway propagation (TRP) forecasting** at module/pack level, usually as multi-step future-temperature prediction or propagation-state prediction.
3. **Time-to-event / safety-margin prediction**, such as time to vent, time to TR, warning lead time, or vent-to-peak interval.
4. **Physics-informed surrogate / digital-twin modelling**, where neural models replace or accelerate multiphysics simulation.

The current BatteryDiffusion data and RA-CDiff design match category 1 most directly, category 3 naturally as a secondary task, category 2 only as a module-level extension, and category 4 only if a physical simulator or governing-equation residual is added later.

## 2. Literature evidence for the selected application

Representative recent studies support the early-warning framing:

- Li et al., *Journal of Cleaner Production* (2024), DOI `10.1016/j.jclepro.2024.142422`: expansion force provided up to 682 s warning before TR, with abnormal force observable at relatively low temperature.
- Li et al., *Applied Energy* (2024), DOI `10.1016/j.apenergy.2024.122998`: expansion-force-based warning across SOC conditions; the reported force indicator preceded TR by >642 s.
- Jin et al., *Energy* (2024), DOI `10.1016/j.energy.2024.133685`: expansion force was used for rapid early TR detection and was reported to respond earlier than surface temperature and terminal voltage.
- Wei et al., *Energy Technology* (2024), DOI `10.1002/ente.202401238`: a three-level warning algorithm fused voltage, temperature and expansion force and reported a third-stage warning hundreds of seconds before complete failure.
- *Green Energy and Intelligent Transportation* (2025), DOI `10.1016/j.geits.2025.100368`: internal pressure + temperature were used as Attention-GRU inputs for multi-level TR risk warning.
- *Energy* (2023), DOI `10.1016/j.energy.2023.127747`: voltage-temperature time-frequency deep learning was used for TR alarming with reported lead times of 8–13 min.
- *Journal of Energy Storage* (2026), DOI `10.1016/j.est.2025.120305`: transfer-learning-based early TR prediction explicitly targets the small-sample regime and reports 540 s early warning in 86 Ah LFP cells.

For propagation forecasting, Ouyang et al., *Energy* (2023), DOI `10.1016/j.energy.2023.127168`, used **25 experiments + 130 simulations** to train a multi-task CNN-LSTM for multi-step cell-temperature/TRP prediction. This is important precedent that scarce destructive experiments are commonly supplemented by non-experimental data, but it also shows that TRP requires richer module-level coverage than the present primary cohort.

For physics-informed prediction, the 2023 *Journal of Energy Storage* MPINN work (DOI `10.1016/j.est.2023.106654`) encoded energy-balance and Arrhenius physics. This is a different methodological branch from the present empirically constrained RA-CDiff and should not be conflated with it.

## 3. Primary paper application is now frozen

The primary application is:

> **Data-scarce multimodal pre-vent early warning for lithium-ion battery thermal runaway experiments.**

The operational interpretation is:

> Given a causal history of temperature plus a mechanical/gas-pressure signal, estimate whether the package-provided venting event will occur within a future warning horizon.

The event label retains the dataset terminology `vent_gas`. It is **not** renamed as TR onset unless an individual source explicitly provides that equivalence.

### Primary repeated-experiment benchmark

The strongest current benchmark is the 8-test Virtual Vehicle BAK N21700CG-50 cohort because it provides repeated experiments and explicit `vent_gas` event tables.

The already frozen protocol remains unchanged:

- causal context: **120 s**;
- stride: **10 s**;
- target: first package-provided `vent_gas` event;
- warning horizon: **300 s**;
- maximum temperature in the input window: **<=150 C**;
- post-event windows excluded;
- independent evaluation unit: **entire real experiment**;
- main scarcity regime: **2 independent real training experiments**;
- test data: held-out **real** experiment only.

No horizon/cap/context tuning is permitted after the existing predictive results.

## 4. Role of RA-CDiff in the application

RA-CDiff is not the endpoint of the paper. It is the data-scarcity mitigation mechanism inside the early-warning pipeline:

`few real destructive experiments -> fit RA-CDiff on training experiments only -> generate screened local variants -> train warning model -> test on unseen real destructive experiment`

The intended claim is therefore about **generalization under destructive-experiment scarcity**, not about synthetic-data realism alone.

The current 8-experiment study already provides a promising feasibility signal for the fusion representation in the k=2 regime:

- real-only fusion_absolute: mean AUROC 0.943, worst AUROC 0.746;
- RA-CDiff fusion_absolute: mean AUROC 0.984, worst AUROC 0.931;
- real-only mean AUPRC 0.920;
- RA-CDiff mean AUPRC 0.973.

These results remain subject to the already documented class-composition fairness correction before any final "RA-CDiff > classical augmentation" claim.

## 5. Secondary tasks

### Task 2 — time-to-vent / remaining safety time

This is the most natural secondary application after binary warning:

`causal T + P/F history -> remaining time until first vent event`

Why it is useful:

- binary warning answers **whether** action is needed;
- time-to-event answers **how much time remains** for mitigation, isolation, cooling, shutdown or evacuation.

This task will be support-audited before any predictive model is fit. No time horizon will be selected based on regression performance.

Recommended metrics once frozen: MAE, RMSE, median absolute error, worst-experiment MAE, and experiment-level bootstrap uncertainty. Window count must not be treated as independent n.

### Task 3 — TR propagation forecasting

This remains an extension, not the main paper task. It is appropriate only for module/pack cohorts with multiple spatial temperature channels.

Candidate formulation:

`past multi-cell temperatures (+ force/pressure) -> future multi-cell temperature trajectories / propagation state`

The Hanyang M1/M2 module data are useful for proof-of-concept, but two independent module experiments are insufficient for a strong standalone TRP claim. Additional module-level real or physics-simulation data would be required.

## 6. What is explicitly not the main application

The following are not the primary paper story:

- unconditional generation of visually realistic TR curves;
- random-window train/test classification within one destructive experiment;
- continuous SOH prediction from this TR dataset;
- treating synthetic windows as new independent experiments;
- claiming the current empirical filters constitute a governing-equation PINN;
- using post-event information to make a nominal "early" warning.

## 7. Paper-level story

The preferred paper narrative is now:

> **Destructive battery thermal-runaway experiments are expensive and scarce. We develop a real-anchored conditional diffusion augmentation strategy that generates screened local multimodal variants from a few training experiments. Its practical value is tested in pre-vent early warning, where models are trained from very few real experiments and evaluated only on unseen real destructive tests. Secondary analysis estimates remaining time to vent, while module-level TR propagation is retained as an extension rather than overstated from insufficient independent experiments.**

A suitable working title is:

> **Real-Anchored Diffusion Augmentation for Data-Scarce Multimodal Battery Thermal-Runaway Early Warning**

## 8. Next mandatory work

1. Complete the class-composition-matched correction for the Virtual Vehicle augmentation comparison.
2. Perform a support-only audit for Task 2 time-to-vent before fitting any regression model.
3. Integrate independent external temperature-pressure/force TR experiments for cross-domain validation.
4. Keep Hanyang pouch/module data as complementary multimodal and module-level evidence rather than inflating the primary repeated-experiment sample size.
5. Only after these are stable should architecture work resume; the paper question is data scarcity + safety warning, not model shopping.
