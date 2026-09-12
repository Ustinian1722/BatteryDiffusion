# External thermal-runaway dataset shortlist

Purpose: increase the number of **independent real TR experiments** rather than inflating sample size through synthetic windows. The current Hanyang/OSF dataset remains the primary development dataset; external sources are candidates for independent training/validation or cross-trigger robustness.

## Priority 1 — Warwick 21700 internal temperature + gas pressure

**Dataset:** *Dataset of internal temperature and gas pressure in cylindrical lithium-ion cells during thermal runaway* (Data in Brief, 2025)

- Data repository: Mendeley Data, DOI **10.17632/rgfhdhcd9k.2**.
- Three thermal-runaway tests on instrumented **21700** cells.
- Measurements include cell voltage, internal gas pressure, multiple surface temperatures, internal midpoint temperature, and vent temperatures.
- The associated data paper explicitly describes **pre-vent, soft-vent, and flame/explosion** stages.
- External heating is used as the TR trigger.

**Compatibility:** excellent for the present NMC-2170 temperature/pressure branch and for testing whether a generator/predictor learned on one experimental apparatus transfers to another. It is also attractive for vent-aligned tasks because the source data are explicitly organized around vent stages.

**Recommended role:** first external dataset to integrate.

## Priority 2 — Virtual Vehicle BAK N21700 repeated TR package

**Dataset:** *Report Package – Thermal Runaway of Li-Ion cell BAK N21700CG-50 at 60% SOC*.

- Zenodo DOI **10.5281/zenodo.18849418**.
- Eight repeated TR tests on BAK N21700CG-50 cells at 60% SOC.
- Package description states that raw data include cell temperature, vent-gas temperature, pressure and gas release, with event information such as CID, burst plate and thermal runaway.
- Approximate archive size is 5 GB.

**Compatibility:** potentially the strongest source for increasing independent 21700 experiment count and for event-based validation. SOC differs from the current primary dataset, which makes it valuable for robustness but means it should not simply be pooled without a domain/protocol indicator.

**Recommended role:** high-value second integration after the smaller Warwick dataset; download/storage cost is much larger.

## Priority 3 — NMC111/NMC811 thermal-abuse pressure data

**Dataset:** *Modeling Thermal Runaway Mechanisms and Pressure Dynamics in Prismatic Lithium-Ion Batteries*.

- Zenodo DOI **10.5281/zenodo.13981390**.
- Two experiments, one NMC111 and one NMC811.
- Temperature and pressure are recorded during thermal-abuse tests.

**Compatibility:** signal modalities match the thermo-pressure branch, but form factor and chemistry differ. Better suited to cross-chemistry/domain-shift evaluation than direct homogeneous pooling.

## Priority 4 — Mechanically induced TR force/temperature database

**Dataset:** *Dataset of mechanically induced thermal runaway measurement and severity level on Li-ion batteries*.

- Mendeley Data DOI **10.17632/sn2kv34r4h.1**; also surfaced through Battery Archive.
- Mechanical indentation experiments include force, displacement, temperature and voltage, with metadata such as chemistry, form factor, SOC, indenter size and speed.
- Data originate from Sandia National Laboratories and Oak Ridge National Laboratory experiments.

**Compatibility:** highly relevant to the mechanical-sensing theme, but the trigger mechanism is indentation rather than external heating. It should therefore be treated as a **cross-trigger/domain-robustness** dataset, not merged naively with heating-induced TR trajectories.

## Useful but not immediately usable

### USTC temperature-pressure early-warning dataset

Zenodo records associated with *Ultra-Early Detection of Thermal Runaway Precursors in Lithium-Ion Batteries* describe time-resolved temperature and internal pressure under 1C overcharge, 400 W overheating, ARC and normal conditions. The files are currently marked **restricted**, so they are not part of the integration plan unless access becomes available.

### RWTH thermal-runaway experiment database

Zenodo DOI **10.5281/zenodo.18232447** contains a curated database of TR experiments from the literature plus statistical analyses. This is useful for parameter ranges, experiment metadata and contextual benchmarking, but it is not a substitute for raw synchronized time-series trajectories for the present diffusion pipeline.

## Integration order

1. **Warwick/Mendeley 21700** — small, directly modality-compatible, vent-stage information.
2. **Virtual Vehicle 8×21700** — larger independent experiment count and event annotations, but 5 GB.
3. **Zenodo NMC111/NMC811** — cross-form-factor/chemistry transfer.
4. **Battery Archive mechanical indentation** — cross-trigger mechanical-sensing robustness.

## Methodological rule

External experiments must remain identifiable as independent experimental units with their original apparatus/trigger/chemistry metadata. Any normalization, generator fitting or domain adaptation must be trained without using the designated external test experiment. Synthetic trajectories cannot be used to claim an increased real experimental sample size.
