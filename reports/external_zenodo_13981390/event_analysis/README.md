# External Zenodo 13981390 — published-event validation

Temperature and pressure were aligned to a common 1 Hz time base. The source MAT files are downloaded only transiently in CI.

The companion open-access paper explicitly reports first venting at **2 h 51 min (10,260 s)** for NMC111 and **22 min 30 s (1,350 s)** for NMC811. These are source-provided labels, not labels inferred from our pressure signal.

| experiment | published vent | vent T | detected pressure drop | error | pressure peak | overlap covers published TR? |
|---|---:|---:|---:|---:|---:|---|
| NMC111 | 10260s | 152°C | 10262s | +2s | 4.462 | no |
| NMC811 | 1350s | 185°C | 1351s | +1s | 4.660 | yes |

## Interpretation

The independently derived strongest local pressure drop agrees with the published vent landmark to approximately the sampling resolution in both experiments. This materially strengthens their suitability as **external event-labeled thermo-pressure experiments**. NMC111 pressure recording ends before the paper-reported TR time, so it is suitable for pre-vent/vent studies but not for pressure behavior through the complete TR peak.

## Compatibility decision

Both experiments support the current 256-s causal-window machinery and have authoritative first-vent labels. Because they are prismatic NMC111/NMC811 cells with different heating rates, SOCs and apparatus, they should be treated as external/domain-shift experiments rather than homogeneous replicas of the primary Hanyang cohort.
