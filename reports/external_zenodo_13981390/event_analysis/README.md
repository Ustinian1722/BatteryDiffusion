# External Zenodo 13981390 — event-structure audit

Temperature and pressure were aligned to a common 1 Hz time base for compatibility analysis. The source MAT files are downloaded only transiently in CI.

**Important:** the pressure-drop times below are algorithmic vent-like candidates, not published vent annotations. They must not be used as ground truth without source-paper verification.

| experiment | duration | T peak | T-peak time | P peak | P-peak time | vent-like drop | drop→Tpeak |
|---|---:|---:|---:|---:|---:|---:|---:|
| NMC111 | 11299s | 161.5°C | 11300s | 4.462 | 10258s | 10261s | 1039s |
| NMC811 | 2922s | 356.0°C | 1692s | 4.660 | 1348s | 1350s | 342s |

## Compatibility decision

Both records are long enough for the current 256-s causal-window machinery and expose synchronized temperature + pressure after alignment. Because these are prismatic cells with different chemistry/apparatus from the Hanyang 2170 cohort, they are best used as cross-domain thermo-pressure validation rather than pooled as homogeneous training replicas.
