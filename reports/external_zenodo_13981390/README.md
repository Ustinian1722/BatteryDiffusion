# External dataset audit — Zenodo 13981390

Record: **Modeling Thermal Runaway Mechanisms and Pressure Dynamics in Prismatic Lithium-Ion Batteries**.

The CI audit downloads the two public MAT files to a temporary runner directory and commits only schema/statistical metadata; raw external files are not redistributed here.

## Files

### NMC111
- bytes: 703,131
- MAT reader: `scipy_loadmat`
- top-level keys: `['T_average_nmc111', 'Time_T_nmc111_hour', 'Time_p_nmc111_hour', 'pressure_nmc111']`

### NMC811
- bytes: 1,055,535
- MAT reader: `scipy_loadmat`
- top-level keys: `['T_average_nmc811', 'Time_nmc811_100Hz', 'V_nmc811', 'pressure_nmc811']`

## Intended role

These two experiments are candidate **cross-form-factor thermo-pressure external tests**. They should not be pooled blindly with the Hanyang 2170 cohort because cell format/chemistry/apparatus differ. The next step is to identify exact time/temperature/pressure arrays and event timing from the MAT schema before defining any transfer experiment.
