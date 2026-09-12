# Remaining-cohort audit

## Pouch-cell mechanical channels

### `A2_Force.xlsx`
- rows: **9244**; missing: **2741**; missing runs: **1**; longest gap: **2741 samples**
- q01 / median / q99: **-59.2932 / 48.9522 / 982.3565**
- longest exact +1-per-sample ramp: **None**

### `D1_Pressure.xlsx`
- rows: **9244**; missing: **64**; missing runs: **1**; longest gap: **64 samples**
- q01 / median / q99: **-36.0797 / 76.8347 / 9151.2100**
- longest exact +1-per-sample ramp: **(7730, 9179, 1450)**

## Module temperature time base

### M1
- first field `ms`: 5 unique values, 4413 resets, ms-like=True
- reconstructed duration: **4413.60 s**; median dt: **0.2000 s**; inferred rate: **5.00 Hz**
- force duration: **4965.00 s**; duration ratio T/F: **0.889**

### M2
- first field `ms`: 5 unique values, 4717 resets, ms-like=True
- reconstructed duration: **4717.60 s**; median dt: **0.2000 s**; inferred rate: **5.00 Hz**
- force duration: **3875.00 s**; duration ratio T/F: **1.217**

## Guardrail

This report does not repair or interpolate raw signals. Long gaps and suspicious ramps remain quarantined until a defensible cleaning/alignment rule is established.
