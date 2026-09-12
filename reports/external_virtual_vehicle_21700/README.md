# Virtual Vehicle BAK N21700CG-50 external package audit

Zenodo record: **18849418**; package contains eight repeated BAK N21700CG-50 thermal-runaway tests at 60% SOC.

- Archive: `report_BAK_N21700CG-50_60SOC_TS0330.zip` (~4.62 GiB)
- ZIP central-directory audit: **success**
- Archive file entries indexed without full download: **817**
- Candidate structured/raw data files: **20**
- The 5 GB raw package is not copied into this repository.

## Why this matters

This source adds **eight independent real destructive experiments**, which is more valuable for generalization claims than creating additional overlapping or synthetic windows. The Zenodo description states that the package includes cell temperature, vent-gas temperature, pressure, gas release and event information such as CID, burst-plate and thermal runaway.

## Next integration gate

Use `candidate_data_files.csv` to locate the smallest machine-readable raw/event files. Only those files should be selectively range-extracted for schema/event-time mapping; photos and videos are not needed for the first predictive protocol.
