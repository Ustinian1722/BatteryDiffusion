# Warwick/Mendeley 21700 external dataset audit

Dataset record: **rgfhdhcd9k**, public version 1; the 2025 Data in Brief descriptor cites DOI **10.17632/rgfhdhcd9k.2** while linking to the version-1 public page.

- Status: **raw-file discovery blocked/unavailable on CI; source-supported mapping only**
- Public files discovered/downloaded: **0**
- Numeric arrays/columns inspected directly: **0**
- Candidate thermo-pressure/time/voltage arrays directly inspected: **0**
- Raw external files are never committed by this audit.

## Source-supported structure

The open data descriptor reports three independent Sony VTC6A 21700 tests at 100% SOC, triggered by 40 W external heating. The processed MATLAB data contain internal pressure, voltage, internal/surface temperatures and two vent-temperature channels. Pressure/voltage channels use a 1 kHz acquisition time base and temperature channels a 10 Hz time base. Exact field names and units are recorded in `source_supported_schema.csv`.

## Access note

If `file_manifest.csv` is empty, that is an access-layer result rather than evidence that the dataset has no files. Mendeley may block the interactive page from hosted runner IPs. We therefore keep source-derived schema separate from raw-file-derived schema and do not fabricate file identifiers.

## Intended use

Once raw file access is resolved, these three tests are the highest-priority cylindrical external thermo-pressure cohort. Event landmarks will be frozen from the source paper / pressure-drop trace before predictive evaluation.
