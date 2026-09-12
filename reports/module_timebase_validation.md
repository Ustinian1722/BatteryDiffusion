# Module time-base validation against published TR landmarks

Source paper: **Multi-modal thermal runaway dataset of fresh and aged lithium-ion battery cells and modules**, *Scientific Data* (2026), DOI: `10.1038/s41597-026-07857-1`.

The downloaded module temperature CSVs do not expose a conventional monotonic `Test_time(s)` column. Instead, their first field is `ms` with five repeated values per second. The repository audit therefore reconstructs elapsed time by counting `ms` resets and adding `ms/1000`, which yields a 5 Hz temperature time base.

This reconstruction can be independently checked against the thermal-runaway landmarks reported in the source paper:

| Experiment | Published first vent time | Published vent-to-peak interval for cell 1 | Implied cell-1 peak time | Reconstructed temperature evidence |
|---|---:|---:|---:|---|
| M1 fresh module | 2137 s | 478 s | ~2615 s | TC1-6 peak neighborhood is ~2608-2617 s; hottest cell-1 TC peaks at ~2614.4 s |
| M2 aged module | 1569 s | 1039 s | ~2608 s | TC1-6 peak neighborhood is ~2587-2674 s; hottest cell-1 TC peaks at ~2599.8 s |

For M1, the reconstructed cell-1 peak agrees with the paper-implied value to within roughly one second. M2 is also in the same peak neighborhood, with the hottest trigger-cell thermocouple peaking within roughly ten seconds of the paper-implied value while other trigger-cell thermocouples peak over the surrounding tens of seconds.

This agreement is strong evidence that the repeated `ms` field is a milliseconds-within-second field and that the reset-count reconstruction preserves the physical test-time scale. The force files have different start/end coverage, so module fusion uses only their exact overlapping test-time interval rather than stretching either signal to match the other.

## Thermocouple grouping used downstream

The source paper states that module thermocouples were attached as follows:

- cell 1: TC1-3 on the heating-film side and TC4-6 on the opposite side;
- cell 2: TC7-9;
- cell 3: TC10-11;
- cell 4: TC12-15.

The module preprocessing therefore reduces the 15 thermocouples to four hot-spot-preserving per-cell maximum-temperature channels and aligns them with module force at 1 Hz. This retains propagation ordering while avoiding an unnecessarily high-dimensional 16-channel diffusion model in an `n=2` module-experiment regime.

## Guardrail

This validation concerns **time-base reconstruction only**. It does not increase the number of independent module experiments. M1 and M2 remain two experimental units, and any synthetic module windows are local training augmentation only.
