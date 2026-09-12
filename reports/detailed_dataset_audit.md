# Detailed dataset audit

- Independent experiment IDs present in the archive: **9**
- Multimodal experiments (temperature + mechanical): **7**
- Signal files audited: **16**

> Note: the published paper reports 8 TR experiments, while this downloaded archive exposes 9 experiment IDs. The repository therefore preserves the archive as-is and records IDs explicitly rather than silently reconciling them.

## Independent-unit warning

Time samples or sliding windows from the same experiment are not independent experimental units.
Any downstream evaluation must split by `experiment_id` before windowing or fitting a generator/predictor.

## Files

- `experiment_inventory.csv`: one row per experiment ID.
- `signal_file_audit.csv`: one row per recorded signal file.
