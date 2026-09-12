# Detailed dataset audit

- Independent experiment IDs: **9**
- Multimodal experiments (temperature + mechanical): **7**
- Signal files audited: **16**

## Independent-unit warning

Time samples or sliding windows from the same experiment are not independent experimental units. 
Any downstream evaluation must split by `experiment_id` before windowing or generative-model fitting.

## Files

- `experiment_inventory.csv`: one row per independent experiment.
- `signal_file_audit.csv`: one row per recorded signal file.
