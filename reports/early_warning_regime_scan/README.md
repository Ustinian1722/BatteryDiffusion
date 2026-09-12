# Thermally subtle impending-vent regime scan

A sample is retained only if the **maximum temperature anywhere in its 256-s causal input window** is at or below the specified ceiling. This prevents an apparently strong early-warning result from being driven solely by obvious high-temperature escalation.

All reported predictive scores are leave-one-real-experiment-out; configurations without both classes in every held-out experiment are excluded from the top candidate table.

## Best fully evaluable definitions

| horizon | T cap | T AUROC | force AUROC | fusion AUROC | force gain | fusion gain | worst fusion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 900s | 120°C | 0.969 | 0.507 | 0.999 | -0.462 | +0.030 | 0.998 |
| 900s | 150°C | 0.971 | 0.728 | 0.987 | -0.243 | +0.016 | 0.948 |
| 900s | 200°C | 0.975 | 0.773 | 0.982 | -0.202 | +0.007 | 0.928 |
| 600s | 150°C | 1.000 | 0.936 | 0.999 | -0.064 | -0.000 | 0.998 |
| 600s | 200°C | 1.000 | 0.918 | 0.987 | -0.082 | -0.013 | 0.950 |
| 300s | 200°C | 1.000 | 0.975 | 0.975 | -0.025 | -0.025 | 0.900 |

## Guardrail

This scan is exploratory task-definition work on four experiments. A chosen threshold/horizon must be frozen before any final model comparison, and later final claims require either external data or explicitly small-sample scope.
