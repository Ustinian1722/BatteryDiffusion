# Virtual Vehicle time-to-vent support audit

No predictor is fit in this audit. Labels use the package-provided first `vent_gas` event. Candidate definitions are compared only by real-experiment support and label-range coverage.

- Independent experiments: **8**
- Fully supported candidate definitions: **24**

## Highest-support candidates

| context | T cap | max target tau | min windows/test | median windows/test | worst tau span |
|---:|---:|---:|---:|---:|---:|
| 60 s | 150 C | 900 s | 37 | 39.0 | 360.0 s |
| 60 s | 150 C | 1200 s | 37 | 39.0 | 360.0 s |
| 60 s | 150 C | 600 s | 33 | 39.0 | 320.0 s |
| 120 s | 150 C | 600 s | 31 | 33.0 | 300.0 s |
| 120 s | 150 C | 900 s | 31 | 33.0 | 300.0 s |
| 120 s | 150 C | 1200 s | 31 | 33.0 | 300.0 s |
| 256 s | 150 C | 600 s | 17 | 19.0 | 160.0 s |
| 256 s | 150 C | 900 s | 17 | 19.0 | 160.0 s |
| 256 s | 150 C | 1200 s | 17 | 19.0 | 160.0 s |
| 60 s | 120 C | 900 s | 17 | 19.5 | 160.0 s |
| 60 s | 120 C | 1200 s | 17 | 19.5 | 160.0 s |
| 60 s | 120 C | 600 s | 12 | 18.0 | 110.0 s |
| 120 s | 120 C | 600 s | 11 | 13.5 | 100.0 s |
| 120 s | 120 C | 900 s | 11 | 13.5 | 100.0 s |
| 120 s | 120 C | 1200 s | 11 | 13.5 | 100.0 s |

## Guardrail

This audit is support-only. A regression protocol must be frozen from these support statistics before any time-to-vent model is scored. Window counts remain repeated decision points, not independent destructive experiments.
