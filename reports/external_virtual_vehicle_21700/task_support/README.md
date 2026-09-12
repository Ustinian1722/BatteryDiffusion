# Virtual Vehicle TS0330 task-support audit

This audit uses all eight independent BAK N21700CG-50 tests. The package-provided first `vent_gas` marker is treated only as the candidate event landmark for support counting. No prediction model is fit here.

Compact export conventions were audited directly: time is represented in milliseconds, pressure in Pa (converted to absolute bar for readability), and cell-case temperature in degC. Trajectories are interpolated to a common 1 Hz grid before causal-window counting.

- Experiments: **8**
- Fully evaluable task definitions (both classes in all 8 tests): **3**

## Best-supported definitions

| window | horizon | T cap | min positive/test | min negative/test | total windows |
|---:|---:|---:|---:|---:|---:|
| 256 s | 300 s | 150 C | 3 | 5 | 172 |
| 120 s | 300 s | 150 C | 3 | 18 | 283 |
| 60 s | 300 s | 150 C | 3 | 24 | 333 |

## Guardrail

This file is label-support analysis only. Any secondary external predictive protocol must be frozen using these counts before model scores are computed. The earlier 900 s / 120 C frozen test is not retroactively changed.
