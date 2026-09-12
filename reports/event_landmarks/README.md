# Event-landmark audit

The source paper explicitly reports venting times for A2, D1, M1 and M2. Those values are stored as published labels. For B2/C1/C2, the repository currently stores only algorithmic pressure-drop candidates; they are **not yet frozen as ground truth**.

## Published labels

| ID | vent time (s) | vent-to-peak (s) | implied peak (s) |
|---|---:|---:|---:|
| A2 | 1812 | 614 | 2426 |
| D1 | 1703 | 1037 | 2740 |
| M1 | 2137 | 478 | 2615 |
| M2 | 1569 | 1039 | 2608 |

## 2170 pressure-drop candidates

### B2
- candidate 1: t=1346.0 s, dP/dt=-1.2632, P_smooth=6.405, T-peak=1435.0 s
- candidate 2: t=1356.0 s, dP/dt=-0.0581, P_smooth=3.6681, T-peak=1435.0 s
- candidate 3: t=1366.5 s, dP/dt=-0.0156, P_smooth=3.3275, T-peak=1435.0 s
- candidate 4: t=1387.0 s, dP/dt=-0.0087, P_smooth=3.1675, T-peak=1435.0 s
- candidate 5: t=1377.0 s, dP/dt=-0.0075, P_smooth=3.2269, T-peak=1435.0 s

### C1
- candidate 1: t=1599.0 s, dP/dt=-1.0848, P_smooth=5.5406, T-peak=1632.0 s
- candidate 2: t=1609.0 s, dP/dt=-0.0535, P_smooth=3.2338, T-peak=1632.0 s
- candidate 3: t=1619.0 s, dP/dt=-0.0218, P_smooth=2.9152, T-peak=1632.0 s
- candidate 4: t=1176.0 s, dP/dt=-0.0169, P_smooth=-0.0656, T-peak=1632.0 s
- candidate 5: t=1153.0 s, dP/dt=-0.0142, P_smooth=-0.0488, T-peak=1632.0 s

### C2
- candidate 1: t=814.0 s, dP/dt=-0.0082, P_smooth=-0.0578, T-peak=1607.0 s
- candidate 2: t=1102.0 s, dP/dt=-0.0079, P_smooth=-0.0568, T-peak=1607.0 s
- candidate 3: t=1304.0 s, dP/dt=-0.0079, P_smooth=0.0158, T-peak=1607.0 s
- candidate 4: t=1144.0 s, dP/dt=-0.0077, P_smooth=-0.0618, T-peak=1607.0 s
- candidate 5: t=1400.0 s, dP/dt=-0.007, P_smooth=0.0376, T-peak=1607.0 s

## Downstream implication

Time-to-vent / impending-vent prediction is label-supported for the pouch cell and module experiments. A 2170 version is feasible only after the pressure-drop candidates are visually/algorithmically validated and frozen before any predictive experiment.
