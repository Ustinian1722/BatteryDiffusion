# Augmentation quality audit

- Real windows: **174**
- Accepted synthetic windows: **4**
- Overall feature SMD mean/max: **0.647 / 1.262**
- Overall feature range coverage: **78.3%**
- Overall real-real / synthetic-real NN median: **0.0246 / 0.2982**
- Overall NN ratio: **12.10**
- Overall RBF-MMD²: **0.1759**

## Automatic diagnostic

- Synthetic condition coverage is incomplete: age0_stage0, age0_stage1, age0_stage2.
- Synthetic windows may be too far from the real manifold; inspect OOD behavior.

## Interpretation guardrail

All metrics here are descriptive at window level. They do not change the number of independent TR experiments and must not be reported as experiment-level validation.
