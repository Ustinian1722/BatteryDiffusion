# Augmentation quality audit

- Real windows: **174**
- Accepted synthetic windows: **121**
- Overall feature SMD mean/max: **0.881 / 2.372**
- Overall feature range coverage: **82.9%**
- Overall real-real / synthetic-real NN median: **0.0246 / 0.3789**
- Overall NN ratio: **15.38**
- Overall RBF-MMD²: **0.5974**

## Automatic diagnostic

- Synthetic condition coverage is incomplete: age1_stage0.
- Synthetic windows may be too far from the real manifold; inspect OOD behavior.

## Interpretation guardrail

All metrics here are descriptive at window level. They do not change the number of independent TR experiments and must not be reported as experiment-level validation.
