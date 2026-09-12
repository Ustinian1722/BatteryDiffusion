# Augmentation quality audit

- Real windows: **122**
- Accepted synthetic windows: **239**
- Overall feature SMD mean/max: **0.893 / 2.179**
- Overall feature range coverage: **87.9%**
- Overall real-real / synthetic-real NN median: **0.0333 / 0.0436**
- Overall NN ratio: **1.31**
- Overall RBF-MMD²: **0.3695**

## Automatic diagnostic

- No automatic red flag was triggered by the descriptive thresholds, but downstream usefulness is not yet established.

## Interpretation guardrail

All metrics here are descriptive at window level. They do not change the number of independent TR experiments and must not be reported as experiment-level validation.
