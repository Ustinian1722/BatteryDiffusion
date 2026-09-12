# Augmentation quality audit

- Real windows: **122**
- Accepted synthetic windows: **274**
- Overall feature SMD mean/max: **0.784 / 1.864**
- Overall feature range coverage: **88.5%**
- Overall real-real / synthetic-real NN median: **0.0333 / 0.0309**
- Overall NN ratio: **0.93**
- Overall RBF-MMD²: **0.2558**

## Automatic diagnostic

- No automatic red flag was triggered by the descriptive thresholds, but downstream usefulness is not yet established.

## Interpretation guardrail

All metrics here are descriptive at window level. They do not change the number of independent TR experiments and must not be reported as experiment-level validation.
