# Learning-curve protocol amendment — equal augmentation budget

Date: 2026-09-12

This amendment is committed **before any learning-curve predictive score has been inspected**.

## Design issue identified

The first launched learning-curve workflow (`Evaluate Virtual Vehicle experiment-count learning curve`, initial run) used a fixed `--candidates 160` for k=2, 4 and 6 real training experiments. Although retained synthetic windows were capped by the number of real training windows, a fixed candidate pool can make the realized synthetic:real ratio fall as k increases simply because larger-k folds contain more real windows.

That would confound the intended scientific question — whether augmentation becomes less useful as **independent real experiment coverage** increases — with a trivial reduction in augmentation budget.

Therefore the initial fixed-160 run is declared **design-invalid for the scarcity learning-curve claim regardless of its eventual numerical result**. Its outputs, if committed, are audit-only and must not be used in the paper.

## Corrected equal-budget protocol

The task, held-out experiments, deterministic cyclic training subsets, predictor, RA-CDiff training hyperparameters and exact-anchor classical comparator are unchanged.

Only the *number of candidate synthetic proposals* is increased with k to ensure the quality gate has enough proposals to reach the existing retained cap of one synthetic window per real training window whenever possible:

- k=2: 240 candidates;
- k=4: 480 candidates;
- k=6: 720 candidates.

After quality/task screening, retained synthetic count is still capped at `n_real_train_windows`. Thus the target augmentation ratio is 1:1 at every k. Actual retained ratios are archived per fold; any shortfall remains visible and will not be silently back-filled with lower-quality samples.

Candidate count affects only how many post-training proposals are screened; it does not change the trained denoiser, diffusion schedule, noise-start range, task definition or downstream learner.

The corrected output is stored separately under `reports/virtual_vehicle_experiment_count_learning_curve_equal_budget/` so it cannot be confused with the discarded fixed-candidate run.
