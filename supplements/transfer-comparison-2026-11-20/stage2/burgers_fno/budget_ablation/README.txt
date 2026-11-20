Fixed-epoch snapshot ablation, independently reconstructed Burgers FNO
==================================================================
This is a follow-up diagnostic of the primary study's saved models. It is not
the original manuscript model and not a new independently trained dataset/run
at every budget. Three training seeds share the primary source dataset.

All saved epochs15,30,50,75 are evaluated at viscosity ratio2 with identical
eight held-out initial conditions, 50 rollout steps, source normalization,
and seven wrappers. No new training, target-dependent hyperparameter tuning,
or primary-result modification was performed. A fixed-epoch snapshot is not
the best source-validation checkpoint within a truncated training budget.

Outputs:
- ablation_config.json locks these choices and source artifact hashes.
- per_step_metrics.csv stores33600 individual IC-step records.
- per_seed_summary.csv/json store84 model-budget-method rows.
- aggregate_summary.csv/json report per-budget mean/sampleSD across3 seeds.
- execution_summary.json records672 trajectories and exact equality with the
  primary epoch75 results.
- fixed_budget_comparison.png/pdf visualize Avg50; explicit failures are
  listed in the caption instead of being replaced by survivor means.

Avg50 results favor Strang overDelta at all tested budgets in the across-seed
means. The relative within-seed Avg50 reductions average19.36%,20.77%,23.51%,
28.40% at15/30/50/75 respectively. This is conditional evidence that stronger
source training does not remove the need for classical composition baselines.
It is not proof of universal dominance: Step50 rankings and individual seed
differences are more mixed, and all models use the same small source dataset.
