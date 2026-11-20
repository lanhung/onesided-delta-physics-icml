Independent Burgers FNO transfer experiment
==========================================
Completed 2026-11-20. This folder contains REAL trained neural-model experiments
on CPU. They are an independent reconstruction, not a reproduction of the
uploaded manuscript's FNO checkpoint, dataset, source, or reported numbers.

What was fixed before target evaluation
--------------------------------------
One shared source dataset: 24 training and6 validation trajectories,20 source
transitions each,64 periodic points,nu0=.01,h=.02. Initial states contain four
random sine/cosine modes divided by mode number, normalized to RMS.5.
Training IC seeds1000..1023,validation seeds2000..2005; held-out test IC seeds
3000..3007. The same8 test initial conditions are used for all methods, ratios,
and independently initialized models.

FNO: state+coordinate input,width32,modes16,4 spectral-plus-pointwise layers,
GELU and128-wide projection head. It predicts a normalized physical increment.
State/increment global means and population standard deviations come ONLY from
training pairs. The wrapper returns the full next physical state.

Three model-training seeds0,1,2 affect initialization and batch shuffle, using
identical source data. Each trains75epochs,batch32,Adam(lr=.001,weight_decay=1e-6),
StepLR(step_size=25,gamma=.5). The minimum source-validation one-step physical
RelL2 selects the frozen checkpoint. This source-only schedule is a declared
reconstruction choice, not a recovered original75-epoch schedule.

All three selected checkpoint epochs were75. Validation errors were:
  seed0: .0026216334
  seed1: .0025344800
  seed2: .0022140791
Validation curves still decline gradually near the end;75epochs does not prove
complete optimization convergence. Parameter tensors contain74209 elements,
including complex Fourier weights; total real scalar degrees of freedom139745.

All target trajectories were generated AFTER all model training and checkpoint
selection. No target values selected the schedule, architecture, checkpoint,
normalization, correction strength, spectral cutoff, or wrapper method.

Evaluation and validity
-----------------------
Viscosity ratios1,1.25,1.5,1.75,2,3,4;50-step autoregressive rollouts; all seven
wrappers: Frozen,Delta,Explicit,Lie-pre,Lie-post,Strang,closed-form implicit.
147 seed/ratio/method runs=1176 individual50-step trajectories,58800 per-IC-step
metric rows. True model-forward hooks verify50 batched forwards per run for
every wrapper; this differs from400 scheduled trajectory-steps per run.

Network and wrapper inference use float32,CPU,2 Torch threads. Accuracy metrics
accumulate in float64 against the original float64 reference fields. There is
no projection or spectral clipping after a neural step. The original nonlinear-
input Recast/Aitken is NOT included because its implementation is unavailable.

Reference Burgers flow is the stage1 conservative2/3-dealiased Fourier RK4
discretization. Training source dt<=.0005; target truth dt<=.000125. Target
dt .00025->.000125 refinement differs by at most1.053e-10 relativeL2 over all
50steps,8ICs and7ratios. This establishes a small temporal discretization error
for the fixed64-point system. It does not establish spatial convergence of the
continuum PDE. The input and nonlinear output are dealias-filtered, and the
full Laplacian damps retained high-mode state errors; no full-state projection.

Metrics: RelL2 is per-IC spatial norm error divided by the target norm; absolute
physicalL2 equals RMS on the unit-length single-channel domain. PSRE uses full
orthonormal FFT over the spatial axis. Avg50 averages all50 steps and all8ICs
within a model seed. The table gives mean and sampleSD across3 training seeds;
these seeds are distinct from the numerical-control IC seeds elsewhere.

Failure: first nonfinite prediction or maximum absolute field>1e4, recorded per
IC; all later state/metric entries for it areNaN/missing. A full-test aggregate
is missing if ANY of its8ICs fail. Failed examples are zeroed only for arithmetic
inside the batch-independent eval model; they never reenter success statistics.

Key results at viscosity ratio2
--------------------------------
Method          Avg50 mean      training-seed SD  Step50 mean     failed/24
Frozen          .2788437034     .0037570943       .4167119755      0
Delta           .0214814945     .0017806860       .0301285360      0
Lie-pre         .0175240640     .0003963851       .0321491658      0
Lie-post        .0168563097     .0006341880       .0308957241      0
Strang          .0153293374     .0004074455       .0308598028      0
Implicit        .0162122277     .0003705240       .0306251055      0
Explicit        missing        missing           missing         24

Strang has lower Avg50 thanDelta in all3 training seeds at ratio2; its
within-seed reductions are28.6817%,32.7184%,23.8083% (mean28.4028%). However,
Delta has a slightly smaller across-seed Step50 mean thanStrang at this ratio.
Thus the result is not that Strang dominates every metric or every seed.
The full rollout curves and uncertainty should accompany any method ranking.

At ratio4 Avg50 means are: Frozen .9225214123,Delta .0295581708,
Lie-pre .0229486410,Lie-post .0226618279,Strang .0200101197,
Implicit .0258974218. Explicit fails all24 seed-IC pairs at ratios>=1.5.
At ratio1 all wrapper outputs recover the same Frozen prediction exactly.

Interpretation
--------------
This independent trained-model study strengthens the case for including
classical one-call baselines: Delta is very effective relative to Frozen,
while splitting and implicit correction can further reduce the average rollout
error. It does not decide the winner for the unavailable originalF-FNO or for
Navier-Stokes, and does not establish broad statistical significance from only
three training seeds on one shared source dataset.

No GPU run was executed. The approximately100s total recorded execution is
not a controlled inference benchmark and must not be converted to speedup.

Reproduce
---------
From this directory, in Python with NumPy,torch2.7.1+cpu andmatplotlib available:
  python3 run_burgers_fno.py --out results
  python3 make_figures.py --results results

Use a different output folder to preserve the supplied result files.
A source-only1epoch pilot is:
  python3 run_burgers_fno.py --out pilot --pilot
The pilot generated no target trajectories and did not alter hyperparameters.

File map
--------
run_burgers_fno.py: complete data generation,model,training,evaluation and metrics.
make_figures.py: plots derived from saved metrics; they do not alter experiments.
results/locked_config.json: declared design.
results/environment.json: runtime,determinism and executed script hash.
results/source_data.npz: shared source training andvalidation physical fields.
results/normalization.json: frozen source-only normalization statistics.
results/truth_r*.npz: exact test arrays [T+1,B,C,N] andIC identifiers.
results/temporal_refinement.json: per-ratio time-refinement results andtruth hashes.
results/seed_*/best_source_validation.pt: actual selected neural weights.
results/seed_*/epoch_15.pt etc.: fixed-budget snapshots,NOT alternate target-selected weights.
results/seed_*/training_curve.csv: all75epoch source metrics.
results/seed_*/training_metadata.json: checkpoint,data andnormalizer hashes.
results/seed_*/rollout_*.npz: every prediction andfailure index.
results/per_step_metrics.csv: all58800rows.
results/per_seed_summary.csv:147rows retaining failures andactualforward counts.
results/aggregate_summary.csv/json: descriptive mean/sampleSD andwithin-seed gains.
results/execution_summary.json: evidence class,counts andscope.
results/*png/pdf: source curves,viscosity comparison andratio2rollout plots.

Follow-up fixed-epoch training-budget ablation
-------------------------------------------
  python3 run_budget_ablation.py
This separately evaluates the saved epoch15/30/50/75 snapshots from the same
75epoch training trajectories at ratio2. It does not retrain or change the
primary study. Snapshots are actual fixed epochs, not best validation weights
within each budget. Full per-IC-step metrics and a plot are in budget_ablation/.
Its epoch75 metrics exactly match the primary result in all3 seeds/all7 methods.

Avg50 means, in epoch order15/30/50/75:
  Frozen:   .2783120 / .2813619 / .2775326 / .2788437
  Delta:    .0330397 / .0248121 / .0253096 / .0214815
  Lie-post: .0270091 / .0205812 / .0209721 / .0168563
  Strang:   .0266626 / .0195903 / .0194132 / .0153293
  Implicit: .0270293 / .0203529 / .0204002 / .0162122
Explicit fails all24 seed-IC pairs at every tested budget. Strang's across-seed
mean Avg50 is smallest at all four budgets. Step50 rankings differ and some
differences are small; report both metrics and individual seed values.

Independent evaluation-path cross-check
---------------------------------------
The neighboring runner package independently loads the saved seed0 model and
ratio2 truth through its generic CLI. All seven methods' full stored rollouts
match this script element-by-element, including explicit failure positions.
Avg50/Step50 differ by at most5.55e-17 from metric accumulation order. See
../runner/burgers_seed0_ratio2_results and its cross-check record.
