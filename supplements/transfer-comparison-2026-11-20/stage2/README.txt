DELTA PHYSICS: EXECUTED STAGE 2 STUDY
===================================
Report date: 2026-11-20 (Asia/Shanghai)

This package contains actual newly trained Burgers FNO weights and experiments,
mathematical implementation checks, a reusable evaluator, and 2D Navier-Stokes
numerical-reference controls. It does not contain the manuscript's unavailable
original source/checkpoints or an implementation of its original Aitken solver.

Read delta_physics_stage2_report_zh.txt first for results, limitations and next
experiments. Do not mix numerical-reference controls with learned-model results.

Executed work
-------------
1. validate_torch_cpu.py: 168 analytical CPU cases on Torch2.7.1+cpu;
   float32/64,1D/2D,non-unit and rectangular domains. GPU unavailable.
2. burgers_fno/run_burgers_fno.py: independent FNO1d reconstruction;
   3 training seeds,75epochs,7 viscosities,7 methods,8 paired test ICs,50steps.
   147 combinations;1,176 trajectories;58,800 per-IC-step metric rows.
3. burgers_fno/run_budget_ablation.py: fixed snapshots of the same runs at
   epochs15/30/50/75,ratio2;84 combinations and33,600 metric rows.
4. runner/evaluate.py: unified paired accuracy evaluator. NumPy and Torch CPU
   controls executed. Reconstructed neural seed0/ratio2 reproduced bitwise
   through a separate checkpoint adapter, including failed-trajectory NaNs.
5. ns_controls/run_ns_controls.py:64/128 grids,ratios1.5/2/4,3 shared continuous
   ICs,7 methods,50steps. Reference flow is numerical, not learned.
6. benchmark_burgers_cpu.py: actual CPU wall-clock experiments on neural seed0,
   ratio2,B1/B8;3warmup+10counterbalanced-order repeats;optimized float32 RK4
   at3internal steps. Four-thread and single-thread Torch settings are separate.
   SciPyFFT uses1worker. No GPU timing or original-model speedup is claimed.

Important provenance
--------------------
The reconstructed FNO uses32width,16modes,4Fourier+pointwise layers and a128-wide
projection head. Its source schedule is StepLR25x0.5, fixed before training.
Source data and validation data are shared across the3 model seeds. This is
declared independent reconstruction, not an exact replication of the paper.
Targets were generated after all3 models were trained. Source validation alone
selected checkpoints. All3 selected epoch75; validation was still improving,
so "fully converged training" has not been demonstrated.

All physical correction wrappers use one actual model evaluation per step.
The outside-network implicit formula is NOT the paper v2 nonlinear-input Recast
equation. Original Recast/Aitken remains unavailable; it is never substituted.

Failures remain in results; null aggregates are not replaced by survivor means.
Truth uses float64. Neural prediction uses float32 with complex64 Fourier
weights; do not cast a complex FNO module to a real dtype.

Reproduce into fresh directories
--------------------------------
Install compatible numpy,scipy,matplotlib and a CPU/CUDA PyTorch distribution.
Recorded environment: Python3.12.14,NumPy2.3.5,SciPy1.17.0,Torch2.7.1+cpu.
The supplied requirements-stage2.txt records versions, not a GPU installation
recipe. Choose the appropriate official PyTorch build for your machine.

From this package directory:
  python validate_torch_cpu.py --out new_torch_checks.json
  python burgers_fno/run_burgers_fno.py --out burgers_fno/reproduced_results
  python burgers_fno/run_budget_ablation.py --source burgers_fno/reproduced_results --out burgers_fno/reproduced_budget
  python ns_controls/run_ns_controls.py --out ns_controls/reproduced_results
  python runner/prepare_burgers_case.py --results burgers_fno/reproduced_results --seed 0 --ratio 2 --out runner/reproduced_case
  python runner/evaluate.py --config runner/reproduced_case/config.json --out runner/reproduced_evaluation --save-rollouts
  python benchmark_burgers_cpu.py --results burgers_fno/reproduced_results --torch-threads 4 --out reproduced_benchmark_4threads
  python benchmark_burgers_cpu.py --results burgers_fno/reproduced_results --torch-threads 1 --out reproduced_benchmark_1thread

Run timings after all other experiment processes finish. They exclude loading,
training,truth generation and metric calculation; all50physical output states
are computed, but only the final state is retained in timed runs. Whole-rollout
accuracy/failure checks occur separately before timing. Absolute latencies and
small differences between wrappers are machine/implementation dependent.

File map
--------
burgers_fno/results/: real checkpoints,source data,normalizer,target truth,
  full predictions,training curves,per-step/per-seed/aggregate metrics.
burgers_fno/budget_ablation/: fixed-epoch diagnostic and complete curves.
ns_controls/results/: numerical controls,convergence,sparse field snapshots,
  all per-step metrics and explicit-stiffness diagnostics.
runner/: strict adapters,example config/data,and verified integration outputs.
cpu_benchmark/: Torch4-thread CPU result and raw timings.
cpu_benchmark_threads1/: Torch1-thread sensitivity result and raw timings.
figures/: summary figures generated from these recorded results.
tables/: exact CSV exports of neural, fixed-epoch, CPU timing and NS controls.
make_report.py: rebuild the Chinese report and table exports from results.
make_figures.py: rebuild the four summary PNG/PDF figures from results.
shared/: NumPy and Torch one-call correction implementations.
execution_manifest.json: evidence categories and SHA256 identity of deliverables.

The benchmark source used for each recorded timing run is preserved as that
run's benchmark_script_used.txt. Its hash matches the script_sha256 in the
corresponding benchmark.json even if the root script later adds CLI options.

Next steps requiring original assets
-----------------------------------
Load the original NS/F-FNO source,checkpoints,source normalizers,and original
test trajectories through an original_neural adapter. First reproduce existing
Frozen/Delta results. Then compare all single-call methods and the exact original
input-Recast implementation on the same checkpoints. See runner/README.txt for
the contract and next_experiments.json for the prioritized research matrix.
