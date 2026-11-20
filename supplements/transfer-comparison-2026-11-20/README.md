# Frozen-model transfer: supplementary comparisons and controls

This supplement preserves the mathematical checks, executable comparison code, independently trained Burgers FNO models, numerical controls, and CPU measurements completed on 18–20 November 2026. Its main finding is that classical one-call compositions are substantive competitors to the additive Δ-Physics correction. The ranking depends on the metric and the quality of the frozen source predictor.

**Start with [FINDINGS.md](FINDINGS.md) for the interpretation and mathematical qualifications, or the [complete Chinese report](stage2/delta_physics_stage2_report_zh.txt) for the detailed execution record.** 中文导航：本页给出结果、文件入口与复现命令；完整中文报告保留原文，当前仓库资产状态以本页的更新说明为准。

## Provenance and current repository status

The `stage1/` and `stage2/` directories preserve the original deliverables and their embedded SHA-256 manifests. This README and `FINDINGS.md` provide a new navigation and interpretation layer; they do not revise the archived experiment records.

The formula review used the supplied [v2 manuscript snapshot](reviewed_manuscript/onesided-delta-physics-icml-v2.pdf), which differs from the repository's current `paper_draft/main.pdf`. Conclusions about that reviewed version should not be read as an audit of every later manuscript or repository change. See [IMPORT_NOTES.md](IMPORT_NOTES.md) for the input-snapshot and import record.

**Historical asset statements need their original context.** The archived reports, manifests, and integration notes were written before the original repository was supplied to that analysis. Statements such as “original assets unavailable,” “original Aitken unavailable,” or Stage 1's “PyTorch untested” describe the environment and inputs at the time of those records. They are not an inventory of the current repository.

The current repository already contains [the original model package](../../fno_dual/), [the NS64 75-epoch seed-0 results and model assets](../../results/ffno_ns64_budget75_seed0/), [the input-Recast/vector-Aitken script](../../run_ffno_equation_recast_3seed.py), and [trajectory data](../../data/trajectories/). Their presence does not turn the independent supplement into an original-model reproduction. **This upload did not rerun those original repository models, their Recast implementation, or GPU experiments.** An audited comparison on the original checkpoints remains separate work. Stage 2 did execute the supplemental PyTorch wrappers on CPU, superseding Stage 1's earlier runtime-status limitation for those tested paths.

## Evidence categories

| Component | What actually ran | Interpretation |
| --- | --- | --- |
| [Stage 1 controls](stage1/README.txt) | NumPy heat-flow checks, a six-dimensional nonlinear local-order test, and a Burgers numerical-reference rollout | Mathematical and implementation controls; no learned predictor |
| [Stage 2 analytical checks](stage2/torch_cpu_checks.json) | 168 PyTorch CPU cases, float32/float64, one/two dimensions, non-unit and rectangular domains | Wrapper validation; no original-checkpoint validation and no CUDA test |
| [Burgers FNO main study](stage2/burgers_fno/results/execution_summary.json) | Three newly trained FNO models, seven viscosity ratios, seven methods, eight paired test initial conditions, 50 steps | `independently_reconstructed_neural_not_manuscript_reproduction` |
| [Fixed-epoch study](stage2/burgers_fno/budget_ablation/) | Epoch 15/30/50/75 snapshots of those same three training runs, at viscosity ratio 2 | Correlated training-budget diagnostics; not 12 independent training seeds |
| [NS64/NS128 controls](stage2/ns_controls/README.txt) | A high-accuracy numerical source-flow callback, two grids, three viscosity ratios, three common continuous initial conditions | `numerical_reference_control_not_neural`; the seeds identify initial conditions |
| [CPU benchmark](stage2/cpu_benchmark_threads1/benchmark.json) | Reconstructed neural seed 0 and an optimized spectral RK4 solver; batches 1 and 8; one- and four-thread Torch settings | Measured implementation latency on one CPU configuration |
| [Unified evaluation](stage2/runner/README.txt) | Heat controls and the reconstructed Burgers seed-0/ratio-2 case | Adapter and rollout cross-check; no original-model run |
| [Next experiments](stage2/next_experiments.json) | A specified research matrix | Planned work, without experiment results |

The main Burgers study contains 147 seed–ratio–method combinations, 1,176 trajectories, and 58,800 per-initial-condition/per-step metric rows. The fixed-epoch diagnostic adds 84 combinations and 33,600 metric rows, with epoch 75 overlapping the main study. The NS numerical controls contain 126 trajectories and 6,300 metric rows. The two CPU thread settings each retain 180 timed repetitions. See the [Stage 2 manifest](stage2/execution_manifest.json).

## Main recorded results

### Three-seed learned Burgers comparison

At viscosity ratio $\nu/\nu_0=2$, grid size 64, output step $h=0.02$, and 50 output steps, the recorded results are:

| Method | Avg-50, mean ± sample SD | Step-50, mean ± sample SD | Failed seed–IC pairs |
| --- | ---: | ---: | ---: |
| Frozen FNO | 0.278844 ± 0.003757 | 0.416712 ± 0.009290 | 0/24 |
| Δ-Physics | 0.021481 ± 0.001781 | 0.030129 ± 0.004237 | 0/24 |
| Lie, pre | 0.017524 ± 0.000396 | 0.032149 ± 0.002510 | 0/24 |
| Lie, post | 0.016856 ± 0.000634 | 0.030896 ± 0.003668 | 0/24 |
| Strang | 0.015329 ± 0.000407 | 0.030860 ± 0.003092 | 0/24 |
| Outside-network implicit | 0.016212 ± 0.000371 | 0.030625 ± 0.003455 | 0/24 |
| Explicit correction | Missing after failure | Missing after failure | 24/24 |

Source: [main CSV](stage2/tables/burgers_fno_results.csv), [per-seed results](stage2/burgers_fno/results/per_seed_summary.csv), and [comparison figure](stage2/figures/burgers_fno_comparison.png). The SD is across **three training seeds**, which share one source dataset. It is not a confidence interval. No statistical-significance claim is made.

Strang reduces Avg-50 relative to Δ in each of the three seeds, with a mean paired percentage reduction of **28.40%**. Δ has the slightly lower across-seed Step-50 mean. These are different criteria; the experiment does not establish that one method dominates every metric. The outside-network implicit baseline also improves Avg-50 relative to Δ, by **24.18%** using the same paired-percentage convention. Both use one actual network call per physical step. The full definitions and scope are in [FINDINGS.md](FINDINGS.md).

### CPU timing and numerical controls

For **the reconstructed Burgers seed 0 only**, on the 64-point problem at ratio 2 and batch 8, single-thread Torch and one-worker SciPy FFT gave these median 50-step times:

| Method | Median time | Avg-50 in the benchmark accuracy precheck |
| --- | ---: | ---: |
| Δ-Physics | 69.266 ms | 0.0210083 |
| Strang | 69.727 ms | 0.0149828 |
| Outside-network implicit | 69.367 ms | 0.0158042 |
| Direct spectral RK4, internal $\Delta t=0.002$ | 42.534 ms | $3.48084\times10^{-7}$ |

Source: [timing CSV](stage2/tables/burgers_cpu_timing.csv) and [single-thread benchmark record](stage2/cpu_benchmark_threads1/benchmark.json). The direct solver is faster and more accurate against the same time-refined, fixed-grid reference in this setting. This is not a result about GPU performance, high-resolution workloads, other PDEs, or amortized training cost. Accuracy here is from one trained model, not the three-seed mean above.

In the **NS128 numerical-reference control** at ratio 2, Avg-50 is 0.00732319 for Δ and 0.0000134112 for Strang. This isolates composition error with a highly accurate numerical source flow. It is not a neural F-FNO result. The paired NS64/NS128 fields are smooth, low-bandwidth cases; their agreement is a resolution-consistency check, not evidence of a difficult high-resolution learning task. See the [NS CSV](stage2/tables/ns_numerical_control_results.csv) and [NS documentation](stage2/ns_controls/README.txt).

## Tables and figures

All four primary CSV exports and all four summary figure sets are directly linked here. Figures are generated from recorded results; no planned results appear in them.

| Content | Exact CSV | Figure |
| --- | --- | --- |
| Three-seed Burgers, all viscosity ratios | [burgers_fno_results.csv](stage2/tables/burgers_fno_results.csv) | [PNG](stage2/figures/burgers_fno_comparison.png) · [PDF](stage2/figures/burgers_fno_comparison.pdf) |
| Fixed-epoch Burgers and source training curves | [burgers_fixed_epoch_results.csv](stage2/tables/burgers_fixed_epoch_results.csv) | [PNG](stage2/figures/burgers_training_curves.png) · [PDF](stage2/figures/burgers_training_curves.pdf) |
| CPU timing, both thread settings and batches | [burgers_cpu_timing.csv](stage2/tables/burgers_cpu_timing.csv) | [PNG](stage2/figures/burgers_cpu_timing.png) · [PDF](stage2/figures/burgers_cpu_timing.pdf) |
| NS64/NS128 numerical controls | [ns_numerical_control_results.csv](stage2/tables/ns_numerical_control_results.csv) | [PNG](stage2/figures/ns_resolution_controls.png) · [PDF](stage2/figures/ns_resolution_controls.pdf) |

The second summary figure shows the source-validation curves. A separate [fixed-budget comparison PNG](stage2/burgers_fno/budget_ablation/fixed_budget_comparison.png) and [PDF](stage2/burgers_fno/budget_ablation/fixed_budget_comparison.pdf) visualize the fixed-epoch transfer results.

## Code, weights, full outputs, and integrity

| Artifact | Entry point or location |
| --- | --- |
| Full interpretation | [FINDINGS.md](FINDINGS.md); [original Chinese Stage 2 report](stage2/delta_physics_stage2_report_zh.txt); [Stage 1 protocol](stage1/experiment_protocol_zh.txt) |
| Main neural experiment | [run_burgers_fno.py](stage2/burgers_fno/run_burgers_fno.py); [locked configuration](stage2/burgers_fno/results/locked_config.json); [training summary and checkpoint hashes](stage2/burgers_fno/results/training_summary.json) |
| Trained weights and complete neural predictions | [seed 0](stage2/burgers_fno/results/seed_0/), [seed 1](stage2/burgers_fno/results/seed_1/), [seed 2](stage2/burgers_fno/results/seed_2/); each includes the selected checkpoint and fixed-epoch snapshots |
| Source data, target truth, and normalizer | [results directory](stage2/burgers_fno/results/), including `source_data.npz`, `truth_r*.npz`, and [normalization.json](stage2/burgers_fno/results/normalization.json) |
| Complete neural metrics | [per-seed CSV](stage2/burgers_fno/results/per_seed_summary.csv); [per-step CSV](stage2/burgers_fno/results/per_step_metrics.csv); [aggregate JSON](stage2/burgers_fno/results/aggregate_summary.json) |
| Fixed-epoch experiment | [run_budget_ablation.py](stage2/burgers_fno/run_budget_ablation.py); [results](stage2/burgers_fno/budget_ablation/) |
| One-call correction implementations | [NumPy](stage2/shared/spectral_transfer_numpy.py); [PyTorch](stage2/shared/spectral_transfer_torch.py); [CPU validation script](stage2/validate_torch_cpu.py) |
| Strict evaluator and checkpoint adapter | [evaluate.py](stage2/runner/evaluate.py); [burgers_reconstruction_adapter.py](stage2/runner/burgers_reconstruction_adapter.py); [verified cross-check](stage2/runner/burgers_seed0_ratio2_results/crosscheck_against_training_script.json) |
| CPU timing source and raw repeats | [benchmark_burgers_cpu.py](stage2/benchmark_burgers_cpu.py); [one thread](stage2/cpu_benchmark_threads1/); [four threads](stage2/cpu_benchmark/) |
| NS numerical experiment | [run_ns_controls.py](stage2/ns_controls/run_ns_controls.py); [results, diagnostics, and sparse field snapshots](stage2/ns_controls/results/) |
| Early mathematical controls | [run_controls.py](stage1/run_controls.py); [failure-path checks](stage1/check_failure_paths.py); [control_results.json](stage1/results/control_results.json); [local-order CSV](stage1/results/local_order.csv); [local-order figure](stage1/results/local_order.png) |
| Report/figure generation | [make_report.py](stage2/make_report.py); [make_figures.py](stage2/make_figures.py) |
| Original artifact hashes | [Stage 1 manifest](stage1/execution_manifest.json); [Stage 2 manifest](stage2/execution_manifest.json) |
| Reviewed manuscript input | [Supplied v2 PDF snapshot](reviewed_manuscript/onesided-delta-physics-icml-v2.pdf); [import notes](IMPORT_NOTES.md) |
| Environment and execution instructions | [Stage 2 README](stage2/README.txt); [recorded requirements](stage2/requirements-stage2.txt); [Stage 1 README](stage1/README.txt) |
| Work not yet executed | [next_experiments.json](stage2/next_experiments.json) |

Each CPU run keeps the exact executed source in its own `benchmark_script_used.txt`; its hash is recorded in that run's `benchmark.json`. The four-thread run preceded the added thread-selection CLI flag, so its source-snapshot hash intentionally differs from the later script. The manifests preserve artifact identity and the original execution context; historical availability statements inside them are qualified above.

## Reproduce into fresh output directories

The commands below are taken from the archived [Stage 2 README](stage2/README.txt), with an explicit repository-relative starting directory. They rerun the **independent supplement**, not the original repository experiments. The recorded environment is Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8, and PyTorch 2.7.1+cpu. Use the [requirements file](stage2/requirements-stage2.txt) as the version inventory and select a compatible PyTorch distribution for your machine; it is not a CUDA reproduction recipe.

From the repository root:

```bash
cd supplements/transfer-comparison-2026-11-20/stage2

python validate_torch_cpu.py --out new_torch_checks.json
python burgers_fno/run_burgers_fno.py --out burgers_fno/reproduced_results
python burgers_fno/run_budget_ablation.py --source burgers_fno/reproduced_results --out burgers_fno/reproduced_budget
python ns_controls/run_ns_controls.py --out ns_controls/reproduced_results

python runner/prepare_burgers_case.py --results burgers_fno/reproduced_results --seed 0 --ratio 2 --out runner/reproduced_case
python runner/evaluate.py --config runner/reproduced_case/config.json --out runner/reproduced_evaluation --save-rollouts
```

After all other experiment processes finish, run the two timing configurations separately:

```bash
python benchmark_burgers_cpu.py --results burgers_fno/reproduced_results --torch-threads 4 --out reproduced_benchmark_4threads
python benchmark_burgers_cpu.py --results burgers_fno/reproduced_results --torch-threads 1 --out reproduced_benchmark_1thread
```

All output directories must be fresh. Timing excludes training, checkpoint/data loading, truth generation, and metric computation. It computes all 50 physical outputs but retains only the final field during the timed interval. Whole-rollout accuracy and failures are checked separately before timing. Do not compare absolute latencies across machines as if they were algorithmic constants.

For Stage 1, from the repository root:

```bash
cd supplements/transfer-comparison-2026-11-20/stage1
python run_controls.py --out reproduced_results
python check_failure_paths.py
```

The archived report/figure generators use fixed archived-result paths and can rewrite the report, tables, or figures. Run them in a separate working copy if regenerating presentation artifacts, so that the preserved original files continue to match their manifests. No experiment was launched as part of preparing these two supplementary Markdown files.
