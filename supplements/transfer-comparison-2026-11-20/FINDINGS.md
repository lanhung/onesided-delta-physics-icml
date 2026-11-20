# Findings, mathematical scope, and next experiments

This document explains the archived evidence in [Stage 1](stage1/README.txt) and [Stage 2](stage2/README.txt). The full execution narrative remains available in the [Chinese report](stage2/delta_physics_stage2_report_zh.txt). The conclusions below separate exact mathematical statements, independent learned-model results, numerical-reference controls, and work that has not been executed.

**The evidence supports comparing several ways to combine an analytic physical correction with a frozen predictor.** Δ improves substantially over Frozen in the reconstructed Burgers experiment, while Strang and outside-network implicit correction improve the trajectory-average error further. The endpoint ranking differs. CPU measurements also show that a small-grid direct solver can be faster than these learned wrappers. None of these findings is a new evaluation of the original repository's F-FNO checkpoints.

## 1. Historical records and current assets

The archived reports refer to [the supplied v2 manuscript snapshot](reviewed_manuscript/onesided-delta-physics-icml-v2.pdf), identified by SHA-256 `ca6f5da8564eb7287d14808be5633a09ff5495996a98e118696471219ecdc473`. That reviewed input differs from the repository's current `paper_draft/main.pdf`. The formula discussion concerns the supplied version and does not constitute an audit of all subsequent manuscript or repository changes. The PDF is preserved separately from the unchanged `stage1/` and `stage2/` packages. The [Stage 1 manifest](stage1/execution_manifest.json), [Stage 2 manifest](stage2/execution_manifest.json), and [import notes](IMPORT_NOTES.md) record the relevant provenance and file identities.

Statements about missing original assets are historical. At the time of those independent analyses, the original repository had not been supplied. The repository containing this supplement now includes [model code](../../fno_dual/), [NS64 75-epoch seed-0 assets](../../results/ffno_ns64_budget75_seed0/), [the input-Recast/vector-Aitken script](../../run_ffno_equation_recast_3seed.py), and [trajectory data](../../data/trajectories/). Their current presence resolves the earlier lack of access to those files; it does not certify that all assets needed for a new three-seed, 75-epoch comparison have been audited or that such a comparison has run.

This upload preserves completed supplemental work and adds documentation. It did **not** rerun the original models or Aitken solver. No GPU experiment is included. Stage 1's statement that its Torch code had only been syntax-checked is also historical: Stage 2 subsequently performed [168 analytical CPU checks](stage2/torch_cpu_checks.json) and an independent neural integration test. The original records are retained unchanged so that their hashes and execution history remain reviewable.

## 2. Input-Recast and outside-network implicit correction solve different equations

Let $\Phi_\theta$ map a physical state to the full next physical state over the learned step $h$, and let $B=(\nu-\nu_0)\Delta$.

The v2 input-Recast equation discussed in the archived analysis is

$$
y=\Phi_\theta(u+hBy).
$$

The unknown $y$ changes the network input. A nonlinear network evaluation generally changes at every fixed-point iteration; it cannot be cached as the constant $\Phi_\theta(u)$. The current repository's [Recast script](../../run_ffno_equation_recast_3seed.py) explicitly documents this input equation and vector Aitken relaxation. Its presence enables a future implementation audit, but this supplement did not execute it.

The additional baseline in this supplement instead solves

$$
y=\Phi_\theta(u)+hBy,
\qquad
\widehat y(k)=\frac{\widehat{\Phi_\theta(u)}(k)}{1+h(\nu-\nu_0)|k|^2}.
$$

For the periodic, constant-coefficient Laplacian and $\nu\geq\nu_0$, this denominator is positive. The formula needs one network evaluation and one FFT/IFFT pair. It is labeled `implicit` in the supplemental code and means **outside-network closed-form implicit correction** throughout the tables.

A linear heat predictor already distinguishes the equations. Write $a=h\nu_0|k|^2$, $z=h(\nu-\nu_0)|k|^2$, and $s=e^{-a}$. Their Fourier multipliers are:

| Mapping | Multiplier |
| --- | --- |
| Exact target heat flow | $s e^{-z}$ |
| Additive Δ, Lie, or Strang with exact source heat flow | $s e^{-z}$ |
| Outside-network implicit | $s/(1+z)$ |
| Input-Recast with the linear source heat predictor | $s/(1+zs)$ |

For $a=z=0.5$, these last three distinct values are approximately 0.36787944, 0.40435377, and 0.46539308. The [Stage 1 controls](stage1/results/control_results.json) verify both fixed-point forms against their respective linear formulas. A linear-flow diagnostic is not a reproduction of the manuscript's nonlinear neural Aitken iteration.

Existence of the outside-network closed form is also distinct from convergence of a chosen iteration. Its unrelaxed affine Picard iteration has spectral radius $z_{\max}$ and needs $z_{\max}<1$ to contract. The recorded ratio-2 heat controls give approximately 4.04259 on the NS64 grid and 8.08518 on the Burgers grid, so unrelaxed Picard need not converge even though the closed form is well defined. In the linear input-Recast heat control, the corresponding factor is instead $\max_k z e^{-a}$, bounded by $1/e$ at ratio 2. These linear diagnostics do not establish contraction of a nonlinear neural Recast solver. See the [Stage 1 protocol](stage1/experiment_protocol_zh.txt) and [control JSON](stage1/results/control_results.json).

Consequently, an error reported for input-Recast cannot be assigned to outside-network implicit by algebraic relabeling. Nor does the outside-network formula show that all original Recast iterations were redundant. What it does show is that a one-call implicit competitor should be measured explicitly. A call-count advantage over an iterative method does not determine the best one-call method or establish a speedup over direct numerical integration.

## 3. One-sided use is a stability policy, not a general necessity theorem

The additive correction uses the **difference of two forward heat propagators**,

$$
D_h=e^{h\nu\Delta}-e^{h\nu_0\Delta}.
$$

For $\nu,\nu_0\geq0$ and periodic $L^2$, each Fourier multiplier lies in $[0,1]$, so $\|D_h\|_{2\to2}\leq1$. This is true in either parameter direction. In particular, when $0\leq\nu<\nu_0$, the additive correction is bounded and is not the inverse-heat operator $e^{h(\nu-\nu_0)\Delta}$. With an exact source heat flow, adding $D_hu$ gives the exact target heat flow even in the lower-viscosity direction. The [Stage 1 heat checks](stage1/results/control_results.json) and [Stage 2 analytical checks](stage2/torch_cpu_checks.json) include that diagnostic.

This bound does not make the complete nonlinear mapping $\Phi_\theta(u)+D_hu$ contractive, energy stable, or accurate over long autoregressive rollouts. Its behavior also depends on network error, nonlinearity, accumulated state error, and the visited state distribution. A one-sided policy can therefore be justified by the empirical stability of particular learned models, without claiming that lower-viscosity additive transfer is mathematically impossible.

The distinction matters for other wrappers. For $\nu<\nu_0$, the splitting factor $e^{hB}$ has inverse-heat amplification; its norm grows with the highest retained frequency. The implicit denominator can approach or cross zero. The provided wrappers reject these unsupported lower-viscosity splitting/implicit paths rather than silently applying clipping. A separate lower-viscosity additive option exists, but **the main three-seed learned Burgers matrix evaluated ratios 1 through 4**. It supplies no new learned-model evidence that lower-viscosity rollouts are stable. See the [implementation interface](stage1/INTEGRATION_EN.txt), [shared implementations](stage2/shared/), and [locked main configuration](stage2/burgers_fno/results/locked_config.json).

## 4. Local errors: additive and Lie terms, and conditions for Strang

Consider a fixed spatial discretization of

$$
\dot u=N(u)+\nu Lu,
\qquad L=\Delta,
$$

and define $F_0(u)=N(u)+\nu_0Lu$, $B=(\nu-\nu_0)L$. Let $\varphi_0^h$ and $\varphi_\nu^h$ be the exact source and target flows. For smooth enough $N$ and states, with fixed parameters and the same discretization, Taylor expansion gives the following **exact-target minus method** defects:

$$
\begin{aligned}
\varphi_\nu^h(u)-\left[\varphi_0^h(u)
 +(e^{h\nu L}-e^{h\nu_0L})u\right]
 &=\frac{h^2}{2}\left[N'(u)Bu+BN(u)\right]+O(h^3),\\
\varphi_\nu^h(u)-e^{hB}\varphi_0^h(u)
 &=\frac{h^2}{2}\left[N'(u)Bu-BN(u)\right]+O(h^3),\\
\varphi_\nu^h(u)-\varphi_0^h(e^{hB}u)
 &=\frac{h^2}{2}\left[BN(u)-N'(u)Bu\right]+O(h^3).
\end{aligned}
$$

Here $N'(u)$ is the derivative of the nonlinear vector field. The cancellation of linear cross terms uses that $B$ and $\nu_0L$ commute. Additive correction contains a sum of nonlinear–linear interactions; the two Lie orders have opposite commutator-type leading defects. Their relative sizes depend on the problem and state. There is no universal ordering of their error constants.

With exact subflows, the symmetric composition

$$
e^{hB/2}\varphi_0^h(e^{hB/2}u)
$$

has local error $O(h^3)$ under the smoothness needed for the splitting expansion. A global second-order statement additionally requires a consistent family as $h\to0$, stability, and appropriate regularity. For the continuous PDE, bounds uniform in spatial resolution require additional operator-domain and regularity arguments; the fixed-discretization expansion alone does not supply them.

In the [six-dimensional Stage 1 control](stage1/results/local_order.csv), fitted local slopes were 1.99287 for additive Δ, 1.99111 for Lie-post, 1.98992 for Lie-pre, and 2.99023 for Strang. Subtracting the predicted leading term gave slopes 2.99472 for Δ and 2.99379 for Lie-post. The fit used the four smallest steps among $h=0.1,0.05,\ldots,0.003125$, with high-accuracy DOP853 source/target flows. See the [control JSON](stage1/results/control_results.json) and [local-order figure](stage1/results/local_order.png). These are observed local orders for a smooth finite-dimensional numerical control, not convergence orders of a trained neural model.

### A learned source flow changes the conclusion

Write $\Phi_\theta(u)=\varphi_0^h(u)+e_h(u)$. Relative to each exact-source composition, the additional model-error terms are

| Wrapper | Added model-error contribution at one step |
| --- | --- |
| Additive Δ | $e_h(u)$ |
| Lie-post | $e^{hB}e_h(u)$ |
| Lie-pre | $e_h(e^{hB}u)$ |
| Strang | $e^{hB/2}e_h(e^{hB/2}u)$ |

For forward diffusion, the analytic factors are nonexpansive in periodic $L^2$, but the full error also contains composition error and accumulated rollout error. No ranking follows from the model-error term alone. Strang's cubic exact-flow defect is useful only when source-model error and stability allow it to matter; for example, an appropriate uniformly accurate $e_h=O(h^3)$ family would be needed to retain that local order through the source approximation.

A checkpoint trained at one fixed $h$ is not a family of half-step models. Changing only a wrapper parameter from $h$ to $h/2$ does not establish convergence. The implemented Strang wrapper uses an analytic half-diffusion, **one full learned step $h$**, and a second analytic half-diffusion. It never assumes an unavailable half-step network. Also, Lie-post and Δ both feed the current physical state directly into the network. Avoiding pre-diffusion of the input is not an advantage unique to Δ. These distinctions are described in the [Stage 1 protocol](stage1/experiment_protocol_zh.txt) and implemented in the [shared wrappers](stage2/shared/).

## 5. What the three-seed Burgers experiment establishes

The experiment independently implements and trains a standard one-dimensional FNO for periodic unforced Burgers on a 64-point unit interval, with $\nu_0=0.01$ and $h=0.02$. It uses width 32, 16 Fourier modes, four Fourier-plus-pointwise layers, and a 128-wide projection head. The network predicts a normalized increment, which is decoded into the full physical next state before applying a physical wrapper.

The three model seeds share 24 source training trajectories and six source validation trajectories. Each trajectory supplies 20 transitions; there are 480 training pairs. Eight held-out initial conditions are paired across all methods, ratios, and model seeds. Training lasts 75 epochs, with a fixed StepLR schedule and checkpoint selection using source validation only. Target trajectories are generated after all three training runs. All selected checkpoints occur at epoch 75, while validation is still improving, so the record does not establish fully converged source training. See the [locked configuration](stage2/burgers_fno/results/locked_config.json), [training summary](stage2/burgers_fno/results/training_summary.json), and [training curves](stage2/figures/burgers_training_curves.png).

Reference trajectories and metric accumulation use float64; the network uses float32 with complex64 Fourier weights. The reference is a time-refined solution of the same 64-point semidiscrete equation. Comparing target integration steps 0.00025 and 0.000125 gives maximum relative discrepancy at most $1.053\times10^{-10}$ across the recorded target matrix. This is temporal validation at a fixed grid, not a spatial-convergence proof. The [main results directory](stage2/burgers_fno/results/) retains source data, target trajectories, normalizers, weights, complete predictions, and metrics.

### Metrics and the ratio-2 ranking

For each model seed, Avg-50 averages the relative $L^2$ error across eight initial conditions and all 50 outputs. Step-50 averages only the last output across those eight initial conditions. The reported mean and sample SD then summarize the **three model seeds**. Initial conditions and times are not extra independent training replicates.

| Method | Avg-50: mean ± sample SD | Step-50: mean ± sample SD | Failures out of 24 seed–IC pairs |
| --- | ---: | ---: | ---: |
| Frozen | 0.278844 ± 0.003757 | 0.416712 ± 0.009290 | 0 |
| Δ | 0.021481 ± 0.001781 | **0.030129 ± 0.004237** | 0 |
| Lie-pre | 0.017524 ± 0.000396 | 0.032149 ± 0.002510 | 0 |
| Lie-post | 0.016856 ± 0.000634 | 0.030896 ± 0.003668 | 0 |
| Strang | **0.015329 ± 0.000407** | 0.030860 ± 0.003092 | 0 |
| Outside-network implicit | 0.016212 ± 0.000371 | 0.030625 ± 0.003455 | 0 |
| Explicit | Missing after failure | Missing after failure | 24 |

Source: [main CSV](stage2/tables/burgers_fno_results.csv). The bold values are the lowest complete means in their respective columns; they do not denote a significance test.

Δ improves Avg-50 over Frozen by a mean paired 92.29%. Strang improves Avg-50 over Δ in all three seeds, with paired reductions of approximately 28.68%, 32.72%, and 23.81%, averaging 28.40%. Outside-network implicit improves that metric by a mean paired 24.18%. These percentages are computed within seed and then averaged; they need not equal the percentage computed from two aggregate means. See [per-seed results](stage2/burgers_fno/results/per_seed_summary.csv) and the [fixed-epoch aggregate](stage2/burgers_fno/budget_ablation/aggregate_summary.csv).

The Step-50 mean reverses the Δ-versus-Strang ordering: Δ is approximately 0.030129, Strang 0.030860. Endpoint seed-level rankings are not identical either. The [full metric curves](stage2/burgers_fno/results/per_step_metrics.csv) and [comparison figure](stage2/figures/burgers_fno_comparison.png) support a conditional conclusion: Strang is stronger for trajectory-average error in this setting; Δ retains a slightly lower final-step mean. Three training seeds permit descriptive comparisons here, not an asserted statistical-significance conclusion or a universal winner.

Across the tested ratios 1.25, 1.5, 1.75, 2, 3, and 4, Strang has the lowest complete Avg-50 mean in the [main table](stage2/tables/burgers_fno_results.csv). At ratio 1 all seven methods exactly recover the same frozen prediction. Explicit correction remains finite at ratio 1.25, while all 24 seed–IC pairs fail at each tested ratio from 1.5 through 4. This is an observed failure at the specified grid, output step, and model; it does not exclude all possible explicit correction regimes.

### Training-budget diagnostic

The fixed-epoch snapshots give these ratio-2 Avg-50 means:

| Method | Epoch 15 | Epoch 30 | Epoch 50 | Epoch 75 |
| --- | ---: | ---: | ---: | ---: |
| Δ | 0.033040 | 0.024812 | 0.025310 | 0.021481 |
| Strang | 0.026663 | 0.019590 | 0.019413 | 0.015329 |
| Outside-network implicit | 0.027029 | 0.020353 | 0.020400 | 0.016212 |

Source: [fixed-epoch CSV](stage2/tables/burgers_fixed_epoch_results.csv). Strang's mean paired improvement relative to Δ is 19.36%, 20.77%, 23.51%, and 28.40%, respectively. The same three optimization trajectories supply every column; these are correlated snapshots, with the epoch-75 column duplicated in the main experiment. They suggest that stronger source fitting does not remove the need for classical composition baselines in this reconstruction. They do not establish the ranking on the original NS/F-FNO model.

### Failures and integration checks

A trajectory fails at the first nonfinite output or when its maximum absolute physical value exceeds $10^4$. Its later values stay missing/NaN. An aggregate requiring a failed sample remains null rather than averaging only survivors. Absolute error, reference norms, spectral error, and mean drift are retained alongside relative error, which matters when strongly diffusive targets become small.

The [unified runner](stage2/runner/evaluate.py) separately reloads the reconstructed seed-0 checkpoint at ratio 2. Its rollouts match the training-script outputs bit for bit, including failed-path NaN locations; the maximum scalar Avg-50/Step-50 discrepancy is $5.55112\times10^{-17}$ from reduction order. Actual model hooks count 50 batched forwards per 50-step method. See the [cross-check record](stage2/runner/burgers_seed0_ratio2_results/crosscheck_against_training_script.json). This verifies the supplemental adapter path, not an original-checkpoint adapter. The physical state, source normalization, coordinates, dtype, and complex Fourier weights must all be preserved correctly when adding one.

## 6. The CPU result is specific and practically consequential

The benchmark uses reconstructed neural seed 0, grid 64, $\nu/\nu_0=2$, $h=0.02$, 50 outputs, and batches 1 and 8. Each configuration receives three full-rollout warmups and ten measured repetitions with randomized method order. The processor is recorded as AMD EPYC 9V74; the container exposes nine logical CPUs. The one-thread Torch comparison and the four-thread sensitivity study both use one SciPy FFT worker. See the [benchmark metadata](stage2/cpu_benchmark_threads1/benchmark.json) and [complete timing CSV](stage2/tables/burgers_cpu_timing.csv).

For batch 8 with single-thread Torch:

| Method | Median 50-step time, ms | IQR, ms | Precheck Avg-50 | Precheck Step-50 |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 66.261 | 3.842 | 0.2804895 | 0.4174910 |
| Δ | 69.266 | 3.757 | 0.0210083 | 0.0269442 |
| Lie-pre | 69.422 | 4.299 | 0.0176057 | 0.0292936 |
| Lie-post | 70.073 | 2.613 | 0.0161369 | 0.0266606 |
| Strang | 69.727 | 2.097 | 0.0149828 | 0.0272940 |
| Outside-network implicit | 69.367 | 5.259 | 0.0158042 | 0.0266912 |
| RK4, internal step 0.002 | 42.534 | 3.614 | $3.48084\times10^{-7}$ | $4.39579\times10^{-7}$ |
| RK4, internal step 0.001 | 83.513 | 4.799 | $5.20620\times10^{-7}$ | $8.29696\times10^{-7}$ |
| RK4, internal step 0.0005 | 173.262 | 10.073 | $5.73912\times10^{-7}$ | $6.95144\times10^{-7}$ |

The direct solver uses float32/complex64 spectral state, the same conservative nonlinearity and dealiasing, and the same physical outputs. At internal step 0.002 it is approximately **1.63× faster than Δ and 1.64× faster than Strang**, while also more accurate against the same semidiscrete reference. Its upper timing quartile is below the corresponding lower quartiles for Δ and Strang. This small-grid CPU case therefore provides no learned inference-speed advantage over the tested direct solver.

Timing includes the initial-state copy, normalization, corrections, every physical output computation, and final-state retention. It excludes training, model/data loading, reference generation, metric calculations, per-step failure checks, and stacking the full output trajectory. Full-rollout accuracy/failure screening occurs separately; failed explicit trajectories remain recorded and are excluded from successful-inference timing. The direct solver does not skip the 50 required physical outputs. These choices are recorded in each [benchmark folder](stage2/cpu_benchmark_threads1/).

This is an implementation-level result for the specified hardware, grid, precision, model seed, batch, output policy, and code. It does not establish GPU rankings, behavior at NS/high resolution, an algorithmic complexity bound, or training-cost amortization. The four-thread neural measurements are slower and noisier in this small problem, demonstrating sensitivity to threading; wrapper timing intervals overlap, so small wrapper-to-wrapper differences are not reliable speed rankings. The smaller float32 RK4 steps do not monotonically reduce error, which is compatible with accumulated rounding effects; those timings are not a convergence-order experiment. The independent float64 refinement checks carry the reference-accuracy argument. See the [timing figure](stage2/figures/burgers_cpu_timing.png).

## 7. NS64/NS128 isolates numerical composition error

The NS controls use a high-accuracy pseudo-spectral source-viscosity solver in place of $\Phi_\theta$. There is **no neural network** in these runs. The three seeds denote shared continuous initial conditions sampled on both grids. The setup is a forced vorticity equation on the periodic unit square, $\nu_0=0.001$, $h=0.05$, and 50 outputs to time 2.5. Source and target flows use dealiased float64 spectral RK4, with full Laplacian diffusion retained beyond the nonlinear cutoff. See the [NS configuration and protocol](stage2/ns_controls/README.txt).

At $128\times128$, the three-IC Avg-50 means are:

| Method | Ratio 1.5 | Ratio 2 | Ratio 4 |
| --- | ---: | ---: | ---: |
| Frozen numerical source flow | 0.11377244 | 0.21886256 | 0.59387505 |
| Δ | 0.00470432 | 0.00732319 | 0.01092693 |
| Lie-pre | 0.00087397 | 0.00143221 | 0.00266076 |
| Lie-post | 0.00087700 | 0.00143614 | 0.00267119 |
| Strang | $8.49374\times10^{-6}$ | $1.34112\times10^{-5}$ | $2.25068\times10^{-5}$ |
| Outside-network implicit | 0.00088696 | 0.00188251 | 0.00662518 |
| Explicit | 3/3 failed | 3/3 failed | 3/3 failed |

Source: [NS numerical-control CSV](stage2/tables/ns_numerical_control_results.csv) and [resolution-control figure](stage2/figures/ns_resolution_controls.png). The very small Strang error shows the importance of symmetric composition when source-flow error is already small. It cannot be transferred as an expected improvement factor for an imperfect neural predictor. These Stage 2 NS runs compare methods at fixed output $h$; they do not constitute a new NS time-step convergence-order experiment.

Target time-step refinement gives maximum relative difference $1.82933\times10^{-12}$. Full source-viscosity rollout refinement gives $1.76571\times10^{-11}$, and probes at the actual source-predictor inputs give at most $8.29003\times10^{-9}$, all below the declared $10^{-7}$ gate. An earlier source internal step 0.005 missed that gate on an explicit-correction path; the final source step was uniformly reduced to 0.0025 for the entire comparison. The reason and accepted results are retained in the [preliminary-run note](stage2/ns_controls/results/preliminary_numerical_floor_note.json), [source refinement](stage2/ns_controls/results/source_flow_convergence.json), and [convergence record](stage2/ns_controls/results/convergence.json).

The initial-condition bandwidth is only four Fourier modes per direction. The common-band NS64/NS128 target differences at saved times are at most approximately $8.80\times10^{-9}$, $9.27\times10^{-10}$, and $1.45\times10^{-12}$ for ratios 1.5, 2, and 4. These are already smooth, well-resolved tests. No 256-grid anchor is present; the evidence does not prove spatial convergence of the 128 grid or performance on a more demanding high-resolution learning problem.

The explicit baseline also illustrates why failure thresholds and stability analysis differ. At ratio 1.5 the NS64 trajectories remain below the failure threshold for 50 steps, while NS128 fails at step 23 for all three ICs. Above the nonlinear cutoff, the linear amplification is

$$
g(q)=R_4(-\nu_0q\,\delta t)^{n_{\rm internal}}
     -h(\nu-\nu_0)q,
$$

where $R_4$ is the RK4 stability polynomial. The maximum modulus is about 2.004 on NS64 and 8.085 on NS128 in this case. This explains a high-frequency growth mechanism; it does not predict the exact first failure time. Remaining finite for a finite horizon is not itself a stability proof. See the [stiffness diagnostic](stage2/ns_controls/results/explicit_stiffness.json).

## 8. Prioritized work that has not been executed

The archived [next-experiment matrix](stage2/next_experiments.json) is a plan. Its original missing-asset language is qualified by the current repository inventory in Section 1. The useful next steps are:

| Priority | Experiment | Decision it would support |
| --- | --- | --- |
| 0 | Audit and adapt the now-present original NS/F-FNO assets; first reproduce their Frozen/Δ outputs, then compare all seven supplemental wrappers and the actual input-Recast/Aitken implementation on the same checkpoints and paired trajectories | Whether the independent Burgers ranking transfers to the original neural setting; include three training seeds where the required checkpoints exist or are trained |
| 1A | Train three source models natively at 128; use initial conditions with broader active physical bandwidth, independent temporal refinement, and selected 256-grid reference anchors | High-resolution parameter transfer after source training at that resolution |
| 1B | Use frozen 64-grid weights at 128 where the actual architecture permits it, retain source statistics, and report source-viscosity resolution error before additional viscosity-transfer error | Joint resolution and parameter transfer without retraining; report separately from native-128 training |
| 2 | Hyperviscous NS with fixed operator $-(-\Delta)^2$, symbol $-\lvert k\rvert^4$, and a newly trained source model at $\kappa_0$ | Transfer for a different dissipative operator; requires a new symbol interface and validated stiff time integration |
| 3 | Two-component reaction–diffusion, such as Gray–Scott with fixed reactions; first scale both diffusivities together, then vary one species | Transfer with species-dependent diffusion and nonlinear coupling; requires channel-specific operators and component/pattern diagnostics |
| Later | KdV and Kuramoto–Sivashinsky with their own source models and operator-specific theory | Dispersive phase transfer and mixed growing/decaying spectra; the dissipative one-sided argument cannot simply be reused |

For native high-resolution work, a larger array is not sufficient: target spectra and refinement differences must show that the problem has meaningful active bandwidth. For 64→128 transfer, check Fourier modes, padding, coordinate construction, and the meaning of source normalization at the new grid. Keep source-resolution error distinct from parameter-shift error.

Hyperviscosity uses a coefficient with units of length$^4$/time, so its numerical value is not an ordinary viscosity. Changing only its magnitude is different from switching a Laplacian model to a fourth-order operator; the latter requires explicitly analyzing the symbol of $L_{\rm target}-L_{\rm source}$, which need not be dissipative at every frequency. The current Laplacian wrapper does not directly implement these extensions. Reaction–diffusion needs component-wise errors, pattern length scales, phase-sensitive field errors, and negative-concentration fractions, with any positivity projection declared and applied consistently.

For KdV, retain unaligned field error and spectral phase diagnostics; alignment may be an additional diagnostic, not a replacement. A unitary analytic flow does not automatically make the additive combination energy stable. For KS, distinguish short-horizon prediction, forecast horizon, and long-time statistics, and inspect both implicit denominators and splitting amplification. These are proposed requirements, not executed results.

Across the planned comparisons, preserve paired seed-level Avg-50 and Step-50, absolute errors, complete curves, failure rates, and measured latency. Extend to 100/200-step horizons where useful and compare numerical solvers at matched error targets. CPU/GPU and batch 1/8/32 measurements require the actual corresponding hardware and runs. If the metrics disagree, report the conditional result instead of selecting a single favorable metric.

## 9. Implication for the research claim

The supported question is how to combine an analytic propagator with an imperfect frozen time-step predictor to obtain useful parameter transfer. Δ remains a useful one-call baseline. Lie, Strang, and outside-network implicit correction are necessary competitors, while input-Recast remains a distinct iterative method requiring its own measured behavior.

The completed evidence does not justify making a roughly ninefold network-call comparison against an iterative solver the sole basis for effectiveness or efficiency. It does justify examining why exact-source Strang is very accurate, why learned-model endpoint and trajectory-average rankings differ, how source-model quality changes the comparison, and when a direct solver remains preferable. The [four primary CSVs and figures](README.md#tables-and-figures), original weights, complete outputs, and preserved manifests make those conclusions inspectable without replacing historical records or claiming experiments that did not run.
