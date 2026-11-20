NAVIER--STOKES NUMERICAL-REFERENCE CONTROLS
========================================

Evidence boundary
-----------------
These are independently executed numerical experiments. The frozen reference
callback is a high-accuracy pseudo-spectral Navier--Stokes solver, NOT an F-FNO,
NOT an original manuscript checkpoint, and NOT a source-trained neural model.
The three seeds identify initial conditions only. They are not training seeds.
No results here establish neural inference speed, GPU performance, generalization
of a learned model, or reproduction of Table 2 / the 75-epoch checkpoint.

Purpose
-------
Compare additive propagator difference, explicit correction, pre/post Lie,
Strang, and outside-network closed-form implicit correction under a common
numerical reference flow, at 64 x 64 and 128 x 128, viscosity ratios 1.5, 2, 4.
This isolates composition/truncation effects before introducing learned-model
error. Source-viscosity recovery is checked separately and is exactly identical
at wrapper level. The nonlinear-input Aitken Recast baseline from v2 Appendix C
is not implemented or silently replaced by the outside-network implicit method.

Execution
---------
Requirements: Python 3, NumPy, SciPy. Matplotlib is needed for the optional plot.
From this directory:

  python3 run_ns_controls.py --pilot
  python3 run_ns_controls.py --out results
  python3 plot_ns_results.py --results results
  python3 inspect_explicit_stiffness.py

The shared spectral_transfer_numpy.py file must remain at ../shared/. The script
adds that path itself. It can also be launched using its absolute path.

Problem and paired samples
--------------------------
Periodic unit square; vorticity equation
  omega_t + div(v omega) = nu Delta(omega) + f,
  f = 0.1 [sin(2*pi*(x+y)) + cos(2*pi*(x-y))].
Vorticity convention: omega = d_x(v_y) - d_y(v_x).
For nonzero Fourier wavevectors, psi_hat = omega_hat / |k|^2 and
  v_x_hat = i*k_y*psi_hat, v_y_hat = -i*k_x*psi_hat.
The mean velocity is set to zero. Grid coordinates exclude the repeated endpoint.

Source nu0 = 0.001; output step h = 0.05; 50 output steps, physical time 2.5.
Initial-condition seeds 0, 1, 2 use fixed continuous Fourier coefficients with
max(|m_x|, |m_y|) <= 4. Coefficient scale is 1/(1+|m|^2); analytic continuous
variance sets unit standard deviation. The exact same coefficients are sampled
on both grids. This specified distribution was independently constructed; the
original unpublished test trajectory IDs were unavailable.

Reference integration
---------------------
Conservative Fourier flux nonlinearity, strict tensor-product 2/3 dealiasing of
nonlinear inputs AND output; real FFT; float64; one SciPy FFT worker.
Only the nonlinear term is filtered. Full-state Laplacian diffusion is retained,
including frequencies above the nonlinear cutoff, identically for all wrappers.
No projection is added uniquely to stabilize the explicit method. Odd-derivative
Nyquist multipliers are zero; the even-derivative Laplacian keeps the full |k|^2.

Classical RK4 source predictor dt = 0.0025, probe dt = 0.00125.
Target truth is checked at dt = 0.00125 versus 0.000625; an additional linear
stability check reduces dt if needed. All truth runs stay below nu*dt*q_max=2.5.
Whole source-viscosity trajectories are also compared at the two source steps.
The predeclared time-refinement acceptance threshold is relative error 1e-7.
Source-step probes at output steps 1, 25, 50 evaluate the actual predictor inputs
seen by each method (including its pre-filter where appropriate).

The preliminary run used source dt=0.005. The explicit method at N=64, ratio1.5,
step50 amplified high-frequency content enough to reach source-step refinement
error 1.9203e-7. The run stopped at the declared gate. The source step was then
halved uniformly for every method, independent of method ranking. The preliminary
gate failure is documented in results/preliminary_numerical_floor_note.json;
those preliminary scores are not accepted results. Final results/config.json and
complete.json files identify accepted cases.

Sanity checks
-------------
Unforced Laplacian eigenmode agrees with analytic heat evolution; a forced
eigenmode agrees with its exact forced linear solution; Fourier velocity curl
matches vorticity and divergence vanishes; the inviscid semidiscrete enstrophy
rate is near roundoff; paired-grid initial conditions agree after restriction;
every source-viscosity wrapper equals the same callback result exactly.

64 versus 128 target fields are compared by orthogonal Fourier restriction with
the proper FFT normalization. The coarse Nyquist axes are excluded. Comparisons
at stored output steps 0, 1, 10, 25, 50 are evidence of resolution consistency;
they do not prove that the 128 grid is spatially converged. No 256-grid anchor is
included in this task.

Metrics and failures
--------------------
All 50 output steps are recorded for each IC, grid, ratio, method. RelL2 is the
discrete L2 error divided by target L2; on the unit square, the quadrature L2
absolute error equals RMS grid error. PSRE compares |FFT(omega)|^2 using the full
orthonormal FFT. Enstrophy is 0.5*mean(omega^2). Kinetic energy is separately
0.5*mean(v_x^2+v_y^2); these are not interchangeable.

Failure is defined before evaluation: nonfinite output or max|omega|>1e4 at an
output time. A failed trajectory remains failed thereafter. It has null metrics
and NaN saved fields from its first failure onward. A zero placeholder is used
internally only to preserve independent successful batch lanes; it is never
counted as a prediction or silently averaged into the result. If any trajectory
fails, the corresponding across-IC Avg50/Step50 means are null. Failure counts
remain explicit. Intermediate IC scores and first-failure steps are preserved.

Standard deviations are across the three initial conditions, not across trained
models. Each method makes 50 batched predictor callback calls for the main
rollout. Refinement probes are additional validation work, not hidden network
evaluations or inference measurements. Execution durations include reference
integration and validation and must not be interpreted as neural inference time.

Output files
------------
results/config.json: parameters, software versions, evidence label, script hash.
results/sanity.json: analytic/discrete checks and continuous IC coefficients.
results/source_flow_convergence.json: complete nu0 rollout refinement.
results/convergence.json: target/source probes and common-band grid comparison.
results/progress.json: whether the complete requested matrix finished.
results/summary.csv: per grid/ratio/method mean and SD across ICs, explicit failures.
results/summary_by_ic.csv: each initial condition's Avg50, Step50 and failure step.
results/n*_r*/per_step.csv: all 50-step curves and physical diagnostics.
results/n*_r*/target_states.npz: target fields at output steps0/1/10/25/50.
results/n*_r*/*_snapshots.npz: corresponding method fields and first failures.
results/n*_r*/final_states.npz: successful method final states for comparisons.
results/n*_r*/complete.json: only written after time-refinement gates pass.
results/explicit_stiffness.json: analytic high-mode amplification diagnostic.

Raw field snapshots use float64. Complete numerical metric curves are retained;
the full 50-step physical field arrays are reproducible but omitted from disk to
limit artifact size. This storage policy is identical for every method.

Why the explicit baseline can fail earlier at higher resolution
-------------------------------------------------------------
Above the nonlinear cutoff, the source reference flow is linear RK4 diffusion.
Its explicit-corrected amplification is
  g(q) = R4(-nu0*q*dt)^n_internal - h*(nu-nu0)*q,
where R4(z)=1+z+z^2/2+z^3/6+z^4/24 and q=|k|^2. The second term uses the much
larger output step h. The maximum |g| at ratio1.5 is approximately2.004 on64^2
and8.085 on128^2. Roundoff or approximation errors can seed these growing modes.
Thus remaining finite at step50 on64^2 is not proof of stability. This analytic
diagnostic explains a mechanism; it does not predict an exact failure step or
establish behavior for trained models.

The --pilot command uses source dt=.005 for an inexpensive throughput estimate.
The accepted matrix uses the refined dt=.0025 specified in config.json. Pilot
duration is not a matched inference benchmark and is not used in any accuracy
table or speedup statement.
