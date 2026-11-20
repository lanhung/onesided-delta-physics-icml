Delta-Physics v2 independent validation and experiment integration kit
===================================================================

Evidence boundary
-----------------
This kit DOES NOT contain the original FNO/F-FNO, its checkpoints, normalizers,
training data, or test trajectories. It DOES NOT reproduce or replace manuscript
Tables 2 or 4. The uploaded v2 PDF has 14 pages; its Appendix C uses an input-side
nonlinear fixed-point map, not an external affine diffusion correction.

What was executed
-----------------
NumPy/SciPy float64 controls were executed on CPU:
  1. Fourier units, zero-parameter-shift identity, constant-mode preservation.
  2. Exact heat-flow comparisons on 64x64, 128x128 and 64-point grids.
  3. External implicit closed form versus a cached, relaxed affine iteration.
  4. Input-recast Picard versus its DIFFERENT heat-flow closed form.
  5. Local error orders for a six-dimensional smooth nonlinear system.
  6. 50-step Burgers control with a high-accuracy numerical reference propagator.

The Burgers seeds 0, 1, 2 are INITIAL-CONDITION seeds, not training seeds.
Time refinement was checked; spatial convergence was not established in this
fixed-grid control. Failed trajectories remain explicitly marked as failed.
No GPU timing or learned-model accuracy is reported by these controls.

Quick start
-----------
Use an environment with numpy, scipy and matplotlib, then:
  python3 run_controls.py --out results

For algebraic checks only:
  python3 run_controls.py --out quick_results --skip-burgers

The full command regenerates JSON, CSV, NPZ trajectories, PNG plots and vector PDF
plots. requirements-controls.txt records the versions used for this run.

Files
-----
spectral_transfer_numpy.py: Tested CPU reference for seven one-predictor methods.
run_controls.py: Executed controls, including a dealiased Burgers RK4 reference.
spectral_transfer_torch.py: Native Torch integration module; see execution limits
                           in INTEGRATION_EN.txt. No Torch runtime test here.
experiment_matrix.json: Planned manuscript comparison; status is pending assets.
experiment_protocol_zh.txt: Full Chinese audit, control results and experiment plan.
execution_manifest.json: Evidence status and input/source hashes.
results/control_results.json: Full machine-readable control results and metadata.
results/local_order.csv: All one-step local-order errors.
results/burgers_control_curve.csv: Per-IC, per-step errors and failure status.
results/burgers_control_summary.csv: Summary of the independent Burgers control.
results/burgers_control_states.npz: Initial fields, target states and predictions.
results/*.png / *.pdf: Plots explicitly labeled as non-neural controls.

Physical predictor contract
---------------------------
predict(u) must accept physical fields and return the COMPLETE next physical
field at the fixed network time step h. It must not return normalized increments.
Wrap the original model with its own frozen training normalizers and coordinate
handling. Only physical state channels are passed through Fourier corrections.
Do not fit or replace normalizers using target test data.

The main upward-transfer comparison uses nu >= nu0. The optional two-sided Delta
diagnostic does not assert that arbitrary learned lower-viscosity rollouts are
stable. Extra negative-time heat splitting is deliberately excluded.

Manuscript Recast implementation
--------------------------------
Preserve and audit the original vector-Aitken implementation after obtaining the
repository. input_recast_picard in this kit verifies only the equation on a
linear heat reference; it is NOT that implementation. The external implicit
method is a separate baseline, and no equality to the manuscript Recast errors
or runtime is assumed.

Next inputs needed
------------------
Original model and rollout source, at least the existing 75-epoch checkpoint,
its normalizers, original test trajectories or deterministic generation config,
and hardware details. Complete same-checkpoint comparisons before extending the
main claim to higher resolution or new operators.
