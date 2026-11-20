UNIFIED PAIRED ACCURACY RUNNER
============================

Status
------
The NumPy and PyTorch CPU paths have both been executed on an exact heat-flow
control dataset. Nine integration checks pass in Python 3.12 with NumPy 2.3.5
and PyTorch 2.7.1+cpu. CUDA is unavailable in this execution environment.
The independently reconstructed Burgers FNO has also been evaluated through this
runner: training seed 0, viscosity ratio 2, 8 initial conditions, 50 steps, all
seven methods. Complete rollouts, including failure/NaN positions, match the
training script bit for bit; the largest Avg50/Step50 scalar discrepancy is
5.56e-17 from reduction order. Actual model hooks count 50 forwards per method.
The evidence is in burgers_seed0_ratio2_results/crosscheck_against_training_script.json.
This runner has NOT evaluated an original manuscript checkpoint. A control uses
an analytical predictor, not a trained network. No original F-FNO checkpoint or
original Aitken solver is provided. This runner does not provide timing benchmarks.

The supported evidence labels are:
  numerical_control
  independently_reconstructed_neural
  original_neural
The second label is for a new implementation/training run and does not certify a
reproduction of the manuscript. All neural modes require the same asset checks.

Quick start from the delta_physics_stage2 directory
------------------------------------------------
  python runner/check_runner.py
  python runner/make_heat_demo.py --out runner/new_demo
  python runner/evaluate.py --config runner/new_demo/config_numpy.json --validate-only
  python runner/evaluate.py --config runner/new_demo/config_numpy.json --out runner/new_numpy --save-rollouts
  python runner/evaluate.py --config runner/new_demo/config_torch.json --out runner/new_torch --save-rollouts

Output directories must be fresh. Existing evidence is never silently overwritten.
The delivered demo and demo_results_* folders contain the outputs actually run.
Regenerate into NEW directories for an independent run. relative adapter paths
in generated configs remain portable when the directory structure is preserved.

--validate-only verifies files, hashes, timestamps, declared physics and adapter
metadata, and loads the predictor, but performs NO predictor inference. Its status
is assets_and_metadata_validated_no_inference, never completed_experiment.
Invalid/missing assets cause exit code 2 and no experiment output directory.
Successful execution means the run completed, not that every trajectory remained
stable: inspect failed_trajectories and null aggregate metrics in summary.json.

Dataset contract
----------------
NPZ with allow_pickle=False, containing:
  trajectory: float32 or float64 [T+1, B, C, *spatial], physical states
  trajectory_ids: B unique nonempty strings, independent initial conditions
  times: float array [T+1], physical times 0, h, 2h, ...
T >= horizon_steps. Array order, channels, grid, dtype and timestamps are checked.
Only the requested T+1 states are evaluated; initial state is excluded from metrics.
Do not concatenate training-seed runs as if they were independent test trajectories.
The sidecar dtype is reference_dtype. The full reference remains in this precision;
only the inference initial state is explicitly cast to the configured model dtype.
Metric arithmetic uses float64. Saved predicted rollouts retain the model dtype.

Separate JSON sidecar requires:
  h, nu, lengths, spatial_shape, channels, dtype, units="physical",
  includes_initial_state=true, generator, reference_accuracy_status,
  projection_policy
These declarations must match the requested experiment. Record actual spatial and
temporal convergence evidence in reference_accuracy_status; a declaration alone
does not prove reference accuracy. No target states may be used for tuning.

Config contract
---------------
Use demo/config_numpy.json as the minimal concrete example. Fields include:
  schema_version=1
  evidence_type and backend (numpy/torch), device, dtype
  reference_dtype: defaults to dtype; may be float64 with float32 neural inference
  target_trajectory_tuning_allowed=false
  h, nu0, nu, lengths, spatial_shape, channels, horizon_steps
  methods: frozen, delta, explicit, lie_post, lie_pre, strang, implicit
  delta_one_sided (boolean; default true)
  failure_max_abs (finite physical threshold; default 10000)
  dataset: {path, sha256}; dataset_metadata: {path, sha256}
  adapter: {path} OR {module}; additional adapter settings are allowed
  training_seed: null for numerical controls, nonnegative integer for neural modes
  ic_seeds: optional list with B entries; it is not a training-seed count

For either neural label, artifacts must additionally include:
  checkpoint: {path, sha256}
  normalizer: {path, sha256}
The runner hashes actual files and rejects mismatches before model loading. Even
an identity normalizer must have an explicit serialized artifact documenting that
convention. Relative paths are resolved against the config directory. Hashes and
metadata identify provided artifacts; they do not establish authorship, provenance,
correct loading, or absence of target-data tuning by themselves.

Physical predictor adapter
--------------------------
The adapter exports:
  load_predictor(config) -> (physical_predict, metadata)

The callable receives [B,C,*spatial] and returns the FULL NEXT PHYSICAL STATE,
not an increment and not normalized network output. It must preserve shape,
dtype and device, be sample-independent and stateless, and not mutate its input.
Input-mutation checks are enabled because this is an accuracy runner. A trained
model must be in eval mode, with original fixed source normalization and original
coordinate construction; neither coordinates nor normalizer statistics are
diffused by the transfer wrappers. No target-normalizer recomputation is allowed.

For a normalized-increment predictor, reconstruct the actual paper convention:
  physical_next = u + increment_mean
                    + increment_std * model(normalize_source(u), coordinates)
The original API, tensor ordering, coordinate encoding and broadcasts must be
preserved. If the original model already returns physical_next, do not add u again.

Metadata returned by the adapter requires the following matching declarations:
  evidence_type, h, nu0, lengths, evaluation_spatial_shape, channels, dtype,
  backend, units="physical", output_kind="full_next_state",
  stateless=true, sample_independent=true, projection_policy
Neural modes also require checkpoint_sha256, normalizer_sha256, eval_mode=true,
training_seed matching the validated config/artifacts.
Include training_resolution and architecture description as additional provenance
when evaluating 64-to-128 resolution transfer. Evaluation grid is checked separately.

Actual neural forward counts
----------------------------
For either neural evidence label, physical_predict must expose:
  get_forward_count() -> monotonically increasing Python int
Instrument the original torch.nn.Module using a forward_pre_hook/forward_hook,
so the counter increments on every actual model evaluation, not just on entry to
the wrapper. A callback that internally calls the network twice is rejected by
the one-call-method contract. An adapter-declared counter is only as trustworthy
as its implementation; retain the hook and adapter source for review.

Each method calls physical_predict once per full physical step. B=8,T=50 means
50 batched callbacks and 400 requested sample-steps. Numerical controls report
actual_network_forward_calls=null because no network exists. Metrics and summary
record counters separately. Pure Strang still uses one FULL h neural step, with
analytic half-diffusion on its input and output; it does not require an h/2 network.

Original nonlinear-input Recast
------------------------------
The v2 equation is y = Phi(u + h*(nu-nu0)*Delta(y)). The implicit method in this
runner instead solves y = Phi(u) + h*(nu-nu0)*Delta(y), using one FFT/IFFT pair
and one predictor call. These are different equations and cannot share accuracy
results by assumption. Requesting original_input_recast_aitken is rejected with
an explicit error. Integrate and audit the actual manuscript Aitken code before
adding that baseline; include initialization/residual forwards, stopping norm,
relaxation clamps, batch stopping, nonconverged steps and iteration counts.

Metrics and failure handling
----------------------------
Absolute L2 is a physical-grid quadrature:
  sqrt(product(lengths)/N_spatial * sum_over_channels_and_grid((prediction-truth)^2))
Relative L2 divides by the corresponding reference norm. If reference norm is
exactly zero, relative L2 is undefined: output null and zero_reference_norm status,
including 0/0. Absolute L2 remains meaningful. No tiny denominator is substituted.

PSRE uses full FFTs, norm="forward", spatial axes only, with power |F(u)|^2.
The numerator/denominator are Euclidean norms across all channels and Fourier
modes. Zero reference power yields null/status, not an invented relative error.
Power scaling avoids avoidable intermediate overflow without altering the ratio.
This power-only metric does not detect phase errors on its own.

A trajectory fails at its first nonfinite prediction or max_abs > failure_max_abs.
Failed samples remain NaN/missing in saved rollouts and subsequent metrics. The
runner retains a fixed batch shape and replaces failed samples by zero internally
ONLY to let other sample-independent trajectories finish. These placeholders
never become successful predictions. The first failure step is written on every
row for that initial condition. Aggregate metrics become null if any constituent
is undefined or failed; no silent mean over survivors is computed. Counts of
failed trajectories, undefined metrics and successful sample-steps accompany it.
No clipping, denoising, positivity projection or spectral truncation is added.

Outputs
-------
  summary.json: complete config, hashes, source metadata, runtime and method means
  metrics.csv: one row per method, initial condition, physical step
  rollouts.npz: optional method arrays [T+1,B,C,*spatial]; failures are NaN
Each config represents ONE training seed and ONE viscosity. To report 3 seeds,
evaluate 3 independently trained checkpoints on paired test sets and aggregate
the resulting seed-level quantities. Never treat timesteps or Fourier modes as
independent training replications. Retain per-seed numbers and paired differences.

Executed reconstruction integration
----------------------------------
From the package root, with the accompanying Burgers study results present:
  python runner/prepare_burgers_case.py --results burgers_fno/results --seed 0 --ratio 2 --out runner/new_burgers_case
  python runner/evaluate.py --config runner/new_burgers_case/config.json --out runner/new_burgers_results --save-rollouts

The adapter imports the actual reconstructed model implementation, validates
checkpoint-normalizer equality and physical parameters, preserves complex64
Fourier weights, freezes/evaluates the model, and instruments real forwards.
Never cast a complete FNO module to a real dtype: that can discard imaginary
parts of complex Fourier parameters. This adapter changes only the device and
checks native float32/complex64 parameter precision.

The verified seed0/ratio2 Avg50 values are: frozen 0.280489457585879,
delta 0.021008342751721, Lie-post 0.016136901081867, Lie-pre 0.017605658618021,
Strang 0.014982783444629, implicit 0.015804234990470. Explicit correction failed
for all eight initial conditions and has no aggregate error value. These numbers
describe this reconstructed model, not the manuscript's unavailable checkpoint.

Limitations
-----------
This implementation covers periodic constant scalar-coefficient Laplacian changes.
It does not yet implement fourth-order hyperviscosity, per-channel diffusivities,
nonperiodic boundaries or phase operators. Under nu<nu0 only frozen/delta are
permitted, and two-sided additive experiments require delta_one_sided=false.
No inverse-heat splitting or singular implicit denominator is silently regularized.

Torch inference stays on the configured device, but states are copied to CPU for
metric collection every step. Therefore this program MUST NOT be used for GPU
latency/speedup claims. Use the separately verified synchronized timing protocol
with identical storage policy, batch, precision, hardware and physical horizon.
