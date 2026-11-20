"""Forced 2D Navier--Stokes numerical-reference controls, NOT neural results.

Reference flow: conservative, 2/3-dealiased Fourier spatial discretization and
classical explicit RK4. States have shape [batch, Nx, Ny], physical vorticity.
Run from any directory: python run_ns_controls.py --out results --pilot
Then: python run_ns_controls.py --out results
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import fft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from spectral_transfer_numpy import METHODS, PeriodicLaplacian, TransferStepper

EVIDENCE = "numerical_reference_control_not_neural"
H = 0.05
NU0 = 1e-3
SEEDS = (0, 1, 2)
THRESHOLD = 1e4
SNAPSHOT_STEPS = (0, 1, 10, 25, 50)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def relative(a, b):
    axes = tuple(range(1, a.ndim))
    num = np.sqrt(np.sum((a - b) ** 2, axis=axes))
    den = np.sqrt(np.sum(b ** 2, axis=axes))
    return num / np.maximum(den, np.finfo(float).tiny)


def initial_conditions(n):
    """One shared continuous trigonometric polynomial per initial-condition seed.

    The independent half-plane modes avoid duplicate sine/cosine pairs. Modes
    have max(|mx|, |my|) <= 4, and coefficient standard deviation is 1/(1+|m|²).
    Analytic continuous variance sets unit standard deviation on every grid;
    normalization never depends on target data or grid resolution.
    """
    x = np.arange(n) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    fields, all_coefficients = [], []
    modes = [(mx, my) for mx in range(0, 5) for my in range(-4, 5)
             if (mx > 0 or my > 0)]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        coefficients = []
        field = np.zeros((n, n))
        variance = 0.0
        for mx, my in modes:
            a, b = rng.normal(size=2) / (1 + mx * mx + my * my)
            phase = 2 * np.pi * (mx * xx + my * yy)
            field += a * np.cos(phase) + b * np.sin(phase)
            variance += 0.5 * (a * a + b * b)
            coefficients.append([mx, my, float(a), float(b)])
        scale = np.sqrt(variance)
        fields.append(field / scale)
        all_coefficients.append({"initial_condition_seed": seed,
                                 "normalization_scale": float(scale),
                                 "coefficients_before_normalization": coefficients})
    return np.stack(fields), all_coefficients


class NSReference:
    """Real FFT Fourier--Galerkin nonlinear advection plus full-grid diffusion.

    The state is not projected after each RK4 step. Only the nonlinear inputs
    and output use the strict 2/3 mask. Thus a wrapper's high-mode errors still
    undergo physical Laplacian diffusion and are not silently removed.
    """
    def __init__(self, n, forcing=True):
        self.n = int(n)
        mx = np.fft.fftfreq(n) * n
        my = np.fft.rfftfreq(n) * n
        kx, ky = 2 * np.pi * mx[:, None], 2 * np.pi * my[None, :]
        self.q = kx * kx + ky * ky
        dx, dy = kx.copy(), ky.copy()
        if n % 2 == 0:
            dx[n // 2, 0] = 0
            dy[0, -1] = 0
        self.ikx, self.iky = 1j * dx, 1j * dy
        self.inv_q = np.divide(1., self.q, out=np.zeros_like(self.q), where=self.q != 0)
        self.mask = (np.abs(mx[:, None]) < n / 3) & (np.abs(my[None, :]) < n / 3)
        x = np.arange(n) / n
        xx, yy = np.meshgrid(x, x, indexing="ij")
        f = .1 * (np.sin(2 * np.pi * (xx + yy)) + np.cos(2 * np.pi * (xx - yy)))
        self.force_hat = self.forward(f if forcing else np.zeros_like(f))
        self.axes = (-2, -1)

    def forward(self, u):
        return fft.rfft2(u, axes=(-2, -1), workers=1)

    def inverse(self, uh):
        return fft.irfft2(uh, s=(self.n, self.n), axes=(-2, -1), workers=1)

    def rhs(self, uh, nu):
        filtered = uh * self.mask
        psi = filtered * self.inv_q
        stack = np.stack((filtered, self.iky * psi, -self.ikx * psi), axis=0)
        w, vx, vy = self.inverse(stack)
        flux = self.forward(np.stack((vx * w, vy * w), axis=0))
        nonlinear = -(self.ikx * flux[0] + self.iky * flux[1]) * self.mask
        return nonlinear - nu * self.q * uh + self.force_hat

    def step_hat(self, uh, nu, dt):
        k1 = self.rhs(uh, nu)
        k2 = self.rhs(uh + .5 * dt * k1, nu)
        k3 = self.rhs(uh + .5 * dt * k2, nu)
        k4 = self.rhs(uh + dt * k3, nu)
        return uh + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def flow(self, u, nu, duration, max_dt):
        steps = int(np.ceil(duration / max_dt - 1e-12))
        dt = duration / steps
        zmax = nu * float(self.q.max()) * dt
        if zmax > 2.5:
            raise ValueError(f"RK4 diffusion safety limit exceeded: {zmax}")
        uh = self.forward(u)
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(steps):
                uh = self.step_hat(uh, nu, dt)
        return self.inverse(uh)

    def trajectory(self, u, nu, steps, h, max_dt):
        trajectory = [u.copy()]
        for _ in range(steps):
            u = self.flow(u, nu, h, max_dt)
            if not np.isfinite(u).all():
                raise RuntimeError("Reference solver became nonfinite.")
            trajectory.append(u)
        return np.stack(trajectory)

    def kinetic_energy(self, u):
        uh = self.forward(u)
        vx = self.inverse(self.iky * self.inv_q * uh)
        vy = self.inverse(-self.ikx * self.inv_q * uh)
        return .5 * np.mean(vx * vx + vy * vy, axis=(-2, -1))


def spectral_restrict(u, n):
    """Orthogonal Fourier projection from an even fine grid to a coarser grid.

    Modes touching either coarse Nyquist axis are removed, making the real
    projection unambiguous. These modes lie outside the retained nonlinear
    subspace of the 64-grid control.
    """
    fine = u.shape[-1]
    assert u.shape[-2] == fine and fine >= n and n % 2 == 0
    uh = np.fft.fft2(u, axes=(-2, -1)) / fine ** 2
    modes = np.fft.fftfreq(n) * n
    idx = modes.astype(int) % fine
    coarse = uh[..., idx[:, None], idx[None, :]].copy()
    coarse[..., n // 2, :] = 0
    coarse[..., :, n // 2] = 0
    return np.fft.ifft2(coarse * n ** 2, axes=(-2, -1)).real


def metrics(pred, truth, solver):
    rms = np.sqrt(np.mean((pred - truth) ** 2, axis=(-2, -1)))
    truth_rms = np.sqrt(np.mean(truth ** 2, axis=(-2, -1)))
    pred_power = np.abs(np.fft.fft2(pred, axes=(-2, -1), norm="ortho")) ** 2
    truth_power = np.abs(np.fft.fft2(truth, axes=(-2, -1), norm="ortho")) ** 2
    power_error = relative(pred_power, truth_power)
    return {"relative_l2": rms / truth_rms, "physical_l2_unit_area": rms,
            "psre_full_fft": power_error,
            "enstrophy": .5 * np.mean(pred ** 2, axis=(-2, -1)),
            "kinetic_energy": solver.kinetic_energy(pred),
            "mean_vorticity": np.mean(pred, axis=(-2, -1))}


def sanity_controls():
    n = 32
    solver = NSReference(n, forcing=False)
    x = np.arange(n) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    # A Laplacian eigenfunction has zero advective term (omega proportional psi).
    u = np.stack([np.sin(2 * np.pi * (xx + 2 * yy))])
    nu, h = .001, .05
    exact = u * np.exp(-nu * h * (2 * np.pi) ** 2 * 5)
    numerical = solver.flow(u, nu, h, .0025)
    heat_error = float(relative(numerical, exact).max())
    if heat_error > 2e-13:
        raise AssertionError(f"NS single-mode heat check failed: {heat_error}")
    uh = solver.forward(u)
    nonlinear = solver.rhs(uh, 0)
    nonlinear_norm = float(np.linalg.norm(nonlinear) / np.linalg.norm(uh))
    if nonlinear_norm > 1e-12:
        raise AssertionError("Eigenfunction advection should vanish.")
    u64, coefficients = initial_conditions(64)
    u128, _ = initial_conditions(128)
    initial_projection_error = float(relative(spectral_restrict(u128, 64), u64).max())
    if initial_projection_error > 2e-14:
        raise AssertionError("ICs differ across resolutions.")
    forced = NSReference(64)
    forced0 = np.zeros_like(u64)
    forced_num = forced.flow(forced0, NU0, H, .0025)
    fq = 2 * (2 * np.pi) ** 2
    forced_exact = forced.inverse(forced.force_hat)[None] * (-np.expm1(-NU0 * fq * H) / (NU0 * fq))
    forced_exact = np.broadcast_to(forced_exact, forced_num.shape)
    forced_error = float(relative(forced_num, forced_exact).max())
    if forced_error > 2e-13:
        raise AssertionError("Forced eigenfunction analytic solution failed.")
    # Standard vorticity omega = d_x v_y - d_y v_x; divergence-free velocity.
    test_hat = forced.forward(u64) * forced.mask
    psi_hat = test_hat * forced.inv_q
    vx_hat, vy_hat = forced.iky * psi_hat, -forced.ikx * psi_hat
    curl = forced.ikx * vy_hat - forced.iky * vx_hat
    div = forced.ikx * vx_hat + forced.iky * vy_hat
    curl_error = float(np.linalg.norm(curl - test_hat) / np.linalg.norm(test_hat))
    div_error = float(np.linalg.norm(div) / np.linalg.norm(test_hat))
    unforced = NSReference(64, forcing=False)
    inviscid_rate = float(np.max(np.abs(np.mean(u64 * unforced.inverse(unforced.rhs(test_hat, 0.)), axis=(-2, -1)))))
    if max(curl_error, div_error, inviscid_rate) > 1e-12:
        raise AssertionError("Velocity or inviscid enstrophy-rate check failed.")
    grid = PeriodicLaplacian((64, 64), (1., 1.))
    phi = lambda v: forced.flow(v, NU0, H, .005)
    b = phi(u64)
    source_equal = {}
    for m in METHODS:
        wrap = TransferStepper(phi, grid, H, NU0, NU0, m)
        source_equal[m] = float(np.max(np.abs(wrap(u64) - b)))
        assert wrap.calls == 1 and source_equal[m] == 0
    return {"single_mode_heat_rel_error": heat_error,
            "forced_eigenmode_rel_error": forced_error,
            "single_mode_advection_relative_norm": nonlinear_norm,
            "velocity_curl_relative_error": curl_error,
            "velocity_divergence_relative_error": div_error,
            "inviscid_enstrophy_rate_max_abs": inviscid_rate,
            "paired_64_128_initial_projection_rel_error": initial_projection_error,
            "source_viscosity_exact_wrapper_identity_max_abs": source_equal,
            "initial_coefficients": coefficients}


def pilot():
    result = []
    for n in (64, 128):
        solver = NSReference(n)
        u, _ = initial_conditions(n)
        t0 = time.perf_counter()
        solver.flow(u, NU0, H, .005)
        elapsed = time.perf_counter() - t0
        result.append({"grid": n, "batch_size": len(SEEDS),
                       "one_source_step_seconds": elapsed,
                       "rough_7_methods_3_ratios_50_steps_seconds": elapsed * 1050,
                       "note": "Execution estimate only, not a neural inference benchmark."})
    return result


def run_case(n, ratio, steps, out, prediction_dt=.0025, truth_dt=.00125):
    """Use fine and twice-finer target integration, and paired predictor checks."""
    tag = f"n{n}_r{ratio:g}".replace(".", "p")
    case_dir = out / tag
    case_dir.mkdir(parents=True, exist_ok=True)
    solver = NSReference(n)
    grid = PeriodicLaplacian((n, n), (1., 1.))
    u0, _ = initial_conditions(n)
    nu = NU0 * ratio
    t0 = time.perf_counter()
    # Ensure even the coarser truth run stays inside the RK4 negative-axis limit.
    coarse_dt = min(truth_dt, 2.4 / (nu * float(solver.q.max())))
    coarse_dt = H / int(np.ceil(H / coarse_dt))
    fine_dt = coarse_dt / 2
    print(f"{tag}: target refinement ({coarse_dt:g}, {fine_dt:g})", flush=True)
    coarse = solver.trajectory(u0, nu, steps, H, coarse_dt)
    truth = solver.trajectory(u0, nu, steps, H, fine_dt)
    target_refinement = [float(relative(coarse[s], truth[s]).max()) for s in range(1, steps + 1)]
    np.savez_compressed(case_dir / "target_states.npz", trajectory=truth[list(SNAPSHOT_STEPS)],
                        output_steps=np.array(SNAPSHOT_STEPS),
                        initial_condition_seeds=np.array(SEEDS))
    per_step_rows, summary, final_states = [], [], {"target": truth[-1], "u0": u0}
    source_refinement_rows = []
    failed_method_states = {}
    for method in METHODS:
        print(f"{tag}: {method}", flush=True)
        phi = lambda v: solver.flow(v, NU0, H, prediction_dt)
        stepper = TransferStepper(phi, grid, H, NU0, nu, method)
        state = u0.copy()
        first_failure = np.zeros(len(SEEDS), dtype=int)
        trajectory = []
        method_rows = []
        for step in range(1, steps + 1):
            before = state.copy()
            pred = stepper(state)
            invalid = ~np.isfinite(pred).all(axis=(-2, -1)) | (np.max(np.abs(pred), axis=(-2, -1)) > THRESHOLD)
            new_fail = (first_failure == 0) & invalid
            first_failure[new_fail] = step
            failed = first_failure > 0
            state = pred.copy()
            # Preserve failed output as NaN; zero placeholders only avoid one
            # failed sample contaminating the remaining independent batch work.
            state[failed] = 0
            displayed = pred.copy()
            displayed[failed] = np.nan
            trajectory.append(displayed)
            if step in (1, 25, steps):
                probe = before.copy()
                if method == "lie_pre":
                    probe = grid.apply(probe, stepper.extra_symbol)
                elif method == "strang":
                    probe = grid.apply(probe, stepper.half_symbol)
                coarse_probe = phi(probe)
                fine_probe = solver.flow(probe, NU0, H, prediction_dt / 2)
                probe_errors = relative(coarse_probe, fine_probe)
                for i, seed in enumerate(SEEDS):
                    if first_failure[i] == 0:
                        source_refinement_rows.append({"method": method, "step": step,
                            "initial_condition_seed": seed,
                            "source_step_refinement_rel_error": float(probe_errors[i])})
            clean = pred.copy()
            clean[failed] = 0
            stats = metrics(clean, truth[step], solver)
            for i, seed in enumerate(SEEDS):
                row = {"evidence_type": EVIDENCE, "grid": n, "ratio": ratio,
                       "method": method, "initial_condition_seed": seed, "step": step,
                       "time": step * H, "failed": bool(failed[i]),
                       "first_failure_step": int(first_failure[i]) or None}
                row.update({key: None if failed[i] else float(val[i]) for key, val in stats.items()})
                method_rows.append(row)
        rollout = np.stack(trajectory)
        snapshot_states = np.stack([u0 if s == 0 else rollout[s - 1] for s in SNAPSHOT_STEPS])
        np.savez_compressed(case_dir / f"{method}_snapshots.npz", trajectory=snapshot_states,
                            output_steps=np.array(SNAPSHOT_STEPS),
                            first_failure_step=first_failure,
                            initial_condition_seeds=np.array(SEEDS))
        if not np.any(first_failure):
            final_states[method] = rollout[-1]
        else:
            failed_method_states[method] = first_failure.tolist()
        for seed in SEEDS:
            rows = [r for r in method_rows if r["initial_condition_seed"] == seed]
            ff = int(first_failure[SEEDS.index(seed)])
            summary.append({"evidence_type": EVIDENCE, "grid": n, "ratio": ratio,
                "method": method, "initial_condition_seed": seed,
                "avg50_rell2": None if ff else float(np.mean([r["relative_l2"] for r in rows])),
                "step50_rell2": None if ff else rows[-1]["relative_l2"],
                "step50_psre": None if ff else rows[-1]["psre_full_fft"],
                "first_failure_step": ff or None,
                "predictor_calls_batched_rollout": stepper.calls,
                "physical_trajectory_steps_attempted": steps,
                "seed_type": "initial_condition_not_training"})
        per_step_rows.extend(method_rows)
        # Persist complete methods progressively; interrupted work is not final.
        write_csv(case_dir / "per_step.csv", per_step_rows)
        write_csv(case_dir / "summary_by_ic.csv", summary)
    np.savez_compressed(case_dir / "final_states.npz", **final_states)
    conv = {"evidence_type": EVIDENCE, "grid": n, "ratio": ratio,
            "target_dt_coarse": coarse_dt, "target_dt_fine": fine_dt,
            "target_refinement_max_rel_error": max(target_refinement),
            "target_refinement_by_step_max_over_ics": target_refinement,
            "source_predictor_dt": prediction_dt,
            "source_probe_refinement_max_rel_error": max(r["source_step_refinement_rel_error"] for r in source_refinement_rows),
            "source_probe_rows": source_refinement_rows,
            "failed_methods_first_failure_steps": failed_method_states,
            "execution_seconds_not_neural_benchmark": time.perf_counter() - t0}
    write_json(case_dir / "convergence.json", conv)
    # Conservative numerical-floor checks; if violated do not mark run complete.
    if conv["target_refinement_max_rel_error"] > 1e-7:
        raise AssertionError("Target time refinement exceeds predeclared 1e-7 tolerance.")
    if conv["source_probe_refinement_max_rel_error"] > 1e-7:
        raise AssertionError("Reference predictor step refinement exceeds 1e-7 tolerance.")
    write_json(case_dir / "complete.json", {"status": "complete", "evidence_type": EVIDENCE})
    return summary, conv


def aggregate(rows):
    output = []
    keys = sorted(set((r["grid"], r["ratio"], r["method"]) for r in rows))
    for n, ratio, method in keys:
        selected = [r for r in rows if (r["grid"], r["ratio"], r["method"]) == (n, ratio, method)]
        fail = sum(r["first_failure_step"] is not None for r in selected)
        row = {"evidence_type": EVIDENCE, "grid": n, "ratio": ratio, "method": method,
               "ic_count": len(selected), "failure_count": fail,
               "avg50_rell2_mean": None, "avg50_rell2_sd_over_ic": None,
               "step50_rell2_mean": None, "step50_rell2_sd_over_ic": None,
               "step50_psre_mean": None}
        if not fail:
            for key in ("avg50_rell2", "step50_rell2"):
                values = [r[key] for r in selected]
                row[key + "_mean"] = float(np.mean(values))
                row[key + "_sd_over_ic"] = float(np.std(values, ddof=1))
            row["step50_psre_mean"] = float(np.mean([r["step50_psre"] for r in selected]))
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.5, 2., 4.])
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        result = pilot()
        write_json(args.out / "pilot.json", result)
        print(json.dumps(result, indent=2))
        return
    if args.steps != 50:
        raise ValueError("This controlled protocol names Avg50 and Step50; require 50 steps.")
    config = {"evidence_type": EVIDENCE, "neural_checkpoint_used": False,
        "training_seeds_used": False, "initial_condition_seeds": SEEDS,
        "grids": args.grids, "ratios": args.ratios, "nu0": NU0,
        "h": H, "steps": args.steps, "domain_lengths": [1., 1.],
        "methods": METHODS, "failure_rule": "nonfinite or max(abs(vorticity)) > 1e4 at an output step",
        "failed_aggregate_policy": "null entire mean if any trajectory fails; keep individual failures",
        "source_dt": .0025, "source_probe_dt": .00125,
        "field_snapshots_at_output_steps": SNAPSHOT_STEPS,
        "metric_curves": "all 50 output steps, all ICs, all methods",
        "truth_max_dt_coarse": .00125, "truth_max_dt_fine": .000625,
        "reference_time_refinement_acceptance": 1e-7,
        "method_selection_tuned_on_target_data": False,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "fft_workers": 1, "dtype": "float64",
        "execution_started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.out / "config.json", config)
    sanity = sanity_controls()
    write_json(args.out / "sanity.json", sanity)
    all_rows, all_conv, source_convergence = [], [], []
    for n in args.grids:
        solver = NSReference(n)
        u0, _ = initial_conditions(n)
        print(f"n{n}: complete nu0 trajectory refinement", flush=True)
        source_coarse = solver.trajectory(u0, NU0, args.steps, H, .0025)
        source_fine = solver.trajectory(u0, NU0, args.steps, H, .00125)
        source_errors = np.stack([relative(source_coarse[s], source_fine[s]) for s in range(1, args.steps + 1)])
        source_case = {"grid": n, "nu": NU0, "dt_coarse": .0025, "dt_fine": .00125,
                       "max_relative_difference_all_output_steps_and_ics": float(source_errors.max()),
                       "relative_difference_by_output_step_and_ic": source_errors.tolist()}
        source_convergence.append(source_case)
        write_json(args.out / "source_flow_convergence.json", source_convergence)
        if source_errors.max() > 1e-7:
            raise AssertionError("Full nu0 flow refinement exceeds 1e-7 tolerance.")
        for ratio in args.ratios:
            progress = {"status": "running", "current_grid": n, "current_ratio": ratio,
                        "complete_cases": len(all_conv), "expected_cases": len(args.grids) * len(args.ratios)}
            write_json(args.out / "progress.json", progress)
            rows, conv = run_case(n, ratio, args.steps, args.out)
            all_rows.extend(rows)
            all_conv.append(conv)
            write_csv(args.out / "summary_by_ic.csv", all_rows)
            write_csv(args.out / "summary.csv", aggregate(all_rows))
    spatial = []
    if 64 in args.grids and 128 in args.grids:
        for ratio in args.ratios:
            tag = f"r{ratio:g}".replace(".", "p")
            coarse = np.load(args.out / f"n64_{tag}" / "target_states.npz")["trajectory"]
            fine = np.load(args.out / f"n128_{tag}" / "target_states.npz")["trajectory"]
            by_time = [relative(coarse[t], spectral_restrict(fine[t], 64)) for t in range(len(SNAPSHOT_STEPS))]
            errors = np.stack(by_time)
            spatial.append({"ratio": ratio,
                "projection": "orthogonal fine-to-coarse Fourier projection; coarse Nyquist axes removed",
                "snapshot_output_steps": SNAPSHOT_STEPS,
                "max_relative_difference_over_stored_snapshot_times_and_ics": float(errors.max()),
                "final_relative_difference_by_ic": errors[-1].tolist(),
                "scope": "64 versus 128 resolution consistency, not proof of 128 spatial convergence"})
    write_json(args.out / "convergence.json", {"time": all_conv, "source_flow_time": source_convergence,
                                             "spatial_64_128": spatial})
    write_json(args.out / "progress.json", {"status": "complete", "evidence_type": EVIDENCE,
                                            "complete_cases": len(all_conv)})
    print(json.dumps({"status": "complete", "cases": len(all_conv), "summary": aggregate(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
