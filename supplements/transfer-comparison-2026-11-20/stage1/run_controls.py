"""Run independent mathematical/implementation controls, not neural experiments.

Usage: python3 run_controls.py --out results
Requires numpy, scipy and matplotlib. No network access or model is required.
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from spectral_transfer_numpy import (METHODS, PeriodicLaplacian, TransferStepper,
                                     affine_picard_cached, input_recast_picard)


def rel(a, b):
    return float(np.linalg.norm((a - b).ravel()) /
                 max(np.linalg.norm(b.ravel()), np.finfo(float).tiny))


def spectral_controls():
    rng = np.random.default_rng(20261118)
    rows = []
    for name, shape, nu0, h in [("NS64", (64, 64), .001, .05),
                                ("NS128", (128, 128), .001, .05),
                                ("Burgers64", (64,), .01, .02)]:
        grid = PeriodicLaplacian(shape, (1.,) * len(shape))
        u = rng.normal(size=(2,) + shape)
        phi = lambda v: grid.heat(v, h * nu0)
        nu = 2 * nu0
        b = phi(u)
        truth = grid.heat(u, h * nu)
        zero_shift, exact_heat, calls = {}, {}, {}
        for method in METHODS:
            zero = TransferStepper(phi, grid, h, nu0, nu0, method)
            zero_shift[method] = rel(zero(u), b)
            corrected = TransferStepper(phi, grid, h, nu0, nu, method)
            prediction = corrected(u)
            calls[method] = corrected.calls
            if method in ("delta", "lie_post", "lie_pre", "strang"):
                exact_heat[method] = rel(prediction, truth)
        # The additive formula, without a policy switch, is exact for heat flow
        # for lower viscosity as well. This is not a learned-rollout guarantee.
        lower = TransferStepper(phi, grid, h, nu0, .5 * nu0, "delta", False)(u)
        lower_error = rel(lower, grid.heat(u, .5 * h * nu0))
        implicit = TransferStepper(phi, grid, h, nu0, nu, "implicit")(u)
        z = h * (nu - nu0) * grid.k2
        implicit_residual = rel(implicit - h * (nu - nu0) * grid.laplacian(implicit), b)
        omega = 2 / (2 + float(z.max()))
        cached, diagnostics = affine_picard_cached(b, grid, h, nu - nu0, omega=omega)
        s = np.exp(-h * nu0 * grid.k2)
        input_exact = grid.apply(u, s / (1 + z * s))
        input_iter, input_diag = input_recast_picard(phi, u, grid, h, nu - nu0)
        constant = np.ones_like(u)
        constant_error = max(float(np.max(np.abs(TransferStepper(phi, grid, h, nu0, nu, m)(constant)
                                                   - constant))) for m in METHODS)
        assert max(zero_shift.values()) == 0
        assert max(exact_heat.values()) < 2e-14
        assert lower_error < 2e-14
        assert implicit_residual < 2e-14
        assert diagnostics["converged"] and rel(cached, implicit) < 2e-12
        assert input_diag["converged"] and rel(input_iter, input_exact) < 2e-12
        assert set(calls.values()) == {1}
        assert constant_error < 1e-14
        rows.append({"case": name, "equation_tested": "pure heat; named grid uses manuscript discretization parameters only", "shape": shape, "nu0": nu0, "h": h,
                     "zero_shift_rel_error": zero_shift, "pure_heat_rel_error": exact_heat,
                     "lower_viscosity_additive_rel_error": lower_error,
                     "predictor_calls_per_step": calls, "constant_mode_max_abs_error": constant_error,
                     "implicit_equation_rel_residual": implicit_residual,
                     "cached_affine_vs_closed_rel_error": rel(cached, implicit),
                     "cached_affine_iteration": diagnostics, "relaxation": omega,
                     "raw_affine_picard_spectral_radius": float(z.max()),
                     "input_recast_picard_spectral_radius": float((z * s).max()),
                     "input_recast_vs_its_linear_closed_form": rel(input_iter, input_exact),
                     "input_recast_diagnostic": input_diag,
                     "input_recast_vs_external_implicit_rel_difference": rel(input_exact, implicit)})
    # One-mode finite-step energy counterexample, with a valid diffusion C.
    c, frozen_multiplier = -.25, -.9
    energy = {"h_nu0_k2": float(np.log(2)), "nu_over_nu0": 2,
              "correction_multiplier": c, "frozen_multiplier": frozen_multiplier,
              "corrected_multiplier": frozen_multiplier + c,
              "quadratic_form_uCu_at_u1": c,
              "frozen_energy_at_u1": .5 * frozen_multiplier ** 2,
              "corrected_energy_at_u1": .5 * (frozen_multiplier + c) ** 2}
    assert energy["corrected_energy_at_u1"] > energy["frozen_energy_at_u1"]
    a, z = .5, .5
    scalar = {"a": a, "z": z, "frozen": float(np.exp(-a)),
              "target_heat": float(np.exp(-a-z)),
              "external_implicit": float(np.exp(-a)/(1+z)),
              "input_recast": float(np.exp(-a)/(1+z*np.exp(-a)))}
    return {"grids": rows, "energy_counterexample": energy, "one_mode_comparison": scalar}


def local_order_controls(out):
    rng = np.random.default_rng(7)
    n = 6
    L = -np.array([0., .7, 1.6, 3., 5., 8.])
    nu0, nu = .17, .30
    B = (nu - nu0) * L
    M = rng.normal(size=(n, n)) / np.sqrt(n)
    u = rng.normal(size=n) * .3

    def nonlinear(v):
        return .4 * (M @ np.tanh(v))

    def flow(v, visc, h):
        sol = solve_ivp(lambda t, y: visc * L * y + nonlinear(y), (0., h), v,
                        method="DOP853", rtol=2.5e-14, atol=1e-15, max_step=h/8)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1]

    h_values = .1 * .5 ** np.arange(6)
    rows = []
    leading_add = .5 * (.4 * M @ ((1 - np.tanh(u)**2) * (B * u)) + B * nonlinear(u))
    leading_post = .5 * (.4 * M @ ((1 - np.tanh(u)**2) * (B * u)) - B * nonlinear(u))
    for h in h_values:
        ref = flow(u, nu0, h)
        target = flow(u, nu, h)
        candidates = {
            "frozen": ref,
            "delta": ref + (np.exp(h * nu * L) - np.exp(h * nu0 * L)) * u,
            "explicit": ref + h * B * u,
            "lie_post": np.exp(h * B) * ref,
            "lie_pre": flow(np.exp(h * B) * u, nu0, h),
            "strang": np.exp(.5*h*B) * flow(np.exp(.5*h*B) * u, nu0, h),
            "implicit": ref / (1 - h*B),
        }
        for method, pred in candidates.items():
            rows.append({"h": float(h), "method": method,
                         "absolute_l2_error": float(np.linalg.norm(target - pred))})
        rows += [{"h": float(h), "method": "delta_after_leading_term",
                  "absolute_l2_error": float(np.linalg.norm(target-candidates["delta"]-h*h*leading_add))},
                 {"h": float(h), "method": "lie_post_after_leading_term",
                  "absolute_l2_error": float(np.linalg.norm(target-candidates["lie_post"]-h*h*leading_post))}]
    slopes = {}
    for method in (*METHODS, "delta_after_leading_term", "lie_post_after_leading_term"):
        selected = [r for r in rows if r["method"] == method][-4:]
        slopes[method] = float(np.polyfit(np.log([r["h"] for r in selected]),
                                         np.log([r["absolute_l2_error"] for r in selected]), 1)[0])
    assert .95 < slopes["frozen"] < 1.05
    assert 2.9 < slopes["strang"] < 3.1
    assert all(1.9 < slopes[m] < 2.1 for m in ("delta", "explicit", "lie_post", "lie_pre", "implicit"))
    assert 2.9 < slopes["delta_after_leading_term"] < 3.1
    assert 2.9 < slopes["lie_post_after_leading_term"] < 3.1
    write_csv(out / "local_order.csv", rows)
    return {"description": "Six-dimensional nonlinear ODE with exact-reference numerical flow; not a neural checkpoint.",
            "seed": 7, "nu0": nu0, "nu": nu, "L_diagonal": L.tolist(),
            "mixing_matrix": M.tolist(), "initial_state": u.tolist(),
            "solver": "DOP853 rtol=2.5e-14 atol=1e-15 max_step=h/8",
            "fit": "OLS log(error) against log(h), four smallest h values",
            "fitted_local_orders": slopes, "rows": rows}


class BurgersReference:
    """Dealiased conservative pseudo-spectral Burgers RK4 in float64."""

    def __init__(self, n=64, length=1.):
        self.n = n
        k = 2*np.pi*np.fft.fftfreq(n, length/n)
        self.k2 = k*k
        derivative_k = k.copy()
        if n % 2 == 0:
            derivative_k[n//2] = 0.
        self.ik = 1j*derivative_k
        self.mask = (np.abs(np.fft.fftfreq(n)*n) < n/3)

    def rhs(self, u, nu):
        uh = np.fft.fft(u, axis=-1)
        filtered = np.fft.ifft(uh*self.mask, axis=-1).real
        nonlinearity = -.5*self.ik*np.fft.fft(filtered*filtered, axis=-1)*self.mask
        return np.fft.ifft(nonlinearity-nu*self.k2*uh, axis=-1).real

    def advance(self, u, nu, h, max_dt):
        count = int(np.ceil(h / max_dt))
        dt = h / count
        u = u.copy()
        for _ in range(count):
            a = self.rhs(u, nu)
            b = self.rhs(u+.5*dt*a, nu)
            c = self.rhs(u+.5*dt*b, nu)
            d = self.rhs(u+dt*c, nu)
            u += dt*(a+2*b+2*c+d)/6
        return u


def burgers_control(out):
    n, h, nu0, nu, steps = 64, .02, .01, .02, 50
    grid = PeriodicLaplacian((n,), (1.,))
    solver = BurgersReference(n)
    x = np.arange(n)/n
    initial = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        u = sum((rng.normal()*np.sin(2*np.pi*j*x) + rng.normal()*np.cos(2*np.pi*j*x))/j
                for j in range(1, 5))
        u = .5*u/np.sqrt(np.mean(u*u))
        initial.append(u)
    u0 = np.array(initial)

    def trajectory(nu_value, dt):
        u = u0.copy()
        states = [u.copy()]
        for _ in range(steps):
            u = solver.advance(u, nu_value, h, dt)
            states.append(u.copy())
        return np.array(states)

    start = time.perf_counter()
    truth = trajectory(nu, .00025)
    refined_truth = trajectory(nu, .000125)
    ref_coarse = trajectory(nu0, .0005)
    ref_fine = trajectory(nu0, .00025)
    truth_refine_error = max(rel(a, b) for a, b in zip(truth[1:], refined_truth[1:]))
    reference_refine_error = max(rel(a, b) for a, b in zip(ref_coarse[1:], ref_fine[1:]))
    assert truth_refine_error < 1e-7 and reference_refine_error < 1e-7
    # The finest target solution is used for all methods.
    truth = refined_truth
    raw, summary, stored = [], [], {"truth": truth, "initial": u0}
    for method in METHODS:
        predictor = lambda v: solver.advance(v, nu0, h, .0005)
        stepper = TransferStepper(predictor, grid, h, nu0, nu, method)
        u = u0.copy()
        states = [u.copy()]
        errors = []
        failure_steps = [None] * len(u0)
        alive = np.ones(len(u0), dtype=bool)
        for step in range(1, steps+1):
            with np.errstate(over="ignore", invalid="ignore"):
                u = stepper(u)
            bad = alive & (~np.all(np.isfinite(u), axis=-1) | (np.max(np.abs(u), axis=-1) > 1e4))
            for idx in np.flatnonzero(bad):
                failure_steps[idx] = step
            alive[bad] = False
            current_error = np.full(len(u0), np.nan)
            current_error[alive] = np.linalg.norm(u[alive]-truth[step,alive], axis=-1)/np.linalg.norm(truth[step,alive], axis=-1)
            state = u.copy()
            state[~alive] = np.nan
            states.append(state)
            # Keep failed trajectories out of further arithmetic, but never
            # count them as successful or average only the surviving subset.
            u[~alive] = 0.
            errors.append(current_error)
            for seed, error in enumerate(current_error):
                raw.append({"method": method, "initial_condition_seed": seed, "step": step,
                            "time": h*step, "rel_l2": "" if not np.isfinite(error) else float(error),
                            "status": "failed" if failure_steps[seed] is not None else "ok"})
        errors = np.array(errors)
        stored[method] = np.array(states)
        valid = all(s is None for s in failure_steps)
        first_failure = min([s for s in failure_steps if s is not None], default=None)
        summary.append({"method": method, "avg50_mean": float(errors.mean()) if valid else None,
                        "step50_mean": float(errors[-1].mean()) if valid else None,
                        "avg50_per_ic": [float(errors[:,i].mean()) if s is None else None for i,s in enumerate(failure_steps)],
                        "step50_per_ic": [float(errors[-1,i]) if s is None else None for i,s in enumerate(failure_steps)],
                        "failure_step": first_failure, "failure_steps_per_ic": failure_steps,
                        "n_failed": sum(s is not None for s in failure_steps),
                        "reference_propagator_calls": stepper.calls,
                        "status": "ok" if valid else "failed"})
    write_csv(out / "burgers_control_curve.csv", raw)
    write_csv(out / "burgers_control_summary.csv", [{k: v for k, v in r.items() if k not in
              ("avg50_per_ic", "step50_per_ic", "failure_steps_per_ic")} for r in summary])
    np.savez_compressed(out / "burgers_control_states.npz", **stored)
    return {"evidence_type": "numerical-reference control, NOT manuscript FNO results",
            "reference_map": "nu0 pseudo-spectral RK4 flow, max_dt=0.0005; no learned network",
            "grid": n, "length": 1., "h": h, "nu0": nu0, "nu": nu,
            "steps": steps, "initial_condition_seeds": [0,1,2], "initial_rms": .5,
            "initial_condition": "Four random sine/cosine modes with amplitudes divided by mode number; normalized RMS=0.5",
            "training_seeds": None, "truth_dt": .000125,
            "truth_time_refinement_max_rel_l2": truth_refine_error,
            "reference_time_refinement_max_rel_l2": reference_refine_error,
            "spatial_convergence": "not established; this is a fixed-grid semidiscrete control",
            "failure_policy": "First nonfinite value or max absolute field >1e4; remaining metrics missing, never omitted from failure count",
            "summary": summary, "control_elapsed_seconds": time.perf_counter()-start}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(out, local, burgers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"frozen":"#777777", "delta":"#0068a3", "explicit":"#bb5037",
              "lie_post":"#c48a17", "lie_pre":"#956ba4", "strang":"#138875", "implicit":"#d05e8c"}
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "figure.dpi":160, "savefig.dpi":220})
    fig, ax = plt.subplots(figsize=(8,5))
    for method in METHODS:
        rows = [r for r in local["rows"] if r["method"] == method]
        ax.loglog([r["h"] for r in rows], [r["absolute_l2_error"] for r in rows],
                  "o-", ms=4, lw=1.6, color=colors[method],
                  label=f'{method}: slope {local["fitted_local_orders"][method]:.3f}')
    ax.set(xlabel="Time step h", ylabel="One-step absolute L2 error",
           title="Local error order with a near-exact reference flow")
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=9, loc="best")
    fig.text(.5,.012,"6D nonlinear control. No neural checkpoint; slopes are local, not global orders.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(out/"local_order.png")
    fig.savefig(out/"local_order.pdf")
    plt.close(fig)
    curves = list(csv.DictReader(open(out/"burgers_control_curve.csv", encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8,5))
    for method in METHODS:
        points = []
        for step in range(1,51):
            vals = [float(r["rel_l2"]) for r in curves
                    if r["method"]==method and int(r["step"])==step and r["rel_l2"]]
            points.append(float(np.mean(vals)) if len(vals)==3 else np.nan)
        ax.semilogy(np.arange(1,51),points,color=colors[method],lw=1.7,label=method)
    ax.set(xlabel="Autoregressive step",ylabel="Mean relative L2 error (3 initial conditions)",
           title="Burgers control: numerical reference propagator at nu0")
    ax.set_ylim(1e-6,1e2)
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=9, loc="best")
    failed=[r for r in burgers["summary"] if r["failure_step"] is not None]
    note="NOT learned-model results. " + "; ".join(f'{r["method"]}: first failure step {r["failure_step"]}; {r["n_failed"]}/3 failed' for r in failed)
    fig.text(.5,.012,note,ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(out/"burgers_control_rollout.png")
    fig.savefig(out/"burgers_control_rollout.pdf")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results"))
    parser.add_argument("--skip-burgers",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    result={"evidence_type":"independent controls; no original neural checkpoint or test trajectories",
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "environment":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
                           "platform":platform.platform(),"logical_cpus":os.cpu_count()},
            "spectral_controls":spectral_controls()}
    result["local_order"]=local_order_controls(args.out)
    print(json.dumps({"spectral_checks":"passed","local_orders":result["local_order"]["fitted_local_orders"]}),flush=True)
    if not args.skip_burgers:
        result["burgers_control"]=burgers_control(args.out)
        figures(args.out,result["local_order"],result["burgers_control"])
        print(json.dumps({"burgers_control_summary":result["burgers_control"]["summary"]}),flush=True)
    (args.out/"control_results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("All requested control checks passed. These are not manuscript neural results.",flush=True)


if __name__=="__main__":
    main()
