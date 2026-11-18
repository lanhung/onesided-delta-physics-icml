"""Compare ways of attaching the known viscosity change to a frozen neural time-stepper.

Every method reuses the same frozen checkpoint and calls the network once per physical
step. Let Phi(u) = u + F(u) be the frozen step trained at nu0, S_a = exp(a * dt * Lap),
and B = (nu - nu0) * Lap.

    increment       Phi(u)                                    frozen model
    delta_physics   Phi(u) + (S_nu - S_nu0) u                 additive correction (the paper)
    lie_post        S_{nu-nu0} Phi(u)                         extra diffusion after the network
    lie_pre         Phi(S_{nu-nu0} u)                         extra diffusion before the network
    strang          S_{(nu-nu0)/2} Phi(S_{(nu-nu0)/2} u)      symmetric splitting
    implicit_post   (I - dt B)^{-1} Phi(u)                    backward-Euler extra diffusion
    explicit_add    Phi(u) + dt B u                           forward-Euler extra diffusion

All corrections are applied only for nu > nu0; at nu <= nu0 every method equals the frozen
model (for nu < nu0, S_{nu-nu0} would be the unbounded backward heat propagator).

The test trajectories, checkpoints and normalisers are exactly those of the original
runners: NS test seed 1000 + seed, training pool seed = seed (used only to rebuild
normalisers when a checkpoint directory has no normalizers.pt). The script checks that
its increment and delta_physics rows reproduce the saved seed_results.csv of the
checkpoint directory, when present.

Examples (CPU; add nothing for GPU, it is picked automatically):

    # 2D Navier-Stokes, the three 15-epoch seeds of the main experiment
    python run_splitting_baselines.py --pde ns

    # 2D Navier-Stokes, the 75-epoch seed-0 checkpoint
    python run_splitting_baselines.py --pde ns --checkpoint-dir results/ffno_ns64_budget75_seed0 \
        --seeds 0 --out results/splitting_baselines_ns64_budget75_seed0

    # 1D viscous Burgers, three seeds
    python run_splitting_baselines.py --pde burgers
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from fno_dual.data import generate_burgers_trajectories, generate_navier_stokes_trajectories
from fno_dual.ffno import FFNO2d
from fno_dual.fno1d import FNO1d
from fno_dual.normalizer import UnitGaussianNormalizer
from fno_dual.physics import diffusion_semigroup, diffusion_semigroup_1d

METHODS = ("increment", "delta_physics", "lie_post", "lie_pre", "strang",
           "implicit_post", "explicit_add")
KEY_METRICS = ("rollout_rel_l2_average", "rollout_rel_l2_final",
               "rollout_psre_average", "rollout_psre_final")

DEFAULTS = {
    "ns": dict(checkpoint_dir="results/ffno_ns64_high_viscosity_3seed",
               out="results/splitting_baselines_ns64_3seed",
               grid=64, step_dt=0.05, solver_dt=0.005, reference_viscosity=1e-3,
               viscosity_ratios=[1.0, 1.5, 1.75, 2.0], width=16, modes=8, depth=8),
    "burgers": dict(checkpoint_dir="results/burgers_high_viscosity_3seed",
                    out="results/splitting_baselines_burgers_3seed",
                    grid=64, step_dt=0.02, solver_dt=0.002, reference_viscosity=1e-2,
                    viscosity_ratios=[1.0, 1.25, 1.5, 1.75, 2.0], width=32, modes=16, depth=4),
}


# --------------------------------------------------------------------------- spectral helpers
def _k_squared(field, dims):
    """|k|^2 on the unit periodic box, same convention as fno_dual.physics."""
    if dims == 2:
        height, width = field.shape[-2:]
        ky = 2.0 * torch.pi * torch.fft.fftfreq(height, d=1.0 / height,
                                               device=field.device, dtype=field.dtype)
        kx = 2.0 * torch.pi * torch.fft.fftfreq(width, d=1.0 / width,
                                               device=field.device, dtype=field.dtype)
        return ky[:, None].square() + kx[None, :].square()
    length = field.shape[-1]
    k = 2.0 * torch.pi * torch.fft.rfftfreq(length, d=1.0 / length,
                                           device=field.device, dtype=field.dtype)
    return k.square()


def apply_multiplier(field, multiplier_of_k2, dims):
    """Apply a real Fourier multiplier m(|k|^2) to a real periodic field."""
    k2 = _k_squared(field, dims)
    if dims == 2:
        spectrum = torch.fft.fft2(field, dim=(-2, -1), norm="ortho")
        return torch.fft.ifft2(spectrum * multiplier_of_k2(k2), dim=(-2, -1), norm="ortho").real
    spectrum = torch.fft.rfft(field, dim=-1, norm="ortho")
    return torch.fft.irfft(spectrum * multiplier_of_k2(k2), n=field.shape[-1], dim=-1, norm="ortho")


def heat(field, viscosity, delta_t, dims):
    """S = exp(-viscosity * |k|^2 * delta_t); viscosity must be >= 0."""
    return apply_multiplier(field, lambda k2: torch.exp(-viscosity * delta_t * k2), dims)


def self_check(dims, device):
    """Make sure the local helpers agree with fno_dual.physics before any experiment runs."""
    generator = torch.Generator().manual_seed(0)
    if dims == 2:
        field = torch.randn(2, 16, 16, generator=generator).to(device)
        reference = diffusion_semigroup(field, 3e-3, 0.05)
    else:
        field = torch.randn(2, 32, generator=generator).to(device)
        reference = diffusion_semigroup_1d(field, 3e-2, 0.02)
    mine = heat(field, 3e-3 if dims == 2 else 3e-2, 0.05 if dims == 2 else 0.02, dims)
    error = (mine - reference).abs().max().item()
    if error > 1e-5:
        raise RuntimeError(f"spectral helper disagrees with fno_dual.physics (max error {error})")
    return error


# --------------------------------------------------------------------------- models and data
def model_inputs(states, input_normalizer, dims):
    if dims == 2:
        from run_ffno_dense_viscosity_scan import model_inputs as ns_inputs
        return ns_inputs(states, input_normalizer)
    from run_burgers_high_viscosity_3seed import model_inputs as burgers_inputs
    return burgers_inputs(states, input_normalizer)


def frozen_step(model, states, input_normalizer, increment_normalizer, dims):
    increment = increment_normalizer.decode(model(model_inputs(states, input_normalizer, dims)))
    return states + increment


def generator_for(pde):
    return generate_navier_stokes_trajectories if pde == "ns" else generate_burgers_trajectories


def flatten_pairs(trajectories):
    spatial = trajectories.shape[2:]
    return (trajectories[:, :-1].reshape(-1, *spatial), trajectories[:, 1:].reshape(-1, *spatial))


def load_model(seed, args, device):
    seed_dir = Path(args.checkpoint_dir) / f"seed_{seed}"
    if args.pde == "ns":
        model = FFNO2d(width=args.width, modes=args.modes, depth=args.depth, factor=args.factor)
    else:
        model = FNO1d(width=args.width, modes=args.modes, depth=args.depth)
    model = model.to(device)
    model.load_state_dict(torch.load(seed_dir / "increment.pt", map_location=device, weights_only=True))
    model.eval()

    input_normalizer = UnitGaussianNormalizer(torch.zeros(2, device=device))
    increment_normalizer = UnitGaussianNormalizer(torch.zeros(2, device=device))
    saved = seed_dir / "normalizers.pt"
    if saved.exists():
        statistics = torch.load(saved, map_location=device, weights_only=True)
        input_normalizer.mean, input_normalizer.std = statistics["input_mean"], statistics["input_std"]
        increment_normalizer.mean = statistics["increment_mean"]
        increment_normalizer.std = statistics["increment_std"]
        source = "normalizers.pt"
    else:
        # Rebuild exactly as the training runners did: training pool generated with `seed`.
        pool = generator_for(args.pde)(
            args.n_train + args.n_val, args.grid, args.reference_viscosity,
            args.train_steps, args.step_dt, args.solver_dt, seed, device,
        )
        states, nxt = flatten_pairs(pool[: args.n_train])
        input_normalizer = UnitGaussianNormalizer(states.unsqueeze(-1))
        increment_normalizer = UnitGaussianNormalizer(nxt - states)
        source = "rebuilt from training pool"
    return model, input_normalizer, increment_normalizer, source


# --------------------------------------------------------------------------- the methods
def corrected_step(method, model, state, viscosity, args, input_normalizer, increment_normalizer, dims):
    dt, nu0 = args.step_dt, args.reference_viscosity
    step = lambda x: frozen_step(model, x, input_normalizer, increment_normalizer, dims)
    if method == "increment" or viscosity <= nu0:
        return step(state)
    shift = viscosity - nu0
    if method == "delta_physics":
        return step(state) + heat(state, viscosity, dt, dims) - heat(state, nu0, dt, dims)
    if method == "lie_post":
        return heat(step(state), shift, dt, dims)
    if method == "lie_pre":
        return step(heat(state, shift, dt, dims))
    if method == "strang":
        return heat(step(heat(state, shift, dt / 2, dims)), shift, dt / 2, dims)
    if method == "implicit_post":
        return apply_multiplier(step(state), lambda k2: 1.0 / (1.0 + dt * shift * k2), dims)
    if method == "explicit_add":
        return step(state) + apply_multiplier(state, lambda k2: -dt * shift * k2, dims)
    raise ValueError(method)


@torch.no_grad()
def rollout(method, model, initial_state, steps, viscosity, args, input_normalizer,
            increment_normalizer, dims):
    state, predictions = initial_state, []
    for _ in range(steps):
        state = corrected_step(method, model, state, viscosity, args,
                               input_normalizer, increment_normalizer, dims)
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def metrics_for(prediction, trajectories, dims):
    if not torch.isfinite(prediction).all():
        return {name: float("nan") for name in KEY_METRICS}, True
    if dims == 2:
        from run_ffno_dense_viscosity_scan import extended_metrics
        return extended_metrics(prediction, trajectories), False
    from run_burgers_high_viscosity_3seed import rollout_metrics_1d
    return rollout_metrics_1d(prediction, trajectories), False


# --------------------------------------------------------------------------- reporting
def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    std = tensor.std(unbiased=True).item() if len(values) > 1 else float("nan")
    return tensor.mean().item(), std


def aggregate(rows):
    numeric = [k for k, v in rows[0].items()
               if isinstance(v, (int, float)) and k not in {"seed", "viscosity_ratio", "viscosity"}]
    output = []
    for ratio in sorted({r["viscosity_ratio"] for r in rows}):
        for method in METHODS:
            picked = [r for r in rows if r["viscosity_ratio"] == ratio and r["method"] == method]
            if not picked:
                continue
            row = {"method": method, "viscosity_ratio": ratio, "n_seeds": len(picked)}
            for name in numeric:
                row[f"{name}_mean"], row[f"{name}_std"] = mean_std(
                    [float(p.get(name, float("nan"))) for p in picked])
            output.append(row)
    return output


def paired_vs_delta(rows):
    """Per-seed relative change of each method against delta_physics (negative = better)."""
    output = []
    for ratio in sorted({r["viscosity_ratio"] for r in rows}):
        for method in METHODS:
            if method == "delta_physics":
                continue
            row = {"method": method, "viscosity_ratio": ratio}
            for name in KEY_METRICS:
                changes = []
                for seed in sorted({r["seed"] for r in rows}):
                    ours = next(r for r in rows if r["seed"] == seed and r["method"] == "delta_physics"
                                and r["viscosity_ratio"] == ratio)
                    other = next(r for r in rows if r["seed"] == seed and r["method"] == method
                                 and r["viscosity_ratio"] == ratio)
                    changes.append(100.0 * (other[name] - ours[name]) / ours[name])
                row[f"{name}_change_vs_delta_percent_mean"], row[f"{name}_change_vs_delta_percent_std"] = \
                    mean_std(changes)
            output.append(row)
    return output


def reproduction_check(rows, checkpoint_dir):
    """Compare recomputed increment/delta_physics metrics with the saved runner output."""
    saved_path = Path(checkpoint_dir) / "seed_results.csv"
    if not saved_path.exists():
        return [{"note": f"{saved_path} not found; reproduction not checked"}]
    with saved_path.open(encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    report = []
    for row in rows:
        if row["method"] not in ("increment", "delta_physics"):
            continue
        match = [s for s in saved if int(float(s["seed"])) == row["seed"] and s["method"] == row["method"]
                 and abs(float(s["viscosity_ratio"]) - row["viscosity_ratio"]) < 1e-9]
        if not match:
            continue
        for name in ("rollout_rel_l2_average", "rollout_rel_l2_final"):
            if name in match[0] and match[0][name] != "":
                report.append({"seed": row["seed"], "method": row["method"],
                               "viscosity_ratio": row["viscosity_ratio"], "metric": name,
                               "saved": float(match[0][name]), "recomputed": row[name],
                               "abs_difference": abs(float(match[0][name]) - row[name])})
    return report


def summary_markdown(aggregates, paired, args, largest):
    lines = [f"# Splitting baselines: {args.pde}, checkpoints `{args.checkpoint_dir}`", "",
             f"Seeds: {args.seeds}. One network evaluation per step for every method. "
             f"Mean ± sample s.d. over seeds at nu/nu0 = {largest:g}.", "",
             "| Method | Avg-50 RelL2 | Step-50 RelL2 | Avg-50 PSRE | rollout time (s) | Avg-50 RelL2 vs Δ-Physics (%) |",
             "|---|---|---|---|---|---|"]
    for method in METHODS:
        a = next((r for r in aggregates if r["method"] == method and r["viscosity_ratio"] == largest), None)
        if a is None:
            continue
        p = next((r for r in paired if r["method"] == method and r["viscosity_ratio"] == largest), None)
        change = ("—" if p is None else
                  f"{p['rollout_rel_l2_average_change_vs_delta_percent_mean']:+.2f} ± "
                  f"{p['rollout_rel_l2_average_change_vs_delta_percent_std']:.2f}")
        lines.append(
            f"| {method} | {a['rollout_rel_l2_average_mean']:.5f} ± {a['rollout_rel_l2_average_std']:.5f} "
            f"| {a['rollout_rel_l2_final_mean']:.5f} ± {a['rollout_rel_l2_final_std']:.5f} "
            f"| {a['rollout_psre_average_mean']:.5f} ± {a['rollout_psre_average_std']:.5f} "
            f"| {a['wall_clock_seconds_mean']:.2f} | {change} |")
    lines += ["", "Negative change = lower error than Δ-Physics. Timings are hardware dependent; "
              "`reference_solver_seconds` in seed_results.csv times the pseudo-spectral data generator "
              "(including initial-condition sampling) for the same test set."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pde", choices=("ns", "burgers"), default="ns")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--out")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--viscosity-ratios", nargs="+", type=float)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for key, value in DEFAULTS[args.pde].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    dims = 2 if args.pde == "ns" else 1
    torch.set_num_threads(args.threads)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    helper_error = self_check(dims, device)
    config = {**vars(args), "device": device, "helper_self_check_max_error": helper_error,
              "torch_version": torch.__version__}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)

    rows = []
    for seed in args.seeds:
        model, input_normalizer, increment_normalizer, source = load_model(seed, args, device)
        print({"seed": seed, "normalizers": source}, flush=True)
        for ratio in args.viscosity_ratios:
            viscosity = ratio * args.reference_viscosity
            start = time.perf_counter()
            trajectories = generator_for(args.pde)(
                args.n_test, args.grid, viscosity, args.eval_steps,
                args.step_dt, args.solver_dt, 1000 + seed, device,
            )
            solver_seconds = time.perf_counter() - start
            predictions = {}
            for method in args.methods:
                start = time.perf_counter()
                prediction = rollout(method, model, trajectories[:, 0], args.eval_steps, viscosity,
                                     args, input_normalizer, increment_normalizer, dims)
                if device == "cuda":
                    torch.cuda.synchronize()
                wall_clock = time.perf_counter() - start
                metrics, nonfinite = metrics_for(prediction, trajectories, dims)
                predictions[method] = prediction
                rows.append({"seed": seed, "method": method, "viscosity_ratio": ratio,
                             "viscosity": viscosity, "nn_evaluations_per_step": 1,
                             "wall_clock_seconds": wall_clock,
                             "reference_solver_seconds": solver_seconds,
                             "nonfinite": int(nonfinite), **metrics})
            if ratio == 1.0:
                worst = max((predictions[m] - predictions["increment"]).abs().max().item()
                            for m in args.methods)
                print({"seed": seed, "ratio": ratio, "max_difference_from_frozen": worst}, flush=True)
            print({"seed": seed, "ratio": ratio, **{
                m: round(r["rollout_rel_l2_average"], 5) for m, r in
                ((r["method"], r) for r in rows if r["seed"] == seed and r["viscosity_ratio"] == ratio)}},
                flush=True)
            write_csv(out_dir / "seed_results.csv", rows)

    aggregates = aggregate(rows)
    paired = paired_vs_delta(rows) if "delta_physics" in args.methods else []
    check = reproduction_check(rows, args.checkpoint_dir)
    write_csv(out_dir / "aggregate_metrics.csv", aggregates)
    if paired:
        write_csv(out_dir / "paired_vs_delta_physics.csv", paired)
    write_csv(out_dir / "reproduction_check.csv", check)
    largest = max(args.viscosity_ratios)
    (out_dir / "summary.md").write_text(summary_markdown(aggregates, paired, args, largest), encoding="utf-8")
    worst = max((c.get("abs_difference", 0.0) for c in check), default=0.0)
    print({"reproduction_max_abs_difference": worst}, flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
