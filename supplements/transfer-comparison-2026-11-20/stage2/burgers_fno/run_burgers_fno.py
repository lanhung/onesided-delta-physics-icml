"""Independent Burgers FNO reconstruction; NOT the manuscript's checkpoint.

Run `python run_burgers_fno.py --out results --pilot` for one source-only epoch.
Run `python run_burgers_fno.py --out results` for the locked three-training-seed
study. All source datasets are shared across training seeds; target trajectories
are generated only after training and never used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from spectral_transfer_torch import METHODS, TorchTransferStepper

EVIDENCE = "independently_reconstructed_neural_not_manuscript_reproduction"
CONFIG = {
    "evidence_type": EVIDENCE,
    "pde": "1d_periodic_unforced_burgers",
    "grid": 64, "domain_length": 1.0, "nu0": 0.01, "h": 0.02,
    "training_seeds": [0, 1, 2],
    "train_ic_seeds": list(range(1000, 1024)),
    "validation_ic_seeds": list(range(2000, 2006)),
    "test_ic_seeds": list(range(3000, 3008)),
    "source_data_shared_across_training_seeds": True,
    "initial_condition": "4 random sine/cos modes with coefficient divided by mode, mean0, RMS0.5",
    "source_transitions_per_trajectory": 20,
    "source_solver_max_dt": 0.0005,
    "test_steps": 50, "target_solver_max_dt": 0.000125,
    "target_refinement_coarse_dt": 0.00025,
    "viscosity_ratios": [1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0],
    "architecture": {"name": "independent_standard_FNO1d", "width": 32,
                     "modes": 16, "fourier_layers": 4, "head_width": 128,
                     "activation": "GELU after first3 spectral+pointwise layers and head",
                     "inputs": "normalized state plus periodic-domain coordinate x",
                     "output": "normalized physical increment"},
    "epochs": 75, "batch_size": 32, "optimizer": "Adam",
    "learning_rate": 0.001, "weight_decay": 1e-6,
    "schedule": "StepLR(step_size=25,gamma=0.5); after epoch optimizer updates",
    "checkpoint_selection": "lowest source-validation mean one-step physical RelL2 across75epochs",
    "target_data_tuning": False,
    "normalization": "global scalar state/increment means and population standard deviations from training pairs only",
    "dtype_training_and_network_inference": "torch.float32",
    "dtype_reference_and_metric_accumulation": "float64",
    "torch_threads": 2,
    "method_names": list(METHODS),
    "failure_policy": "first nonfinite value or maxabs>1e4, per trajectory; aggregate missing if any IC fails",
    "post_prediction_projection": "none, all methods",
    "fixed_grid_scope": "time-refined semidiscrete64-point truth; spatial convergence not established",
    "inference_timing": "not benchmarked in accuracy run; no neural speedup claim",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BurgersReference:
    """Copied vetted stage1 float64 conservative2/3-dealiased Fourier RK4.

    Full states are not projected: only nonlinear input/output use the mask.
    Odd derivative Nyquist is zero; Laplacian uses the full k² spectrum.
    """
    def __init__(self, n=64, length=1.0):
        k = 2 * np.pi * np.fft.fftfreq(n, length / n)
        self.k2 = k * k
        derivative_k = k.copy()
        if n % 2 == 0:
            derivative_k[n // 2] = 0.0
        self.ik = 1j * derivative_k
        self.mask = np.abs(np.fft.fftfreq(n) * n) < n / 3

    def rhs(self, u, nu):
        uh = np.fft.fft(u, axis=-1)
        filtered = np.fft.ifft(uh * self.mask, axis=-1).real
        nonlinear = -.5 * self.ik * np.fft.fft(filtered * filtered, axis=-1) * self.mask
        return np.fft.ifft(nonlinear - nu * self.k2 * uh, axis=-1).real

    def advance(self, u, nu, h, max_dt):
        count = int(np.ceil(h / max_dt))
        dt = h / count
        if dt * nu * self.k2.max() > 2.5:
            raise ValueError("Explicit RK4 exceeds the declared diffusion stability margin")
        u = u.copy()
        for _ in range(count):
            a = self.rhs(u, nu)
            b = self.rhs(u + .5 * dt * a, nu)
            c = self.rhs(u + .5 * dt * b, nu)
            d = self.rhs(u + dt * c, nu)
            u += dt * (a + 2 * b + 2 * c + d) / 6
        return u


def initial_conditions(seeds, n=64):
    x = np.arange(n) / n
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        u = sum((rng.normal() * np.sin(2 * np.pi * j * x)
                 + rng.normal() * np.cos(2 * np.pi * j * x)) / j
                for j in range(1, 5))
        u *= .5 / np.sqrt(np.mean(u * u))
        out.append(u)
    return np.stack(out)


def trajectory(initial, nu, steps, max_dt):
    solver = BurgersReference(CONFIG["grid"], CONFIG["domain_length"])
    u = initial.copy()
    states = [u.copy()]
    for _ in range(steps):
        u = solver.advance(u, nu, CONFIG["h"], max_dt)
        if not np.isfinite(u).all():
            raise RuntimeError("Reference trajectory is nonfinite")
        states.append(u.copy())
    return np.stack(states)


def source_dataset(out):
    train = trajectory(initial_conditions(CONFIG["train_ic_seeds"]), CONFIG["nu0"],
                       CONFIG["source_transitions_per_trajectory"], CONFIG["source_solver_max_dt"])
    val = trajectory(initial_conditions(CONFIG["validation_ic_seeds"]), CONFIG["nu0"],
                     CONFIG["source_transitions_per_trajectory"], CONFIG["source_solver_max_dt"])
    x = train[:-1].reshape(-1, 1, CONFIG["grid"])
    d = (train[1:] - train[:-1]).reshape(x.shape)
    stats = {"state_mean": float(x.mean()), "state_std": float(x.std()),
             "increment_mean": float(d.mean()), "increment_std": float(d.std()),
             "provenance": "training pairs only; no validation/test statistics"}
    if min(stats["state_std"], stats["increment_std"]) <= 0:
        raise ValueError("Nonpositive source standard deviation")
    np.savez_compressed(out / "source_data.npz", train=train, validation=val)
    write_json(out / "normalization.json", stats)
    return train, val, stats


class SpectralConv1d(nn.Module):
    def __init__(self, width=32, modes=16):
        super().__init__()
        self.modes = modes
        scale = 1 / math.sqrt(2 * width)
        self.weight = nn.Parameter(scale * torch.complex(
            torch.randn(width, width, modes), torch.randn(width, width, modes)))

    def forward(self, u):
        uh = torch.fft.rfft(u, dim=-1)
        zh = torch.zeros(u.shape[0], u.shape[1], uh.shape[-1],
                         dtype=uh.dtype, device=u.device)
        zh[..., :self.modes] = torch.einsum("bik,iok->bok", uh[..., :self.modes], self.weight)
        return torch.fft.irfft(zh, n=u.shape[-1], dim=-1)


class FNO1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = nn.Linear(2, 32)
        self.spectral = nn.ModuleList([SpectralConv1d() for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv1d(32, 32, 1) for _ in range(4)])
        self.head = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, normalized_state):
        x = torch.arange(normalized_state.shape[-1], device=normalized_state.device,
                         dtype=normalized_state.dtype) / normalized_state.shape[-1]
        coordinates = x.expand(normalized_state.shape[0], 1, -1)
        u = self.lift(torch.cat((normalized_state, coordinates), dim=1).transpose(1, 2)).transpose(1, 2)
        for idx, (fourier, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            u = fourier(u) + pointwise(u)
            if idx != 3:
                u = torch.nn.functional.gelu(u)
        return self.head(u.transpose(1, 2)).transpose(1, 2)


def physical_predictor(model, stats):
    def predict(u):
        return u + stats["increment_mean"] + stats["increment_std"] * model(
            (u - stats["state_mean"]) / stats["state_std"])
    return predict


def seed_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return FNO1d()


@torch.inference_mode()
def validation(model, vx, vy, stats):
    model.eval()
    prediction = physical_predictor(model, stats)(vx).double()
    target = vy.double()
    relative = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1) / torch.linalg.vector_norm(target.flatten(1), dim=1)
    return float(relative.mean())


def train_seed(seed, train, val, stats, out, epochs=75, pilot=False):
    model = seed_model(seed)
    optim = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=25, gamma=.5)
    tx = torch.tensor(train[:-1].reshape(-1, 1, 64), dtype=torch.float32)
    ty = torch.tensor((train[1:] - train[:-1]).reshape(-1, 1, 64), dtype=torch.float32)
    ty = (ty - stats["increment_mean"]) / stats["increment_std"]
    vx = torch.tensor(val[:-1].reshape(-1, 1, 64), dtype=torch.float32)
    vy = torch.tensor(val[1:].reshape(-1, 1, 64), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 50000)
    rows, best_error, best_epoch = [], float("inf"), None
    started = time.perf_counter()
    seed_dir = out / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)
    best_path = seed_dir / "best_source_validation.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tx), generator=generator)
        accumulated = 0.0
        lr = float(optim.param_groups[0]["lr"])
        for start in range(0, len(tx), CONFIG["batch_size"]):
            ids = order[start:start + CONFIG["batch_size"]]
            pred = model((tx[ids] - stats["state_mean"]) / stats["state_std"])
            loss = torch.mean((pred - ty[ids]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite source training loss seed{seed} epoch{epoch}")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            accumulated += float(loss.detach()) * len(ids)
        value = validation(model, vx, vy, stats)
        rows.append({"training_seed": seed, "epoch": epoch, "learning_rate": lr,
                     "train_normalized_increment_mse": accumulated / len(tx),
                     "source_validation_one_step_rel_l2": value,
                     "elapsed_seconds_in_this_run": time.perf_counter() - started})
        if value < best_error:
            best_error, best_epoch = value, epoch
            torch.save({"state_dict": model.state_dict(), "training_seed": seed,
                        "epoch": epoch, "source_validation_rel_l2": value,
                        "evidence_type": EVIDENCE, "config": CONFIG,
                        "normalization": stats}, best_path)
        if epoch in (15, 30, 50, 75):
            torch.save({"state_dict": model.state_dict(), "training_seed": seed,
                        "epoch": epoch, "source_validation_rel_l2": value,
                        "evidence_type": EVIDENCE}, seed_dir / f"epoch_{epoch}.pt")
        schedule.step()
        if epoch == 1 or epoch % 5 == 0:
            write_csv(seed_dir / "training_curve.csv", rows)
            print(json.dumps({"training_seed": seed, "epoch": epoch,
                              "source_val_rel": value, "selected_epoch_so_far": best_epoch,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    write_csv(seed_dir / "training_curve.csv", rows)
    chosen = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(chosen["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    record = {"training_seed": seed, "completed_epochs": epochs, "selected_epoch": best_epoch,
              "selected_source_validation_rel_l2": best_error,
              "checkpoint_path": str(best_path.relative_to(out)),
              "checkpoint_sha256": digest(best_path),
              "normalization_sha256": digest(out / "normalization.json"),
              "source_data_sha256": digest(out / "source_data.npz"),
              "parameter_tensor_elements": sum(p.numel() for p in model.parameters()),
              "real_parameter_degrees_of_freedom": sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()),
              "training_elapsed_seconds": time.perf_counter() - started,
              "pilot": pilot, "evidence_type": EVIDENCE}
    write_json(seed_dir / "training_metadata.json", record)
    return model, record


def generate_targets(out):
    initial = initial_conditions(CONFIG["test_ic_seeds"])
    truths, refinement = {}, []
    for ratio in CONFIG["viscosity_ratios"]:
        nu = CONFIG["nu0"] * ratio
        fine = trajectory(initial, nu, CONFIG["test_steps"], CONFIG["target_solver_max_dt"])
        coarse = trajectory(initial, nu, CONFIG["test_steps"], CONFIG["target_refinement_coarse_dt"])
        den = np.linalg.norm(fine[1:], axis=-1)
        err = np.linalg.norm(coarse[1:] - fine[1:], axis=-1) / den
        if err.max() > 1e-7:
            raise AssertionError("Target temporal refinement exceeds predeclared1e-7 threshold")
        tag = f"r{ratio:g}".replace(".", "p")
        np.savez_compressed(out / f"truth_{tag}.npz", truth=fine[:, :, None, :],
                            ic_seeds=np.array(CONFIG["test_ic_seeds"]))
        truths[ratio] = fine
        refinement.append({"ratio": ratio, "nu": nu,
                           "max_rel_l2_coarse_vs_fine_over_all_50_steps_and_8_ics": float(err.max()),
                           "coarse_dt": CONFIG["target_refinement_coarse_dt"],
                           "fine_dt": CONFIG["target_solver_max_dt"],
                           "truth_file": f"truth_{tag}.npz", "truth_sha256": digest(out / f"truth_{tag}.npz")})
        print(json.dumps({"target_ratio_generated": ratio, "max_refinement_relative": float(err.max())}), flush=True)
    source_coarse = trajectory(initial, CONFIG["nu0"], 50, CONFIG["source_solver_max_dt"])
    source_fine = truths[1.0]
    source_error = np.linalg.norm(source_coarse[1:] - source_fine[1:], axis=-1) / np.linalg.norm(source_fine[1:], axis=-1)
    if source_error.max() > 1e-7:
        raise AssertionError("Source flow temporal refinement exceeded1e-7")
    write_json(out / "temporal_refinement.json", {"target_cases": refinement,
               "source_dt0p0005_vs0p000125_max_rel": float(source_error.max()),
               "scope": "fixed64-point semidiscrete time convergence; no spatial convergence claim"})
    return truths


def metrics(pred, truth):
    p, t = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    rms_error = np.sqrt(np.mean((p - t) ** 2))
    norm = np.sqrt(np.mean(t ** 2))
    pp = np.abs(np.fft.fft(p, norm="ortho")) ** 2
    tp = np.abs(np.fft.fft(t, norm="ortho")) ** 2
    power_norm = np.linalg.norm(tp)
    return {"relative_l2": float(rms_error / norm) if norm > 0 else None,
            "absolute_l2_unit_length": float(rms_error),
            "psre": float(np.linalg.norm(pp - tp) / power_norm) if power_norm > 0 else None,
            "reference_l2_unit_length": float(norm),
            "mean_drift_from_truth": float(p.mean() - t.mean()),
            "state_l2_unit_length": float(np.sqrt(np.mean(p * p)))}


@torch.inference_mode()
def evaluate_seed(seed, model, stats, truths, out, training_record):
    seed_dir = out / f"seed_{seed}"
    all_rows, summaries = [], []
    actual_forwards = [0]
    def hook(_module, _inputs, _output):
        actual_forwards[0] += 1
    handle = model.register_forward_hook(hook)
    predictor = physical_predictor(model, stats)
    identities = []
    initial = torch.tensor(truths[1.0][0, :, None, :], dtype=torch.float32)
    frozen_reference = predictor(initial.clone())
    for method in METHODS:
        stepper = TorchTransferStepper(predictor, (1.,), CONFIG["h"], CONFIG["nu0"], CONFIG["nu0"], method)
        candidate = stepper(initial.clone())
        identities.append({"method": method, "equal_at_source": bool(torch.equal(candidate, frozen_reference))})
    if not all(r["equal_at_source"] for r in identities):
        raise AssertionError("Source-viscosity wrappers differ")
    write_json(seed_dir / "source_identity_check.json", identities)
    for ratio in CONFIG["viscosity_ratios"]:
        truth = truths[ratio]
        for method in METHODS:
            stepper = TorchTransferStepper(predictor, (1.,), CONFIG["h"], CONFIG["nu0"], CONFIG["nu0"] * ratio, method)
            u = torch.tensor(truth[0, :, None, :], dtype=torch.float32)
            alive = np.ones(len(u), dtype=bool)
            failure_steps = [None] * len(u)
            errors = np.full((CONFIG["test_steps"], len(u)), np.nan)
            states = [u.numpy().copy()]
            calls_before = actual_forwards[0]
            for step in range(1, CONFIG["test_steps"] + 1):
                u = stepper(u)
                p = u.numpy()[:, 0, :].astype(np.float64)
                bad = alive & (~np.isfinite(p).all(axis=-1) | (np.max(np.abs(p), axis=-1) > 1e4))
                for idx in np.flatnonzero(bad):
                    failure_steps[idx] = step
                alive[bad] = False
                saved = u.numpy().copy()
                saved[~alive] = np.nan
                states.append(saved)
                for idx, ic_seed in enumerate(CONFIG["test_ic_seeds"]):
                    row = {"evidence_type": EVIDENCE, "training_seed": seed, "test_ic_seed": ic_seed,
                           "ratio": ratio, "nu": CONFIG["nu0"] * ratio, "method": method,
                           "step": step, "time": CONFIG["h"] * step,
                           "status": "ok" if alive[idx] else "failed",
                           "first_failure_step": failure_steps[idx]}
                    if alive[idx]:
                        row.update(metrics(p[idx], truth[step, idx]))
                        errors[step - 1, idx] = row["relative_l2"]
                    else:
                        row.update({k: None for k in ("relative_l2", "absolute_l2_unit_length", "psre",
                                    "reference_l2_unit_length", "mean_drift_from_truth", "state_l2_unit_length")})
                    all_rows.append(row)
                # Model is batch-independent in eval mode; failed examples are
                # zeroed only for arithmetic and remain failed/missing forever.
                if not alive.all():
                    u[torch.as_tensor(~alive)] = 0.
            model_calls = actual_forwards[0] - calls_before
            if model_calls != CONFIG["test_steps"] or stepper.calls != model_calls:
                raise AssertionError("Instrumented model forward count mismatch")
            valid = alive.all()
            summaries.append({"evidence_type": EVIDENCE, "training_seed": seed, "ratio": ratio,
                              "method": method, "avg50_rel_l2": float(errors.mean()) if valid else None,
                              "step50_rel_l2": float(errors[-1].mean()) if valid else None,
                              "n_failed_ics": int((~alive).sum()), "n_test_ics": len(u),
                              "first_failure_step": min((s for s in failure_steps if s is not None), default=None),
                              "actual_batched_network_forwards": model_calls,
                              "callback_calls": stepper.calls,
                              "trajectory_steps_scheduled": len(u) * CONFIG["test_steps"],
                              "selected_epoch": training_record["selected_epoch"],
                              "checkpoint_sha256": training_record["checkpoint_sha256"]})
            tag = f"r{ratio:g}_{method}".replace(".", "p")
            np.savez_compressed(seed_dir / f"rollout_{tag}.npz", prediction=np.stack(states),
                                ic_seeds=np.array(CONFIG["test_ic_seeds"]),
                                first_failure_step=np.array([-1 if s is None else s for s in failure_steps]))
        write_csv(seed_dir / "per_step_metrics.csv", all_rows)
        write_json(seed_dir / "summary.json", summaries)
        print(json.dumps({"evaluated_training_seed": seed, "ratio": ratio,
                          "methods": [{"method": r["method"], "avg50": r["avg50_rel_l2"], "failed": r["n_failed_ics"]}
                                      for r in summaries if r["ratio"] == ratio]}), flush=True)
    handle.remove()
    return all_rows, summaries


def aggregate(summaries):
    aggregate_rows = []
    for ratio in CONFIG["viscosity_ratios"]:
        for method in METHODS:
            chosen = [r for r in summaries if r["ratio"] == ratio and r["method"] == method]
            complete = len(chosen) == 3 and all(r["avg50_rel_l2"] is not None for r in chosen)
            gains, paired = [], []
            if complete:
                for row in chosen:
                    base = next(r for r in summaries if r["training_seed"] == row["training_seed"] and r["ratio"] == ratio and r["method"] == "frozen")
                    delta = next(r for r in summaries if r["training_seed"] == row["training_seed"] and r["ratio"] == ratio and r["method"] == "delta")
                    if base["avg50_rel_l2"] is not None and base["avg50_rel_l2"] > 0:
                        gains.append(100 * (base["avg50_rel_l2"] - row["avg50_rel_l2"]) / base["avg50_rel_l2"])
                    if delta["avg50_rel_l2"] is not None:
                        paired.append(row["avg50_rel_l2"] - delta["avg50_rel_l2"])
            aggregate_rows.append({"evidence_type": EVIDENCE, "ratio": ratio, "method": method,
                                   "avg50_mean_over_training_seeds": float(np.mean([r["avg50_rel_l2"] for r in chosen])) if complete else None,
                                   "avg50_sample_sd_over_training_seeds": float(np.std([r["avg50_rel_l2"] for r in chosen], ddof=1)) if complete else None,
                                   "step50_mean_over_training_seeds": float(np.mean([r["step50_rel_l2"] for r in chosen])) if complete else None,
                                   "step50_sample_sd_over_training_seeds": float(np.std([r["step50_rel_l2"] for r in chosen], ddof=1)) if complete else None,
                                   "mean_gain_percent_within_seed_vs_frozen": float(np.mean(gains)) if len(gains) == 3 else None,
                                   "mean_paired_avg50_difference_minus_delta": float(np.mean(paired)) if len(paired) == 3 else None,
                                   "failed_ics_total_over_seed_ic_pairs": sum(r["n_failed_ics"] for r in chosen),
                                   "training_seeds_completed": len(chosen), "test_ics_per_training_seed": 8,
                                   "aggregate_status": "complete" if complete else "missing_due_to_failure_or_incomplete"})
    return aggregate_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(out / "locked_config.json", CONFIG)
    write_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "numpy": np.__version__, "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
               "device_used": "cpu", "threads": torch.get_num_threads(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": digest(__file__)})
    begun = time.perf_counter()
    train, val, stats = source_dataset(out)
    models, records = {}, []
    for seed in ([0] if args.pilot else CONFIG["training_seeds"]):
        model, record = train_seed(seed, train, val, stats, out, epochs=1 if args.pilot else CONFIG["epochs"], pilot=args.pilot)
        models[seed] = model
        records.append(record)
        write_json(out / "training_summary.json", records)
    if args.pilot:
        print(json.dumps({"pilot_complete": True, "total_seconds": time.perf_counter() - begun,
                          "one_epoch_training_seconds": records[0]["training_elapsed_seconds"],
                          "rough_three_seed_75epoch_seconds": records[0]["training_elapsed_seconds"] * 225,
                          "no_target_evaluation": True}), flush=True)
        return
    truths = generate_targets(out)
    all_rows, all_summaries = [], []
    for seed in CONFIG["training_seeds"]:
        raw, summary = evaluate_seed(seed, models[seed], stats, truths, out, records[seed])
        all_rows.extend(raw)
        all_summaries.extend(summary)
    write_csv(out / "per_step_metrics.csv", all_rows)
    write_csv(out / "per_seed_summary.csv", all_summaries)
    aggregates = aggregate(all_summaries)
    write_csv(out / "aggregate_summary.csv", aggregates)
    write_json(out / "aggregate_summary.json", aggregates)
    write_json(out / "execution_summary.json", {"evidence_type": EVIDENCE,
               "completed": True, "training_seeds": CONFIG["training_seeds"],
               "test_ic_seeds": CONFIG["test_ic_seeds"], "number_of_method_seed_ratio_runs": len(all_summaries),
               "number_of_per_ic_step_metric_rows": len(all_rows),
               "number_of_50step_trajectories": len(all_summaries) * 8,
               "elapsed_seconds_exploratory_execution_not_benchmark": time.perf_counter() - begun,
               "original_manuscript_checkpoint_used": False, "cuda_inference_executed": False,
               "source_data_hash": digest(out / "source_data.npz"),
               "normalization_hash": digest(out / "normalization.json"),
               "config_hash": digest(out / "locked_config.json"), "script_hash": digest(__file__),
               "statistical_scope": "3 training seeds on one fixed source dataset, 8 common held-out ICs; descriptive mean/sampleSD"})
    print(json.dumps({"completed": True, "evidence_type": EVIDENCE, "result": str(out)}), flush=True)


if __name__ == "__main__":
    main()
