"""Paired accuracy evaluation for periodic scalar-coefficient viscosity transfer.

This is an accuracy runner, not a timing benchmark. Inputs and outputs use physical
units. A dataset must include the initial state: [T+1, B, C, *spatial].
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SHARED = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED))
from spectral_transfer_numpy import METHODS, PeriodicLaplacian, TransferStepper


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON numeric constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def checked_artifact(spec, base, name):
    if not isinstance(spec, dict) or not spec.get("path") or not spec.get("sha256"):
        raise ValueError(f"{name}: require an existing path and expected sha256.")
    path = (base / spec["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"{name}: file unavailable: {path}")
    actual = sha256(path)
    if actual != str(spec["sha256"]).lower():
        raise ValueError(f"{name}: SHA256 mismatch; refusing unpaired evaluation.")
    return path, {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def positive_float(value, label, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number.")
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{label}: expected {'nonnegative' if allow_zero else 'positive'} finite value.")
    return float(value)


def positive_int(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}.")
    return value


def identical(actual, expected, label):
    if isinstance(expected, float):
        ok = isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual == expected
    else:
        ok = actual == expected
    if not ok:
        raise ValueError(f"{label}: metadata {actual!r} differs from requested {expected!r}.")


def load_adapter(spec, base):
    if "path" in spec:
        path = (base / spec["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"Adapter unavailable: {path}")
        module_spec = importlib.util.spec_from_file_location("transfer_user_adapter", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        name = spec.get("module")
        if not isinstance(name, str) or not name:
            raise ValueError("adapter requires module or local path.")
        sys.path.insert(0, str(base))
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
    if not callable(getattr(module, "load_predictor", None)):
        raise ValueError("Adapter must export load_predictor(config).")
    return module, {"path": str(path), "sha256": sha256(path)}


class CheckedPredictor:
    """Validate physical callback contract; count forwards separately from samples."""
    def __init__(self, predict, backend, original_neural=False):
        self.predict, self.backend = predict, backend
        self.original_neural = original_neural
        self.calls = 0
        self.network_calls = 0 if original_neural else None
        if original_neural and not callable(getattr(predict, "get_forward_count", None)):
            raise ValueError("Original neural adapter must expose get_forward_count(), incremented by a model forward hook.")

    def __call__(self, u):
        before = u.copy() if self.backend == "numpy" else u.clone()
        count_before = self.predict.get_forward_count() if self.original_neural else None
        self.calls += 1
        y = self.predict(u)
        if self.backend == "numpy":
            if not isinstance(y, np.ndarray) or y.shape != u.shape or y.dtype != u.dtype:
                raise ValueError("Predictor must preserve NumPy state shape and dtype.")
            unchanged = np.array_equal(before, u, equal_nan=True)
        else:
            import torch
            if not isinstance(y, torch.Tensor) or y.shape != u.shape or y.dtype != u.dtype or y.device != u.device:
                raise ValueError("Predictor must preserve Torch state shape, dtype and device.")
            unchanged = bool(torch.equal(before, u))
        if not unchanged:
            raise ValueError("Predictor mutated its input; corrections require the original physical state.")
        if self.original_neural:
            count_after = self.predict.get_forward_count()
            if any(isinstance(x, bool) or not isinstance(x, int) for x in (count_before, count_after)):
                raise ValueError("get_forward_count must return a Python integer.")
            if count_after - count_before != 1:
                raise ValueError("Each physical predictor callback must perform exactly one instrumented network forward.")
            self.network_calls += count_after - count_before
        return y


def prepare(config_path):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = load_json(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema_version; require 1.")
    kind = config.get("evidence_type")
    if kind not in ("numerical_control", "original_neural", "independently_reconstructed_neural"):
        raise ValueError("evidence_type must be numerical_control, original_neural or independently_reconstructed_neural.")
    if config.get("target_trajectory_tuning_allowed") is not False:
        raise ValueError("Declare target_trajectory_tuning_allowed=false for this zero-shot protocol.")
    backend = config.get("backend")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be numpy or torch.")
    if config.get("dtype") not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64.")
    reference_dtype = config.get("reference_dtype", config["dtype"])
    if reference_dtype not in ("float32", "float64"):
        raise ValueError("reference_dtype must be float32 or float64.")
    if backend == "numpy" and config.get("device", "cpu") != "cpu":
        raise ValueError("NumPy backend is CPU only.")
    h = positive_float(config.get("h"), "h")
    nu0 = positive_float(config.get("nu0"), "nu0", allow_zero=True)
    nu = positive_float(config.get("nu"), "nu", allow_zero=True)
    steps = positive_int(config.get("horizon_steps"), "horizon_steps")
    lengths = config.get("lengths")
    shape = config.get("spatial_shape")
    if not isinstance(lengths, list) or not lengths or not isinstance(shape, list) or len(shape) != len(lengths):
        raise ValueError("Provide matching nonempty lengths and spatial_shape lists.")
    lengths = [positive_float(x, "lengths entry") for x in lengths]
    shape = [positive_int(x, "spatial_shape entry", minimum=2) for x in shape]
    channels = positive_int(config.get("channels"), "channels")
    limit = positive_float(config.get("failure_max_abs", 1e4), "failure_max_abs")
    methods = config.get("methods")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a nonempty list with no duplicates.")
    if "original_input_recast_aitken" in methods:
        raise ValueError("Original input-Recast/Aitken is unavailable here: integrate its original solver and inner forward counter separately; implicit is a different method.")
    if any(method not in METHODS for method in methods):
        raise ValueError(f"Supported methods: {METHODS}.")
    if nu < nu0 and any(method not in ("frozen", "delta") for method in methods):
        raise ValueError("Lower-viscosity experiments support only frozen/delta under a declared additive policy.")
    if not isinstance(config.get("delta_one_sided", True), bool):
        raise ValueError("delta_one_sided must be boolean.")
    if kind != "numerical_control":
        positive_int(config.get("training_seed"), "training_seed", minimum=0)
    elif config.get("training_seed") is not None:
        raise ValueError("Numerical controls have no training seed; use ic_seeds instead.")

    wrapper_path = SHARED / ("spectral_transfer_numpy.py" if backend == "numpy" else "spectral_transfer_torch.py")
    provenance = {"config": {"path": str(config_path), "sha256": sha256(config_path)},
                  "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
                  "transfer_wrapper": {"path": str(wrapper_path), "sha256": sha256(wrapper_path)}}
    # Resolve every required artifact before importing a model or creating outputs.
    if kind != "numerical_control":
        for name in ("checkpoint", "normalizer"):
            _, provenance[name] = checked_artifact(config.get("artifacts", {}).get(name), base, name)
    dataset_path, provenance["dataset"] = checked_artifact(config.get("dataset"), base, "dataset")
    metadata_path, provenance["dataset_metadata"] = checked_artifact(config.get("dataset_metadata"), base, "dataset_metadata")
    metadata = load_json(metadata_path)
    expected = {"h": h, "nu": nu, "lengths": lengths, "spatial_shape": shape,
                "channels": channels, "units": "physical", "includes_initial_state": True,
                "dtype": reference_dtype}
    for key, value in expected.items():
        identical(metadata.get(key), value, "dataset " + key)
    if not metadata.get("generator") or not metadata.get("reference_accuracy_status") or not metadata.get("projection_policy"):
        raise ValueError("Dataset metadata must identify generator, reference_accuracy_status and projection_policy.")
    with np.load(dataset_path, allow_pickle=False) as stored:
        if any(key not in stored for key in ("trajectory", "trajectory_ids", "times")):
            raise ValueError("NPZ must contain trajectory, trajectory_ids and times arrays.")
        truth = np.array(stored["trajectory"], copy=True)
        ids = np.array(stored["trajectory_ids"], copy=True)
        times = np.array(stored["times"], copy=True)
    if truth.dtype != np.dtype(reference_dtype):
        raise ValueError("Ground truth dtype differs from config; no silent conversion.")
    if truth.ndim != len(shape) + 3 or tuple(truth.shape[2:]) != (channels, *shape) or truth.shape[0] < steps + 1 or truth.shape[1] < 1:
        raise ValueError("Expected trajectory shape [at least T+1, B>=1, C, *spatial].")
    if not np.isfinite(truth[:steps + 1]).all():
        raise ValueError("Ground truth has nonfinite values; dataset cannot define a valid comparison.")
    if times.dtype.kind != "f" or times.shape != (truth.shape[0],) or not np.isfinite(times).all():
        raise ValueError("times must be one finite floating-point physical time per saved state.")
    if not np.allclose(times[:steps + 1], h * np.arange(steps + 1), rtol=1e-12, atol=1e-14):
        raise ValueError("Ground-truth timestamps must start at zero with the exact requested physical step h.")
    if np.max(np.abs(truth[:steps + 1])) > limit:
        raise ValueError("Failure threshold excludes the reference trajectory; choose a documented physical threshold.")
    if ids.dtype.kind not in "US" or ids.shape != (truth.shape[1],) or len(set(ids.tolist())) != len(ids):
        raise ValueError("trajectory_ids must be one unique non-object string per initial condition.")
    if any(not str(x) for x in ids):
        raise ValueError("trajectory_ids cannot be empty.")
    if config.get("ic_seeds") is not None and len(config["ic_seeds"]) != len(ids):
        raise ValueError("ic_seeds length must match dataset batch.")

    adapter, provenance["adapter"] = load_adapter(config.get("adapter", {}), base)
    # The model loader is intentionally not given target states; paths and protocol
    # are visible, so absence of tuning remains an auditable user responsibility.
    loader_config = dict(config)
    loader_config["config_directory"] = str(base)
    loaded = adapter.load_predictor(loader_config)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("load_predictor(config) must return (physical_predict, metadata).")
    predict, model_metadata = loaded
    if not callable(predict) or not isinstance(model_metadata, dict):
        raise ValueError("Adapter returned invalid predictor or metadata.")
    model_expected = {"h": h, "nu0": nu0, "lengths": lengths,
                      "evaluation_spatial_shape": shape, "channels": channels,
                      "dtype": config["dtype"], "backend": backend, "units": "physical",
                      "output_kind": "full_next_state", "stateless": True,
                      "sample_independent": True, "evidence_type": kind}
    for key, value in model_expected.items():
        identical(model_metadata.get(key), value, "predictor " + key)
    if not model_metadata.get("projection_policy"):
        raise ValueError("Predictor metadata must declare its projection/filtering policy.")
    if kind != "numerical_control":
        for key in ("checkpoint", "normalizer"):
            identical(model_metadata.get(key + "_sha256"), provenance[key]["sha256"], "predictor " + key)
        identical(model_metadata.get("eval_mode"), True, "predictor eval_mode")
        identical(model_metadata.get("training_seed"), config["training_seed"], "predictor training_seed")
    checked = CheckedPredictor(predict, backend, kind != "numerical_control")
    if backend == "torch":
        try:
            import torch
        except ImportError as exc:
            raise ValueError("PyTorch runtime is unavailable; Torch experiments were not run.") from exc
        device = torch.device(config.get("device", "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable.")
    config.update(h=h, nu0=nu0, nu=nu, lengths=lengths, spatial_shape=shape, reference_dtype=reference_dtype,
                  failure_max_abs=limit, horizon_steps=steps, channels=channels)
    return config, truth[:steps + 1], [str(x) for x in ids], checked, provenance, metadata, model_metadata


def stable_norm(values):
    """Euclidean norm with scaling to avoid squaring a large finite value first."""
    flat = np.ravel(values)
    scale = float(np.max(np.abs(flat)))
    return 0. if scale == 0. else float(scale * np.sqrt(np.sum(np.square(flat / scale))))


def ratio_or_undefined(numerator, denominator):
    if denominator == 0:
        return None, "zero_reference_norm"
    value = numerator / denominator
    return (float(value), "ok") if math.isfinite(value) else (None, "nonfinite_metric")


def metrics(pred, truth, lengths):
    """One sample [C,*space]; aggregate channels by summation, spatial quadrature."""
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    dims = len(lengths)
    axes = tuple(range(-dims, 0))
    quadrature = math.prod(lengths) / math.prod(truth.shape[-dims:])
    error = stable_norm(pred - truth) * math.sqrt(quadrature)
    target_norm = stable_norm(truth) * math.sqrt(quadrature)
    prediction_norm = stable_norm(pred) * math.sqrt(quadrature)
    relative, relative_status = ratio_or_undefined(error, target_norm)
    # Normalize both spectra by a common amplitude before squaring; the PSRE
    # quotient is unchanged and intermediate powers avoid avoidable overflow.
    hp = np.fft.fftn(pred, axes=axes, norm="forward")
    ht = np.fft.fftn(truth, axes=axes, norm="forward")
    scale = max(float(np.max(np.abs(hp))), float(np.max(np.abs(ht))))
    if scale == 0:
        psre, psre_status = None, "zero_reference_norm"
    else:
        pp, pt = np.abs(hp / scale) ** 2, np.abs(ht / scale) ** 2
        psre, psre_status = ratio_or_undefined(stable_norm(pp - pt), stable_norm(pt))
    return {"relative_l2": relative, "relative_l2_status": relative_status,
            "absolute_l2": error, "psre": psre, "psre_status": psre_status,
            "field_l2": prediction_norm, "reference_l2": target_norm}


def mean_if_complete(values):
    if not values or any(x is None or not math.isfinite(x) for x in values):
        return None
    return float(np.mean(values))


def evaluate_prepared(prepared):
    config, truth, ids, predictor, provenance, data_metadata, model_metadata = prepared
    steps, batch = config["horizon_steps"], len(ids)
    numpy_backend = config["backend"] == "numpy"
    grid = PeriodicLaplacian(tuple(config["spatial_shape"]), tuple(config["lengths"]))
    rows, summaries, raw_states = [], [], {}
    for method in config["methods"]:
        if numpy_backend:
            state = truth[0].astype(config["dtype"], copy=True)
            stepper = TransferStepper(predictor, grid, config["h"], config["nu0"], config["nu"], method,
                                      config.get("delta_one_sided", True))
        else:
            import torch
            from spectral_transfer_torch import TorchTransferStepper
            state = torch.from_numpy(truth[0].astype(config["dtype"], copy=True)).to(config.get("device", "cpu"))
            stepper = TorchTransferStepper(predictor, config["lengths"], config["h"], config["nu0"], config["nu"], method,
                                           config.get("delta_one_sided", True))
        first_failure = np.zeros(batch, dtype=int)
        failure_reason = [None] * batch
        output = np.empty(truth.shape, dtype=config["dtype"])
        output[0] = truth[0].astype(config["dtype"], copy=False)
        method_rows = []
        before_calls = predictor.calls
        before_network = predictor.network_calls
        for step in range(1, steps + 1):
            call_before = predictor.calls
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                state = stepper(state)
            predicted = np.array(state, copy=True) if numpy_backend else state.detach().cpu().numpy().copy()
            finite = np.all(np.isfinite(predicted), axis=tuple(range(1, predicted.ndim)))
            max_abs = np.max(np.abs(predicted), axis=tuple(range(1, predicted.ndim)))
            now_failed = (~finite | (max_abs > config["failure_max_abs"])) & (first_failure == 0)
            for sample in np.flatnonzero(now_failed):
                first_failure[sample] = step
                failure_reason[sample] = "nonfinite_prediction" if not finite[sample] else "max_abs_exceeds_threshold"
            failed = first_failure > 0
            predicted[failed] = np.nan
            output[step] = predicted
            for sample, trajectory_id in enumerate(ids):
                row = {"evidence_type": config["evidence_type"], "training_seed": config.get("training_seed"),
                       "ic_seed": (config.get("ic_seeds") or [None] * batch)[sample],
                       "trajectory_id": trajectory_id, "method": method, "step": step,
                       "physical_time": step * config["h"], "nu0": config["nu0"], "nu": config["nu"],
                       "failed": bool(failed[sample]), "first_failure_step": int(first_failure[sample]) or None,
                       "failure_reason": failure_reason[sample]}
                if failed[sample]:
                    row.update({key: None for key in ("relative_l2", "absolute_l2", "psre", "field_l2", "reference_l2")})
                    row.update(relative_l2_status="trajectory_failed", psre_status="trajectory_failed")
                else:
                    row.update(metrics(predicted[sample], truth[step, sample], config["lengths"]))
                method_rows.append(row)
            if predictor.calls - call_before != 1:
                raise RuntimeError("Unexpected callback count; one-call wrapper contract failed.")
            # Keep a fixed batch shape. Failed samples are neutral placeholders for
            # arithmetic only, and remain failed/NaN in every output and summary.
            # Adapters must be sample-independent and stateless for this to be valid.
            if failed.any():
                if numpy_backend:
                    state = state.copy()
                    state[failed] = 0.
                else:
                    state = state.clone()
                    state[torch.as_tensor(failed, device=state.device)] = 0.
        # Fill first failure on all rows, including rows preceding the failure.
        by_id = {identity: i for i, identity in enumerate(ids)}
        for row in method_rows:
            row["first_failure_step"] = int(first_failure[by_id[row["trajectory_id"]]]) or None
        rows.extend(method_rows)
        final_rows = [row for row in method_rows if row["step"] == steps]
        summary = {"method": method, "evidence_type": config["evidence_type"],
                   "training_seed": config.get("training_seed"), "trajectory_count": batch,
                   "steps": steps, "failed_trajectories": int(np.count_nonzero(first_failure)),
                   "first_failure_step_by_trajectory": {identity: int(first_failure[i]) or None for i, identity in enumerate(ids)},
                   "callback_calls": predictor.calls - before_calls,
                   "actual_network_forward_calls": None if before_network is None else predictor.network_calls - before_network,
                   "requested_sample_steps": steps * batch,
                   "arithmetic_sample_slots_including_failed_placeholders": steps * batch,
                   "successful_sample_steps": sum(not row["failed"] for row in method_rows),
                   "mean_callback_calls_per_batch_step": (predictor.calls - before_calls) / steps,
                   "avg_relative_l2": mean_if_complete([row["relative_l2"] for row in method_rows]),
                   "final_relative_l2": mean_if_complete([row["relative_l2"] for row in final_rows]),
                   "avg_absolute_l2": mean_if_complete([row["absolute_l2"] for row in method_rows]),
                   "final_absolute_l2": mean_if_complete([row["absolute_l2"] for row in final_rows]),
                   "avg_psre": mean_if_complete([row["psre"] for row in method_rows]),
                   "undefined_relative_l2_rows": sum(row["relative_l2"] is None for row in method_rows),
                   "aggregation": "all initial conditions and all requested steps; null if any constituent is missing/undefined; no survivor-only mean"}
        summaries.append(summary)
        raw_states[method] = output
    report = {"status": "completed", "evidence_type": config["evidence_type"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "original_neural_experiment": config["evidence_type"] == "original_neural",
              "independently_reconstructed_neural_experiment": config["evidence_type"] == "independently_reconstructed_neural",
              "timing_benchmark_performed": False,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "backend": config["backend"], "device": config.get("device", "cpu")},
              "config": config, "provenance": provenance,
              "dataset_metadata": data_metadata, "predictor_metadata": model_metadata,
              "metric_definitions": {
                  "metric_accumulation_dtype": "float64; the reference trajectory retains reference_dtype, only initial inference state is cast to dtype",
                  "absolute_l2": "sqrt(product(lengths)/spatial_grid_count * sum_over_channels_and_grid(error**2))",
                  "relative_l2": "absolute_l2 / reference_l2; zero denominator is always null with explicit status, including 0/0; absolute_l2 remains available",
                  "psre": "||abs(fftn(pred,norm=forward))**2-abs(fftn(truth,norm=forward))**2||_2 / ||abs(fftn(truth,norm=forward))**2||_2; channels and all full-FFT modes pooled",
                  "avg": "uniform mean over requested time steps 1..T and paired initial conditions; initial state excluded"},
              "limitations": ["Metadata declarations and hashes identify artifacts; they do not prove model provenance or absence of target-data tuning.",
                              "Accuracy runner includes CPU metric collection for Torch and must not be used as a GPU latency benchmark.",
                              "Original input-Recast/Aitken is intentionally unavailable until its exact implementation is supplied."],
              "summaries": summaries}
    if not numpy_backend:
        import torch
        report["runtime"].update(torch=str(torch.__version__), torch_cpu_threads=torch.get_num_threads(),
                                 cuda_available=torch.cuda.is_available())
    return report, rows, raw_states


def write_outputs(output_directory, report, rows, raw_states, save_rollouts):
    out = Path(output_directory)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is nonempty; choose a fresh run directory to preserve evidence.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if save_rollouts:
        np.savez_compressed(out / "rollouts.npz", **raw_states)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", help="Fresh output directory; required unless --validate-only")
    parser.add_argument("--validate-only", action="store_true", help="Validate assets/metadata and load predictor; perform no model inference")
    parser.add_argument("--save-rollouts", action="store_true")
    args = parser.parse_args(argv)
    try:
        prepared = prepare(args.config)
        if args.validate_only:
            print(json.dumps({"status": "assets_and_metadata_validated_no_inference", "evidence_type": prepared[0]["evidence_type"],
                              "trajectory_shape": list(prepared[1].shape), "methods": prepared[0]["methods"],
                              "warning": "This validates availability and declared metadata, not predictor correctness or experiment results."}))
            return 0
        if not args.out:
            raise ValueError("--out is required for evaluation.")
        if Path(args.out).exists() and any(Path(args.out).iterdir()):
            raise ValueError("Output directory is nonempty; choose a fresh run directory.")
        report, rows, raw = evaluate_prepared(prepared)
        write_outputs(args.out, report, rows, raw, args.save_rollouts)
        print(json.dumps({"status": report["status"], "evidence_type": report["evidence_type"],
                          "output_directory": str(Path(args.out).resolve()), "method_count": len(report["summaries"])}))
        return 0
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(json.dumps({"status": "not_executed_or_not_completed", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
