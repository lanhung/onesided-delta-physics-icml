"""PyTorch-native periodic viscosity wrappers; RUNTIME UNTESTED in this environment.

physical_predict(u) returns the full physical next state, with one frozen model
call and no input mutation. State shape is [B, C, *spatial], float32 or float64.
Coordinates and fixed training normalization belong inside physical_predict.
No checkpoint, training, Recast/Aitken reproduction, or measured GPU result.
"""
import math
import statistics
import time
from collections.abc import Callable, Sequence

import torch

METHODS = ("frozen", "delta", "explicit", "lie_pre", "lie_post", "strang", "implicit")


class TorchTransferStepper:
    """One physical_predict invocation per step, with all FFT work on u.device.

    Only frozen/delta permit nu < nu0. delta_one_sided=False enables the bounded
    two-sided additive diagnostic; inverse-heat splitting is deliberately absent.
    Call model.eval() before constructing this wrapper. Inference uses no grad.
    """

    def __init__(self, physical_predict: Callable, lengths: Sequence[float],
                 h: float, nu0: float, nu: float, method: str = "delta",
                 delta_one_sided: bool = True):
        self.lengths = tuple(float(length) for length in lengths)
        self.h, self.nu0, self.nu = float(h), float(nu0), float(nu)
        if not self.lengths or any(not math.isfinite(x) or x <= 0 for x in self.lengths):
            raise ValueError("Provide one positive finite physical length per spatial axis.")
        if any(not math.isfinite(x) for x in (self.h, self.nu0, self.nu)):
            raise ValueError("h and viscosities must be finite.")
        if self.h <= 0 or self.nu0 < 0 or self.nu < 0:
            raise ValueError("Require h > 0 and nonnegative viscosities.")
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
        if self.nu < self.nu0 and method not in ("frozen", "delta"):
            raise ValueError("Splitting/explicit/implicit comparison requires nu >= nu0.")
        if not callable(physical_predict):
            raise TypeError("physical_predict must be callable.")
        self.physical_predict, self.method = physical_predict, method
        self.delta_one_sided, self.calls = bool(delta_one_sided), 0
        self.axes = tuple(range(-len(self.lengths), 0))
        self._cache_key, self._cache = None, None

    def _validate(self, u):
        if not isinstance(u, torch.Tensor) or u.ndim != len(self.lengths) + 2:
            raise ValueError("Expected a Tensor with shape [B, C, *spatial].")
        if u.dtype not in (torch.float32, torch.float64):
            raise TypeError("Use real float32 or float64; mixed/half precision is unsupported.")
        if any(n < 1 for n in u.shape[:2]) or any(n < 2 for n in u.shape[2:]):
            raise ValueError("Batch/channels must be nonempty and spatial sizes >= 2.")

    def _predict(self, u):
        self.calls += 1
        y = self.physical_predict(u)
        if not isinstance(y, torch.Tensor) or y.shape != u.shape:
            raise ValueError("physical_predict must return a full state of unchanged shape.")
        if y.device != u.device or y.dtype != u.dtype:
            raise ValueError("physical_predict must preserve state device and dtype.")
        return y

    def _symbols(self, u):
        shape = tuple(u.shape[2:])
        key = (shape, u.device, u.dtype)
        if key == self._cache_key:
            return self._cache
        q = torch.zeros(shape, device=u.device, dtype=u.dtype)
        for axis, (n, length) in enumerate(zip(shape, self.lengths)):
            k = 2 * math.pi * torch.fft.fftfreq(n, d=length / n,
                                               device=u.device, dtype=u.dtype)
            view = [1] * len(shape)
            view[axis] = n
            q = q + k.square().reshape(view)
        z = self.h * (self.nu - self.nu0) * q
        if self.nu >= self.nu0:
            delta = torch.exp(-self.h * self.nu0 * q) * torch.expm1(-z)
        else:
            # Avoid catastrophic cancellation and 0 * inf at fine resolutions.
            delta = -torch.exp(-self.h * self.nu * q) * torch.expm1(z)
        symbols = {"delta": delta, "explicit": -z}
        if self.nu >= self.nu0:
            symbols.update(extra=torch.exp(-z), half=torch.exp(-z / 2),
                           implicit=torch.reciprocal(1 + z))
        self._cache_key, self._cache = key, symbols
        return symbols

    def _filter(self, u, symbol):
        spectrum = torch.fft.fftn(u, dim=self.axes, norm="backward")
        return torch.fft.ifftn(spectrum * symbol, dim=self.axes,
                              norm="backward").real.contiguous()

    @torch.inference_mode()
    def __call__(self, u):
        self._validate(u)
        # Return the predictor output directly: no FFT, addition, or near-zero test.
        if self.method == "frozen" or self.nu == self.nu0:
            return self._predict(u)
        if self.method == "delta" and self.delta_one_sided and self.nu < self.nu0:
            return self._predict(u)
        s = self._symbols(u)
        if self.method == "delta":
            return self._predict(u) + self._filter(u, s["delta"])
        if self.method == "explicit":
            return self._predict(u) + self._filter(u, s["explicit"])
        if self.method == "lie_pre":
            return self._predict(self._filter(u, s["extra"]))
        if self.method == "lie_post":
            return self._filter(self._predict(u), s["extra"])
        if self.method == "strang":
            return self._filter(self._predict(self._filter(u, s["half"])), s["half"])
        return self._filter(self._predict(u), s["implicit"])


@torch.inference_mode()
def rollout(stepper, u0, steps: int, keep_trajectory: bool = False):
    """Repeat a full physical step; output is final state or [T, B, C, *spatial]."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    u = u0.clone()  # Device-resident reset is included in benchmark timings.
    states = [] if keep_trajectory else None
    for _ in range(steps):
        u = stepper(u)
        if keep_trajectory:
            states.append(u)
    return torch.stack(states, dim=0) if keep_trajectory else u


@torch.inference_mode()
def benchmark_cuda_rollout(stepper, u0, steps: int, warmup: int = 3,
                           repeats: int = 10, keep_trajectory: bool = False):
    """Synchronized wall time for repeated complete CUDA-resident rollouts.

    No state transfer, NumPy conversion, loading, or training is timed. Includes
    Python launch overhead, the device-side initial clone, and optional trajectory
    storage. Models, u0, dtype, batch, horizon and storage policy must match across
    methods. The peak covers PyTorch allocator tracked memory on that device.
    This helper has NOT been executed or GPU-validated in the supplied environment.
    """
    if not isinstance(u0, torch.Tensor) or not u0.is_cuda:
        raise ValueError("A CUDA input is required; this is not a CPU benchmark.")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (warmup, repeats)):
        raise ValueError("warmup and repeats must be integers.")
    if warmup < 1 or repeats < 2:
        raise ValueError("Require warmup >= 1 complete rollout and repeats >= 2.")
    device = u0.device
    for _ in range(warmup):
        warm_output = rollout(stepper, u0, steps, keep_trajectory)
        del warm_output
    torch.cuda.synchronize(device)
    timings, counts, peaks, extra_peaks, baselines = [], [], [], [], []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        calls_before = stepper.calls
        start = time.perf_counter()
        output = rollout(stepper, u0, steps, keep_trajectory)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device)
        timings.append(elapsed)
        counts.append(stepper.calls - calls_before)
        baselines.append(baseline)
        peaks.append(peak)
        extra_peaks.append(max(0, peak - baseline))
        final = output[-1] if keep_trajectory else output
        finite = bool(torch.isfinite(final).all().item())  # Outside timed interval.
        del final, output
        if not finite:
            raise RuntimeError("A rollout produced a nonfinite final state; do not report it as successful.")
    q1, _, q3 = statistics.quantiles(timings, n=4, method="inclusive")
    median = statistics.median(timings)
    return {
        "timing_kind": "synchronized_wall_clock_cuda_resident",
        "method": stepper.method, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "dtype": str(u0.dtype), "shape": list(u0.shape), "steps": steps,
        "h": stepper.h, "nu0": stepper.nu0, "nu": stepper.nu,
        "lengths": list(stepper.lengths), "keep_trajectory": keep_trajectory,
        "warmup_rollouts": warmup, "timed_rollouts": repeats,
        "seconds": timings, "median_seconds": median, "iqr_seconds": q3 - q1,
        "q1_seconds": q1, "q3_seconds": q3,
        "median_ms_per_batch_step": 1000 * median / steps,
        "median_ms_per_trajectory_step": 1000 * median / (steps * u0.shape[0]),
        "predict_calls_per_rollout": counts,
        "mean_calls_per_batch_step": statistics.mean(counts) / steps,
        "peak_allocated_bytes": max(peaks), "peak_increment_bytes": max(extra_peaks),
        "baseline_allocated_bytes_by_repeat": baselines,
        "peak_allocated_bytes_by_repeat": peaks,
        "finite_final_states": True,
    }
