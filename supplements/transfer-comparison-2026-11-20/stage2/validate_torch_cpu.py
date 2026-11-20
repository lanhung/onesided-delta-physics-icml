"""Executed analytical CPU controls for the PyTorch viscosity wrappers.

This validates code on Fourier modes, not manuscript model performance.
Run from any directory: python validate_torch_cpu.py --out torch_cpu_checks.json
"""
import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))
from spectral_transfer_torch import METHODS, TorchTransferStepper, benchmark_cuda_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("torch_cpu_checks.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = []
    for shape, lengths in [((45,), (2.5,)), ((48, 32), (2.5, .75)),
                           ((31, 35), (1.3, 2.7))]:
        axes = tuple(range(-len(shape), 0))
        meshes = np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij")
        mode_a = tuple(range(1, len(shape) + 1))
        mode_b = tuple(range(3, len(shape) + 3))
        wave_a = np.cos(2 * np.pi * sum(m * x for m, x in zip(mode_a, meshes)))
        wave_b = np.sin(2 * np.pi * sum(m * x for m, x in zip(mode_b, meshes)))
        q_a = sum((2 * np.pi * m / l) ** 2 for m, l in zip(mode_a, lengths))
        q_b = sum((2 * np.pi * m / l) ** 2 for m, l in zip(mode_b, lengths))
        leading = (2, 2) + (1,) * len(shape)
        means = np.array([[.2, -.3], [1.1, .4]]).reshape(leading)
        amps_a = np.array([[.7, 1.2], [-.4, .5]]).reshape(leading)
        amps_b = np.array([[.3, -.8], [.9, .2]]).reshape(leading)
        state = means + amps_a * wave_a + amps_b * wave_b
        h, nu0 = .015, .013
        for dtype in (torch.float32, torch.float64):
            u = torch.tensor(state, dtype=dtype)
            q = torch.zeros(shape, dtype=dtype)
            for axis, (n, length) in enumerate(zip(shape, lengths)):
                k = 2 * math.pi * torch.fft.fftfreq(n, d=length/n, dtype=dtype)
                view = [1] * len(shape)
                view[axis] = n
                q += k.square().reshape(view)

            class HeatPredictor(torch.nn.Module):
                def forward(self, v):
                    return torch.fft.ifftn(torch.fft.fftn(v, dim=axes) *
                                           torch.exp(-h*nu0*q), dim=axes).real

            model = HeatPredictor().eval()
            actual = [0]
            hook = model.register_forward_hook(lambda *_: actual.__setitem__(0, actual[0] + 1))
            with torch.inference_mode():
                source = model(u)
            for ratio in (1., 1.0000000001, 2., 4.):
                nu = nu0 * ratio
                for method in METHODS:
                    stepper = TorchTransferStepper(model, lengths, h, nu0, nu, method)
                    before, unchanged = actual[0], u.clone()
                    predicted = stepper(u)
                    assert actual[0] - before == stepper.calls == 1
                    assert torch.equal(u, unchanged), "Wrapper modified its input."
                    assert predicted.dtype == dtype and predicted.device.type == "cpu"
                    assert not predicted.requires_grad
                    if ratio == 1.:
                        assert torch.equal(predicted, source), "Source recovery must be exact."

                    def factor(q_mode):
                        s = math.exp(-h*nu0*q_mode)
                        z = h*(nu-nu0)*q_mode
                        if method == "frozen":
                            return s
                        if method == "explicit":
                            return s-z
                        if method == "implicit":
                            return s/(1+z)
                        return math.exp(-h*nu*q_mode)

                    expected = means + amps_a*factor(q_a)*wave_a + amps_b*factor(q_b)*wave_b
                    error = float(np.linalg.norm(predicted.numpy()-expected) / np.linalg.norm(expected))
                    threshold = 2e-5 if dtype == torch.float32 else 2e-13
                    assert error < threshold, (shape, dtype, ratio, method, error)
                    records.append({"shape": list(shape), "lengths": list(lengths),
                                    "dtype": str(dtype), "ratio": ratio, "method": method,
                                    "relative_l2_vs_analytic_modes": error,
                                    "predictor_forward_calls": actual[0]-before})
            hook.remove()

    # A bounded lower-viscosity additive difference must not form 0 * infinity.
    u = torch.randn(2, 2, 512, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(12))
    h, nu0, nu = 100., .01, 1e-5
    k = 2*math.pi*torch.fft.fftfreq(512, d=1/512, dtype=torch.float64)
    source = lambda x: torch.fft.ifft(torch.fft.fft(x)*torch.exp(-h*nu0*k*k)).real
    two_sided = TorchTransferStepper(source, (1.,), h, nu0, nu, "delta", False)(u)
    target = torch.fft.ifft(torch.fft.fft(u)*torch.exp(-h*nu*k*k)).real
    lower_error = float(torch.linalg.vector_norm(two_sided-target)/torch.linalg.vector_norm(target))
    assert bool(torch.isfinite(two_sided).all()) and lower_error < 1e-13
    one_sided = TorchTransferStepper(source, (1.,), h, nu0, nu, "delta", True)(u)
    assert torch.equal(one_sided, source(u))
    guards = {}
    for method in ("implicit", "explicit", "lie_pre", "lie_post", "strang"):
        try:
            TorchTransferStepper(source, (1.,), h, nu0, nu, method)
        except ValueError:
            guards[method] = "rejected_lower_viscosity"
        else:
            raise AssertionError("Unsafe domain not rejected")
    try:
        benchmark_cuda_rollout(None, u, 2)
    except ValueError:
        guards["cuda_benchmark_with_cpu_input"] = "rejected"
    else:
        raise AssertionError("CPU input passed CUDA benchmark")
    report = {
        "evidence_type": "analytical_implementation_control",
        "manuscript_checkpoint_tested": False,
        "status": "passed",
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "device": "cpu",
        "cuda_available": torch.cuda.is_available(), "gpu_tests_executed": False,
        "case_count": len(records),
        "max_relative_error_float32": max(r["relative_l2_vs_analytic_modes"] for r in records if r["dtype"] == "torch.float32"),
        "max_relative_error_float64": max(r["relative_l2_vs_analytic_modes"] for r in records if r["dtype"] == "torch.float64"),
        "lower_viscosity_additive_relative_error": lower_error,
        "guards": guards, "cases": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
