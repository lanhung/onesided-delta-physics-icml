"""Exact heat-flow control adapter. This is NOT a neural model/checkpoint."""
import math
import numpy as np


def load_predictor(config):
    shape = config["spatial_shape"]
    axes = tuple(range(-len(shape), 0))
    metadata = {
        "evidence_type": "numerical_control", "h": config["h"], "nu0": config["nu0"],
        "lengths": config["lengths"], "evaluation_spatial_shape": shape,
        "channels": config["channels"], "dtype": config["dtype"], "backend": config["backend"],
        "units": "physical", "output_kind": "full_next_state", "stateless": True,
        "sample_independent": True, "projection_policy": "none; exact full-grid heat semigroup",
        "predictor_kind": "analytical_heat_semigroup_no_network",
    }
    if config["backend"] == "numpy":
        wave_numbers = [2 * np.pi * np.fft.fftfreq(n, length / n)
                        for n, length in zip(shape, config["lengths"])]
        q = sum(k*k for k in np.meshgrid(*wave_numbers, indexing="ij", sparse=True))
        symbol = np.exp(-config["h"] * config["nu0"] * q)

        def predict(u):
            return np.fft.ifftn(np.fft.fftn(u, axes=axes) * symbol, axes=axes).real.astype(u.dtype)
    else:
        import torch
        device = torch.device(config.get("device", "cpu"))
        dtype = getattr(torch, config["dtype"])
        q = torch.zeros(shape, device=device, dtype=dtype)
        for dimension, (n, length) in enumerate(zip(shape, config["lengths"])):
            wave_number = 2 * math.pi * torch.fft.fftfreq(n, length / n, device=device, dtype=dtype)
            layout = [1] * len(shape)
            layout[dimension] = n
            q = q + wave_number.reshape(layout).square()
        symbol = torch.exp(-config["h"] * config["nu0"] * q)

        def predict(u):
            return torch.fft.ifftn(torch.fft.fftn(u, dim=axes) * symbol, dim=axes).real.contiguous()
    return predict, metadata
