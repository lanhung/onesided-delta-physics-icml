"""Write a paired exact-heat dataset and NumPy/Torch configs for evaluate.py."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from evaluate import METHODS, sha256


def create_demo(directory):
    destination = Path(directory).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Demo directory must be fresh.")
    destination.mkdir(parents=True, exist_ok=True)
    h, nu0, nu, steps = .02, .005, .01, 20
    shape, lengths = [16, 12], [1., 1.5]
    x = np.arange(shape[0]) / shape[0]
    y = np.arange(shape[1]) / shape[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    states = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        state = np.zeros(shape)
        for kx, ky in ((1, 0), (0, 2), (2, 1), (3, -2)):
            state += rng.normal() * np.sin(2 * np.pi * (kx * xx + ky * yy) + rng.uniform(-np.pi, np.pi))
        states.append(state / np.std(state))
    initial = np.stack(states)[:, None]
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], lengths[0] / shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], lengths[1] / shape[1])
    q = kx[:, None] ** 2 + ky[None, :] ** 2
    initial_hat = np.fft.fftn(initial, axes=(-2, -1))
    times = h * np.arange(steps + 1)
    truth = np.stack([np.fft.ifftn(initial_hat * np.exp(-nu * t * q), axes=(-2, -1)).real for t in times])
    truth[0] = initial
    data_path = destination / "heat_target.npz"
    np.savez_compressed(data_path, trajectory=truth, trajectory_ids=np.array(["heat_ic_0", "heat_ic_1", "heat_ic_2"]), times=times)
    metadata = {"h": h, "nu": nu, "lengths": lengths, "spatial_shape": shape, "channels": 1,
                "units": "physical", "includes_initial_state": True, "dtype": "float64",
                "generator": "analytic Fourier heat semigroup evaluated directly from initial state at every output time",
                "reference_accuracy_status": "exact finite-grid heat flow up to floating-point FFT/exponential roundoff",
                "projection_policy": "none; periodic grid, no repeated endpoint"}
    metadata_path = destination / "heat_target.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for backend in ("numpy", "torch"):
        config = {"schema_version": 1, "evidence_type": "numerical_control", "backend": backend,
                  "device": "cpu", "dtype": "float64", "target_trajectory_tuning_allowed": False,
                  "h": h, "nu0": nu0, "nu": nu, "lengths": lengths, "spatial_shape": shape,
                  "channels": 1, "horizon_steps": steps, "methods": list(METHODS),
                  "ic_seeds": [0, 1, 2], "training_seed": None, "failure_max_abs": 1e4,
                  "delta_one_sided": True,
                  "adapter": {"path": os.path.relpath(Path(__file__).with_name("heat_adapter.py").resolve(), destination)},
                  "dataset": {"path": data_path.name, "sha256": sha256(data_path)},
                  "dataset_metadata": {"path": metadata_path.name, "sha256": sha256(metadata_path)}}
        (destination / f"config_{backend}.json").write_text(json.dumps(config, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(create_demo(args.out))
