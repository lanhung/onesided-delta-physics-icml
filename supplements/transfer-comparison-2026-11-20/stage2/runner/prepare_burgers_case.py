"""Package one independently reconstructed checkpoint/target case for evaluate.py."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from evaluate import METHODS, load_json, sha256


def package_case(result_directory, seed, ratio, destination):
    result_directory, destination = Path(result_directory).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Case directory must be fresh.")
    locked = load_json(result_directory / "locked_config.json")
    training = load_json(result_directory / f"seed_{seed}" / "training_metadata.json")
    if training.get("pilot") or training["completed_epochs"] != locked["epochs"]:
        raise ValueError("Training is incomplete or a pilot; cannot label the 75-epoch case complete.")
    tag = f"r{ratio:g}".replace(".", "p")
    raw_path = result_directory / f"truth_{tag}.npz"
    with np.load(raw_path, allow_pickle=False) as stored:
        truth, ic_seeds = stored["truth"], stored["ic_seeds"].tolist()
    checkpoint = result_directory / f"seed_{seed}" / "best_source_validation.pt"
    normalizer = result_directory / "normalization.json"
    if sha256(checkpoint) != training["checkpoint_sha256"] or sha256(normalizer) != training["normalization_sha256"]:
        raise ValueError("Training metadata no longer matches the selected artifacts.")
    refinement = load_json(result_directory / "temporal_refinement.json")
    chosen_refinement = next(case for case in refinement["target_cases"] if case["ratio"] == ratio)
    if sha256(raw_path) != chosen_refinement["truth_sha256"]:
        raise ValueError("Target reference changed since the recorded time-refinement check.")
    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "paired_target.npz"
    np.savez_compressed(dataset_path, trajectory=truth,
                        trajectory_ids=np.array([f"burgers_test_ic_{seed}" for seed in ic_seeds]),
                        times=locked["h"] * np.arange(truth.shape[0]))
    metadata = {"h": locked["h"], "nu": ratio * locked["nu0"], "lengths": [locked["domain_length"]],
                "spatial_shape": [locked["grid"]], "channels": 1, "units": "physical", "includes_initial_state": True,
                "dtype": str(truth.dtype), "generator": "independent float64 Fourier RK4 Burgers reference",
                "reference_accuracy_status": "time refinement only on fixed grid; spatial convergence not established",
                "temporal_refinement": chosen_refinement,
                "projection_policy": "2/3-dealiased conservative nonlinearity, full Laplacian; no full-state projection",
                "source_raw_file": os.path.relpath(raw_path, destination), "source_raw_sha256": sha256(raw_path)}
    metadata_path = destination / "paired_target.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    training_script = result_directory.parent / "run_burgers_fno.py"
    config = {"schema_version": 1, "evidence_type": "independently_reconstructed_neural", "backend": "torch",
              "device": "cpu", "dtype": "float32", "reference_dtype": str(truth.dtype), "training_seed": seed,
              "ic_seeds": ic_seeds, "target_trajectory_tuning_allowed": False,
              "h": locked["h"], "nu0": locked["nu0"], "nu": ratio * locked["nu0"],
              "lengths": [locked["domain_length"]], "spatial_shape": [locked["grid"]], "channels": 1,
              "horizon_steps": locked["test_steps"], "methods": list(METHODS), "delta_one_sided": True,
              "failure_max_abs": 1e4,
              "adapter": {"path": os.path.relpath(Path(__file__).with_name("burgers_reconstruction_adapter.py"), destination),
                          "settings": {"training_script": os.path.relpath(training_script, destination),
                                       "training_script_sha256": sha256(training_script), "torch_cpu_threads": locked["torch_threads"]}},
              "dataset": {"path": dataset_path.name, "sha256": sha256(dataset_path)},
              "dataset_metadata": {"path": metadata_path.name, "sha256": sha256(metadata_path)},
              "artifacts": {"checkpoint": {"path": os.path.relpath(checkpoint, destination), "sha256": sha256(checkpoint)},
                            "normalizer": {"path": os.path.relpath(normalizer, destination), "sha256": sha256(normalizer)}}}
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return destination / "config.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=2.)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(package_case(args.results, args.seed, args.ratio, args.out))
