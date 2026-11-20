"""Load the separately reconstructed Burgers FNO with an actual forward hook."""
import importlib.util
import json
from pathlib import Path
import torch

from evaluate import sha256


class InstrumentedPhysicalPredictor:
    def __init__(self, model, stats, physical_predictor_factory):
        self.model = model
        self.count = 0
        self.physical = physical_predictor_factory(model, stats)
        self.hook_handle = model.register_forward_hook(self._on_model_forward)

    def _on_model_forward(self, _module, _args, _output):
        self.count += 1

    def get_forward_count(self):
        return self.count

    @torch.inference_mode()
    def __call__(self, u):
        return self.physical(u)


def load_predictor(config):
    if config["evidence_type"] != "independently_reconstructed_neural":
        raise ValueError("This adapter is for an independent reconstruction, not the original manuscript model.")
    if config["backend"] != "torch" or config["dtype"] != "float32":
        raise ValueError("Reconstruction was trained/evaluated in Torch float32.")
    base = Path(config["config_directory"])
    settings = config["adapter"].get("settings", {})
    module_path = (base / settings["training_script"]).resolve()
    if sha256(module_path) != settings["training_script_sha256"]:
        raise ValueError("Reconstructed model source changed since config generation.")
    module_spec = importlib.util.spec_from_file_location("reconstructed_burgers_model", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    checkpoint_path = (base / config["artifacts"]["checkpoint"]["path"]).resolve()
    normalizer_path = (base / config["artifacts"]["normalizer"]["path"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    stats = json.loads(normalizer_path.read_text())
    if checkpoint["normalization"] != stats:
        raise ValueError("Loaded normalizer differs from checkpoint's source-training statistics.")
    trained = checkpoint["config"]
    if (config["spatial_shape"] != [trained["grid"]] or config["channels"] != 1 or
            config["h"] != trained["h"] or config["nu0"] != trained["nu0"] or
            config["lengths"] != [trained["domain_length"]]):
        raise ValueError("This verified adapter requires the reconstructed model's original physical grid and step.")
    torch.set_num_threads(settings.get("torch_cpu_threads", trained["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    model = module.FNO1d()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # Fourier weights are complex64. A blanket Module.to(dtype=float32) would
    # discard their imaginary parts, so preserve the checkpoint's native dtypes.
    model.to(device=config.get("device", "cpu"))
    if any(parameter.dtype != (torch.complex64 if parameter.is_complex() else torch.float32)
           for parameter in model.parameters()):
        raise ValueError("Checkpoint parameter precision differs from the locked float32/complex64 architecture.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    predict = InstrumentedPhysicalPredictor(model, stats, module.physical_predictor)
    metadata = {
        "evidence_type": "independently_reconstructed_neural", "original_manuscript_checkpoint": False,
        "h": trained["h"], "nu0": trained["nu0"], "lengths": [trained["domain_length"]],
        "evaluation_spatial_shape": [trained["grid"]], "training_resolution": [trained["grid"]],
        "channels": 1, "dtype": "float32", "backend": "torch", "units": "physical",
        "output_kind": "full_next_state", "stateless": True, "sample_independent": True,
        "projection_policy": "none; fixed source normalization and coordinates inside physical predictor",
        "checkpoint_sha256": sha256(checkpoint_path), "normalizer_sha256": sha256(normalizer_path),
        "training_seed": checkpoint["training_seed"], "selected_epoch": checkpoint["epoch"],
        "source_validation_rel_l2": checkpoint["source_validation_rel_l2"],
        "eval_mode": not model.training, "architecture": trained["architecture"],
        "training_script_sha256": sha256(module_path), "forward_count_method": "torch model register_forward_hook",
    }
    return predict, metadata
