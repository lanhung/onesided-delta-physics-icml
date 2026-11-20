"""Meaningful integration/guardrail checks; no learned-model accuracy is asserted."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate import (CheckedPredictor, evaluate_prepared, load_json, main, metrics,
                      prepare, sha256)
from make_heat_demo import create_demo


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = create_demo(Path(self.temporary.name) / "demo")
        self.config_path = self.directory / "config_numpy.json"

    def tearDown(self):
        self.temporary.cleanup()

    def edit(self, operation):
        config = load_json(self.config_path)
        operation(config)
        self.config_path.write_text(json.dumps(config))

    def test_heat_end_to_end_and_source_identity(self):
        prepared = prepare(self.config_path)
        report, rows, raw = evaluate_prepared(prepared)
        self.assertEqual(len(rows), 7 * 20 * 3)
        for summary in report["summaries"]:
            self.assertEqual(summary["callback_calls"], 20)
            self.assertEqual(summary["requested_sample_steps"], 60)
            self.assertIsNone(summary["actual_network_forward_calls"])
            self.assertIsNone(summary["training_seed"])
            if summary["method"] in ("delta", "lie_pre", "lie_post", "strang"):
                self.assertLess(summary["final_relative_l2"], 3e-14)
        self.assertFalse(report["original_neural_experiment"])
        # Source-viscosity recovery is exact for a deterministic adapter, with the
        # same reference trajectory used only to keep this comparison paired.
        config, truth, ids, predictor, provenance, dm, mm = prepared
        config["nu"] = config["nu0"]
        _, _, source_raw = evaluate_prepared((config, truth, ids, predictor, provenance, dm, mm))
        for method, states in source_raw.items():
            np.testing.assert_array_equal(states, source_raw["frozen"])

    def test_per_sample_failure_retained(self):
        prepared = list(prepare(self.config_path))
        original = prepared[3].predict

        class Fault:
            def __init__(self):
                self.calls = 0

            def __call__(self, state):
                self.calls += 1
                result = original(state)
                if self.calls == 2:
                    result[1] = np.nan
                return result

        prepared[0]["methods"] = ["frozen"]
        prepared[3] = CheckedPredictor(Fault(), "numpy")
        report, rows, raw = evaluate_prepared(tuple(prepared))
        summary = report["summaries"][0]
        self.assertEqual(summary["failed_trajectories"], 1)
        self.assertEqual(summary["first_failure_step_by_trajectory"]["heat_ic_1"], 2)
        self.assertIsNone(summary["avg_relative_l2"])
        self.assertIsNone(summary["final_relative_l2"])
        self.assertTrue(np.isnan(raw["frozen"][2:, 1]).all())
        self.assertTrue(np.isfinite(raw["frozen"][:, 0]).all())
        self.assertTrue(all(row["first_failure_step"] == 2 for row in rows if row["trajectory_id"] == "heat_ic_1"))

    def test_zero_reference_norm_is_explicit(self):
        zero = np.zeros((1, 8))
        for prediction in (zero, np.ones_like(zero)):
            result = metrics(prediction, zero, [2.])
            self.assertIsNone(result["relative_l2"])
            self.assertIsNone(result["psre"])
            self.assertEqual(result["relative_l2_status"], "zero_reference_norm")
        self.assertEqual(metrics(zero, zero, [2.])["absolute_l2"], 0.)
        self.assertAlmostEqual(metrics(np.ones_like(zero), zero, [2.])["absolute_l2"], np.sqrt(2))

    def test_neural_artifacts_required_before_outputs(self):
        self.edit(lambda config: config.update(evidence_type="original_neural", training_seed=0))
        output = self.directory / "should_not_exist"
        self.assertEqual(main(["--config", str(self.config_path), "--out", str(output)]), 2)
        self.assertFalse(output.exists())
        self.edit(lambda config: config.update(evidence_type="independently_reconstructed_neural"))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            prepare(self.config_path)

    def test_hash_and_grid_mismatch_rejected(self):
        original = self.config_path.read_text()
        self.edit(lambda config: config["dataset"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare(self.config_path)
        self.config_path.write_text(original)
        self.edit(lambda config: config.update(lengths=[1., 2.]))
        with self.assertRaisesRegex(ValueError, "lengths"):
            prepare(self.config_path)

    def test_duplicate_ids_and_timestamps_rejected(self):
        config = load_json(self.config_path)
        data_path = self.directory / config["dataset"]["path"]
        with np.load(data_path) as data:
            trajectory, ids, times = data["trajectory"], data["trajectory_ids"], data["times"]
        np.savez(data_path, trajectory=trajectory, trajectory_ids=np.array(["duplicate"] * 3), times=times)
        self.edit(lambda cfg: cfg["dataset"].update(sha256=sha256(data_path)))
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare(self.config_path)
        np.savez(data_path, trajectory=trajectory, trajectory_ids=ids, times=times * 2)
        self.edit(lambda cfg: cfg["dataset"].update(sha256=sha256(data_path)))
        with self.assertRaisesRegex(ValueError, "timestamps"):
            prepare(self.config_path)

    def test_no_aitken_substitution_and_invalid_physics(self):
        original = self.config_path.read_text()
        self.edit(lambda cfg: cfg.update(methods=["original_input_recast_aitken"]))
        with self.assertRaisesRegex(ValueError, "different method"):
            prepare(self.config_path)
        self.config_path.write_text(original)
        self.edit(lambda cfg: cfg.update(h=0))
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare(self.config_path)

    def test_predictor_mutation_and_inner_call_counter(self):
        state = np.ones((2, 1, 8))

        def bad_predictor(u):
            u += 1
            return u

        with self.assertRaisesRegex(ValueError, "mutated"):
            CheckedPredictor(bad_predictor, "numpy")(state.copy())

        class Counted:
            def __init__(self):
                self.counter = 0

            def get_forward_count(self):
                return self.counter

            def __call__(self, u):
                self.counter += 1
                return u.copy()

        checked = CheckedPredictor(Counted(), "numpy", True)
        checked(state)
        self.assertEqual(checked.network_calls, 1)

    def test_torch_cpu_matches_numpy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is unavailable; this is not a passed runtime check.")
        torch.set_num_threads(4)
        numpy_report, _, numpy_raw = evaluate_prepared(prepare(self.config_path))
        torch_report, _, torch_raw = evaluate_prepared(prepare(self.directory / "config_torch.json"))
        for method in numpy_raw:
            np.testing.assert_allclose(torch_raw[method], numpy_raw[method], rtol=1e-12, atol=2e-13)
        for left, right in zip(numpy_report["summaries"], torch_report["summaries"]):
            self.assertAlmostEqual(left["avg_relative_l2"], right["avg_relative_l2"], delta=2e-13)
            self.assertEqual(right["callback_calls"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
