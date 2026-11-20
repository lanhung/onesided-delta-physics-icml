"""Follow-up fixed-epoch ablation, independently reconstructed Burgers FNO.

Evaluate all saved epochs15/30/50/75 at ratio2, without retraining, checkpoint
selection using target data, or modifications to the primary experiment files.
Fixed-epoch snapshots are NOT best-source-validation-within-budget selections.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_burgers_fno import (CONFIG, EVIDENCE, FNO1d, METHODS, TorchTransferStepper,
                             digest, metrics, physical_predictor, write_csv, write_json)

BUDGETS = (15, 30, 50, 75)


def run(source, out):
    out.mkdir(parents=True, exist_ok=True)
    stats = json.loads((source / "normalization.json").read_text())
    truth_path = source / "truth_r2.npz"
    with np.load(truth_path, allow_pickle=False) as data:
        truth = data["truth"][:, :, 0, :]
        ic_seeds = data["ic_seeds"].tolist()
    locked = {"evidence_type": EVIDENCE, "experiment": "followup_fixed_epoch_checkpoint_ablation",
              "epochs": list(BUDGETS), "ratio": 2.0, "training_seeds": [0, 1, 2],
              "ic_seeds": ic_seeds, "methods": list(METHODS),
              "checkpoint_policy": "actual fixed epoch snapshot; not best validation checkpoint within each budget",
              "source_dataset_and_schedule": "same primary75epoch training run, snapshots of its trajectory",
              "target_tuning": False, "new_training": False,
              "truth_sha256": digest(truth_path),
              "normalization_sha256": digest(source / "normalization.json"),
              "scope": "follow-up diagnostic on saved models; not a new independent training replicate or manuscript reproduction"}
    write_json(out / "ablation_config.json", locked)
    started = time.perf_counter()
    summaries, raw = [], []
    for seed in (0, 1, 2):
        for budget in BUDGETS:
            path = source / f"seed_{seed}" / f"epoch_{budget}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["training_seed"] != seed or checkpoint["epoch"] != budget:
                raise ValueError("Saved fixed-epoch metadata disagrees")
            model = FNO1d()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            predictor = physical_predictor(model, stats)
            forward_count = [0]
            def hook(_module, _inputs, _output):
                forward_count[0] += 1
            handle = model.register_forward_hook(hook)
            for method in METHODS:
                stepper = TorchTransferStepper(predictor, (1.,), CONFIG["h"], CONFIG["nu0"], .02, method)
                u = torch.tensor(truth[0, :, None, :], dtype=torch.float32)
                alive = np.ones(len(u), dtype=bool)
                failed_at = [None] * len(u)
                errors = np.full((50, len(u)), np.nan)
                count0 = forward_count[0]
                with torch.inference_mode():
                    for step in range(1, 51):
                        u = stepper(u)
                        prediction = u.numpy()[:, 0, :].astype(np.float64)
                        bad = alive & (~np.isfinite(prediction).all(axis=-1) |
                                       (np.max(np.abs(prediction), axis=-1) > 1e4))
                        for idx in np.flatnonzero(bad):
                            failed_at[idx] = step
                        alive[bad] = False
                        for idx, ic_seed in enumerate(ic_seeds):
                            row = {"evidence_type": EVIDENCE, "training_seed": seed,
                                   "fixed_checkpoint_epoch": budget, "ratio": 2., "method": method,
                                   "test_ic_seed": ic_seed, "step": step,
                                   "status": "ok" if alive[idx] else "failed",
                                   "first_failure_step": failed_at[idx]}
                            if alive[idx]:
                                row.update(metrics(prediction[idx], truth[step, idx]))
                                errors[step - 1, idx] = row["relative_l2"]
                            else:
                                row.update({k: None for k in ("relative_l2", "absolute_l2_unit_length", "psre",
                                            "reference_l2_unit_length", "mean_drift_from_truth", "state_l2_unit_length")})
                            raw.append(row)
                        if not alive.all():
                            u[torch.as_tensor(~alive)] = 0.
                calls = forward_count[0] - count0
                assert calls == 50 and stepper.calls == 50
                valid = bool(alive.all())
                summaries.append({"evidence_type": EVIDENCE, "training_seed": seed,
                                  "fixed_checkpoint_epoch": budget, "ratio": 2., "method": method,
                                  "avg50_rel_l2": float(errors.mean()) if valid else None,
                                  "step50_rel_l2": float(errors[-1].mean()) if valid else None,
                                  "n_failed_ics": int((~alive).sum()), "n_test_ics": len(u),
                                  "first_failure_step": min((s for s in failed_at if s is not None), default=None),
                                  "source_validation_one_step_rel_l2_at_epoch": checkpoint["source_validation_rel_l2"],
                                  "checkpoint_sha256": digest(path), "actual_batched_model_forwards": calls})
            handle.remove()
            print(json.dumps({"training_seed": seed, "fixed_epoch_completed": budget}), flush=True)
    aggregate = []
    for epoch in BUDGETS:
        for method in METHODS:
            selected = [r for r in summaries if r["fixed_checkpoint_epoch"] == epoch and r["method"] == method]
            valid = len(selected) == 3 and all(r["avg50_rel_l2"] is not None for r in selected)
            frozen_gains, delta_gains = [], []
            for row in selected:
                frozen = next(r for r in summaries if r["fixed_checkpoint_epoch"] == epoch and r["training_seed"] == row["training_seed"] and r["method"] == "frozen")
                delta = next(r for r in summaries if r["fixed_checkpoint_epoch"] == epoch and r["training_seed"] == row["training_seed"] and r["method"] == "delta")
                if row["avg50_rel_l2"] is not None:
                    frozen_gains.append(100 * (frozen["avg50_rel_l2"] - row["avg50_rel_l2"]) / frozen["avg50_rel_l2"])
                    delta_gains.append(100 * (delta["avg50_rel_l2"] - row["avg50_rel_l2"]) / delta["avg50_rel_l2"])
            aggregate.append({"fixed_checkpoint_epoch": epoch, "ratio": 2., "method": method,
                              "avg50_mean": float(np.mean([r["avg50_rel_l2"] for r in selected])) if valid else None,
                              "avg50_sample_sd": float(np.std([r["avg50_rel_l2"] for r in selected], ddof=1)) if valid else None,
                              "step50_mean": float(np.mean([r["step50_rel_l2"] for r in selected])) if valid else None,
                              "step50_sample_sd": float(np.std([r["step50_rel_l2"] for r in selected], ddof=1)) if valid else None,
                              "mean_within_seed_avg50_gain_percent_vs_frozen": float(np.mean(frozen_gains)) if len(frozen_gains) == 3 else None,
                              "mean_within_seed_avg50_gain_percent_vs_delta": float(np.mean(delta_gains)) if len(delta_gains) == 3 else None,
                              "source_val_mean_at_fixed_epoch": float(np.mean([r["source_validation_one_step_rel_l2_at_epoch"] for r in selected])),
                              "failed_seed_ic_pairs": sum(r["n_failed_ics"] for r in selected)})
    write_csv(out / "per_step_metrics.csv", raw)
    write_csv(out / "per_seed_summary.csv", summaries)
    write_json(out / "per_seed_summary.json", summaries)
    write_csv(out / "aggregate_summary.csv", aggregate)
    write_json(out / "aggregate_summary.json", aggregate)
    # The primary best checkpoints happened to be epoch75 for every seed.
    # Require exact reproduction of their saved metrics at that same epoch.
    matches = []
    for seed in (0, 1, 2):
        primary = json.loads((source / f"seed_{seed}" / "summary.json").read_text())
        for row in (r for r in summaries if r["training_seed"] == seed and r["fixed_checkpoint_epoch"] == 75):
            reference = next(r for r in primary if r["ratio"] == 2. and r["method"] == row["method"])
            same = row["avg50_rel_l2"] == reference["avg50_rel_l2"] and row["step50_rel_l2"] == reference["step50_rel_l2"] and row["n_failed_ics"] == reference["n_failed_ics"]
            matches.append(same)
    if not all(matches):
        raise AssertionError("Fixed75 ablation differs from primary75 result")
    write_json(out / "execution_summary.json", {"evidence_type": EVIDENCE, "complete": True,
               "runs": len(summaries), "individual_50_step_trajectories": len(summaries) * 8,
               "per_step_metric_rows": len(raw), "epoch75_matches_primary_exactly": all(matches),
               "elapsed_seconds_not_benchmark": time.perf_counter() - started,
               "script_sha256": digest(__file__), "primary_results_modified": False})
    render(out, aggregate)


def render(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures import COLORS, LABELS
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        if all(r["avg50_mean"] is None for r in selected):
            continue
        ax.errorbar([r["fixed_checkpoint_epoch"] for r in selected],
                    [r["avg50_mean"] for r in selected], yerr=[r["avg50_sample_sd"] for r in selected],
                    marker="o", lw=1.6, capsize=3, color=COLORS[method], label=LABELS[method])
    ax.set(xlabel="Saved checkpoint epoch", ylabel="Avg-50 relative L2 at viscosity ratio2",
           title="Independent Burgers FNO: fixed training-budget snapshots", yscale="log")
    ax.set_xticks(BUDGETS)
    ax.grid(alpha=.2, which="both")
    ax.legend(ncols=2, fontsize=9)
    fig.text(.5, .035, "Mean +/- sample SD over3 training seeds; same8ICs. Snapshots from one75epoch schedule.", ha="center", fontsize=8.5)
    fig.text(.5, .01, "Fixed epochs, not best-within-budget; explicit fails all24 seed-IC pairs at each budget.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(out / "fixed_budget_comparison.png", dpi=220)
    fig.savefig(out / "fixed_budget_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "budget_ablation")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.source.resolve(), args.out.resolve())
