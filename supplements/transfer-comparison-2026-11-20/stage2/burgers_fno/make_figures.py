"""Render the independently reconstructed neural study; no original-paper claims."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"frozen": "#777777", "delta": "#0068a3", "explicit": "#ba4e37",
          "lie_pre": "#9670aa", "lie_post": "#bd8a15", "strang": "#16826e", "implicit": "#d05b91"}
LABELS = {"frozen": "Frozen FNO", "delta": "Delta", "explicit": "Explicit",
          "lie_pre": "Lie-pre", "lie_post": "Lie-post", "strang": "Strang", "implicit": "Closed-form implicit"}


def render(folder):
    rows = json.loads((folder / "aggregate_summary.json").read_text())
    config = json.loads((folder / "locked_config.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "savefig.dpi": 220})
    fig, ax = plt.subplots(figsize=(9, 5.4))
    failure_notes = []
    for method, color in COLORS.items():
        selected = [r for r in rows if r["method"] == method]
        selected.sort(key=lambda r: r["ratio"])
        x = np.array([r["ratio"] for r in selected])
        y = np.array([np.nan if r["avg50_mean_over_training_seeds"] is None else r["avg50_mean_over_training_seeds"] for r in selected])
        sd = np.array([np.nan if r["avg50_sample_sd_over_training_seeds"] is None else r["avg50_sample_sd_over_training_seeds"] for r in selected])
        ax.errorbar(x, y, yerr=sd, marker="o", markersize=4, capsize=3, lw=1.5,
                    color=color, label=LABELS[method])
        if any(r["failed_ics_total_over_seed_ic_pairs"] for r in selected):
            failure_notes.append(method)
    ax.set(xlabel="Viscosity ratio", ylabel="Avg-50 relative L2 error",
           title="Independent Burgers FNO reconstruction: three training seeds", yscale="log")
    ax.set_xticks(config["viscosity_ratios"])
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=9, loc="best", ncols=2)
    note = "Mean +/- sample SD across 3 training seeds; 8 common test ICs. Not manuscript checkpoints."
    fig.text(.5, .045, note, ha="center", fontsize=9)
    if failure_notes:
        fig.text(.5, .015, "Failed full-test aggregates are missing, not survivor averages: " + ", ".join(failure_notes), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(folder / "neural_viscosity_comparison.png")
    fig.savefig(folder / "neural_viscosity_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for seed in config["training_seeds"]:
        curve = list(csv.DictReader((folder / f"seed_{seed}" / "training_curve.csv").open()))
        epoch = [int(r["epoch"]) for r in curve]
        axes[0].semilogy(epoch, [float(r["train_normalized_increment_mse"]) for r in curve], label=f"Training seed {seed}")
        axes[1].semilogy(epoch, [float(r["source_validation_one_step_rel_l2"]) for r in curve], label=f"Training seed {seed}")
    axes[0].set(xlabel="Epoch", ylabel="Normalized increment MSE", title="Training loss (source viscosity only)")
    axes[1].set(xlabel="Epoch", ylabel="Mean one-step relative L2", title="Source validation (checkpoint selector)")
    for ax in axes:
        ax.grid(alpha=.2, which="both")
        ax.legend(fontsize=8)
    fig.text(.5, .015, "Fixed 75-epoch budget, StepLR every 25 epochs. No target data used for training or selection.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(folder / "source_training_curves.png")
    fig.savefig(folder / "source_training_curves.pdf")
    plt.close(fig)

    raw = list(csv.DictReader((folder / "per_step_metrics.csv").open()))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method, color in COLORS.items():
        seed_curves = []
        for seed in config["training_seeds"]:
            curve = []
            for step in range(1, 51):
                selected = [r for r in raw if int(r["training_seed"]) == seed and float(r["ratio"]) == 2.
                            and r["method"] == method and int(r["step"]) == step]
                curve.append(float(np.mean([float(r["relative_l2"]) for r in selected]))
                             if len(selected) == 8 and all(r["relative_l2"] for r in selected) else np.nan)
            seed_curves.append(curve)
        values = np.array(seed_curves)
        mean, sd = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
        x = np.arange(1, 51)
        ax.semilogy(x, mean, color=color, lw=1.7, label=LABELS[method])
        if method != "explicit":
            ax.fill_between(x, np.maximum(mean - sd, 1e-8), mean + sd, color=color, alpha=.12)
    ax.set(xlabel="Autoregressive step", ylabel="Relative L2 error", title="Independent trained FNO: viscosity ratio 2")
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=9, ncols=2)
    fig.text(.5, .015, "Within-seed mean over 8 ICs; band is across-training-seed SD. Failures remain missing.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(folder / "neural_rollout_ratio2.png")
    fig.savefig(folder / "neural_rollout_ratio2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    render(parser.parse_args().results)
