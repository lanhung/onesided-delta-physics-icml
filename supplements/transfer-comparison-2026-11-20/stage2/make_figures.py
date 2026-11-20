"""Generate publication-style figures from recorded stage2 results only."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
COLORS = {"frozen": "#52606d", "delta": "#d1493f", "lie_pre": "#9b6a2f",
          "lie_post": "#286f9b", "strang": "#17856e", "implicit": "#7a5ba7",
          "explicit": "#999999"}
LABELS = {"frozen": "Frozen", "delta": "Delta Physics", "lie_pre": "Lie before",
          "lie_post": "Lie after", "strang": "Strang", "implicit": "Closed implicit",
          "explicit": "Explicit"}
METHODS = ("frozen", "delta", "lie_pre", "lie_post", "strang", "implicit")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    base = ROOT / "burgers_fno" / "results"
    rows = json.loads((base / "aggregate_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in METHODS:
        chosen = [r for r in rows if r["method"] == method]
        x = [r["ratio"] for r in chosen]
        y = [r["avg50_mean_over_training_seeds"] for r in chosen]
        axes[0].plot(x, y, "o-", color=COLORS[method], label=LABELS[method], lw=1.8, ms=4)
    axes[0].set(xlabel="Viscosity ratio", ylabel="Avg 50 relative L2", yscale="log",
                title="A. Same frozen weights across viscosity shifts")
    axes[0].set_xticks([1, 1.5, 2, 3, 4])
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=9, ncol=2, loc="upper left")
    with (base / "per_step_metrics.csv").open() as f:
        selected = [r for r in csv.DictReader(f) if float(r["ratio"]) == 2.]
    for method in ("delta", "lie_pre", "lie_post", "strang", "implicit"):
        values = np.zeros((3, 50))
        for seed in range(3):
            for step in range(1, 51):
                entries = [float(r["relative_l2"]) for r in selected
                           if r["method"] == method and int(r["training_seed"]) == seed and int(r["step"]) == step]
                values[seed, step-1] = np.mean(entries)
        mean, sd = values.mean(0), values.std(0, ddof=1)
        axes[1].plot(np.arange(1, 51), mean, color=COLORS[method], label=LABELS[method], lw=1.8)
        if method in ("delta", "strang"):
            axes[1].fill_between(np.arange(1, 51), mean-sd, mean+sd, color=COLORS[method], alpha=.13)
    axes[1].set(xlabel="Rollout step", ylabel="Relative L2", title="B. Ratio 2: the ranking changes over time")
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=9)
    fig.suptitle("Independent Burgers FNO reconstruction | 75 epochs, 3 training seeds, 8 paired test ICs", fontsize=12, y=1.02)
    fig.text(.5, -.025, "Bands: sample SD across training seeds. Explicit failed all 24 seed-IC pairs at ratios 1.5 to 4; ratio 1.25 remained finite.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "burgers_fno_comparison")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for seed in range(3):
        with (base / f"seed_{seed}" / "training_curve.csv").open() as f:
            curve = list(csv.DictReader(f))
        ax.plot([int(r["epoch"]) for r in curve],
                [float(r["source_validation_one_step_rel_l2"]) for r in curve], label=f"Training seed {seed}")
    ax.set(xlabel="Epoch", ylabel="Source validation one-step relative L2", yscale="log",
           title="Source-only validation; all selected checkpoints are epoch 75")
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.5, -.01, "The curves are still decreasing near epoch 75; this run does not establish full training convergence.", ha="center", fontsize=10)
    fig.tight_layout()
    save(fig, "burgers_training_curves")

    ns = ROOT / "ns_controls" / "results" / "summary.csv"
    if ns.exists() and json.loads((ns.parent/"progress.json").read_text()).get("status") == "complete":
        with ns.open() as f:
            ns_rows = list(csv.DictReader(f))
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
        for ax, grid in zip(axes, (64, 128)):
            for method in METHODS:
                chosen = sorted([r for r in ns_rows if r["method"] == method and int(r["grid"]) == grid], key=lambda r: float(r["ratio"]))
                ax.plot([float(r["ratio"]) for r in chosen], [float(r["avg50_rell2_mean"]) for r in chosen],
                        "o-", color=COLORS[method], label=LABELS[method], lw=1.8, ms=5)
            ax.set(title=f"{grid} x {grid}", xlabel="Viscosity ratio", yscale="log", xticks=[1.5, 2, 4])
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Avg 50 relative L2")
        axes[1].legend(fontsize=9, ncol=2, loc="center right", bbox_to_anchor=(.98, .37))
        fig.suptitle("Navier-Stokes numerical reference controls | 3 paired initial conditions; no neural network", y=1.02, fontsize=12)
        fig.text(.5, -.025, "All curves use accurate source flow. Grid consistency here does not demonstrate neural resolution transfer.", ha="center", fontsize=10)
        fig.text(.5, -.07, "Explicit omitted: no failures within 50 steps at 64 x 64, ratio 1.5; 3/3 failures in each other case.", ha="center", fontsize=9)
        fig.tight_layout()
        save(fig, "ns_resolution_controls")

    bench = ROOT / "cpu_benchmark_threads1" / "benchmark.json"
    if bench.exists():
        d = json.loads(bench.read_text())
        lookup = {r["method"]: r for r in d["summary"] if r["batch_size"] == 8}
        methods = ["frozen", "delta", "lie_pre", "lie_post", "strang", "implicit", "rk4_dt0.002"]
        labels = [LABELS.get(m, "Direct spectral RK4") for m in methods]
        colors = [COLORS.get(m, "#254857") for m in methods]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
        y = np.arange(len(methods))
        med = np.array([lookup[m]["median_seconds_per_50step_rollout"]*1000 for m in methods])
        low = np.array([lookup[m]["q1_seconds"]*1000 for m in methods])
        high = np.array([lookup[m]["q3_seconds"]*1000 for m in methods])
        axes[0].barh(y, med, color=colors, alpha=.88, height=.62)
        axes[0].errorbar(med, y, xerr=np.stack([med-low, high-med]), fmt="none", color="#27313a", capsize=3)
        for pos, val in zip(y, med):
            axes[0].text(val+4, pos, f"{val:.2f}", va="center", fontsize=9)
        axes[0].set(xlabel="Milliseconds per 50-output batch rollout", title="A. CPU latency: median and interquartile range", xlim=(0, 91))
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", alpha=.18)
        errors = [lookup[m]["avg50_rel_l2"] for m in methods]
        axes[1].scatter(errors, y, c=colors, s=65, zorder=3)
        for pos, err in zip(y, errors):
            axes[1].annotate(f"{err:.2e}", (err, pos), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
        axes[1].set(xscale="log", xlabel="Avg 50 relative L2 (log scale)", title="B. Separate full-rollout accuracy check", xlim=(1e-7, 2))
        axes[1].grid(axis="x", alpha=.18)
        fig.suptitle("Independent Burgers FNO | seed 0, ratio 2, N=64, batch 8 | Torch 1 thread, SciPy FFT 1 worker", y=1.03, fontsize=12)
        fig.text(.5, -.02, "3 full warmups + 10 repeats. All 50 physical outputs computed; final state retained. RK4 internal dt = 0.002.", ha="center", fontsize=9)
        fig.text(.5, -.065, "This is a CPU implementation comparison, not an original-paper reproduction or a GPU performance claim.", ha="center", fontsize=9)
        fig.tight_layout()
        save(fig, "burgers_cpu_timing")
    print("Created figures from recorded results in", FIG)


if __name__ == "__main__":
    main()
