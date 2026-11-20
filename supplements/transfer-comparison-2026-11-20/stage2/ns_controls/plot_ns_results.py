"""Plot accepted numerical-reference controls; no neural-model claims."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    config = json.loads((args.results / "config.json").read_text())
    progress = json.loads((args.results / "progress.json").read_text())
    if progress["status"] != "complete":
        raise RuntimeError("Only plot a completed accepted experiment matrix.")
    methods = ("frozen", "delta", "explicit", "lie_post", "lie_pre", "strang", "implicit")
    labels = {"frozen": "Frozen numerical flow", "delta": "Additive Delta",
              "explicit": "Explicit", "lie_post": "Lie post", "lie_pre": "Lie pre",
              "strang": "Strang", "implicit": "Closed-form implicit"}
    colors = {"frozen": "#2864A8", "delta": "#D57421", "explicit": "#8A8A8A",
              "lie_post": "#167D83", "lie_pre": "#7E57A5", "strang": "#B63142",
              "implicit": "#6A5438"}
    grids, ratios = config["grids"], config["ratios"]
    fig, axs = plt.subplots(len(grids), len(ratios), figsize=(14.2, 8.8),
                            sharex=True, sharey=True, squeeze=False)
    handles = {}
    ymax = 10.
    for gi, n in enumerate(grids):
        for ri, ratio in enumerate(ratios):
            ax = axs[gi, ri]
            tag = f"n{n}_r{ratio:g}".replace(".", "p")
            rows = list(csv.DictReader((args.results / tag / "per_step.csv").open()))
            summaries = list(csv.DictReader((args.results / tag / "summary_by_ic.csv").open()))
            for method in methods:
                error = []
                for step in range(1, 51):
                    group = [r for r in rows if r["method"] == method and int(r["step"]) == step]
                    vals = [float(r["relative_l2"]) for r in group if r["relative_l2"]]
                    error.append(np.mean(vals) if len(vals) == 3 else np.nan)
                error = np.asarray(error)
                line, = ax.semilogy(np.arange(1, 51) * .05, np.minimum(error, ymax),
                                  color=colors[method], label=labels[method], lw=1.8,
                                  linestyle="--" if method in ("lie_pre", "explicit") else "-")
                handles[method] = line
                capped = np.flatnonzero(error > ymax)
                if len(capped):
                    ax.plot((capped[0] + 1) * .05, ymax, marker="^", ms=7,
                            color=colors[method], clip_on=False)
            failed = [r for r in summaries if r["method"] == "explicit" and r["first_failure_step"]]
            annotation = f"Explicit failures: {len(failed)}/3"
            if failed:
                annotation += "; first step " + str(min(int(r["first_failure_step"]) for r in failed))
            ax.text(.03, .95, annotation, transform=ax.transAxes, va="top", fontsize=9,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
            ax.set_title(f"{n} x {n}  |  viscosity ratio = {ratio:g}", fontsize=12, pad=9)
            ax.set_ylim(1e-7, ymax)
            ax.set_xlim(0, 2.5)
            ax.grid(True, which="major", alpha=.23)
            ax.spines[["top", "right"]].set_visible(False)
            if gi == len(grids) - 1:
                ax.set_xlabel("Physical time", fontsize=11)
            if ri == 0:
                ax.set_ylabel("Mean relative L2 error over 3 ICs", fontsize=11)
    fig.suptitle("Navier-Stokes: numerical reference flow controls", fontsize=18, y=.985)
    fig.text(.5, .944, "Paired initial conditions, 50 output steps; these are NOT trained neural-model results",
             ha="center", fontsize=11, color="#444444")
    fig.legend([handles[m] for m in methods], [labels[m] for m in methods],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .035), fontsize=10)
    fig.text(.5, .012, "Failed trajectories are never omitted from means; curves stop at first failure. Triangles mark values above the plot ceiling.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.075, right=.985, top=.885, bottom=.17, hspace=.28, wspace=.12)
    fig.savefig(args.results / "ns_control_curves.png", dpi=160)
    fig.savefig(args.results / "ns_control_curves.pdf")
    print(args.results / "ns_control_curves.png")


if __name__ == "__main__":
    main()
