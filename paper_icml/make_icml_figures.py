"""Figures for the ICML-format manuscript (paper_icml/).

Reads only committed result files (CSV); needs numpy and matplotlib, not torch.
Three evidence categories are kept apart and named in every panel title:

* original models   -- ../results/ (the paper's F-FNO and FNO checkpoints);
* independent FNOs  -- supplements/.../stage2/burgers_fno (three newly trained Burgers FNOs);
* numerical control -- supplements/.../stage2/ns_controls (no network; exact source flow).

Run from the repository root or from paper_icml/:

    python paper_icml/make_icml_figures.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SUP = ROOT / "supplements" / "transfer-comparison-2026-11-20" / "stage2"
OUT = Path(__file__).resolve().parent / "figures"
NS = RES / "ffno_ns64_high_viscosity_3seed"
BURGERS = RES / "burgers_high_viscosity_3seed"
RECAST = RES / "ffno_ns64_equation_recast_3seed"
CONDITIONED = RES / "ffno_ns64_conditioned_multinu_3seed"
DENSE = RES / "ffno_dense_viscosity_scan"
SPECTRAL = RES / "paper_assets" / "figure_3_spectral_data.csv"
RECON = SUP / "tables" / "burgers_fno_results.csv"
RECON_STEPS = SUP / "burgers_fno" / "results" / "per_step_metrics.csv"
EPOCHS = SUP / "tables" / "burgers_fixed_epoch_results.csv"
NSCTRL = SUP / "tables" / "ns_numerical_control_results.csv"

# Okabe-Ito colour-blind-safe palette, one colour per method throughout the paper.
STYLE = {
    "frozen": ("Frozen", "#666666", "o", "--"),
    "delta": (r"Additive ($\Delta$-Physics)", "#E69F00", "s", "-"),
    "lie_pre": ("Lie, pre", "#56B4E9", "v", "-"),
    "lie_post": ("Lie, post", "#0072B2", "^", "-"),
    "strang": ("Strang", "#009E73", "D", "-"),
    "implicit": ("Implicit, post", "#CC79A7", "P", "-"),
}
FULL, COLUMN = 6.75, 3.25  # ICML text and column widths in inches


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 3.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        out = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            out.append(parsed)
    return out


def series(table, method, key):
    picked = sorted((r for r in table if r["method"] == method), key=lambda r: r["viscosity_ratio"])
    return (np.array([r["viscosity_ratio"] for r in picked]),
            np.array([r[f"{key}_mean"] for r in picked]),
            np.array([r[f"{key}_std"] for r in picked]))


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def plot_method(axis, x, y, method, yerr=None, **kwargs):
    label, color, marker, ls = STYLE[method]
    if yerr is None:
        axis.plot(x, y, color=color, marker=marker, ls=ls, label=label, **kwargs)
    else:
        axis.errorbar(x, y, yerr=yerr, color=color, marker=marker, ls=ls, capsize=2, label=label, **kwargs)


def spectrum_panel(axis, data, show_error=False):
    x = np.array([r["mode_radius"] for r in data])
    entries = [("increment", "frozen"), ("delta", "delta")]
    if not show_error:
        axis.plot(x, [r["truth_power_mean"] for r in data], color="#222222", lw=1.0, label="Ground truth")
    suffix = "error" if show_error else "power"
    for prefix, method in entries:
        mean = np.array([r[f"{prefix}_{suffix}_mean"] for r in data])
        label, color, _, ls = STYLE[method]
        axis.plot(x, mean, color=color, ls=ls, label=label)
        if show_error:
            std = np.array([r[f"{prefix}_{suffix}_std"] for r in data])
            ok = mean - std > 0
            axis.fill_between(x, np.where(ok, mean - std, np.nan), np.where(ok, mean + std, np.nan),
                              color=color, alpha=0.15, lw=0)
    axis.set_yscale("log")
    axis.set_xlabel(r"Radial wavenumber $|k|/2\pi$")
    axis.grid(alpha=0.2, which="major")


def recon_series(method, key="avg50"):
    table = sorted((r for r in rows(RECON) if r["method"] == method), key=lambda r: r["ratio"])
    good = [r for r in table if isinstance(r[f"{key}_mean_over_training_seeds"], float)]
    return (np.array([r["ratio"] for r in good]),
            np.array([r[f"{key}_mean_over_training_seeds"] for r in good]),
            np.array([r[f"{key}_sample_sd_over_training_seeds"] for r in good]))


def teaser():
    """Figure 1: the problem (excess fine-scale energy), the fix, and why the form matters."""
    spec = rows(SPECTRAL)
    burgers = rows(BURGERS / "aggregate_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    spectrum_panel(axes[0], spec)
    axes[0].set_ylabel("Power at step 50")
    axes[0].set_title(r"(a) Original F-FNO, NS, $\nu=2\nu_0$")
    axes[0].set_ylim(1e-16, 1e9)
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.6, fontsize=6.5)

    for method, source in (("frozen", "increment"), ("delta", "delta_physics")):
        x, m, s = series(burgers, source, "rollout_rel_l2_average")
        plot_method(axes[1], x, m, method, yerr=s)
    axes[1].set_title("(b) Original FNO, Burgers")
    axes[1].set_ylabel("Avg-50 RelL2")
    axes[1].set_ylim(bottom=0)

    for method in ("delta", "lie_pre", "lie_post", "strang", "implicit"):
        x, m, s = recon_series(method)
        plot_method(axes[2], x, m, method, yerr=s)
    axes[2].set_title("(c) Independent FNOs, Burgers")
    axes[2].set_ylabel("Avg-50 RelL2")
    axes[2].legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.5)
    axes[2].set_ylim(0.013, 0.043)
    for axis in axes[1:]:
        axis.axvline(1.0, color="#999999", lw=0.6, ls=":")
        axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
        axis.grid(alpha=0.2)
    axes[2].set_xticks([1, 2, 3, 4])
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_teaser")


def horizon_curves():
    """Original models: improvement of the additive correction grows with shift and horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.85))
    for axis, source, title, ylim in (
            (axes[0], NS, "(a) Original F-FNO, 2D Navier–Stokes", (-1.5, 26)),
            (axes[1], BURGERS, "(b) Original FNO, 1D Burgers", (-3, 100))):
        table = sorted(rows(source / "aggregate_improvements.csv"), key=lambda r: r["viscosity_ratio"])
        x = [r["viscosity_ratio"] for r in table]
        for horizon, color, marker in ((1, "#56B4E9", "o"), (10, "#E69F00", "s"),
                                       (20, "#009E73", "^"), (50, "#D55E00", "D")):
            key = f"improvement_rel_l2_t{horizon}_percent"
            axis.errorbar(x, [r[f"{key}_mean"] for r in table], yerr=[r[f"{key}_std"] for r in table],
                          color=color, marker=marker, capsize=2, label=f"step {horizon}")
        axis.axhline(0, color="#222222", lw=0.6)
        axis.set_xticks(x)
        axis.set_ylim(*ylim)
        axis.set_title(title)
        axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Error reduction vs. frozen (%)")
    axes[1].legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_horizon_curves")


def per_step_curves(ratio=2.0):
    """Mean relative L2 per step (over 3 training seeds x 8 ICs) from the independent study."""
    sums, counts = defaultdict(float), defaultdict(int)
    with RECON_STEPS.open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if float(r["ratio"]) != ratio or r["method"] not in STYLE or r["status"] != "ok":
                continue
            key = (r["method"], int(r["step"]))
            sums[key] += float(r["relative_l2"])
            counts[key] += 1
    curves = {}
    for method in STYLE:
        steps = sorted(s for (m, s) in counts if m == method)
        if steps and all(counts[(method, s)] == 24 for s in steps):
            curves[method] = (np.array(steps), np.array([sums[(method, s)] / 24 for s in steps]))
    return curves


def comparison():
    """Which correction? Per-step error, source-model quality, and exact-source composition error."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0))
    curves = per_step_curves()
    for method in ("delta", "lie_post", "strang", "implicit"):
        steps, err = curves[method]
        label, color, _, ls = STYLE[method]
        axes[0].plot(steps, err, color=color, ls=ls, label=label)
    axes[0].set_title("(a) Independent FNOs: per step")
    axes[0].set_xlabel("Rollout step")
    axes[0].set_ylabel("RelL2")
    axes[0].legend(frameon=False, loc="upper left")

    epochs = rows(EPOCHS)
    for method in ("delta", "lie_post", "strang", "implicit"):
        picked = sorted((r for r in epochs if r["method"] == method), key=lambda r: r["fixed_checkpoint_epoch"])
        plot_method(axes[1], [r["fixed_checkpoint_epoch"] for r in picked],
                    [r["avg50_mean"] for r in picked], method, yerr=[r["avg50_sample_sd"] for r in picked])
    axes[1].set_title("(b) Independent FNOs: training")
    axes[1].set_xlabel("Training epoch of the frozen model")
    axes[1].set_ylabel("Avg-50 RelL2")
    axes[1].set_xticks([15, 30, 50, 75])

    control = [r for r in rows(NSCTRL) if r["grid"] == 128.0]
    for method in ("frozen", "delta", "lie_pre", "lie_post", "strang", "implicit"):
        picked = sorted((r for r in control if r["method"] == method and r["failure_count"] == 0),
                        key=lambda r: r["ratio"])
        plot_method(axes[2], [r["ratio"] for r in picked], [r["avg50_rell2_mean"] for r in picked], method)
    axes[2].set_yscale("log")
    axes[2].set_title(r"(c) Numerical control, NS $128^2$")
    axes[2].set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
    axes[2].set_ylabel("Avg-50 RelL2")
    axes[2].set_xticks([1.5, 2, 4])
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout(w_pad=1.3)
    save(fig, "fig4_comparison")


def spectral_error():
    fig, axis = plt.subplots(figsize=(COLUMN, 1.85))
    spectrum_panel(axis, rows(SPECTRAL), show_error=True)
    axis.set_ylabel("Spectral error per band")
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "figA0_spectral_error")


def two_sided_scan():
    table = sorted(rows(DENSE / "improvements.csv"), key=lambda r: r["viscosity_ratio"])
    x = [r["viscosity_ratio"] for r in table]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1))
    for axis in axes:
        for horizon, color, marker in ((1, "#56B4E9", "o"), (10, "#E69F00", "s"),
                                       (20, "#009E73", "^"), (50, "#D55E00", "D")):
            axis.plot(x, [r[f"improvement_rel_l2_t{horizon}_percent"] for r in table],
                      color=color, marker=marker, label=f"step {horizon}")
        axis.axhline(0, color="#222222", lw=0.6)
        axis.axvspan(min(x), 1.0, color="#D55E00", alpha=0.06)
        axis.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
        axis.grid(alpha=0.2)
    axes[0].set_title(r"(a) Full range (shaded: $\nu<\nu_0$)")
    axes[1].set_title("(b) Zoom near the training viscosity")
    axes[1].set_ylim(-20, 30)
    axes[0].set_ylabel("Error reduction vs. frozen (%)")
    axes[0].legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "figA1_two_sided_scan")


def accuracy_cost():
    recast, cond = rows(RECAST / "aggregate_metrics.csv"), rows(CONDITIONED / "aggregate_metrics.csv")

    def at2(table, method):
        return next(r for r in table if r["method"] == method and r["viscosity_ratio"] == 2.0)

    points = [
        ("Frozen", at2(recast, "increment"), 24, "#666666", "o"),
        (r"Additive ($\Delta$-Physics)", at2(recast, "delta_physics"), 24, "#E69F00", "s"),
        ("Input-Recast (iterative)", at2(recast, "equation_recast"), 24, "#D55E00", "^"),
        ("Conditioned F-FNO (5 viscosities)", at2(cond, "multi_nu_conditioned"), 120, "#CC79A7", "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True)
    for label, r, traj, color, marker in points:
        evals = r.get("mean_nn_evaluations_per_physical_step_mean", 1.0)
        for axis, xval in ((axes[0], traj), (axes[1], evals)):
            axis.errorbar(xval, r["rollout_rel_l2_average_mean"], yerr=r["rollout_rel_l2_average_std"],
                          color=color, marker=marker, ms=5, capsize=2, ls="none", label=label)
    axes[0].set_xlabel("Training trajectories")
    axes[1].set_xlabel("Network evaluations per physical step")
    axes[0].set_ylabel(r"Avg-50 RelL2 at $\nu/\nu_0=2$")
    axes[0].set_title("(a) Training data")
    axes[1].set_title("(b) Test-time compute")
    axes[0].set_ylim(0, 0.27)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="center right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "figA2_accuracy_cost")


if __name__ == "__main__":
    style()
    for stale in ("fig4_spectral_error.pdf", "fig4_spectral_error.png"):
        (OUT / stale).unlink(missing_ok=True)
    teaser()
    horizon_curves()
    comparison()
    spectral_error()
    two_sided_scan()
    accuracy_cost()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
