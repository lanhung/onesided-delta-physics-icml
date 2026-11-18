"""Figures for the ICML-format manuscript (paper_icml/).

Reads only the committed result files (CSV); needs numpy and matplotlib, not torch.
Every plotted value comes from ../results/. Run from the repository root or from
paper_icml/:

    python paper_icml/make_icml_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
NS = RES / "ffno_ns64_high_viscosity_3seed"
BURGERS = RES / "burgers_high_viscosity_3seed"
RECAST = RES / "ffno_ns64_equation_recast_3seed"
CONDITIONED = RES / "ffno_ns64_conditioned_multinu_3seed"
DENSE = RES / "ffno_dense_viscosity_scan"
SPECTRAL = RES / "paper_assets" / "figure_3_spectral_data.csv"

# Okabe-Ito colour-blind-safe palette (same hues as the original assets).
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#666666")
FULL, COLUMN = 6.75, 3.25  # ICML text and column widths in inches


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.4, "lines.markersize": 3.8,
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
    picked = sorted((r for r in table if r["method"] == method),
                    key=lambda r: r["viscosity_ratio"])
    return (np.array([r["viscosity_ratio"] for r in picked]),
            np.array([r[f"{key}_mean"] for r in picked]),
            np.array([r[f"{key}_std"] for r in picked]))


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def spectrum_panel(axis, data, show_error=False):
    x = np.array([r["mode_radius"] for r in data])
    entries = [("Increment (frozen)", "increment", BLUE, "--"),
               (r"One-sided $\Delta$-Physics", "delta", ORANGE, "-")]
    if not show_error:
        entries.insert(0, ("Ground truth", "truth", "#222222", "-"))
    suffix = "error" if show_error else "power"
    for label, prefix, color, ls in entries:
        mean = np.array([r[f"{prefix}_{suffix}_mean"] for r in data])
        std = np.array([r[f"{prefix}_{suffix}_std"] for r in data])
        axis.plot(x, mean, color=color, ls=ls, label=label,
                  lw=1.1 if prefix == "truth" else 1.4)
        if show_error:  # +/-1 s.d. band, drawn only where the lower edge is positive
            lower = mean - std
            ok = lower > 0
            axis.fill_between(x, np.where(ok, lower, np.nan), np.where(ok, mean + std, np.nan),
                              color=color, alpha=0.15, lw=0)
    axis.set_yscale("log")
    axis.set_xlabel(r"Radial wavenumber $|k|/2\pi$")
    axis.grid(alpha=0.2, which="major")


def teaser():
    """Figure 1: the failure (excess fine-scale energy) and the fix (flat error)."""
    ns, bu = rows(NS / "aggregate_metrics.csv"), rows(BURGERS / "aggregate_metrics.csv")
    spec = rows(SPECTRAL)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.95),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    spectrum_panel(axes[0], spec)
    axes[0].set_ylabel("Power at step 50")
    axes[0].set_title(r"(a) NS spectrum at $\nu=2\nu_0$, step 50")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.8)
    for axis, table, title in ((axes[1], ns, "(b) 2D Navier–Stokes"),
                               (axes[2], bu, "(c) 1D viscous Burgers")):
        for method, label, color, marker, ls in (
                ("increment", "Increment (frozen)", BLUE, "o", "--"),
                ("delta_physics", r"One-sided $\Delta$-Physics", ORANGE, "s", "-")):
            x, m, s = series(table, method, "rollout_rel_l2_average")
            axis.errorbar(x, m, yerr=s, color=color, marker=marker, ls=ls,
                          capsize=2, label=label)
        axis.axvline(1.0, color=GRAY, lw=0.6, ls=":")
        axis.set_xlabel(r"Test / training viscosity $\nu/\nu_0$")
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    axes[1].set_ylabel("Avg-50 RelL2 (lower is better)")
    axes[2].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig1_teaser")


def horizon_curves():
    """Figure 3: improvement grows with the viscosity shift and the rollout horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95))
    for axis, source, title, ylim in (
            (axes[0], NS, "(a) 2D Navier–Stokes (F-FNO)", (-1.5, 26)),
            (axes[1], BURGERS, "(b) 1D viscous Burgers (FNO)", (-3, 100))):
        table = sorted(rows(source / "aggregate_improvements.csv"),
                       key=lambda r: r["viscosity_ratio"])
        x = [r["viscosity_ratio"] for r in table]
        for horizon, color, marker in ((1, BLUE, "o"), (10, ORANGE, "s"),
                                       (20, GREEN, "^"), (50, RED, "D")):
            key = f"improvement_rel_l2_t{horizon}_percent"
            axis.errorbar(x, [r[f"{key}_mean"] for r in table],
                          yerr=[r[f"{key}_std"] for r in table], color=color,
                          marker=marker, capsize=2, label=f"step {horizon}")
        axis.axhline(0, color="#222222", lw=0.6)
        axis.set_xticks(x)
        axis.set_ylim(*ylim)
        axis.set_title(title)
        axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Error reduction vs. frozen model (%)")
    axes[1].legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_horizon_curves")


def spectral_error():
    """Figure 4: where in the spectrum the error is removed."""
    fig, axis = plt.subplots(figsize=(COLUMN, 1.85))
    spectrum_panel(axis, rows(SPECTRAL), show_error=True)
    axis.set_ylabel("Spectral error per band")
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_spectral_error")


def two_sided_scan():
    """Appendix figure: the symmetric correction fails for nu < nu0."""
    table = sorted(rows(DENSE / "improvements.csv"), key=lambda r: r["viscosity_ratio"])
    x = [r["viscosity_ratio"] for r in table]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1))
    for axis in axes:
        for horizon, color, marker in ((1, BLUE, "o"), (10, ORANGE, "s"),
                                       (20, GREEN, "^"), (50, RED, "D")):
            axis.plot(x, [r[f"improvement_rel_l2_t{horizon}_percent"] for r in table],
                      color=color, marker=marker, label=f"step {horizon}")
        axis.axhline(0, color="#222222", lw=0.6)
        axis.axvspan(min(x), 1.0, color=RED, alpha=0.06)
        axis.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
        axis.grid(alpha=0.2)
    axes[0].set_title(r"(a) Full range (shaded: $\nu<\nu_0$)")
    axes[1].set_title("(b) Zoom near the training viscosity")
    axes[1].set_ylim(-20, 30)
    axes[0].set_ylabel("Error reduction vs. frozen model (%)")
    axes[0].legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "figA1_two_sided_scan")


def accuracy_cost():
    """Appendix figure: accuracy against training data and test-time compute."""
    recast, cond = rows(RECAST / "aggregate_metrics.csv"), rows(CONDITIONED / "aggregate_metrics.csv")

    def at2(table, method):
        return next(r for r in table if r["method"] == method and r["viscosity_ratio"] == 2.0)

    points = [
        ("Increment (frozen)", at2(recast, "increment"), 24, BLUE, "o"),
        (r"One-sided $\Delta$-Physics", at2(recast, "delta_physics"), 24, ORANGE, "s"),
        ("Recast-style (iterative)", at2(recast, "equation_recast"), 24, GREEN, "^"),
        ("Conditioned F-FNO (5 viscosities)", at2(cond, "multi_nu_conditioned"), 120, PURPLE, "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True)
    for label, r, traj, color, marker in points:
        evals = r.get("mean_nn_evaluations_per_physical_step_mean", 1.0)
        for axis, xval in ((axes[0], traj), (axes[1], evals)):
            axis.errorbar(xval, r["rollout_rel_l2_average_mean"],
                          yerr=r["rollout_rel_l2_average_std"], color=color,
                          marker=marker, ms=5, capsize=2, ls="none", label=label)
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
    teaser()
    horizon_curves()
    spectral_error()
    two_sided_scan()
    accuracy_cost()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
