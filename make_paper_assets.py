"""Generate publication-ready figures and tables from frozen experiment CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from fno_dual.data import generate_navier_stokes_trajectories
from fno_dual.metrics import radial_power_spectrum
from run_ffno_dense_viscosity_scan import rollout
from run_ffno_equation_recast_3seed import load_reference


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "paper_assets"
NS = ROOT / "results" / "ffno_ns64_high_viscosity_3seed"
BURGERS = ROOT / "results" / "burgers_high_viscosity_3seed"
RECAST = ROOT / "results" / "ffno_ns64_equation_recast_3seed"
CONDITIONED = ROOT / "results" / "ffno_ns64_conditioned_multinu_3seed"
DENSE = ROOT / "results" / "ffno_dense_viscosity_scan"

COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "gray": "#666666",
}


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.9,
        "lines.linewidth": 2.0, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        output = {}
        for key, value in row.items():
            if key in {"method", "baseline"}:
                output[key] = value
            else:
                try:
                    output[key] = float(value)
                except (TypeError, ValueError):
                    output[key] = value
        converted.append(output)
    return converted


def save_figure(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def rounded_box(axis, xy, width, height, text, facecolor, edgecolor, fontsize=10):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5, facecolor=facecolor, edgecolor=edgecolor,
        transform=axis.transAxes,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=fontsize, transform=axis.transAxes,
    )


def arrow(axis, start, end, color="#333333"):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
        color=color, transform=axis.transAxes,
    ))


def figure_1_method():
    fig, axis = plt.subplots(figsize=(11.4, 5.3))
    axis.set_axis_off()
    rounded_box(axis, (0.02, 0.57), 0.13, 0.15,
                "Current state\n" + r"$\omega_t$", "#F2F2F2", COLORS["gray"], 11)
    rounded_box(
        axis, (0.22, 0.64), 0.27, 0.19,
        "Frozen reference time-stepper\n"
        + r"$\widehat\omega^{\rm inc}_{t+1}=\omega_t+F_\theta(\omega_t)$",
        "#DDEBF7", COLORS["blue"], 10,
    )
    rounded_box(axis, (0.56, 0.66), 0.17, 0.15,
                "Reference prediction\n(no retraining)",
                "#DDEBF7", COLORS["blue"], 10)
    rounded_box(axis, (0.02, 0.20), 0.13, 0.15,
                "Known parameters\n" + r"$(\nu,\nu_0)$",
                "#F2F2F2", COLORS["gray"], 10)
    rounded_box(axis, (0.22, 0.19), 0.22, 0.17,
                "One-sided policy\n" + r"$\mathbf{1}_{\nu>\nu_0}$",
                "#FFF2CC", COLORS["orange"], 10)
    rounded_box(axis, (0.50, 0.34), 0.22, 0.13,
                r"$\nu\leq\nu_0$: bypass" + "\n" + r"$C_{\nu,\nu_0}=0$",
                "#F2F2F2", COLORS["gray"], 10)
    rounded_box(
        axis, (0.47, 0.035), 0.33, 0.23,
        r"$\nu>\nu_0$: analytical dissipation" + "\n"
        + r"$C_{\nu,\nu_0}=(S_\nu-S_{\nu_0})\omega_t$" + "\n"
        + r"$S_\nu=\mathcal{F}^{-1}[e^{-\nu|k|^2\Delta t}\mathcal{F}(\cdot)]$" + "\n"
        + r"$k$: physical wavenumber ($k=2\pi n/L$)",
        "#E2F0D9", COLORS["green"], 9.1,
    )
    plus = Circle(
        (0.82, 0.57), 0.037, facecolor="white", edgecolor=COLORS["red"],
        linewidth=1.8, transform=axis.transAxes,
    )
    axis.add_patch(plus)
    axis.text(0.82, 0.57, "+", ha="center", va="center", fontsize=18,
              color=COLORS["red"], transform=axis.transAxes)
    rounded_box(axis, (0.89, 0.49), 0.10, 0.16,
                "Target step\n" + r"$\widehat\omega_{t+1}$",
                "#FCE4D6", COLORS["red"], 11)
    arrow(axis, (0.15, 0.645), (0.22, 0.72))
    arrow(axis, (0.49, 0.735), (0.56, 0.735), COLORS["blue"])
    arrow(axis, (0.73, 0.735), (0.80, 0.60), COLORS["blue"])
    arrow(axis, (0.86, 0.57), (0.89, 0.57), COLORS["red"])
    arrow(axis, (0.15, 0.275), (0.22, 0.275), COLORS["orange"])
    arrow(axis, (0.44, 0.29), (0.50, 0.40), COLORS["gray"])
    arrow(axis, (0.44, 0.24), (0.47, 0.16), COLORS["green"])
    axis.plot([0.145, 0.18, 0.18], [0.57, 0.42, 0.11], color=COLORS["green"],
              linewidth=1.4, transform=axis.transAxes)
    arrow(axis, (0.18, 0.11), (0.47, 0.11), COLORS["green"])
    arrow(axis, (0.72, 0.40), (0.80, 0.56), COLORS["gray"])
    arrow(axis, (0.80, 0.15), (0.81, 0.53), COLORS["green"])
    axis.text(0.5, 0.94, r"One-sided $\Delta$-Physics",
              ha="center", va="center", fontsize=14, weight="bold",
              transform=axis.transAxes)
    axis.text(
        0.5, 0.89,
        "zero-shot analytical viscosity correction  |  "
        "zero additional learned parameters  |  zero retraining  |  "
        r"exact baseline recovery at $\nu=\nu_0$",
        ha="center", va="center", fontsize=9.3, color=COLORS["gray"],
        transform=axis.transAxes,
    )
    save_figure(fig, "figure_1_method")


def _improvement_panel(axis, source, title):
    rows = read_rows(source)
    ratios = [row["viscosity_ratio"] for row in rows]
    specifications = [
        (1, COLORS["blue"], "o"), (10, COLORS["orange"], "s"),
        (20, COLORS["green"], "^"), (50, COLORS["red"], "D"),
    ]
    for horizon, color, marker in specifications:
        key = f"improvement_rel_l2_t{horizon}_percent"
        axis.errorbar(
            ratios, [row[f"{key}_mean"] for row in rows],
            yerr=[row[f"{key}_std"] for row in rows],
            marker=marker, markersize=5.5, color=color, capsize=3,
            label=f"Step {horizon}",
        )
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    axis.axvline(1.0, color="#888888", linewidth=0.9, linestyle="--")
    axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
    axis.set_title(title)
    axis.set_xticks(ratios)
    axis.grid(alpha=0.22)


def figure_2_cross_pde():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=False)
    _improvement_panel(
        axes[0], NS / "aggregate_improvements.csv",
        r"(a) 64$\times$64 Navier–Stokes",
    )
    _improvement_panel(
        axes[1], BURGERS / "aggregate_improvements.csv",
        "(b) Viscous Burgers",
    )
    axes[0].set_ylabel(
        r"One-sided $\Delta$-Physics improvement over Increment (%)"
    )
    axes[1].set_ylabel(r"Improvement over Increment (%)")
    axes[0].set_ylim(-1.5, 26)
    axes[1].set_ylim(-3, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 1.01))
    fig.text(
        0.5, 0.012,
        "Mean ± sample standard deviation over three independent seeds.",
        ha="center", va="bottom", fontsize=9, color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.91))
    save_figure(fig, "figure_2_cross_pde_improvement")


def find_row(rows, method, ratio):
    return next(row for row in rows
                if row["method"] == method
                and math.isclose(row["viscosity_ratio"], ratio))


def mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def figure_4_tradeoff():
    recast = read_rows(RECAST / "aggregate_metrics.csv")
    conditioned = read_rows(CONDITIONED / "aggregate_metrics.csv")
    ratio = 2.0
    conditioned_row = find_row(conditioned, "multi_nu_conditioned", ratio)
    descriptions = [
        ("Increment", "increment", COLORS["blue"], "o", 24, 1.0),
        (r"One-sided $\Delta$-Physics", "delta_physics", COLORS["orange"], "s", 24, 1.0),
        ("Equation-Recast-style", "equation_recast", COLORS["green"], "^", 24,
         find_row(recast, "equation_recast", ratio)
         ["mean_nn_evaluations_per_physical_step_mean"]),
    ]
    points = []
    for label, method, color, marker, trajectories, evaluations in descriptions:
        row = find_row(recast, method, ratio)
        points.append({
            "label": label, "color": color, "marker": marker,
            "error": row["rollout_rel_l2_average_mean"],
            "error_std": row["rollout_rel_l2_average_std"],
            "trajectories": trajectories, "evaluations": evaluations,
        })
    conditioned_label = "Conditioned F-FNO\n" + r"(multi-$\nu$ supervised)"
    points.append({
        "label": conditioned_label, "color": COLORS["purple"], "marker": "D",
        "error": conditioned_row["rollout_rel_l2_average_mean"],
        "error_std": conditioned_row["rollout_rel_l2_average_std"],
        "trajectories": 120, "evaluations": 1.0,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.6), sharey=True)
    data_offsets = {
        "Increment": (18, 20), r"One-sided $\Delta$-Physics": (38, -30),
        "Equation-Recast-style": (75, 5), conditioned_label: (-135, 8),
    }
    compute_offsets = {
        "Increment": (18, 20), r"One-sided $\Delta$-Physics": (22, -30),
        "Equation-Recast-style": (-125, -30), conditioned_label: (9, -7),
    }
    settings = [
        (axes[0], "trajectories", "Training trajectories", "(a) Data cost",
         data_offsets),
        (axes[1], "evaluations", "NN evaluations per physical step",
         "(b) Test-time compute", compute_offsets),
    ]
    for axis, x_key, xlabel, title, offsets in settings:
        for point in points:
            axis.errorbar(
                point[x_key], point["error"], yerr=point["error_std"],
                fmt=point["marker"], markersize=8,
                color=point["color"], capsize=3,
            )
            dx, dy = offsets[point["label"]]
            axis.annotate(
                point["label"],
                (point[x_key], point["error"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", color=point["color"], lw=0.7),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
            )
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[0].set_xlim(15, 132)
    axes[1].set_xlim(0.4, 9.7)
    axes[0].set_ylabel("Average 50-step RelL2")
    fig.suptitle(
        r"Accuracy–cost tradeoff at $\nu/\nu_0=2$ (64$\times$64 NS)"
    )
    fig.tight_layout()
    save_figure(fig, "figure_4_accuracy_cost")


def _spectral_mechanism_rows():
    cache = OUT / "figure_3_spectral_data.csv"
    if cache.exists():
        return read_rows(cache)
    args = SimpleNamespace(
        width=16, modes=8, depth=8, factor=4,
        reference_viscosity=1e-3, step_dt=0.05,
    )
    device = torch.device("cpu")
    collected = {name: [] for name in ("Ground truth", "Increment", "One-sided Delta-Physics")}
    for seed in (0, 1, 2):
        trajectories = generate_navier_stokes_trajectories(
            8, 64, 2e-3, 50, 0.05, 0.005, 1000 + seed, device,
        )
        model, input_normalizer, increment_normalizer = load_reference(
            seed, NS, args, device,
        )
        increment = rollout(
            model, trajectories[:, 0], 50, "increment", 2e-3, args,
            input_normalizer, increment_normalizer,
        )[:, -1]
        delta = rollout(
            model, trajectories[:, 0], 50, "delta_physics", 2e-3, args,
            input_normalizer, increment_normalizer,
        )[:, -1]
        target = trajectories[:, 50]
        for name, fields in (
            ("Ground truth", target), ("Increment", increment),
            ("One-sided Delta-Physics", delta),
        ):
            collected[name].append(radial_power_spectrum(fields, n_bins=32).cpu())
    spectra = {name: torch.cat(values).numpy() for name, values in collected.items()}
    truth = spectra["Ground truth"]
    truth_norm = np.linalg.norm(truth, axis=1, keepdims=True).clip(min=1e-12)
    max_mode = math.sqrt(32.0 ** 2 + 32.0 ** 2)
    rows = []
    for band in range(32):
        row = {"radial_bin": band, "mode_radius": (band + 0.5) * max_mode / 32.0}
        for name, prefix in (
            ("Ground truth", "truth"), ("Increment", "increment"),
            ("One-sided Delta-Physics", "delta"),
        ):
            values = spectra[name][:, band]
            row[f"{prefix}_power_mean"] = float(values.mean())
            row[f"{prefix}_power_std"] = float(values.std(ddof=1))
            if name != "Ground truth":
                errors = np.abs(values - truth[:, band]) / truth_norm[:, 0]
                row[f"{prefix}_error_mean"] = float(errors.mean())
                row[f"{prefix}_error_std"] = float(errors.std(ddof=1))
        rows.append(row)
    csv_table(cache, rows)
    return rows


def figure_3_spectral_mechanism():
    rows = _spectral_mechanism_rows()
    x = np.array([row["mode_radius"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.35))
    for label, prefix, color in (
        ("Ground truth", "truth", "#222222"),
        ("Increment", "increment", COLORS["blue"]),
        (r"One-sided $\Delta$-Physics", "delta", COLORS["orange"]),
    ):
        mean = np.array([row[f"{prefix}_power_mean"] for row in rows])
        std = np.array([row[f"{prefix}_power_std"] for row in rows])
        axes[0].plot(x, mean, label=label, color=color)
        axes[0].fill_between(x, np.maximum(mean - std, 1e-12), mean + std,
                             color=color, alpha=0.14, linewidth=0)
    for label, prefix, color in (
        ("Increment", "increment", COLORS["blue"]),
        (r"One-sided $\Delta$-Physics", "delta", COLORS["orange"]),
    ):
        mean = np.array([row[f"{prefix}_error_mean"] for row in rows])
        std = np.array([row[f"{prefix}_error_std"] for row in rows])
        axes[1].plot(x, mean, label=label, color=color)
        axes[1].fill_between(x, np.maximum(mean - std, 1e-12), mean + std,
                             color=color, alpha=0.14, linewidth=0)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("(a) Radial power spectrum")
    axes[1].set_title("(b) Spectral-error contribution")
    axes[0].set_ylabel("Radial mean power")
    axes[1].set_ylabel(r"$|E_{pred}(k)-E_{true}(k)|/\|E_{true}\|_2$")
    for axis in axes:
        axis.set_xlabel(r"Radial mode $|k|/(2\pi)$")
        axis.grid(alpha=0.22, which="both")
        axis.legend(frameon=True)
    fig.suptitle(
        r"64$\times$64 NS at $\nu/\nu_0=2$, rollout Step 50 "
        r"(mean $\pm$ sample standard deviation; 24 trajectories pooled "
        r"across three seeds)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "figure_3_ns_spectral_mechanism")


def figure_s1_dense_two_sided():
    rows = read_rows(DENSE / "improvements.csv")
    ratios = [row["viscosity_ratio"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharex=True)
    specifications = (
        (1, COLORS["blue"], "o"), (10, COLORS["orange"], "s"),
        (20, COLORS["green"], "^"), (50, COLORS["red"], "D"),
    )
    for axis in axes:
        for horizon, color, marker in specifications:
            values = [row[f"improvement_rel_l2_t{horizon}_percent"] for row in rows]
            axis.plot(
                ratios, values, marker=marker, color=color,
                label=f"Step {horizon}",
            )
        axis.axhline(0, color="#222222", linewidth=0.9)
        axis.axvline(1, color="#888888", linewidth=0.9, linestyle="--")
        axis.axvspan(min(ratios), 1, color=COLORS["red"], alpha=0.06)
        axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
        axis.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        axis.tick_params(axis="x", labelrotation=0)
        axis.grid(alpha=0.22)
    axes[0].set_title("(a) Full range: low-viscosity instability")
    axes[1].set_title("(b) Boundary zoom")
    axes[0].set_ylabel(r"RelL2 improvement over Increment (%)")
    axes[1].set_ylim(-20, 30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Why one-sided? Dense viscosity ablation", y=1.08,
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, "figure_s1_dense_two_sided_ablation")


def formatted(mean, std):
    return f"{mean:.5f} +/- {std:.5f}"


def formatted_cost(mean, std):
    return f"{mean:.2f} +/- {std:.2f}"


def metric_record(row, display_method, ratio):
    return {
        "nu/nu0": ratio, "Method": display_method,
        "Step 1 RelL2": formatted(row["rel_l2_t1_mean"], row["rel_l2_t1_std"]),
        "Step 10 RelL2": formatted(row["rel_l2_t10_mean"], row["rel_l2_t10_std"]),
        "Step 20 RelL2": formatted(row["rel_l2_t20_mean"], row["rel_l2_t20_std"]),
        "Step 50 RelL2": formatted(row["rel_l2_t50_mean"], row["rel_l2_t50_std"]),
        "Avg-50 RelL2": formatted(
            row["rollout_rel_l2_average_mean"],
            row["rollout_rel_l2_average_std"],
        ),
        "Avg-50 PSRE": formatted(
            row["rollout_psre_average_mean"], row["rollout_psre_average_std"]
        ),
        "Final PSRE": formatted(
            row["rollout_psre_final_mean"], row["rollout_psre_final_std"]
        ),
    }


def compact_metric_record(row, display_method, ratio, regime=None):
    record = {}
    if regime is not None:
        record["Supervision regime"] = regime
    record.update({
        "nu/nu0": ratio, "Method": display_method,
        "Step-50 RelL2": formatted(row["rel_l2_t50_mean"], row["rel_l2_t50_std"]),
        "Avg-50 RelL2": formatted(
            row["rollout_rel_l2_average_mean"], row["rollout_rel_l2_average_std"]
        ),
        "Avg-50 PSRE": formatted(
            row["rollout_psre_average_mean"], row["rollout_psre_average_std"]
        ),
        "Final PSRE": formatted(
            row["rollout_psre_final_mean"], row["rollout_psre_final_std"]
        ),
    })
    return record


def csv_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    fields = list(rows[0])
    output = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    output.extend(
        "| " + " | ".join(str(row[field]) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(output) + "\n"


def latex_escape(value):
    return str(value).replace("_", r"\_").replace("+/-", r"$\pm$")


def latex_table(rows, caption, label):
    if label in {"tab:ns64_main", "tab:burgers_main", "tab:data_compute"}:
        return readable_main_table(rows, caption, label)
    if label in {"tab:ns64_absolute", "tab:burgers_absolute"}:
        return readable_full_table(rows, caption, label)
    fields = list(rows[0])
    alignment = "rl" + "c" * (len(fields) - 2)
    lines = [
        r"\begin{table*}[!htbp]", r"\centering", r"\small",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}", r"\toprule",
        " & ".join(r"$\nu/\nu_0$" if field == "nu/nu0" else latex_escape(field)
                   for field in fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(latex_escape(row[field]) for field in fields) + r" \\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}%", r"}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def readable_full_table(rows, caption, label):
    names = {"One-sided Delta-Physics": r"$\Delta$-Physics",
             "Equation-Recast-style": "Recast-style", "Conditioned F-FNO": "Cond. F-FNO"}
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.12}"]
    for title, fields in (
        ("(a) Horizon-specific relative errors", ["Step 1 RelL2", "Step 10 RelL2", "Step 20 RelL2", "Step 50 RelL2"]),
        ("(b) Average rollout and spectral errors", ["Avg-50 RelL2", "Avg-50 PSRE", "Final PSRE"]),
    ):
        lines += [r"\textit{" + title + r"}\par\smallskip",
                  r"\begin{tabular}{@{}rl" + "c" * len(fields) + r"@{}}", r"\toprule",
                  r"$\nu/\nu_0$ & Method & " + " & ".join(fields) + r" \\", r"\midrule"]
        for row in rows:
            cells = [str(row["nu/nu0"]), names.get(row["Method"], row["Method"])]
            for field in fields:
                mean, std = (v.strip() for v in row[field].split("+/-"))
                cells.append(f"${mean}\\pm{std}$")
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}\par\medskip"]
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def readable_main_table(rows, caption, label):
    """Typeset at an explicit readable font size; never scale the main tables."""
    names = {
        "One-sided Delta-Physics": r"$\Delta$-Physics",
        "Equation-Recast-style": "Recast-style",
        "Conditioned F-FNO": "Cond. F-FNO",
    }
    cost = label == "tab:data_compute"
    fields = (["Method", "Training trajectories", "Training pairs",
               "Optimizer time / seed (s)", "NN evals / step", "Inference time (s)"]
              if cost else [f for f in rows[0] if f != "Supervision regime"])
    headers = (["Method", r"\shortstack{Train\\trajectories}",
                r"\shortstack{Train\\pairs}", r"\shortstack{Optimizer\\time (s)}",
                r"\shortstack{NN evals\\/ step}", r"\shortstack{Inference\\time (s)}"]
               if cost else [r"$\nu/\nu_0$", "Method",
                             r"\shortstack{Step-50\\RelL2}",
                             r"\shortstack{Avg-50\\RelL2}",
                             r"\shortstack{Avg-50\\PSRE}", "Final PSRE"])
    lines = [r"\begin{table*}[!t]", r"\centering", r"\normalsize",
             r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.17}",
             r"\begin{tabular}{@{}" + ("lccccc" if cost else "rlcccc") + r"@{}}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    previous_group = None
    previous_ratio = None
    for row in rows:
        group = row.get("Supervision regime")
        if group and group != previous_group:
            if previous_group:
                lines.append(r"\midrule")
            text = (r"Single-$\nu$, zero-shot" if group.startswith("Single")
                    else r"Multi-$\nu$, supervised (target viscosities included)")
            lines.append(r"\multicolumn{6}{l}{\textit{" + text + r"}} \\")
            previous_group, previous_ratio = group, None
        ratio = row.get("nu/nu0")
        if not cost and previous_ratio is not None and ratio != previous_ratio:
            lines.append(r"\addlinespace[3pt]")
        cells = []
        for field in fields:
            value = row[field]
            if field == "Method":
                cells.append(names.get(value, value))
            elif not cost and "+/-" in str(value):
                mean, std = (float(v) for v in value.split("+/-"))
                cells.append(f"${mean:.4f}\\pm{std:.4f}$")
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
        previous_ratio = ratio
    caption += (r" Abbreviations: $\Delta$-Physics denotes One-sided $\Delta$-Physics; "
                r"Recast-style denotes Equation-Recast-style; Cond. F-FNO denotes "
                r"Conditioned F-FNO." if label != "tab:burgers_main" else
                r" $\Delta$-Physics denotes One-sided $\Delta$-Physics.")
    if cost:
        caption += (r" The first three methods use one training viscosity and 28,370 "
                    r"parameters, with no target-parameter training data; Cond. F-FNO "
                    r"uses five viscosities, target-parameter training data, and 28,386 "
                    r"parameters. Optimizer time excludes data generation.")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}",
              rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def table_1_ns():
    recast = read_rows(RECAST / "aggregate_metrics.csv")
    conditioned = read_rows(CONDITIONED / "aggregate_metrics.csv")
    rows = []
    methods = [
        ("Increment", recast, "increment"),
        ("One-sided Delta-Physics", recast, "delta_physics"),
        ("Equation-Recast-style", recast, "equation_recast"),
        ("Conditioned F-FNO", conditioned, "multi_nu_conditioned"),
    ]
    for ratio in (1.0, 1.5, 1.75, 2.0):
        for display, source, method in methods:
            rows.append(metric_record(find_row(source, method, ratio), display, ratio))
    csv_table(OUT / "table_1_ns64_absolute.csv", rows)
    (OUT / "table_1_ns64_absolute.md").write_text(
        markdown_table(rows), encoding="utf-8"
    )
    (OUT / "table_1_ns64_absolute.tex").write_text(
        latex_table(
            rows,
            r"Absolute 64$\times$64 Navier--Stokes rollout errors "
            r"(mean $\pm$ sample standard deviation over three seeds). "
            r"Conditioned F-FNO uses five training viscosities; the remaining "
            r"methods use only $\nu_0$ training data.",
            "tab:ns64_absolute",
        ), encoding="utf-8",
    )
    main_rows = []
    for regime, regime_methods in (
        ("Single-nu, zero-shot", methods[:3]),
        ("Multi-nu, supervised", methods[3:]),
    ):
        for ratio in (1.0, 1.5, 1.75, 2.0):
            for display, source, method in regime_methods:
                main_rows.append(compact_metric_record(
                    find_row(source, method, ratio), display, ratio, regime
                ))
    csv_table(OUT / "table_1_ns64_main.csv", main_rows)
    (OUT / "table_1_ns64_main.md").write_text(
        markdown_table(main_rows), encoding="utf-8"
    )
    (OUT / "table_1_ns64_main.tex").write_text(
        latex_table(
            main_rows,
            r"Main 64$\times$64 Navier--Stokes results (mean $\pm$ sample "
            r"standard deviation over three seeds). Supervision regimes are "
            r"shown explicitly: the zero-shot methods use only $\nu_0$, whereas "
            r"Conditioned F-FNO uses five viscosities including all test values.",
            "tab:ns64_main",
        ), encoding="utf-8",
    )


def table_2_burgers():
    source = read_rows(BURGERS / "aggregate_metrics.csv")
    rows = []
    for ratio in (1.0, 1.25, 1.5, 1.75, 2.0):
        for display, method in (
            ("Increment", "increment"),
            ("One-sided Delta-Physics", "delta_physics")
        ):
            rows.append(metric_record(find_row(source, method, ratio), display, ratio))
    csv_table(OUT / "table_2_burgers_absolute.csv", rows)
    (OUT / "table_2_burgers_absolute.md").write_text(
        markdown_table(rows), encoding="utf-8"
    )
    (OUT / "table_2_burgers_absolute.tex").write_text(
        latex_table(
            rows,
            r"Absolute viscous Burgers rollout errors "
            r"(mean $\pm$ sample standard deviation over three seeds).",
            "tab:burgers_absolute",
        ), encoding="utf-8",
    )
    main_rows = []
    for ratio in (1.0, 1.25, 1.5, 1.75, 2.0):
        for display, method in (
            ("Increment", "increment"),
            ("One-sided Delta-Physics", "delta_physics"),
        ):
            main_rows.append(compact_metric_record(
                find_row(source, method, ratio), display, ratio
            ))
    csv_table(OUT / "table_2_burgers_main.csv", main_rows)
    (OUT / "table_2_burgers_main.md").write_text(
        markdown_table(main_rows), encoding="utf-8"
    )
    (OUT / "table_2_burgers_main.tex").write_text(
        latex_table(
            main_rows,
            r"Main viscous Burgers results (mean $\pm$ sample standard "
            r"deviation over three seeds).",
            "tab:burgers_main",
        ), encoding="utf-8",
    )


def table_3_cost():
    recast = read_rows(RECAST / "aggregate_metrics.csv")
    costs = read_rows(CONDITIONED / "costs.csv")
    reference_seeds = read_rows(NS / "seed_results.csv")
    reference_train = mean_std([
        next(row for row in reference_seeds if int(row["seed"]) == seed)["train_seconds"]
        for seed in (0, 1, 2)
    ])
    conditioned_train = mean_std([row["train_seconds_conditioned"] for row in costs])
    conditioned_inference = mean_std([
        row["inference_seconds_conditioned_all_tests"] / 4.0 for row in costs
    ])
    rows = []
    for display, method, evaluations in (
        ("Increment", "increment", 1.0),
        ("One-sided Delta-Physics", "delta_physics", 1.0),
        ("Equation-Recast-style", "equation_recast",
         find_row(recast, "equation_recast", 2.0)
         ["mean_nn_evaluations_per_physical_step_mean"]),
    ):
        metric = find_row(recast, method, 2.0)
        rows.append({
            "Method": display, "Target-parameter training data?": "No",
            "Training viscosities": 1,
            "Training trajectories": 24, "Training pairs": 480,
            "Optimizer time / seed (s)": formatted_cost(
                reference_train[0], reference_train[1]
            ),
            "Parameters": "28,370", "NN evals / step": f"{evaluations:.2f}",
            "Inference time (s)": formatted_cost(
                metric["wall_clock_seconds_mean"],
                metric["wall_clock_seconds_std"],
            ),
        })
    rows.append({
        "Method": "Conditioned F-FNO", "Target-parameter training data?": "Yes",
        "Training viscosities": 5,
        "Training trajectories": 120, "Training pairs": 2400,
        "Optimizer time / seed (s)": formatted_cost(
            conditioned_train[0], conditioned_train[1]
        ),
        "Parameters": "28,386", "NN evals / step": "1.00",
        "Inference time (s)": formatted_cost(
            conditioned_inference[0], conditioned_inference[1]
        ),
    })
    csv_table(OUT / "table_3_data_compute.csv", rows)
    (OUT / "table_3_data_compute.md").write_text(
        markdown_table(rows), encoding="utf-8"
    )
    (OUT / "table_3_data_compute.tex").write_text(
        latex_table(
            rows,
            r"Data and compute comparison. Inference time is measured on CPU "
            r"for one viscosity, eight trajectories, and 50 rollout steps at "
            r"$\nu/\nu_0=2$. One-sided $\Delta$-Physics and Equation-Recast-style reuse the "
            r"Increment checkpoint and require no additional training.",
            "tab:data_compute",
        ), encoding="utf-8",
    )


def table_s1_recast_convergence():
    diagnostics = read_rows(RECAST / "iteration_diagnostics.csv")
    aggregate = read_rows(RECAST / "aggregate_metrics.csv")
    rows = []
    for ratio in (1.0, 1.5, 1.75, 2.0):
        selected = [row for row in diagnostics
                    if math.isclose(row["viscosity_ratio"], ratio)]
        iterations = np.array([row["iterations"] for row in selected])
        evaluations = np.array([row["nn_evaluations"] for row in selected])
        nonconverged = sum(str(row["converged"]).lower() != "true" for row in selected)
        metric = find_row(aggregate, "equation_recast", ratio)
        rows.append({
            "nu/nu0": ratio,
            "Mean iterations": f"{iterations.mean():.2f}",
            "P95 iterations": f"{np.percentile(iterations, 95):.1f}",
            "Max iterations": int(iterations.max()),
            "Mean NN evals / step": f"{evaluations.mean():.2f}",
            "Non-converged steps": nonconverged,
            "Inference time (s)": formatted_cost(
                metric["wall_clock_seconds_mean"], metric["wall_clock_seconds_std"]
            ),
        })
    csv_table(OUT / "table_s1_recast_convergence.csv", rows)
    (OUT / "table_s1_recast_convergence.md").write_text(
        markdown_table(rows), encoding="utf-8"
    )
    (OUT / "table_s1_recast_convergence.tex").write_text(
        latex_table(
            rows,
            r"Equation-Recast-style fixed-point diagnostics at tolerance "
            r"$10^{-5}$ and at most 30 iterations. Counts pool three seeds, "
            r"eight trajectories, and 50 physical steps per viscosity.",
            "tab:recast_convergence",
        ), encoding="utf-8",
    )


def write_index():
    content = """# Paper-ready figures and tables

All figures are provided as 320-DPI PNG, vector PDF, and editable SVG.
Tables are provided as CSV, Markdown, and LaTeX (booktabs + graphicx).

## Main-text figures

1. `figure_1_method`: One-sided Delta-Physics overview. The policy depends only
   on known `(nu, nu0)`; the current state separately enters the frozen
   time-stepper and analytical correction.
2. `figure_2_cross_pde_improvement`: NS/Burgers improvement curves with separate
   y-axis ranges, mean +/- sample standard deviation over three seeds.
3. `figure_3_ns_spectral_mechanism`: Step-50 radial spectrum and spectral-error
   contribution for 64 x 64 NS at `nu/nu0=2` (24 trajectories pooled across
   three seeds).
4. `figure_4_accuracy_cost`: accuracy versus training trajectories and NN
   evaluations per physical step at `nu/nu0=2`.

## Main-text tables

1. `table_1_ns64_main`: compact NS results grouped by single-viscosity zero-shot
   versus multi-viscosity supervised regimes.
2. `table_2_burgers_main`: compact Burgers results.
3. `table_3_data_compute`: target-parameter supervision, data, optimizer time,
   parameters, NN evaluations, and CPU inference time.

## Supplementary assets

- `figure_s1_dense_two_sided_ablation`: full-range instability and boundary zoom
  explaining the one-sided switch.
- `table_1_ns64_absolute` and `table_2_burgers_absolute`: full horizons.
- `table_s1_recast_convergence`: fixed-point convergence diagnostics.
- `figure_3_spectral_data.csv`: cached per-bin statistics used by Figure 3.

Recommended first-use name: **One-sided Delta-Physics, a zero-shot analytical
viscosity correction.** In LaTeX prose, render Delta as `\\Delta`.

All reported uncertainty is sample standard deviation over three seeds.
"""
    (OUT / "README.md").write_text(content, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    figure_1_method()
    figure_2_cross_pde()
    figure_3_spectral_mechanism()
    figure_4_tradeoff()
    figure_s1_dense_two_sided()
    table_1_ns()
    table_2_burgers()
    table_3_cost()
    table_s1_recast_convergence()
    write_index()
    print(f"Wrote publication assets to {OUT}")
    return
    improvement_figure(
        NS / "aggregate_improvements.csv", "figure_2_ns_improvement",
        r"64$\times$64 Navier–Stokes",
    )
    improvement_figure(
        BURGERS / "aggregate_improvements.csv",
        "figure_3_burgers_improvement", "Viscous Burgers",
    )
    figure_4_tradeoff()
    table_1_ns()
    table_2_burgers()
    table_3_cost()
    write_index()
    print(f"Wrote publication assets to {OUT}")


if __name__ == "__main__":
    main()
