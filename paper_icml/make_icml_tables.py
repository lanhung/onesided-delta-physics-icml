"""Tables for the ICML-format manuscript, generated from ../results/ (no retyping)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "tables"


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get(table, method, ratio, key):
    for r in table:
        if r["method"] == method and abs(float(r["viscosity_ratio"]) - ratio) < 1e-9:
            return float(r[f"{key}_mean"]), float(r[f"{key}_std"])
    return None


def fmt(value, bold=False):
    if value is None:
        return "--"
    text = f"{value[0]:.4f}\\std{{{value[1]:.4f}}}"
    return f"\\textbf{{{value[0]:.4f}}}\\std{{{value[1]:.4f}}}" if bold else text


def pair(table, ratio, key):
    inc, dlt = get(table, "increment", ratio, key), get(table, "delta_physics", ratio, key)
    if inc is None:
        return ["--", "--"]
    better = dlt[0] < inc[0] - 1e-12
    return [fmt(inc), fmt(dlt, bold=better)]


def table_main():
    ns = rows(RES / "ffno_ns64_high_viscosity_3seed" / "aggregate_metrics.csv")
    bu = rows(RES / "burgers_high_viscosity_3seed" / "aggregate_metrics.csv")
    lines = []
    for ratio in (1.0, 1.25, 1.5, 1.75, 2.0):
        cells = [f"{ratio:g}"]
        cells += pair(ns, ratio, "rollout_rel_l2_final") + pair(ns, ratio, "rollout_psre_average")
        cells += pair(bu, ratio, "rollout_rel_l2_final") + pair(bu, ratio, "rollout_psre_average")
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)
    return rf"""\begin{{table*}}[t]
\caption{{\textbf{{Original models: the additive correction on the frozen network.}} Error after 50 autoregressive steps
(Step-50 RelL2) and average spectral error over the rollout (Avg-50 PSRE); lower is better.
Mean\,$\pm$\,sample standard deviation over three independently trained seeds; bold marks the better
of the two columns. At $\nu=\nu_0$ the two methods are identical by construction. Navier--Stokes was not
evaluated at $\nu/\nu_0=1.25$. Full horizon-wise results are in \cref{{app:full_tables}}.}}
\label{{tab:main}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{tabular}}{{@{{}}c cc cc cc cc@{{}}}}
\toprule
& \multicolumn{{4}}{{c}}{{2D Navier--Stokes, $64\times64$ (F-FNO)}} & \multicolumn{{4}}{{c}}{{1D viscous Burgers, 64 points (FNO)}} \\
\cmidrule(lr){{2-5}} \cmidrule(l){{6-9}}
& \multicolumn{{2}}{{c}}{{Step-50 RelL2}} & \multicolumn{{2}}{{c}}{{Avg-50 PSRE}} & \multicolumn{{2}}{{c}}{{Step-50 RelL2}} & \multicolumn{{2}}{{c}}{{Avg-50 PSRE}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}} \cmidrule(l){{8-9}}
$\nu/\nu_0$ & Frozen & $+\Delta$ & Frozen & $+\Delta$ & Frozen & $+\Delta$ & Frozen & $+\Delta$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def table_alternatives():
    recast = rows(RES / "ffno_ns64_equation_recast_3seed" / "aggregate_metrics.csv")
    cond = rows(RES / "ffno_ns64_conditioned_multinu_3seed" / "aggregate_metrics.csv")
    cost = {r["Method"]: r for r in rows(RES / "paper_assets" / "table_3_data_compute.csv")}
    spec = [
        (r"Frozen F-FNO", recast, "increment", "Increment"),
        (r"\quad + additive ($\Delta$-Physics)", recast, "delta_physics", "One-sided Delta-Physics"),
        (r"\quad + input-Recast (iterative)", recast, "equation_recast", "Equation-Recast-style"),
        (r"Conditioned F-FNO (trained on all 5 $\nu$)", cond, "multi_nu_conditioned", "Conditioned F-FNO"),
    ]
    lines = []
    for label, table, method, cost_key in spec:
        c = cost[cost_key]
        inf = c["Inference time (s)"].replace(" +/- ", r"\std{") + "}"
        cells = [label, c["Target-parameter training data?"], c["Training trajectories"],
                 c["NN evals / step"], inf,
                 fmt(get(table, method, 2.0, "rollout_rel_l2_average")),
                 fmt(get(table, method, 2.0, "rollout_rel_l2_final"))]
        lines.append(" & ".join(cells) + r" \\")
        if method == "equation_recast":
            lines.append(r"\midrule")
    body = "\n".join(lines)
    return rf"""\begin{{table*}}[t]
\caption{{\textbf{{Original F-FNO: costlier alternatives.}} 2D Navier--Stokes at $\nu/\nu_0=2$.
The first three rows reuse the same frozen checkpoint (24 training trajectories at $\nu_0$, 28,370 parameters);
the conditioned model is trained on five viscosities including every test value (28,386 parameters) and also
receives proportionally more optimizer updates, so it is an upper reference rather than an equal-budget competitor.
Inference: CPU wall-clock for 8 trajectories $\times$ 50 steps. Mean\,$\pm$\,s.d.\ over three seeds.}}
\label{{tab:alternatives}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}l c c c c c c@{{}}}}
\toprule
Method & \shortstack{{Target-$\nu$\\data?}} & \shortstack{{Train\\traj.}} & \shortstack{{NN evals\\/ step}} & \shortstack{{Inference\\time (s)}} & Avg-50 RelL2 & Step-50 RelL2 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


SUP = ROOT / "supplements" / "transfer-comparison-2026-11-20" / "stage2"
RECON_LABELS = [("frozen", "Frozen FNO"), ("delta", "Additive"), ("lie_pre", "Lie, pre"),
                ("lie_post", "Lie, post"), ("strang", "Strang"), ("implicit", "Implicit, post"),
                ("explicit", "Explicit")]


def table_recon(ratio=2.0):
    """Independent Burgers FNOs (three newly trained models): all one-call corrections."""
    agg = {r["method"]: r for r in rows(SUP / "tables" / "burgers_fno_results.csv")
           if abs(float(r["ratio"]) - ratio) < 1e-9}
    seeds = rows(SUP / "burgers_fno" / "results" / "per_seed_summary.csv")

    def seed_value(seed, method):
        r = next(x for x in seeds if int(x["training_seed"]) == seed and x["method"] == method
                 and abs(float(x["ratio"]) - ratio) < 1e-9)
        return float(r["avg50_rel_l2"]) if r["avg50_rel_l2"] not in ("", "nan") else None

    complete = [a for a in agg.values() if a["avg50_mean_over_training_seeds"] != ""]
    best_avg = min(float(a["avg50_mean_over_training_seeds"]) for a in complete)
    best_step = min(float(a["step50_mean_over_training_seeds"]) for a in complete)
    lines = []
    for method, label in RECON_LABELS:
        a = agg[method]
        failed = int(float(a["failed_ics_total_over_seed_ic_pairs"]))
        if a["avg50_mean_over_training_seeds"] == "":
            lines.append(f"{label} & \\multicolumn{{2}}{{c}}{{diverged}} & -- & {failed}/24 \\\\")
            continue
        avg = (float(a["avg50_mean_over_training_seeds"]), float(a["avg50_sample_sd_over_training_seeds"]))
        step = (float(a["step50_mean_over_training_seeds"]), float(a["step50_sample_sd_over_training_seeds"]))
        if method in ("frozen", "delta"):
            gain = "--"
        else:
            changes = [100.0 * (seed_value(s, "delta") - seed_value(s, method)) / seed_value(s, "delta")
                       for s in range(3)]
            mean = sum(changes) / 3
            sd = (sum((c - mean) ** 2 for c in changes) / 2) ** 0.5
            gain = f"{mean:.1f}\\std{{{sd:.1f}}}"
        lines.append(" & ".join([label, fmt(avg, bold=abs(avg[0] - best_avg) < 1e-12),
                                 fmt(step, bold=abs(step[0] - best_step) < 1e-12), gain,
                                 f"{failed}/24"]) + r" \\")
        if method == "delta":
            lines.append(r"\midrule")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\caption{{\textbf{{Independent FNOs: which one-call correction?}} Burgers, $\nu/\nu_0={ratio:g}$, three newly trained
FNOs (75 epochs, source-only checkpoint selection), eight paired test trajectories each. RelL2 error; mean\,$\pm$\,s.d.\ over the three
training seeds, which share one source dataset. Gain: within-seed reduction of Avg-50 relative to the additive
correction, averaged over seeds. Fail: diverged seed--trajectory pairs. These are not the original checkpoints of
\cref{{tab:main}}.}}
\label{{tab:recon}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2.6pt}}
\begin{{tabular}}{{@{{}}l c c c c@{{}}}}
\toprule
Correction & Avg-50 & Step-50 & Gain (\%) & Fail \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def sci(value):
    exponent = int(f"{value:e}".split("e")[1])
    return f"${value / 10 ** exponent:.2f}\\times10^{{{exponent}}}$"


def table_cpu():
    """CPU latency of one-call corrections versus a direct spectral solver (independent seed 0)."""
    data = {r["method"]: r for r in rows(SUP / "tables" / "burgers_cpu_timing.csv")
            if r["torch_threads"] == "1" and r["batch_size"] == "8"}
    spec = [("frozen", "Frozen FNO", "1"), ("delta", r"Additive ($\Delta$-Physics)", "1"),
            ("lie_post", "Lie, post", "1"), ("strang", "Strang", "1"),
            ("implicit", "Implicit, post", "1"), ("rk4_dt0.002", r"Spectral RK4, $\delta t=0.002$", "0")]
    lines = []
    for method, label, calls in spec:
        r = data[method]
        median = 1000 * float(r["median_seconds_per_50step_rollout"])
        iqr = 1000 * float(r["iqr_seconds"])
        avg = float(r["avg50_rel_l2"])
        avg_text = f"{avg:.4f}" if avg > 1e-3 else sci(avg)
        if method.startswith("rk4"):
            lines.append(r"\midrule")
        lines.append(f"{label} & {calls} & {median:.1f} ({iqr:.1f}) & {avg_text} \\\\")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\caption{{\textbf{{CPU cost on a small grid.}} Independent Burgers FNO, seed 0, 64 points, $\nu/\nu_0=2$, batch of 8,
50 steps, one CPU thread. Median wall-clock time and IQR over ten timed runs; training, data loading and metrics
excluded. Avg-50 is for this one model, so it differs from the three-seed means in \cref{{tab:recon}}.}}
\label{{tab:cpu}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}l c c c@{{}}}}
\toprule
Method & \shortstack{{NN calls\\/ step}} & Time (ms) & Avg-50 RelL2 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "tab_main.tex").write_text(table_main(), encoding="utf-8")
    (OUT / "tab_alternatives.tex").write_text(table_alternatives(), encoding="utf-8")
    (OUT / "tab_recon.tex").write_text(table_recon(), encoding="utf-8")
    (OUT / "tab_cpu.tex").write_text(table_cpu(), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
