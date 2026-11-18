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
\caption{{\textbf{{Same frozen network, corrected vs.\ uncorrected.}} Error after 50 autoregressive steps
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
        ("Frozen network (Increment)", recast, "increment", "Increment"),
        (r"\quad + One-sided $\Delta$-Physics (ours)", recast, "delta_physics", "One-sided Delta-Physics"),
        (r"\quad + Recast-style fixed-point iteration", recast, "equation_recast", "Equation-Recast-style"),
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
\caption{{\textbf{{What do stronger alternatives cost?}} 2D Navier--Stokes at $\nu/\nu_0=2$.
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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "tab_main.tex").write_text(table_main(), encoding="utf-8")
    (OUT / "tab_alternatives.tex").write_text(table_alternatives(), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
