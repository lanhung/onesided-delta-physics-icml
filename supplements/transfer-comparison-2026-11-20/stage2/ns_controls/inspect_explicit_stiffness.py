"""Linear high-mode amplification for the specified explicit correction control.

Above the nonlinear dealiasing cutoff, the reference solver evolves only the
linear Laplacian (forcing has low modes). The explicit wrapper reintroduces the
stiff parameter difference explicitly at the much larger output time step.
"""
import json
from pathlib import Path
import numpy as np


def main():
    out = Path(__file__).resolve().parent / "results"
    config = json.loads((out / "config.json").read_text())
    rows = []
    h, nu0, source_dt = config["h"], config["nu0"], config["source_dt"]
    internal_steps = int(np.ceil(h / source_dt - 1e-12))
    dt = h / internal_steps
    for n in config["grids"]:
        k = 2 * np.pi * np.fft.fftfreq(n, 1 / n)
        q = k[:, None] ** 2 + k[None, :] ** 2
        a = h * nu0 * q
        d = dt * nu0 * q
        rk = (1 - d + d ** 2 / 2 - d ** 3 / 6 + d ** 4 / 24) ** internal_steps
        m = np.fft.fftfreq(n) * n
        above_cutoff = ~((np.abs(m[:, None]) < n / 3) & (np.abs(m[None, :]) < n / 3))
        for ratio in config["ratios"]:
            z = h * (ratio - 1) * nu0 * q
            exact_reference_amplification = np.exp(-a) - z
            rk_reference_amplification = rk - z
            rows.append({"grid": n, "ratio": ratio,
                "maximum_mode_z_h_delta_nu_q": float(z.max()),
                "max_abs_explicit_amplification_exact_heat_reference": float(np.max(np.abs(exact_reference_amplification))),
                "max_abs_explicit_amplification_RK4_high_modes": float(np.max(np.abs(rk_reference_amplification[above_cutoff]))),
                "linearly_unstable_high_modes_present": bool(np.any(np.abs(rk_reference_amplification[above_cutoff]) > 1)),
                "interpretation": "A finite 50-step run does not establish stability; roundoff or predictor errors can seed growing high modes."})
    payload = {"evidence_type": "analytic_diagnostic_for_numerical_reference_control",
               "formula": "g(q) = R4(-nu0*q*dt)^n_internal - h*(nu-nu0)*q",
               "R4": "1+z+z^2/2+z^3/6+z^4/24", "rows": rows,
               "caution": "This quantifies modal instability, not an exact prediction of nonlinear failure time or a result for trained networks."}
    (out / "explicit_stiffness.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
