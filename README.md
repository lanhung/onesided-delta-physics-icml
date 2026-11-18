# One-sided Δ-Physics

Research artifacts accompanying the current manuscript in [paper_draft/main.tex](paper_draft/main.tex) ([PDF](paper_draft/main.pdf)).

One-sided Δ-Physics is a zero-shot analytical viscosity correction for a frozen Increment time-stepper:

\[
u_{t+1}=u_t+F_\theta(u_t)+\mathbf{1}_{\nu>\nu_0}(S_\nu-S_{\nu_0})u_t,
\qquad \widehat{S_\nu u}(k)=e^{-\nu |k|^2\Delta t}\hat u(k).
\]

The correction adds **zero additional learned parameters**, uses one neural evaluation per physical step, and exactly preserves the reference predictor when bypassed. It does not guarantee energy decay or global stability of the complete neural predictor.

## Contents

| Path | Purpose |
|---|---|
| `fno_dual/ffno.py`, `fno_dual/fno1d.py` | NS factorized F-FNO and Burgers 1D backbone |
| `fno_dual/physics.py` | Analytical diffusion and relative correction |
| `fno_dual/data.py` | PDE solvers and deterministic synthetic trajectory generation |
| `fno_dual/metrics.py`, `normalizer.py` | RelL2, PSRE, rollout metrics and normalization |
| `run_ffno_high_viscosity_3seed.py` | Single-viscosity NS training and paired evaluation |
| `run_burgers_high_viscosity_3seed.py` | Burgers training and paired evaluation |
| `run_ffno_conditioned_multinu_3seed.py` | Multi-viscosity supervised baseline |
| `run_ffno_equation_recast_3seed.py` | Dynamic-step Equation-Recast-style baseline |
| `run_splitting_baselines.py` | Not yet run: compares Δ-Physics with Lie, Strang, implicit and explicit one-evaluation corrections on the existing frozen checkpoints (NS and Burgers); see its docstring |
| `run_ffno_dense_viscosity_scan.py` | Two-sided low-resolution diagnostic (supplement); its reference checkpoint is `results/ffno_physics_split_screen/increment.pt` |
| `results/` | Configurations, seed-level and aggregate measurements, checkpoints, normalizers, diagnostics |
| `results/paper_assets/` | Main/supplementary figures (PDF/SVG/PNG), tables (TeX/CSV/Markdown), spectral statistics |
| `paper_draft/` | Manuscript, bibliography and compiled PDF |
| `paper_icml/` | Reader-friendly ICML-format rewrite of the manuscript, with torch-free figure/table scripts (see its README) |
| `.claude/skills/` | Paper-writing skills used for the rewrite (`ml-paper-writing`, vendored under MIT; `readable-sciml-paper`) |
| `export_datasets.py` | Export the exact configured synthetic train/validation/test datasets |
| `MANIFEST.sha256` | Integrity hashes for the publication snapshot |

Some shared modules retain historical model utilities because the frozen experiment scripts import them. They are not additional components of the final method. Third-party repository clones and unrelated exploratory experiments are excluded.

## Environment and tests

Python 3.10 or later; install `python -m pip install -r requirements.txt`. The tested versions are pinned separately in `requirements-tested.txt` (Python 3.10.11).
Run `python -m pytest tests -q` from the repository root. CPU experiments used four Torch threads; timings are hardware dependent. See `environment.json` for the export environment, when available.

## Reproduce the experiments

The following commands **train new models**. Use different output directories if preserving the bundled snapshot. The NS script defaults to a smaller grid, so the explicit `--grid 64` is essential.

```bash
python run_ffno_high_viscosity_3seed.py --out results/ffno_ns64_high_viscosity_3seed --grid 64 --seeds 0 1 2 --epochs 15 --viscosity-ratios 1 1.5 1.75 2
python run_burgers_high_viscosity_3seed.py
python run_ffno_conditioned_multinu_3seed.py
```

Evaluate the frozen NS checkpoints without retraining (also measures Increment and Δ-Physics):

```bash
python run_ffno_equation_recast_3seed.py --tolerance 1e-5 --max-iterations 30
python run_ffno_dense_viscosity_scan.py
```

The Recast-style baseline is our adaptation to a dynamic Increment predictor, **not an official implementation or exact reproduction of a source-to-solution method**. Its per-step convergence diagnostics and neural-call counts are included.

Regenerate figures/tables and build the manuscript:

```bash
python make_paper_assets.py
cd paper_draft
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Data and experimental interpretation

The original runners generate synthetic trajectories in memory rather than saving raw arrays. All original evaluation data (CSV/JSON), model weights and normalization statistics are included. `export_datasets.py` materializes trajectory arrays with the same solver parameters and RNG seeds; its outputs are regenerated data, not retrospectively claimed original cached files. The runners themselves remain the authoritative reproduction path.

The exported [trajectory archive](data/trajectories) contains 55 compressed NPZ files covering NS64, Burgers and the supplementary NS24 scan. [Validation results](data/validation.json) confirm finite arrays and matching NS reference normalization statistics (standard deviations identical; means differ only at floating-point reduction precision). See [data documentation](data/README.md) for shapes, splits and provenance.

The low-level `relative_diffusion_correction` function and the dense-scan helper deliberately implement a **two-sided** semigroup difference, enabling the supplementary ablation. For deployment of the final one-sided method, call this correction only when `nu > nu0`; otherwise return the Increment prediction directly. The formal main experiments test only `nu >= nu0`.

NS: 64×64, seeds 0/1/2, reference viscosity 0.001, 24 training and 6 validation trajectories per seed, 20 training transitions, 8 test trajectories per viscosity, 50-step autoregressive evaluation. Single-viscosity training uses 480 pairs and 15 epochs. Test viscosity ratios: 1, 1.5, 1.75, 2.

The conditioned baseline uses five training viscosities (ratios 1, 1.25, 1.5, 1.75, 2), 120 trajectories and 2,400 pairs per seed. Every reported test viscosity is supervised during its training. It also receives more optimizer updates; this is **not** an equal-data/equal-compute comparison. The conditioned baseline is substantially more accurate. Δ-Physics improves its corresponding frozen single-viscosity baseline; it does not replace multi-viscosity supervision at equal accuracy.

Burgers uses a 1D backbone, reference viscosity 0.01 and 64 spatial points. See each result directory's `config.json` for solver time steps, architecture and all remaining settings. Error bars are sample standard deviations over three seeds. The dense two-sided scan is a separate single-seed, 24×24 diagnostic; it is not a three-seed NS64 result.

The additional `ffno_ns64_budget75_seed0` experiment extends single-viscosity training to 75 epochs on the same data. It is a **single-seed sensitivity check, not part of the manuscript's main tables**. Its learning-rate schedule changes with the epoch budget. See its `analysis.md` and `analyze_ffno_training_budget.py`; do not interpret singleton aggregate standard deviations as three-seed evidence.

To repeat that sensitivity check:

```bash
python run_ffno_high_viscosity_3seed.py --out results/ffno_ns64_budget75_seed0 --seeds 0 --grid 64 --epochs 75 --viscosity-ratios 1 1.5 1.75 2
python analyze_ffno_training_budget.py
```

Check the downloaded snapshot with `python verify_artifacts.py`. The archived manuscript and original experimental outputs are preserved; the root README, dataset exporter and verification utility were added for publication packaging. The supplementary scan's reference checkpoint lives under the historical directory name `ffno_physics_split_screen`, but only its Increment weights and configuration are included.

The repository contains the current working manuscript, not a claim of peer review or acceptance. No software license is assigned by this export; reuse permissions remain with the authors.
