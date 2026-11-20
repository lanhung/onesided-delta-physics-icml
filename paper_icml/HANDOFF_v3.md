# Handoff: manuscript revision v3 (2026-11-20)

Branch `readable-icml-rewrite` of `lanhung/onesided-delta-physics-icml` (PR #1).
Reference snapshot: `lanhung/main` at `a9288a0` (verified supplement), merged into this branch
at `90461b6` without resetting. `verify_artifacts.py` still verifies all 598 listed artifacts,
so no archived report, checkpoint, array or manifest entry was changed.

## 1. What changed in the paper

- **Framing.** The paper is now a study of one-call physics corrections for a frozen neural
  time-stepper: additive (Δ-Physics), Lie-pre, Lie-post, Strang and implicit, with explicit
  Euler and input-Recast as reference points. Title kept.
- **Structure** follows the exemplar papers in
  `.claude/skills/readable-sciml-paper/references/exemplar-structure.md`: running example in
  the introduction, problem setting with an explicit information budget, method with a
  table of update rules and a diagram, results organised as questions (does it help, which
  form, does it depend on the model, what does it cost, lower viscosity), related work with
  run-in headings, a limitations list, impact statement.
- **Theory.** Proposition 1 no longer claims "safe by construction" or that the correction
  never adds energy; Appendix A gives the one-mode counterexample (energy 0.405 → 0.66125).
  Proposition 2 now covers additive, Lie-post, Lie-pre, implicit and Strang, each checked by
  `check_local_error.py`. The model-error propagation terms are stated without claiming a
  ranking.
- **Evidence kept separate.** Original models (Table 2, Fig. 1a-b), independent Burgers
  FNOs (Table 3, Table 4, Fig. 1c, Fig. 3a-b) and numerical controls (Fig. 3c) never share
  a table. Every caption names its evidence type.
- **Honest comparisons.** Strang is best on Avg-50 (28.40% below additive at ratio 2) but
  additive has the lowest mean Step-50 (0.030129 vs 0.030860); both are reported. The CPU
  benchmark shows the direct spectral solver is faster (42.5 ms vs 66-70 ms) and far more
  accurate; this is stated in the abstract.
- **Recast.** Input-Recast is described as our adaptation, distinct from the closed-form
  implicit correction; no Recast accuracy is attributed to the implicit form, and the
  original repeated network calls are not described as an error.

Main body: 7 pages (limit 8), plus impact statement, references and appendix (15 pages total).

## 2. Evidence inventory

| Evidence | What ran | Location |
|---|---|---|
| Original models | F-FNO (NS 64²) and FNO (Burgers 64), 3 seeds, 15 epochs: frozen, additive, input-Recast, conditioned F-FNO; one 75-epoch NS seed; one 24² two-sided scan | `results/` |
| Independent Burgers FNOs | 3 seeds, 75 epochs, shared dataset: frozen, additive, explicit, Lie-pre, Lie-post, Strang, implicit; ratios 1-4; epoch 15/30/50/75 snapshots; CPU timing | `supplements/transfer-comparison-2026-11-20/stage2/` |
| Numerical controls | NS 64² and 128² with exact source flow, ratios 1.5/2/4, 3 ICs; 6-D local-order control | `.../stage2/ns_controls/`, `.../stage1/` |
| Analytical checks | Taylor expansions of Prop. 2 (5 forms) | `paper_icml/check_local_error.py` |
| **Planned, not run** | Splitting/implicit corrections on the original checkpoints; native 128² neural; 64→128 transfer; hyperviscosity; reaction-diffusion | `run_splitting_baselines.py`, `.../stage2/next_experiments.json` |

## 3. What ran in this session, and what did not

Ran (CPU, no torch): figure and table generation from the archived CSVs, the local-error
check, the number check, and the LaTeX build. **No new experiment was run**, because torch
is not installed in this workspace and package installs are blocked.

Not run, in priority order:

- **P0 — original-checkpoint comparison.** Without it, the claim "Strang beats additive"
  rests on the independent Burgers FNOs and the numerical controls only. This is the first
  thing a reviewer will ask for.
- **P1 — larger or harder problems** (native 128² neural model, stiffer ICs), where the
  direct solver is no longer cheap. Without this the paper has no setting in which a
  corrected network beats the solver on cost.
- **P1 — a PDE-residual fine-tuning baseline** (PINO-style test-time optimisation), the
  closest zero-data alternative.
- **P2 — more seeds with separate datasets**, hyperviscosity, reaction-diffusion.

## 4. Reproducible commands

```bash
# Verify the archive (expects "Verified 598 artifacts.")
python verify_artifacts.py

# Rebuild figures, tables and the PDF (numpy, scipy, matplotlib, TeX Live)
cd paper_icml
python make_icml_figures.py && python make_icml_tables.py
pdflatex main && bibtex main && pdflatex main && pdflatex main
python check_numbers.py       # expected: only 0.9999 (rounded cosine bound)
python check_local_error.py   # Prop. 2 expansions

# P0, needs torch (requirements-tested.txt). Write to new directories only.
cd ..
python run_splitting_baselines.py --pde ns \
    --out experiments/2026-11_original_splitting/ns64_15epoch_3seed
python run_splitting_baselines.py --pde ns --checkpoint-dir results/ffno_ns64_budget75_seed0 \
    --seeds 0 --out experiments/2026-11_original_splitting/ns64_75epoch_seed0
python run_splitting_baselines.py --pde burgers \
    --out experiments/2026-11_original_splitting/burgers64_15epoch_3seed
```

The script checks that its frozen and additive rows reproduce the saved
`seed_results.csv` of each checkpoint directory; if that check fails, stop and investigate
before using the new rows.

## 5. Manuscript recommendations once P0 has run

- If Strang (or another splitting form) also beats additive on the original F-FNO: add a
  row group to Table 2 or a new main-text table labelled "Original models", and update the
  second contribution bullet and abstract to say the result holds on both model families.
- If it does not: say so in Section 5.2 and the abstract, and frame the splitting result as
  model-dependent. Do not merge the two model families into one table either way.
- Keep the direct-solver comparison in the abstract until a setting exists where the
  corrected network is cheaper.
- Swap in the ICML 2027 style file when released, and anonymise the code link.

## 6. Commits in this revision

1. Skill reference: exemplar structure distilled from ICML/NeurIPS/ICLR neural-PDE papers.
2. Checks: Prop. 2 check extended to Lie-pre and implicit; number check extended to the supplement.
3. Figures, tables and references for the one-call comparison (all generated from archived CSVs).
4. Manuscript v3, README and this handoff.
