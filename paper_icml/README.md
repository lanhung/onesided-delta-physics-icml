# ICML-format manuscript (third revision)

`main.tex` is the ICML-format version of the paper, in the ICML 2026 style (a stand-in
until the ICML 2027 style is released; swap `icml2026.sty`/`.bst` when it is).

The third revision reframes the paper as a study of one-call physics corrections for a
frozen neural time-stepper: additive (Δ-Physics), Lie (pre/post), Strang and implicit. It
uses three kinds of evidence and never mixes them in one table:

| Evidence | Source | Used for |
|---|---|---|
| Original models (F-FNO NS, FNO Burgers; 3 seeds) | `../results/` | Table 2, Fig. 1a-b, Sec. 5.1, 5.4, 5.5, App. C-D |
| Independent Burgers FNOs (3 seeds, 75 epochs) | `../supplements/transfer-comparison-2026-11-20/stage2/` | Table 3, Table 4, Fig. 1c, Fig. 3a-b, Sec. 5.2-5.4 |
| Numerical controls (exact source flow) | same supplement, `stage2/ns_controls/` and `stage1/` | Fig. 3c, App. A |

The splitting and implicit corrections have **not** been run on the original checkpoints.
`../run_splitting_baselines.py` does this; it needs the torch environment in
`../requirements-tested.txt`, and its output must go to a new directory.

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (main text, impact statement, appendix) |
| `make_icml_figures.py` | Figures 1, 3, 4-7 from the result CSVs (numpy + matplotlib only, no torch) |
| `make_icml_tables.py` | Tables 2, 3, 4 and 6 from the result CSVs |
| `check_numbers.py` | Flags any number in `main.tex` not found in the old draft, `results/` or the supplement |
| `check_local_error.py` | Numerical check of Proposition 2 (additive, Lie-post, Lie-pre, implicit, Strang; numpy + scipy) |
| `references_extra.bib` | References added in these revisions, each checked online |
| `figures/`, `tables/` | Generated outputs |

Figure 2 (method diagram) is TikZ inside `main.tex`. The appendix reuses the full-horizon and
convergence tables from `../results/paper_assets/`. The bibliography is
`../paper_draft/references.bib` plus `references_extra.bib`.

Build:

```bash
python make_icml_figures.py && python make_icml_tables.py
pdflatex main && bibtex main && pdflatex main && pdflatex main
python check_numbers.py    # expected: only 0.9999 (a rounded bound on cosines from check_local_error.py)
python check_local_error.py
```

The writing follows `.claude/skills/readable-sciml-paper/` (including
`references/exemplar-structure.md`, distilled from ICML/NeurIPS/ICLR neural-PDE papers) and
`.claude/skills/ml-paper-writing/`. Before submission: anonymise the code link, replace the
ICML 2026 style with the 2027 one, and fill in the author block for the camera-ready version.
