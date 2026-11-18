# ICML-format manuscript (reader-friendly rewrite)

`main.tex` is a rewrite of `../paper_draft/main.tex` for an ML audience, in the ICML
2026 style (a stand-in until the ICML 2027 style is released; swap `icml2026.sty`/`.bst`
when it is). The results are unchanged: every number comes from `../results/`.

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (main text, impact statement, appendix) |
| `make_icml_figures.py` | Figures 1, 3, 4, A1, A2 from the result CSVs (numpy + matplotlib only, no torch) |
| `make_icml_tables.py` | Tables 1 and 2 from the result CSVs |
| `check_numbers.py` | Flags any number in `main.tex` not found in the old draft or `results/` |
| `check_local_error.py` | Numerical check of Proposition 2 (local error of additive, Lie and Strang corrections; numpy + scipy) |
| `references_extra.bib` | References added in this revision (classical splitting; Gopakumar et al. 2026), verified online |
| `figures/`, `tables/` | Generated outputs |

Figure 2 (method diagram) is TikZ inside `main.tex`. The appendix reuses the full-horizon and
convergence tables from `../results/paper_assets/`. The bibliography is `../paper_draft/references.bib` plus `references_extra.bib`.

Build:

```bash
python make_icml_figures.py && python make_icml_tables.py
pdflatex main && bibtex main && pdflatex main && pdflatex main
python check_numbers.py
```

The writing follows `.claude/skills/readable-sciml-paper/` and `.claude/skills/ml-paper-writing/`.
Before submission: anonymise the code link, replace the ICML 2026 style with the 2027 one,
and fill in the author block for the camera-ready version.
