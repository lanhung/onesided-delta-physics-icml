# Publication export validation

- All 16 bundled tests passed (`python -m pytest tests -q`). The only warning was the existing PyTorch `weight_norm` deprecation warning.
- The copied manuscript compiled successfully with both pdfLaTeX and XeLaTeX, with BibTeX and two subsequent LaTeX passes. The final XeLaTeX log had no undefined or multiply defined references. The original compiled manuscript PDF is retained rather than replaced by the validation build.
- Original model implementations, experiment runners, checkpoints, normalization files, result tables and figures were copied from the research workspace; no training was performed for this export.
- Raw trajectories were not cached by the original runners. Publication-export arrays are regenerated from the same solvers, configurations and random seeds. Their integrity and normalization checks are recorded separately in `data/validation.json`.
- `MANIFEST.sha256` inventories the released files. Run `python verify_artifacts.py` after downloading a complete snapshot. Original export text files use LF line endings for portable checksum verification; the later supplement preserves its archived line endings through scoped Git attributes.

This validates artifact packaging and reproducibility dependencies, not scientific novelty or independent replication of the reported experiments.

## Supplement import — 2026-11-20

The package under `supplements/transfer-comparison-2026-11-20/` preserves two previously executed research snapshots: 19 stage-1 files and 364 stage-2 files, including their original manifests. All 381 manifest-listed payload hashes matched before and after copying. The user-provided manuscript reviewed in those snapshots is also archived, because its contents differ from `paper_draft/main.pdf`.

Two focused checks were rerun from the imported code's new location: all 168 analytical PyTorch CPU cases passed, and all 9 unified-runner integration checks passed without skips. These checks validate the correction implementation and evaluation plumbing; they do not rerun the reported model training, target-parameter accuracy experiments, or timing benchmarks. No GPU execution or original-checkpoint reproduction was performed during this import. Details are in the supplement's `import_verification.json`.

The imported scientific files retain their original bytes and historical execution metadata. A current navigation layer explains which results are independent neural experiments, numerical controls, or future plans. `MANIFEST.sha256` now includes this supplement as well as the original repository files. The original manuscript, model implementations, weights, data exports, and result files were not modified by this import.
