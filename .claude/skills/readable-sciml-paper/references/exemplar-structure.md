# How strong recent ML-for-PDE papers are structured

Notes from reading ICML/NeurIPS/ICLR papers closest to this project (November 2026).
Use them as a template; do not copy text.

## Papers read for structure

| Paper | Venue | What to borrow |
|---|---|---|
| Serrano et al., *Test-time Generalization for Physics through Neural Operator Splitting* | ICML 2026 | Problem Setting section with a displayed equation, "where ..." definitions, a concrete advection-diffusion example, and an explicit statement of the test-time information budget |
| Takamoto et al., *CAPE: Learning Neural PDE Solvers with Parameter-Guided Channel Attention* | ICML 2023 | Intro that names what existing methods need (architecture changes, parameter inputs) before the proposal |
| Zhou et al., *Unisolver* | ICML 2025 | Figure 1 that places the work between two paradigms; contributions as three bullets |
| Yin et al., *LEADS* | NeurIPS 2021 | Explicit research questions (RQ1, RQ2) stated in the introduction and answered in the experiments |
| Yin et al., *APHYNITY* | ICLR 2021 | Decomposition "known physics + learned part"; Figure 1 contrasts data-only, physics-only and hybrid |
| Lippe et al., *PDE-Refiner* | NeurIPS 2023 | "We analyse X and find Y, so we propose Z": an analysis result motivates the method |
| Brandstetter et al., *Message Passing Neural PDE Solvers* | ICLR 2022 | Background section on classical solvers before the learned one |
| McCabe et al., *Towards Stability of Autoregressive Neural Operators* | TMLR 2023 | Diagnosing error growth in rollouts before fixing it |

## Recurring template

1. **Title**: method name plus a plain claim, or a plain claim alone.
2. **Abstract** (5-7 sentences): setting, the failure, the insight, what we do, headline evidence, the honest qualifier.
3. **Introduction**:
   - Paragraph 1: the setting and the failure, with one concrete example.
   - Paragraph 2: existing remedies, grouped by what they need (target data, retraining, test-time optimisation), and their limits.
   - Paragraph 3: the insight in plain words.
   - Paragraph 4: what we do and the research question(s).
   - Paragraph 5: headline results with numbers.
   - Contributions: 3-5 bullets starting with a verb ("We formulate", "We show", "We find").
4. **Problem setting**: displayed equation, "where" sentence, one worked example, the information available at test time stated as a sentence a reviewer can check.
5. **Method / analysis**: intuition paragraph, the update rule (a small table works when there are several variants), named propositions, proofs in the appendix.
6. **Experiments**: an "Experimental setting" subsection with run-in headings (*Datasets*, *Models*, *Baselines*, *Metrics*), then one subsection per question or scenario, each with a *Setting* and a *Results* paragraph.
7. **Related work**: run-in bold paragraph headings, grouped by approach; the last sentence of each paragraph says how this work differs. Placed either after the introduction or before the conclusion; ICML papers use both.
8. **Limitations**: increasingly a separate section or a clearly labelled paragraph; ICML 2026 papers also include an Impact Statement.

## Style observations

- First person plural; present tense for what the paper shows.
- Sentences in these papers often run to 25-35 words. For readability, aim for 15-25 words, and break long sentences at "which" and "so that".
- One idea per paragraph; the first sentence states it.
- Tables: methods as rows, settings as columns, best value in bold, "mean ± s.d. over N seeds" in the caption.
- Captions start with a bold one-line message, then say what to look at.
