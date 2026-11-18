---
name: readable-sciml-paper
description: Rewrite or draft a scientific-ML / neural-PDE paper so that a general AI/ML audience (ICML, NeurIPS, ICLR, TMLR) can follow it - motivation first, a running example, intuition before equations, honest claims. Use with ml-paper-writing for venue rules and citation checks.
---

# Reader-friendly scientific-ML papers

Scientific-ML papers are often written for applied mathematicians: definitions, then
lemmas, then experiments. ML reviewers read differently. They read title, abstract,
Figure 1 and introduction, and decide there. This skill turns a correct but
"mathematical-style" draft into one an AI reader can follow without losing rigor.

It distills advice from Neel Nanda (narrative), Sebastian Farquhar (abstract),
Gopen & Swan (reader expectations), Zachary Lipton (word choice) and Ethan Perez
(micro-clarity). The full sources and venue checklists are in the companion
`ml-paper-writing` skill (`references/writing-guide.md`, `references/checklists.md`).

## 0. Ground rules

- **Never change a number.** Every value in the text, tables and figures comes from
  the result files (CSV/JSON) in the repo. Regenerate tables with a script, do not
  retype them. After editing, re-check every number in the prose against those files.
- **Never add a citation from memory.** Reuse entries already in the `.bib`; add a new
  one only after finding the paper online, otherwise leave `% TODO verify citation`.
- **Keep claims inside the evidence.** Rewriting for readability must not widen what is
  claimed. If a result is single-seed, a different resolution, or a self-implemented
  baseline, say so where the result is stated, not only in the appendix.

## 1. Compress the paper into a narrative first

Before touching LaTeX, write these down (they become the abstract and the intro):

1. **Setting the reader recognises** - one or two sentences, concrete, no
   "deep learning has revolutionised...". If the first sentence could open any ML
   paper, delete it.
2. **The pain** - what goes wrong today, shown with a number or a picture.
3. **The key insight in plain words** - one sentence a non-specialist can repeat.
4. **The method in one line** - ideally one equation or a 5-line algorithm box.
5. **Evidence** - the 2-3 results that support the claims, with headline numbers.
6. **Scope** - where it does not apply.

If you cannot state the contribution in one sentence, stop and settle that first.

## 2. Use a running example

Pick one concrete scenario and return to it in the intro, method and experiments.
For a parameter-transfer paper: "a surrogate trained on a fluid at viscosity nu_0,
asked about the same fluid at 2 nu_0". Each new concept should be explained on that
example before it is stated in general. A tiny worked case where the method is
exactly right (e.g. the pure linear PDE) is the best bridge from intuition to math.

## 3. Figure 1 sells the problem and the fix

- Prefer a *result-style* teaser (the failure, and the method removing it) or a very
  simple schematic. One message per figure.
- No titles or slogans baked into the image; that text belongs in the caption.
- Captions stand alone: say what to look at and what it shows
  ("the dashed line stays flat: the corrected error no longer grows with the shift").
- Readable in grayscale; markers plus colours; vector PDF; no literal LaTeX escapes
  (`\%`, `--`) in matplotlib labels when usetex is off.

## 4. Abstract (Farquhar 5 sentences)

What we do -> why it is hard/important -> how (with searchable keywords) ->
evidence -> the most striking number, stated with its conditions.

## 5. Introduction (about 1-1.5 pages in two columns)

Context and pain (with the running example) -> why existing routes are costly
(organised by *what extra resource* they need) -> key insight -> method in one
sentence -> headline evidence -> 3 contribution bullets, each a claim plus where it
is supported -> one sentence of scope. Readers should be able to stop here.

## 6. Method: intuition, then formalism, then properties

- Open each subsection with what the piece does and why, in words.
- Introduce notation only when it is used; keep one name per concept for the whole
  paper (make a terminology table for yourself and stick to it).
- Give an algorithm box for anything executed at inference time.
- State properties as short, named propositions; move one-line proofs to the appendix.
- Explain design choices with a physical or geometric picture before the inequality
  (e.g. "anti-diffusion is the heat equation run backwards: it sharpens noise").

## 7. Experiments as questions

Title each subsection with the question it answers ("Does the gain grow with the
shift?", "Is it fixing the mechanism we think?", "What do stronger alternatives
cost?"). For each: the claim, the setting (details to the appendix), what to look at
in the figure/table, and the takeaway. Report seeds and what the error bars are.
Put "what this does *not* show" next to the result when relevant.

## 8. Related work after the experiments (ML style)

Organise by approach, not paper by paper, and end each paragraph with how this work
differs. Name the closest prior/concurrent work explicitly and say plainly what is and
is not new relative to it.

## 9. Limitations and impact

A dedicated limitations section (required by most venues) that pre-empts the obvious
reviewer objections. ICML also needs an Impact Statement (does not count to the limit).

## 10. Sentence-level pass

- Subject and verb close together; new or important information at the end of the sentence.
- Verbs, not nominalisations ("we analyse", not "we perform an analysis").
- Replace "this"/"it" with "this result", "this correction".
- Delete hedges and intensifiers unless they carry meaning; but do not delete
  genuine caveats - move them to the place they apply instead of the contribution list.
- Prefer short paragraphs with one point each.

## 11. Final checks

- Page limit of the target venue, anonymised author block, no identifying URLs.
- Compile cleanly (no undefined references or citations, no overfull figure boxes).
- Look at every rendered page; check every number in the prose against the result files.
