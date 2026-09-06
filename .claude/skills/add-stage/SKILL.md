---
name: add-stage
description: Add a new stage to the study (dataset build, model selection, fine-tuning, explainability, LLM baseline, the pipeline). Use when implementing any roadmap item in README.md, or whenever a new result needs to reach a reader. Covers the six pieces every stage ships and the evidence rules a published number has to satisfy.
---

# Adding a stage

Every stage of this study ships the same six pieces. A stage missing any of them
is not finished, because a reader either cannot reach the result or cannot check
it.

## The six pieces

1. **Committed inputs** under `benchmarks/<stage>/` — the data the result derives
   from. Small, text, readable. If the raw data is too large or is gone, commit
   the summary statistics instead and follow "When the raw data is gone" below.
2. **A module** under `src/emotion_timeline/<stage>/`, with a module docstring
   that explains the non-obvious decision rather than restating the code.
3. **A CLI subcommand** in `cli.py` that recomputes the result from the committed
   inputs. Register it in `build_parser()`. If a reader cannot run it, it is not
   a result.
4. **Tests** in `tests/test_<stage>.py`: the arithmetic, the published figures,
   and any mistake worth pinning so it cannot come back.
5. **A docs chapter** at `docs/<stage>.md`, ending with a section saying what the
   result does **not** establish.
6. **A README section** with the headline number, plus the roadmap checkbox
   ticked.

## Evidence rules

**Every published number is reproducible or cross-checked.** No exceptions. A
figure in the README that nobody can regenerate is decoration.

Prefer, in this order:

- **Recompute it.** A CLI command reads committed inputs and prints the number,
  and a test asserts it. This is what `emotion-timeline wer` does.
- **Cross-check it.** If the raw inputs are gone, find a second record produced
  independently and assert they agree. `test_class_error_rates_match_the_model_card_recalls`
  is the model: the model card records per-class recall, the error report records
  per-class error rate, and they must sum to 1 across all seven classes.
- **Say it is unverified.** If neither is possible, state that plainly in the doc
  and do not put the number in the README.

**Check the number before you publish it.** Re-derive it from the source data
rather than copying it forward. Two of the figures inherited from the coursework
did not survive that: the AssemblyAI word error rate (0.61% became 0.81%) and an
NPEC F1 that no repository recorded at all.

## When the raw data is gone

Some stages only have summary statistics left. That is workable, but it has to be
declared and defended:

- Commit the summary as JSON under `benchmarks/<stage>/`, with a `_comment` key
  saying what is missing and why.
- Write a `check_consistency()` that verifies every arithmetic relation the
  numbers must satisfy — parts summing to totals, rates matching their own
  numerator and denominator, shares matching their base.
- Make the render command **refuse to draw** if any check fails. Publishing a
  contradiction as a picture is worse than publishing nothing.
- Say so in the docs chapter, under a "Provenance" heading.

`src/emotion_timeline/analysis/error_analysis.py` is the worked example.

## Figures

- Rendered by a function in the stage module, registered in a `FIGURES` dict, and
  written by `emotion-timeline figures --out assets`. Never hand-edited, never
  screenshotted.
- Reuse the palette already in `error_analysis.py`: `WRONG` red for the failing
  thing, `RIGHT` green for the working thing, `ACCENT` teal otherwise, used the
  same way in every chart so the colours mean something across figures.
- Title states the finding, not the axes: "Three surface markers each take the
  error rate past 55%", not "Error rate by feature".
- Deterministic output, so CI can diff `assets/` and catch a stale figure.
- Label every bar with its value; a reader should not have to measure.

## Dependencies

Anything needing a GPU or a paid API goes in an optional extra, and its import
stays **inside the function that uses it**, so `import emotion_timeline` never
pulls torch. The core install is numpy, pandas, matplotlib, scipy.

## Before you call it done

```bash
uv run pytest
uvx ruff@0.6.9 check . && uvx ruff@0.6.9 format --check .
uv run emotion-timeline figures --out assets && git diff --stat -- assets
```

The last one must produce no diff. If it does, commit the regenerated figures.
