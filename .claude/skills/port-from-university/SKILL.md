---
name: port-from-university
description: Fetch source material from the two private BredaUniversityADSAI repositories this project is rebuilt from. Use when a stage needs the original notebooks, scripts, metrics or figures. Covers the account switch that makes them readable, a targeted fetch that avoids cloning 1.3 GB, what to leave behind, and the credit obligation.
---

# Porting from the university repositories

This repository is a rebuild. The originals are two **private** repos in the
`BredaUniversityADSAI` organisation, neither cloned locally:

| Repo | What it holds | Size |
| --- | --- | --- |
| `fae2-nlpr-group-group-14-1` | the five-person group project | 1 GB, 347 files |
| `2025-26a-fai2-adsai-OleksiiKrasnoshtanov240247` | the individual NLP block | 316 MB, 206 files |

## They are invisible to the default account

`gh` is normally authenticated as the personal account, which is **not** a member
of that organisation. `gh api repos/BredaUniversityADSAI/...` returns 404 — which
looks exactly like "does not exist", and has already caused one wrong conclusion.

```bash
gh auth switch --user OleksiiKrasnoshtanov240247   # university account
# ... do the reads ...
gh auth switch --user Gfgf96                        # back to the personal account
```

Switch back when finished. Leaving the wrong account active breaks any later
push to the personal repositories.

## Fetch selectively, do not clone

A clone pulls 1.3 GB, almost all of it downloaded YouTube audio. Walk the tree
and fetch only the blobs you need:

```bash
gh api "repos/$REPO/git/trees/main?recursive=1" --jq '.tree[]|select(.type=="blob")|.path'
gh api "repos/$REPO/git/blobs/$SHA" --jq .content | base64 -d > out
```

Skip `.mp3 .mp4 .wav .pt .pth .bin .safetensors .pyc .parquet`, anything under
`__pycache__`, `.ipynb_checkpoints`, `.idea/`, `batch_output/`, `models/saved/`,
and any blob over a few MB. Stage into a scratch directory outside the repo —
**never** into the working tree, where it can be committed by accident.

Notebook outputs are worth reading: several results exist only in a cell output
and nowhere else, including the word error rates and the 428,331-row dataset
shape.

## What must not come across

- **Client material.** `Task6/data/raw/*.xlsx` are labelled transcripts supplied
  by the Content Intelligence Agency. Out, without exception.
- **Media.** ~700 MB of YouTube audio under `Pipeline_src/batch_output/`.
- **External material.** `DataLabs/Task15` is a conference deck, not our work.
- **Model weights.** They ship as release assets with a recorded digest.

## Rewrite, do not copy

Port the idea and the data; write the code fresh against this repository's
conventions. Two reasons, and the second is the one that matters:

1. Coursework code is notebook-shaped — no tests, no CLI, no packaging.
2. **Re-deriving the numbers is how errors surface.** Rewriting the WER
   calculation is what exposed the 240-row comparison bug. Copying it forward
   would have carried the bug into the portfolio.

So: recompute every figure from the source data rather than quoting it. If a
recomputed number disagrees with the original, that discrepancy is a finding —
document it in the docs chapter rather than quietly using the new value.

## Credit what you port

Authorship in the group repo, by directory:

| Component | Author |
| --- | --- |
| `Pipeline_src/`, `Task12/` — the pipeline | Oleksii Krasnoshtanov |
| `Task6/` — nine-family model comparison | Danil Sysenko |
| `Task10/` — explainability | Filipp Lotsmanov |
| `Task 8/` — prompted-LLM baseline | the group |

Anything ported from `Task6` or `Task10` gets a row in the README Credits table.
The group is relaxed about reuse, so this is attribution rather than permission —
but the table stays accurate.
