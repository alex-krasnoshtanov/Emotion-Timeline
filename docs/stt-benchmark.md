# Which speech-to-text system

Everything downstream reads a transcript, so this was the first thing to settle.
Two systems transcribed the same 51-minute Russian-language documentary. Fluent
speakers listened to the audio alongside each transcript and marked, per segment,
how many words were **substituted**, **inserted** or **deleted**.

That hand annotation is the ground truth here. There is no reference transcript
in the data — nobody typed the audio out in full — so word error rate cannot be
recovered by aligning two strings. It can only be recombined from the counts.

## The arithmetic

Each row records the hypothesis text and its three edit counts. WER is defined
over the **reference** length, but the only length available is the
**hypothesis** length, so the reference is recovered from the edits:

```
N_ref = N_hypothesis + deletions - insertions
WER   = (S + I + D) / N_ref
```

A deletion is a reference word the system dropped: absent from the hypothesis,
so it has to be added back. An insertion is a word the system invented: present
in the hypothesis, never in the reference, so it comes off.
`tests/test_wer.py` pins this with a worked example.

## The result

| System | WER | S | I | D | Reference tokens | Segments |
| --- | --- | --- | --- | --- | --- | --- |
| **AssemblyAI Best** | **0.81%** | 7 | 2 | 8 | 2,101 | 95 |
| Whisper large-v3 | 3.33% | 21 | 13 | 36 | 2,105 | 241 |

Scored over the same 18.1 minutes. The two reference lengths land four tokens
apart, which is the control that makes the rates comparable: both systems were
scored on the same quantity of speech.

AssemblyAI makes roughly a quarter as many errors, and its errors are milder —
over half of Whisper's are deletions, which silently lose content rather than
corrupt it.

```bash
uv run emotion-timeline wer --window 0:00-18:09
```

## The correction

The original analysis reported **0.61%** for AssemblyAI. That number is
reproducible — scoring the first 240 rows still returns it — but it does not mean
what it appears to.

It came from taking **the first 240 rows of each spreadsheet**. The two systems
segment differently: Whisper emits 797 segments for this recording where
AssemblyAI emits 316, because Whisper cuts at pauses and AssemblyAI cuts at
speaker turns. So 240 rows are not 240 comparable units:

| | 240 rows covers | Reference tokens in those rows |
| --- | --- | --- |
| Whisper large-v3 | 0:00 – 18:06 | 2,100 |
| AssemblyAI Best | 0:00 – 40:54 | 4,884 |

AssemblyAI was scored over 2.3x as much audio. And that extra stretch was not
annotated for both systems:

| | 0 – 18 min | past 18 min |
| --- | --- | --- |
| Whisper: segments carrying a marked error | 28 of 241 | **0 of 556** |
| AssemblyAI: segments carrying a marked error | 6 of 95 | 9 of 221 |

The annotators worked through the whole file for AssemblyAI and stopped at
18 minutes for Whisper. So the extra 23 minutes inside AssemblyAI's 240 rows are
real, carefully checked, low-error audio — for which Whisper has no score at all.
Averaging it in pulled AssemblyAI's rate from 0.81% down to 0.61%.

Nothing was wrong with the annotation, and no judgement in it was revisited. The
mistake was summing a fixed number of **rows** rather than a fixed window of
**time**.

## What changed in the code

`compare()` takes a time window and never a row count, and `annotated_extent()`
reports where each system's annotation actually stops, so the CLI can warn when a
requested window runs past it:

```
$ uv run emotion-timeline wer --window 0:00-40:00
note: annotation for whisper-large-v3 stops at 17.9 min; beyond that the comparison is uneven.
```

`test_the_window_is_what_makes_it_a_comparison` asserts that the row-count
version gives the wrong answer, so the mistake cannot come back quietly.

## What this does not establish

- **One recording, one language, one domain** — a 51-minute Russian documentary
  with clear studio narration. Neither number is a general claim about either
  system.
- **Annotation is human and single-pass.** No second annotator, so there is no
  inter-annotator agreement figure.
- **Whisper ran with anti-hallucination settings** tuned for this pipeline
  rather than at its defaults.

The decision it supported — transcribe with AssemblyAI, keep Whisper as the
offline fallback — sits well within what the evidence carries.
