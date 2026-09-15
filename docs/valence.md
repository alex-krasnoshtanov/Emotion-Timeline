# Valence and arousal: measured before being shown

The original coursework ran a second model beside the classifier and used its
**arousal** output as an "intensity" score, cut into five levels at 0.2 / 0.4 /
0.6 / 0.8. The model itself is well chosen:
[Mendes & Martins, ECIR 2023](https://arxiv.org/abs/2302.14021), XLM-RoBERTa
trained on 34 psycho-linguistic datasets across 100 languages, and it reads
Russian without translation, which is more than the pipeline's own second opinion
manages. What was missing is the one thing this project insists on: **nobody ever
scored it.**

```bash
uv run emotion-timeline valence              # reads only committed records
uv run emotion-timeline valence --rescore    # recompute all of it (--extra model)
uv run emotion-timeline score-timeline --valence    # opt in, on your own run
```

## Scoring it needs no new data

Russell's circumplex makes two predictions about any working valence–arousal
model, and the seven-class labels this project already has can check both:
valence should rank Joy above Anger, Disgust, Fear and Sadness; arousal should
rank Anger, Fear and Surprise above Sadness and Neutral. Held out on the same
3,715 Russian rows every other approach is scored on:

| Class | n | Valence | Arousal |
| --- | ---: | ---: | ---: |
| Joy | 687 | **0.5814** | 0.4635 |
| Surprise | 245 | 0.4825 | 0.5001 |
| Neutral | 1,163 | 0.4332 | 0.4677 |
| Anger | 360 | 0.3745 | **0.5231** |
| Fear | 285 | 0.3356 | 0.5339 |
| Sadness | 568 | 0.3521 | 0.4930 |
| Disgust | 407 | 0.3219 | 0.5133 |

| Dimension | Separates | AUC |
| --- | --- | ---: |
| **Valence** | Joy over Anger/Disgust/Fear/Sadness | **0.8223** |
| **Arousal** | Anger/Fear/Surprise over Sadness/Neutral | **0.5734** |

**Valence works. Arousal barely does**, at 0.5734 where 0.5 is nothing at all. So
the dimension the original picked as its intensity measure is the weaker of the
two by a wide margin, and it picked it without ever looking.

### The five intensity levels were three

The original's thresholds, applied to a real distribution:

| Level | Arousal | Rows | |
| --- | --- | ---: | ---: |
| 1 | 0.0–0.2 | 185 | 5.0% |
| 2 | 0.2–0.4 | 940 | 25.3% |
| 3 | 0.4–0.6 | **1,628** | **43.8%** |
| 4 | 0.6–0.8 | 825 | 22.2% |
| 5 | 0.8–1.0 | 137 | 3.7% |

Levels 1 and 5 together hold **8.7%** of the data. A five-point scale that spends
91% of its rows in three buckets is a three-point scale with two decorative ends.

## It does not make the classifier better

The obvious hope is that two extra numbers help the seven-class decision,
especially the tie-break that suggests itself, where the two classifiers disagree
and one of them says Neutral. Stacked on top of both classifiers, fitted on
validation and scored on test:

| | Accuracy |
| --- | ---: |
| both classifiers | 0.5009 |
| both classifiers **+ valence and arousal** | 0.5036 |

**+0.0027: 54 rows right, 44 wrong, net 10 in 3,715, p = 0.3634.** Null.

Two rules built directly on the idea do measurably *worse* than doing nothing,
each with its threshold fitted on validation and scored on test:

| Rule | Threshold | Accuracy | vs. the native model | |
| --- | ---: | ---: | ---: | --- |
| low arousal means Neutral | 0.82 | 0.4681 | −0.0135 | 144 right, 194 wrong |
| valence picks the positive candidate | 0.24 | 0.4649 | −0.0167 | 68 right, 130 wrong |

Both are significant in the wrong direction. The arousal threshold landing at
0.82, the top of its range, is the optimiser saying *always choose Neutral* rather
than *use arousal*.

The reason is not mysterious. Valence is close to "is this positive or negative",
which a seven-class emotion classifier already encodes. It is a coarser view of
the same signal, not an independent one.

### One near miss worth recording

While measuring the above, the stacker scored **0.5036 against the native model's
0.4816, with 212 rows right and 130 wrong, p < 0.0001**, which looks like the
combination rule [the Russian chapter](russian.md) says does not exist.

It is the class prior. Balanced accuracy, which ignores that, goes **0.4559 →
0.4527**: no better at telling classes apart, slightly worse. The whole gain is
reaching for Neutral more often, **44.7% of rows against a gold share of 31.3%**,
which pays on a Neutral-heavy test set and would be harmful on a documentary with
a different mix. The chapter's finding stands, and the record keeps the numbers so
nobody has to work them out again.

## So what is it for?

**The label is least informative exactly where valence is most.** On the committed
recording, 26 of 47 scenes are called Neutral, and those 26 span **91% of the
whole episode's valence range**, from 0.154 to 0.824. The classifier says "nothing
happening" across a stretch that runs from the bleakest material in the programme
to the most positive.

| At | Label | Valence | |
| ---: | --- | ---: | --- |
| 1.6m | Neutral | **0.154** | the favelas, and what happens in them |
| 3.7m | Sadness | 0.112 | the killing |
| 5.4m | Fear | 0.131 | the investigator on the dead man |
| 44.0m | Neutral | **0.824** | across the border, the tone lifts |
| 51.3m | Surprise | 0.849 | the day ends |

That is the case for carrying it: it answers a different question, and it answers
it best where the seven classes answer worst. It is drawn as a band under the
timeline, and it is **off by default** (`--valence` on the command, a checkbox in
the browser) because a reader who only wants the label should not pay for a
1.1 GB download to get one.

## What this does not establish

- **That valence is accurate on this recording.** It was scored on translated
  social-media text, which is the only Russian ground truth available and is the
  wrong register for documentary speech. The band is a reading, not a
  measurement, and the figure's caption says so.
- **That arousal is useless.** It is weak *at ranking this project's seven
  classes*, which is what could be tested. Energy in speech is a real thing and a
  text-only model may simply be the wrong instrument for it. Audio would be the
  right one, and [the cut list](../README.md#scope) says why that is not here.
- **That a better VA model would not help the label.** Only this one was tested.
  Its base and large checkpoints agreed closely when the base was chosen. Those
  figures are in the [release notes](https://github.com/alex-krasnoshtanov/Emotion-Timeline/releases/tag/weights-va-v1)
  rather than in a benchmark here, because it was a selection decision rather
  than a published result. It is weak evidence that capacity is not the binding
  constraint, and no more than that.
- **Anything about the 1–5 intensity scale as a product.** The levels collapse on
  this data; whether a rescaled version would be useful to an editor is a
  question about editors, not about this corpus.

## Credit

The checkpoint is not trained here. It is
[gmendes9/multilingual_va_prediction](https://github.com/gmendes9/multilingual_va_prediction)
(MIT), mirrored as `weights-va-v1` with a recorded SHA-256, and repackaged as
safetensors so it can be pinned and loaded without unpickling a `.bin` from a
Google Drive folder. The base checkpoint ships because the authors' large one is
2.09 GB, over GitHub's asset limit, and the two are within noise of each other
here.
