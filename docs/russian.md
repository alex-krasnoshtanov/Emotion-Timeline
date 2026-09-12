# Russian: translate, or classify it directly?

The pipeline this project is named for reads Russian. Everything else here is
trained on English. The original coursework resolved that by judgement — there
was no good Russian emotion dataset, so translate — and never scored the decision
against anything.

```bash
uv run emotion-timeline russian          # reads only committed records
uv run emotion-timeline compare-russian  # reruns all four
```

![B native ruBERT wins outright; combining only helps where both models agree](../assets/russian-approaches.png)

## The set, and the one decision building it needs

[`Djacon/ru-izard-emotions`](https://huggingface.co/datasets/Djacon/ru-izard-emotions),
24,891 rows in, **24,766** out. It is the only labelled Russian emotion corpus
this project has, and the original trained a ruBERT on it once, then ran a
different model on untranslated transcripts with nothing to score against. Held
out instead, it makes the question arithmetic.

| Step | Rows | |
| --- | ---: | --- |
| load | 24,891 | every split, concatenated |
| deduplicate | 24,853 | −38 |
| drop unlabelled | 24,766 | −87, once `guilt` and `shame` go |

`guilt` and `shame` have no seven-class home and the original dropped them too,
which is the only reason the two builds are comparable at all.

**`enthusiasm` is the real decision, and it is four times smaller than it looks.**
5,185 rows carry the column — 21% of the set — so merging it into Joy reads like a
large intervention. It is not: only **1,115** rows actually change class, because
the rest already carry a label the priority collapse prefers. `--drop-enthusiasm`
builds the other version, 1,115 rows shorter with Joy at 3,467 instead of 4,582.
Both numbers are in the record, which is the treatment
[`dataset.md`](dataset.md) gives the 34,940 Love rows it could not place.

### The distribution is the confound, before any model runs

| | This set | The English training set |
| --- | ---: | ---: |
| Neutral | **31.3%** | **3.2%** |

Ten times the share, on the class our model is already worst at — a 43% error rate
in [`fine-tune.md`](fine-tune.md). Every number below is measuring that at least as
much as it is measuring language, and no amount of care about translation changes
it. A test asserts both shares so the point cannot quietly go missing.

## Four approaches, 3,715 held-out rows

| | | Accuracy | Macro F1 | Coverage |
| --- | --- | ---: | ---: | ---: |
| **A** | translate with `opus-mt-ru-en`, then our `distilbert-v1` | 0.3631 | 0.3276 | all |
| **B** | `rubert-base-cased` fine-tuned here | **0.4816** | **0.4640** | all |
| **C** | `tabularisai/multilingual-emotion-classification`, off the shelf | 0.3157 | 0.2826 | all |
| **D** | `Djacon/rubert-tiny2`, the one the pipeline shipped | 0.4538 | 0.3945 | all |

**Classifying Russian directly beats translating it by twelve points.** That is
the decision the original made by judgement, now with a number attached — and the
judgement was wrong. A native model trained on 17,336 Russian rows beats an
English model with 293,426 behind it, reached through a translator.

**It also beats the model the pipeline actually shipped** — and does so despite D
having an advantage it cannot be stripped of: D reports ru-izard's own ten columns,
which is the tell that it was **trained on the corpus this scores it against**.
Its split is unknown and may overlap these rows, so 0.4538 is an upper bound
rather than a measurement. That caveat lives in the record beside the number, not
in this paragraph, so the two cannot be separated.

C's 0.3157 comes with its own discount: **20% of its top answers** were a class our
label map could only approximate. `contempt` becomes Disgust and `frustration`
Anger, which are standard; `gratitude` and `love` both become Joy, and `love` is
the uncomfortable one, since the English build drops Love outright for having no
equivalent. Dropping a training row and dropping a prediction are different
things — a prediction we refuse to map is scored wrong regardless, which measures
our map rather than the model — so it is mapped, and the share is published.

## Does cross-validating two models help?

`Task12/emotion_classifier_ru.py` ends with *"Next stage: Cross-validation between
multiple emotion models (future)"*. It was never built, and it could not have
been settled: there was no Russian ground truth to settle it against.

Only A and B can be combined — both answer in our seven classes, so their
probabilities are comparable once each is scaled by its own temperature, fitted on
its own validation rows. **A's temperature is 2.619 and B's is 1.235**: the
translated pipeline is far more overconfident, and averaging the two raw would
have measured that rather than either model's judgement.

| Rule | Accuracy | Macro F1 | Coverage |
| --- | ---: | ---: | ---: |
| soft vote | 0.4799 | 0.4596 | all |
| confidence pick | 0.4781 | 0.4540 | all |
| **agreement filter** | **0.5604** | **0.5112** | **43.7%** |

**The ensemble does not raise accuracy, and that is the finding.** Both
full-coverage rules land *below* B alone. Averaging a 0.36 model into a 0.48 one
drags it down, and picking by confidence does no better because the weaker model
is also the more confident one. Two opinions beat one when the two are comparable;
these are not.

**Where combining pays is as a filter, not a classifier.** On the 43.7% of rows
where A and B pick the same class, accuracy is 0.5604 — eight points above either
alone. That is not a better model. It is a usable *believe this one / look at that
one* signal, and it is what the timeline needs: a scene two models trained on
different languages and different data agree on is one to trust, and a scene they
split on is where a human should look.

The coverage is published with the accuracy every time, and `check_consistency`
refuses a record carrying one without the other. An accuracy over the rows two
models happened to agree on, quoted alone, is the same mistake as the 240-row word
error rate [this repository opens with](stt-benchmark.md).

## Why everything here is so much lower than 0.9164

Our English model scores 0.9164 on English and the best Russian number is 0.4816.
Four things are stacked and none of them can be separated from the others on this
evidence:

1. **Distribution.** 31.3% Neutral against 3.2%, on the worst class.
2. **Domain.** ru-izard is translated social-media text; the pipeline runs on
   documentary speech. Neither matches the other.
3. **Data volume.** B saw 17,336 rows; the English model saw 293,426.
4. **The collapse asymmetry.** Our labels come from a priority collapse that puts
   Disgust first, so a row annotated both Anger and Disgust is *Disgust*. Every
   model here answers with an argmax instead. D never once predicts Disgust on its
   407 Disgust rows, which is as much a property of that mismatch as of the model.

ruBERT also overfits quickly: five epochs scored 0.4568 where two scored 0.5082,
so the run is two epochs, chosen off the validation curve. Seventeen thousand rows
is not 293,000.

## What this does not establish

- **That any of these numbers describes the pipeline's accuracy.** ru-izard is
  social-media register. The documentary is not. This ranks the approaches; it
  does not predict what any of them does on a transcript.
- **That translation is bad.** It ranks below a native model *on this set*, where
  the native model also had in-domain training data. A better translator, or a
  Russian test set matching the English training register, could reverse it.
- **That D is worse than B.** D's number is an upper bound on a corpus it was
  trained on, and it still loses. That is a stronger statement than the table
  looks, and it is the only direction this comparison can support.
- **That the agreement filter is worth its coverage.** Whether answering 44% of
  segments at 0.56 beats answering all of them at 0.48 depends on what the
  timeline is for, and that is a product decision rather than a measurement.
- **Anything about a model trained on more Russian data.** 24,766 rows is what
  exists. The obvious experiment — more Russian data — cannot be run.
