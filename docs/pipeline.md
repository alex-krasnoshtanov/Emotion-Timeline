# The pipeline: a real recording, scene by scene

Every other chapter here measures something against ground truth. This one cannot:
a 51-minute documentary about Manaus and the Colombian border has no labels, and
nobody is going to annotate it. So this chapter is a demonstration rather than a
result, and the interesting question is not *how accurate is it* — that is
unanswerable here — but **which parts of the answer are worth believing**.

```bash
uv run emotion-timeline timeline        # reads only committed records
uv run emotion-timeline score-timeline  # reruns both models over the transcript
```

![47 scenes over 52 minutes; the two models agree on 49% of them](../assets/emotion-timeline.png)

## The seam

```
 optional, --extra stt, never in CI
   video ──► yt-dlp ──► ffmpeg 16 kHz mono ──► whisper large-v3-turbo
                                                       │
                                        segments.csv ◄──┘   start_s, end_s, text
                                                       │
 always reproducible, no network, tested  ─────────────┤
                                                       ▼
                     scene grouping ──► classify ──► calibrate ──► timeline.csv + .png
```

Three columns are the whole interface. Everything above them needs a network, a
GPU and about 700 MB of audio nobody should commit; everything below them runs
from `benchmarks/stt/assemblyai-best.csv`, which is already in this repository
because the [speech-to-text chapter](stt-benchmark.md) needed it. Whatever
replaces Whisper in 2027 has to write those three columns and nothing else.

## Three components replaced by one parameter each

The original pipeline had a visual scene detector, a separate intensity
classifier, and an HTTP call to a local LLM for translation. None of them
survived, and each replacement is smaller than the thing it replaced.

**Scenes come from silence.** The original ran PySceneDetect over the video to
find visual cuts. At this layer there is no video and none is needed: consecutive
segments with less than one second of silence between them are one scene. That
gives 47 scenes with a median length of 51 seconds. Two seconds gives 31 at 83
seconds and five gives ten, which is a chapter list rather than a timeline. The
threshold is one flag, it is recorded in the record, and a reader can check it
against the transcript by eye — which is three things PySceneDetect was not.

**Intensity is the calibration.** The original gated each emotion behind a
separate intensity model trained on nothing anybody could check. A temperature
fitted on held-out validation rows answers the same question — which scenes to
believe — with a number behind it and one fewer untested component. The strip
under the timeline is that number.

**Translation is a pinned artefact.** `Helsinki-NLP/opus-mt-ru-en`, greedy
decoding, no beams. The original called a local LLM server with a configurable
model name, which cannot be pinned and therefore cannot be checked a year later.

## One model answers; the other is asked anyway

[The Russian chapter](russian.md) measured every way of combining the two models
that answer in our seven classes, and **none of them beat the native model
alone**: soft vote 0.4799, confidence pick 0.4781, against B's 0.4816. So the
timeline's emotion is B's. There is no ensemble here, because the ensemble was
measured and it lost.

The translation path still runs on every segment, and its answer is carried in a
`second_opinion` column, because the one rule that *did* pay was the agreement
filter: where the two agreed, accuracy was 0.5604 against 0.4816 overall. That is
not a better classifier and it is not used as one. It is a *where to look*
signal, and the picture draws it as one — the band is solid where the two models
agree and hatched where they split.

## What it found

| | |
| --- | ---: |
| Segments | 316 |
| Scenes, at a 1s gap | 47 |
| Recording | 51.6 minutes |
| Scenes the two models agree on | 23 (48.9%) |
| Median calibrated confidence | 0.414 |

**Two-thirds of a documentary is Neutral**, which is the correct answer. 32 of
47 scenes are narration: a presenter explaining that Manaus sells rubber and is
surrounded by jungle is not an emotional beat, and a timeline that found one
would be wrong. The 15 that are not Neutral are where the episode turns, and they
land where the footage does:

| At | Scene | Confidence | |
| ---: | --- | ---: | --- |
| 5.4m | **Fear** | 0.4965 | the investigator tells them the dead man was a known criminal |
| 12.5m | **Disgust** | 0.3817 | cocaine trafficking, *"people prefer to stay quiet, because they want to live"* |
| 34.7m | **Joy** | 0.4991 | *"I'm on Colombian soil"* — crossing the border, the atmosphere changes |
| 49.5m | **Joy** | 0.6827 | *"thank you for the interview, for the courage, for the candour"* |
| 49.9m | **Fear** | 0.3957 | *"such anxious feelings stay with you afterwards"* |

The first of those two is the episode's most confident non-Neutral scene at
0.6827; the second follows it immediately and says the opposite, and **both
models agree on both**. Two adjacent scenes, opposite emotions, no disagreement
between models trained on different languages — that is about as much as a run
with no ground truth can offer.

### The second opinion has a tell

A predicts **Disgust on 16 of 47 scenes**; B predicts it on 3. On a recording
about drug trafficking Disgust is not an absurd answer, but sixteen times is not
a reading of the material — it is a systematic lean, and it is the same model
that scored 0.3631 on ru-izard against B's 0.4816. Given that, the sensible use
of A here is as a dissent signal rather than a vote, which is what it is.

### Agreement and confidence move together

Scenes the two models agree on carry a mean calibrated confidence of **0.456**
against **0.388** where they split. Two signals that were never fitted to each
other point the same way, which is mild corroboration that agreement is tracking
something. It is not evidence that either is right.

## What this does not establish

- **Not an accuracy.** There are no labels on this recording. The 48.9%
  agreement rate is a consistency figure: two models can agree and both be wrong,
  and on out-of-domain text they will do that more often than the ru-izard number
  suggests. Nothing on this page should be read as the pipeline's accuracy.
- **The models are not independent.** Both were scored on ru-izard, B was
  trained on it, and A ends in a model trained to the same seven-class collapse.
  Their errors correlate more than "two opinions" implies, so agreement is weaker
  corroboration than it looks.
- **The confidences are low, and that is the honest reading.** A median of 0.414
  after calibration says the model is rarely sure on documentary speech. It is
  trained on social-media register and this is narration. The temperature was
  fitted on ru-izard's validation rows, which are the wrong domain too, so even
  the calibration is borrowed.
- **The scene boundaries are silences, not scenes.** A presenter pausing for
  breath and a hard cut to a different location look identical from a transcript.
  Some of these 47 are neither.
- **The transcript is AssemblyAI's**, with its 0.81% word error rate measured
  over 18 minutes, not all 51. Errors in it propagate here silently.
