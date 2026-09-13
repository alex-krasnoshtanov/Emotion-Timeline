# The pipeline: a real recording, scene by scene

Every other chapter here measures something against ground truth. This one cannot:
a 51-minute documentary about Manaus and the Brazil–Colombia border has no labels,
and nobody is going to annotate it. *How accurate is this* has no answer here.
What it can answer is **how much of this timeline is the recording and how much
is the machinery**, and that turns out to be measurable without a single label.
The answer is less flattering than the picture.

```bash
uv run emotion-timeline timeline --against benchmarks/pipeline/timeline-whisper.json
uv run emotion-timeline timeline --write   # and regenerate the table and the figure
uv run emotion-timeline score-timeline --valence   # reruns the models over the transcript
```

![47 scenes over 52 minutes; the two models agree on 36% of them](../assets/emotion-timeline.png)

## The seam

```
 optional, --extra stt, never in CI
   video ──► yt-dlp ──► ffmpeg 16 kHz mono ──► whisper large-v3-turbo
                                                       │
                                        segments.csv ◄──┘   start_s, end_s, text
                                                       │
 always reproducible, no network, tested  ─────────────┤
                                                       ▼
      scene grouping ──► chunking ──► classify ──► calibrate ──► timeline.csv + .png
```

Three columns are the whole interface. Everything above them needs a network, a
GPU and a few hundred megabytes of audio nobody should commit; everything below
them runs from a committed CSV. Whatever replaces Whisper in 2027 has to write
those three columns and nothing else.

## Three components replaced by one parameter each

The original pipeline had a visual scene detector, a separate intensity
classifier, and an HTTP call to a local LLM for translation. None survived, and
each replacement is smaller than the thing it replaced.

**Scenes come from silence.** The original ran PySceneDetect over the video to
find visual cuts. At this layer there is no video and none is needed: consecutive
segments with less than one second of silence between them are one scene. That
gives 47 scenes with a median length of 51 seconds; two seconds gives 31 at 83
seconds, and five gives ten, which is a chapter list rather than a timeline. The
threshold is one flag, it is recorded, and a reader can check it against the
transcript by eye. PySceneDetect offered none of those.

**Intensity was the original's second model, and it has now been scored.** That
model is a good one: [Mendes & Martins, ECIR 2023](https://arxiv.org/abs/2302.14021),
multilingual, and it reads Russian without translation. Its output was never
checked against anything, though, and its arousal dimension was cut into five
"intensity levels" at 0.2/0.4/0.6/0.8 on no evidence. [`valence.md`](valence.md) scores both
dimensions on held-out Russian and finds the split verdict: **valence separates
the classes at AUC 0.8223, arousal at 0.5734**, so the original built its scale on
the weaker of the two. The five levels turn out to be three; the outer two hold
8.7% of the data.

The strip under this timeline is therefore still **calibrated confidence**: how
sure the classifier is. That is a different quantity from how energetic the
speech is, and it is the one that can be checked. Neither dimension improves the
emotion label (p = 0.3634), so neither is allowed near the prediction.

**The valence band above it is display-only and off by default.** It earns its
place by disagreeing with the label where the label says least: 26 of these 47
scenes are called Neutral, and those 26 span **91% of the whole episode's valence
range**, from 0.154 at the favelas to 0.824 after the border crossing. Turn it on with
`--valence`, or the checkbox in the browser.

**Translation is a pinned artefact.** `Helsinki-NLP/opus-mt-ru-en`, greedy
decoding, no beams. The original called a local LLM server with a configurable
model name, which cannot be pinned and therefore cannot be checked a year later.

## The classification unit is a chunk, not a transcript row

This one came out of running the pipeline twice and is the most useful thing on
the page.

Ask the models about whatever rows the transcriber emitted, and part of every
answer is the transcriber. The same recording through Whisper rather than
AssemblyAI came back **87% Neutral against 68%**. Whisper splits on pauses where
AssemblyAI merges into paragraphs, and a lone sentence reads as neutral where the
paragraph it came from does not.

So each scene's text is repacked into chunks of at most **400 characters**,
splitting between whole segments where possible and between sentences where a
single segment is too long. The budget is measured rather than guessed: at this
corpus's worst observed rate ruBERT spends 0.298 tokens per character, so 400
characters is 119 of its 128 and 140 of the translator's 192, and **nothing is
truncated**. 316 segments become 142 chunks; Whisper's 947 become 132.

**It did not make the two transcripts agree.** All it does is remove one known
reason for them not to.

## How much of this is the transcriber?

Two transcripts of the same 51 minutes, the same pipeline over both:

| | AssemblyAI | Whisper large-v3-turbo |
| --- | ---: | ---: |
| Segments | 316 | 947 |
| Scenes | 47 | 54 |
| Chunks | 142 | 132 |
| Neutral, by runtime | 66.4% | 44.0% |
| The two models agree | 36.2% | 33.3% |

**They put the same emotion on 62.0% of the 2,740 seconds both cover.** Not 95%,
and not noise either. Two-thirds of the disagreement runs one way, with
AssemblyAI Neutral where Whisper is not, and the largest single move is 274
seconds of Neutral becoming Fear.

That is a number no accuracy figure would have shown, it needed no labels, and it
is the ceiling on how precisely any claim about this recording can be made. It is
recomputable: both transcripts and both timelines are committed, and
`emotion-timeline timeline --against` prints it.

## What it found

| | |
| --- | ---: |
| Segments | 316 |
| Scenes, at a 1s silence gap | 47 |
| Recording | 51.6 minutes |
| Scenes the two models agree on | 17 (36.2%) |
| Median calibrated confidence | 0.382 |

**Just over half the episode is Neutral**, which is the correct answer for
narration: a presenter explaining that Manaus sold rubber and is ringed by jungle
is not an emotional beat, and a timeline that found one would be wrong. The 21
scenes that are not Neutral land where the episode turns:

| At | | Confidence | Agreed | |
| ---: | --- | ---: | :-: | --- |
| 3.7m | **Sadness** | 0.3210 | | a killing, and what it says about the neighbourhood |
| 6.1m | **Disgust** | 0.2665 | ✓ | walking away from the murder scene |
| 12.5m | **Disgust** | 0.3817 | ✓ | cocaine, *"people prefer to stay quiet, because they want to live"* |
| 16.4m | **Fear** | 0.4982 | ✓ | *"I understood why they will not talk about it"* |
| 49.5m | **Joy** | **0.7691** | ✓ | *"thank you for the interview, for the courage, for the candour"* |
| 51.3m | **Surprise** | 0.6585 | | *"an interesting day. Busy. And now I can finally relax"* |

The 49.5-minute scene is the most confident non-Neutral reading in the episode and
both models agree on it. So are the two *least* confident, at 0.2297 and 0.2370,
which is a useful reminder that agreement and confidence are different things,
and that neither of them is accuracy.

**One is plainly wrong**, and it is worth naming: at 37.6 minutes *"oh, brilliant!
And they are handing out sweets!"* comes back **Anger**. The two models split on
it, so the picture hatches it, which is the entire argument for carrying a second
opinion at all.

### The second opinion has a tell

A predicts **Disgust on 24 of 47 scenes**; B predicts it on 4. On a recording
about drug trafficking Disgust is not an absurd answer, but half the episode is
too much to be a reading of the material. It is a systematic lean, from the model
that scored 0.3728 on ru-izard against B's 0.4816.

The lean got **worse** when the translator was fixed. An audit found `opus-mt` was
dropping sentences ([`russian.md`](russian.md) has the numbers), and with whole
paragraphs reaching the English model rather than their first sentences, A's
Disgust calls went from 17 to 24 and agreement fell from 46.8% to 36.2%. A fix
that makes two models agree less is the useful kind: the earlier agreement was
partly an artefact of feeding one of them less text.

### Agreement and confidence move together, slightly

Scenes the two models agree on carry a mean calibrated confidence of **0.422**
against **0.384** where they split. Two signals never fitted to each other point
the same way. It is mild corroboration that agreement tracks something, and the
gap is small enough that it is worth no more than that.

### The voice-activity filter drops narration over music

`faster-whisper` takes `vad_filter`, which runs a voice-activity detector and
transcribes only what it calls speech. It is usually on for a good reason: it
stops Whisper hallucinating loops over silence. On this recording it costs more
than it saves.

| | Segments | Speech captured | Largest gap |
| --- | ---: | ---: | ---: |
| AssemblyAI | 316 | 48.6 min | 18.5s |
| Whisper, `vad_filter=True` | 947 | 41.1 min | **44.4s** |
| Whisper, `--no-vad` | 913 | 41.8 min | 25.3s |

The 44-second gap is the episode's opening narration, *"Манаус, самый большой
мегаполис Северной Бразилии…"*, delivered over a music bed. The detector hears
music and drops the voice with it. AssemblyAI transcribes the paragraph;
`--no-vad` recovers most of it.

**And the hallucination it guards against did not appear.** Back-to-back repeated
segments, the signature of a Whisper loop, number two either way. So on this
material the filter has no upside to weigh against the paragraph it ate.

The general point is the one worth keeping: **the filter's failure is silent and
hallucination's failure is loud.** A dropped paragraph leaves a gap nobody
notices; a hallucination loop is visible in the first ten rows of the CSV. For a
pipeline whose claim is that its output can be checked, the loud failure is the
one to prefer. That is one recording, though, so `--no-vad` is a flag rather than
a new default, and the committed transcript is still the filtered one.

## Running it on a video of your own

```bash
uv sync --extra stt --extra model
uv run emotion-timeline transcribe "<url or file>" --out data/segments.csv
uv run emotion-timeline score-timeline --segments data/segments.csv --out mine.json
uv run emotion-timeline timeline --record mine.json --segments data/segments.csv
```

`transcribe` is the only command in this repository that needs a network, and the
only one carrying `# pragma: no cover`. It downloads audio with yt-dlp, converts
it to 16 kHz mono with ffmpeg, runs Whisper large-v3-turbo and writes the three
columns. On this 51-minute recording that took 5 minutes 39 seconds end to end,
including fetching the model. Nothing it produces is committed: `downloads/` and
`data/` are both gitignored, because the upstream repository carries about 700 MB
of YouTube audio and that is the mistake not to repeat.

`--language ru` is passed rather than detected. Auto-detection reads the first
thirty seconds, so a recording that opens on music or a title card can come back
as the wrong language and transcribe into it without complaining. `--no-vad` is
worth trying on anything with a score under the narration, for the reason above.

## The browser front end

```bash
uv sync --extra web --extra stt --extra model
uv run emotion-timeline serve            # http://127.0.0.1:8000

docker compose up                        # or the same thing in a container
```

A link or a file goes in the box, the page polls while Whisper and the two
classifiers work, and the timeline comes back with the transcript beside it. It
is the same code the commands run: `pipeline/score.py` is called by both, so the
page and `timeline.csv` cannot drift apart.

**It opens on a timeline rather than an empty form.** The committed 47-scene
example is drawn on load, with no GPU, no network and nothing downloaded. That is
also what makes the interface testable: `tests/test_web.py` exercises every route
without a model on the machine. What is on screen always names itself, so the
example cannot be read as a run of your own.

**A run is something you can watch and stop.** Three named stages, the elapsed
time measured on the server, and a button that stops it. Cancellation lands on
the progress callback, because there is no polite place to return from inside one
long call into faster-whisper, so a stopped run unwinds at its next line of
output rather than ten minutes later. The job id goes in the URL, so reloading at
minute eight reattaches to the run instead of losing it, and if the server goes
away the page says so rather than sitting on its last state for ever.

**The result comes away with you.** CSV and JSON download from what is on screen.
Until this, the browser was the one surface that could run the pipeline and not
give you the output.

**It binds to loopback, and that is not a default to change casually.** The page
hands a URL to yt-dlp and a file to ffmpeg, so anyone who can reach the port can
make this machine fetch a URL of their choosing. Three checks sit on that
boundary. The link has to be `http`/`https`; the upload has to carry a media
extension and stay under 512 MB; and the uploaded *filename is never used as a
path*, since the extension is taken and the name is generated. Each has a test
named after the thing it refuses. The size limit is now checked in the browser as
well, so a 600 MB file is refused before it is uploaded rather than after.

The scene gap and the voice-activity filter are both on the page, because the
[VAD finding](#the-voice-activity-filter-drops-narration-over-music) means the
right setting depends on the recording. If this checkout is missing something,
say ffmpeg on PATH or the valence checkpoint, the page says so on load instead of
failing at the end of a run. That is also what makes the small container honest:
`:study` carries no transcriber, so it opens the page, draws the committed
example, and disables the run button with a line saying why.

**Two images, for the two things people want.** `ghcr.io/...:study` is core plus
the page, about 400 MB, and runs every command that reads a committed record.
`ghcr.io/...:latest` adds ffmpeg, Whisper and the classifiers, built against
CUDA and run with `--gpus all`. It is several gigabytes, and worth it: the
transcriber is most of the wall clock, and it takes its device from
`torch.cuda.is_available()`, so a CPU-only torch would slow down the one stage
that matters. faster-whisper is CTranslate2 rather than torch and loads cuBLAS
and cuDNN by name, so the image puts torch's copies of both on the loader path;
without that the classifiers would find the card and the transcriber would not.
Both install the
project into `/app` rather than into site-packages, because the commands find
their records relative to their own file: the image ships the repository layout,
which is the thing a wheel cannot. The published images are built by
[`release.yml`](../.github/workflows/release.yml), which runs `timeline` inside
each one and checks the numbers before pushing it.

## What this does not establish

- **None of this is an accuracy.** There are no labels on this recording. 36.2% is a
  consistency figure between two models, and 62.0% a consistency figure between
  two transcripts. Two models can agree and both be wrong, and on out-of-domain
  text they will do that more often than the ru-izard number suggests.
- **The models are not independent.** Both were scored on ru-izard, B was trained
  on it, and A ends in a model trained to the same seven-class collapse. Their
  errors correlate more than "two opinions" implies.
- **A is weaker here than the ru-izard table suggests, and B may be stronger.**
  ru-izard is DeepL-translated English, so it makes A translate twice and lets B
  train on its own test distribution. This recording is native Russian speech,
  where neither handicap applies, so the 0.3728-against-0.4816 gap is the wrong
  prior for what these two are doing here. [`russian.md`](russian.md) says
  why, and nothing available measures the native case.
- **The confidences are low, and that is what they mean.** A median of 0.382
  after calibration says the model is rarely sure on documentary speech. It is
  trained on social-media register and this is narration, and the temperature
  itself was fitted on ru-izard, so even the calibration is borrowed.
- **Two transcripts is not a distribution.** 62.0% is one pair. A third
  transcriber could sit anywhere, and the comparison also confounds the
  transcript with the scene boundaries, since the two do not pause in the same
  places. The number is a floor on the sensitivity, not a measurement of it.
- **The scene boundaries are silences, not scenes.** A presenter pausing for
  breath and a hard cut to a different location look identical from a transcript.
- **The valence band is a reading, not a measurement.** It was scored on
  translated social-media text, which is the wrong register for this, and it has
  never been validated on documentary speech. It is drawn because it separates
  scenes the label cannot, not because anything here says it is right.
- **The AssemblyAI transcript's 0.81% word error rate** was measured over an
  eighteen-minute window, not all 51. Errors outside it propagate here silently.
