"""Timestamped segments in, a per-scene emotion timeline out.

This is the layer the repository is named for, and it is deliberately the dull
one: everything above the segment CSV -- downloading a video, converting audio,
running Whisper -- is optional and untested, and everything from the CSV down is
reproducible from what is committed. The seam is three columns, ``start_s``,
``end_s`` and the text, and whatever replaces Whisper in 2027 only has to write
them.

Three decisions are worth stating, because each replaces something the original
pipeline did with a heavier component.

**Scenes come from silence, not from vision.** The original ran PySceneDetect
over the video to find visual cuts. At this layer there is no video, and there
does not need to be: consecutive segments separated by less than
:data:`GAP_SECONDS` of silence are one scene. One parameter, no dependency, and a
reader can check it against the transcript by eye.

**One model answers, the other is asked anyway.** The Russian comparison measured
every combination rule and none of them beat the native model alone -- soft vote
0.4799 and confidence pick 0.4781 against B's 0.4816 -- so the timeline's emotion
is B's, not an ensemble's. The translation path still runs, and its answer is
carried in a ``second_opinion`` column, because where the two agree accuracy was
0.5604 against 0.4816 overall. That is a *where to look* signal rather than a
better classifier, and it is drawn as one: the band is hatched where they split.

**Confidence replaces the intensity model.** The original gated emotion behind a
separate intensity classifier trained on nothing anybody could check. A
temperature fitted on held-out validation rows does the same job -- which scenes
to believe -- with a number behind it and one fewer component.

**The unit of classification is a chunk, not a transcript row.** Ask the models
about whatever rows the transcriber emitted and part of the answer is the
transcriber: the same recording through Whisper rather than AssemblyAI came back
87% Neutral against 68%, because Whisper splits on pauses where AssemblyAI merges
into paragraphs, and a lone sentence reads as neutral where the paragraph it came
from does not. Repacking each scene into chunks of at most :data:`CHUNK_CHARS`
characters makes the unit one neither transcriber chose, and means nothing is
truncated. The budget is measured rather than guessed: at this corpus's worst
observed rate ruBERT spends 0.298 tokens per character, so 400 characters is 119
of its 128, and 140 of the translator's 192.

**It does not make the two transcripts agree**, and the chapter says so. They
still put the same emotion on only 62% of the runtime. What it removes is one
known reason for that, not the disagreement -- see :func:`runtime_agreement`,
which is committed for both transcripts so the number can be recomputed.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emotion_timeline.data.labels import EMOTIONS

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "pipeline"
DEFAULT_TIMELINE = BENCHMARK / "timeline.json"
DEFAULT_SEGMENTS = ROOT / "benchmarks" / "stt" / "assemblyai-best.csv"

#: Silence longer than this starts a new scene. One second over the committed
#: transcript gives 47 scenes with a median length of 51 seconds; two gives 31
#: at 83 seconds, and five gives ten, which is a chapter list rather than a
#: timeline. Nothing downstream depends on the value -- it is recorded in the
#: record and can be changed with one flag.
GAP_SECONDS = 1.0

#: The classification unit, in characters. See the module docstring: this is the
#: parameter that makes the timeline independent of who segmented the transcript,
#: and it is sized to the smaller of the two models' contexts.
CHUNK_CHARS = 400

#: A transcript writes its text under one of these, depending on who wrote it.
#: The annotated benchmark says ``hypothesis``; ``transcribe`` says ``text``.
TEXT_COLUMNS = ("text", "hypothesis")

CSV_COLUMNS = (
    "scene",
    "start_s",
    "end_s",
    "emotion",
    "confidence",
    "second_opinion",
    "second_confidence",
    "agreed",
    "segments",
    "text",
)


@dataclass(frozen=True, slots=True)
class Segment:
    """One timestamped line of transcript, whoever produced it."""

    start_s: float
    end_s: float
    text: str

    @property
    def duration(self) -> float:
        return max(self.end_s - self.start_s, 0.0)


@dataclass(frozen=True, slots=True)
class Scene:
    """Consecutive segments with no long silence between them."""

    index: int
    segments: tuple[Segment, ...]

    @property
    def start_s(self) -> float:
        return self.segments[0].start_s

    @property
    def end_s(self) -> float:
        return self.segments[-1].end_s

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments if segment.text).strip()


def relative(path: str | Path) -> str:
    """A repo-relative POSIX path, so a record never carries someone's home directory."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_segments(path: str | Path) -> list[Segment]:
    """A segment CSV, in file order.

    Rows with no text are kept: they still occupy wall-clock time, and dropping
    them would silently merge the scenes either side of a long unintelligible
    stretch.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    column = next((name for name in TEXT_COLUMNS if name in rows[0]), None)
    if column is None:
        raise ValueError(f"{path} has no text column; expected one of {TEXT_COLUMNS}")
    return [
        Segment(float(row["start_s"]), float(row["end_s"]), str(row[column] or "").strip())
        for row in rows
    ]


def group(segments: Sequence[Segment], gap: float = GAP_SECONDS) -> list[Scene]:
    """Split a transcript into scenes wherever the silence runs longer than ``gap``."""
    if not segments:
        return []
    scenes: list[list[Segment]] = [[segments[0]]]
    for previous, current in itertools.pairwise(segments):
        if current.start_s - previous.end_s > gap:
            scenes.append([])
        scenes[-1].append(current)
    return [Scene(index, tuple(members)) for index, members in enumerate(scenes)]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of one scene small enough for both models to read whole."""

    scene: int
    text: str
    seconds: float


SENTENCE_END = re.compile(r"(?<=[.!?\u2026]) ")


def sentences(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """``text`` packed into pieces of at most ``max_chars``, splitting between sentences.

    A single sentence longer than the budget is cut mid-word at the end, which is
    what the tokenizer would have done anyway, except that this keeps the tail
    rather than dropping it.
    """
    packed: list[str] = []
    current = ""
    for piece in SENTENCE_END.split(text):
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            packed.append(current)
            current = piece
        else:
            current = candidate
    if current:
        packed.append(current)

    out: list[str] = []
    for piece in packed:
        while len(piece) > max_chars:
            out.append(piece[:max_chars])
            piece = piece[max_chars:]
        if piece:
            out.append(piece)
    return out


def _atoms(segment: Segment, max_chars: int) -> list[Segment]:
    """A segment, or the sentence-sized pieces of one too long to read whole.

    Time is apportioned by character count. That is an approximation -- people do
    not speak at a constant rate -- but a scene boundary is a silence, and these
    pieces are inside one segment, so the error stays within a few seconds and
    never moves a scene.
    """
    if len(segment.text) <= max_chars:
        return [segment]
    parts = sentences(segment.text, max_chars)
    total = sum(len(part) for part in parts) or 1
    out: list[Segment] = []
    cursor = segment.start_s
    for part in parts:
        span = segment.duration * len(part) / total
        out.append(Segment(cursor, cursor + span, part))
        cursor += span
    return out


def chunks(scenes: Sequence[Scene], max_chars: int = CHUNK_CHARS) -> list[Chunk]:
    """Repack each scene's text into pieces of at most ``max_chars``.

    Packs whole segments where it can and splits between sentences where it
    cannot, so the unit the models see is the same size whoever wrote the
    transcript -- which is the entire point, see the module docstring. Never
    packs across a scene boundary.
    """
    out: list[Chunk] = []

    def flush(index: int, members: list[Segment]) -> None:
        if members:
            out.append(
                Chunk(
                    scene=index,
                    text=" ".join(s.text for s in members if s.text).strip(),
                    seconds=sum(s.duration for s in members),
                )
            )

    for scene in scenes:
        pending: list[Segment] = []
        length = 0
        for segment in (a for s in scene.segments for a in _atoms(s, max_chars)):
            extra = len(segment.text) + (1 if pending else 0)
            if pending and length + extra > max_chars:
                flush(scene.index, pending)
                pending, length, extra = [], 0, len(segment.text)
            pending.append(segment)
            length += extra
        flush(scene.index, pending)
    return out


def aggregate(pieces: Sequence[Chunk], probabilities: np.ndarray, scenes: int) -> np.ndarray:
    """One distribution per scene, averaging its chunks' by duration.

    ``probabilities`` is per chunk, in :func:`chunks` order, and already
    calibrated. A chunk of zero length still counts once, so a scene made
    entirely of zero-duration rows averages rather than dividing by zero.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if len(values) != len(pieces):
        raise ValueError(f"{len(values)} rows of probabilities against {len(pieces)} chunks")

    out = np.zeros((scenes, values.shape[1]), dtype=np.float64)
    total = np.zeros(scenes, dtype=np.float64)
    for piece, row in zip(pieces, values, strict=True):
        weight = piece.seconds if piece.seconds > 0 else 1.0
        out[piece.scene] += row * weight
        total[piece.scene] += weight
    if not total.all():
        raise ValueError("a scene has no chunks")
    return out / total[:, None]


def _decide(probabilities: np.ndarray) -> tuple[list[str], list[float]]:
    chosen = np.asarray(probabilities).argmax(axis=1)
    return (
        [EMOTIONS[index] for index in chosen],
        [round(float(row[index]), 4) for row, index in zip(probabilities, chosen, strict=True)],
    )


def build_record(
    scenes: Sequence[Scene],
    pieces: Sequence[Chunk],
    primary: np.ndarray,
    second: np.ndarray,
    *,
    source: str,
    gap: float,
    primary_model: dict[str, Any],
    second_model: dict[str, Any],
    agreement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The committed record: scene boundaries, both answers, and where they split.

    The scene *text* is deliberately absent. It is already committed, in the
    transcript this was built from, and duplicating it here would let the two
    drift apart. :func:`table` puts them back together, and
    :func:`check_consistency` refuses a record whose scenes do not line up with
    the transcript it names.
    """
    primary_labels, primary_confidence = _decide(aggregate(pieces, primary, len(scenes)))
    second_labels, second_confidence = _decide(aggregate(pieces, second, len(scenes)))
    agreed = [a == b for a, b in zip(primary_labels, second_labels, strict=True)]

    return {
        "_comment": [
            "A per-scene emotion timeline over a real 51-minute documentary,",
            "written by `emotion-timeline score-timeline`. The emotion is the",
            "native Russian model's: no combination rule beat it alone. The",
            "translation path's answer rides along as a second opinion, because",
            "where the two agreed accuracy was higher -- which makes disagreement",
            "a place to look rather than a number to average away.",
            "Scene text is not repeated here; it is in the transcript named by",
            "`source`, and `check_consistency` requires the two to line up.",
        ],
        "source": source,
        "gap_seconds": gap,
        "segments": sum(len(scene.segments) for scene in scenes),
        "chunks": len(pieces),
        "chunk_chars": CHUNK_CHARS,
        "scenes": len(scenes),
        "duration_s": round(scenes[-1].end_s - scenes[0].start_s, 3) if scenes else 0.0,
        "primary": dict(primary_model),
        "second_opinion": dict(second_model),
        "agreement": {
            "scenes": sum(agreed),
            "share": round(sum(agreed) / len(agreed), 4) if agreed else 0.0,
            **(agreement or {}),
        },
        "timeline": [
            {
                "scene": scene.index,
                "start_s": round(scene.start_s, 3),
                "end_s": round(scene.end_s, 3),
                "segments": len(scene.segments),
                "emotion": primary_labels[position],
                "confidence": primary_confidence[position],
                "second_opinion": second_labels[position],
                "second_confidence": second_confidence[position],
                "agreed": agreed[position],
            }
            for position, scene in enumerate(scenes)
        ],
    }


@dataclass(frozen=True, slots=True)
class Timeline:
    """The committed timeline, and the digest its figure is stamped with."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TIMELINE) -> Timeline:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def scenes(self) -> list[dict[str, Any]]:
        return list(self.raw["timeline"])

    @property
    def gap(self) -> float:
        return float(self.raw["gap_seconds"])

    @property
    def duration(self) -> float:
        return float(self.raw["duration_s"])

    @property
    def agreement(self) -> dict[str, Any]:
        return dict(self.raw["agreement"])

    def counts(self) -> dict[str, int]:
        """How many scenes each emotion claims, in the seven-class order."""
        found = {name: 0 for name in EMOTIONS}
        for scene in self.scenes:
            found[str(scene["emotion"])] += 1
        return found


def emotion_at(report: Timeline, second: float) -> str | None:
    """The emotion covering a moment, or ``None`` if no scene does.

    Gaps are real: a silence longer than the threshold belongs to no scene, and
    answering for it would be inventing a reading.
    """
    for scene in report.scenes:
        if float(scene["start_s"]) <= second <= float(scene["end_s"]):
            return str(scene["emotion"])
    return None


def runtime_agreement(first: Timeline, second: Timeline, step: float = 1.0) -> dict[str, Any]:
    """How much of the recording two timelines put the same emotion on.

    Sampled on a fixed grid rather than compared scene by scene, because two
    transcribers do not agree on where the scenes are either -- which is part of
    what is being measured. Seconds only one timeline covers are excluded and
    counted, so the coverage travels with the share, the same way it does for the
    agreement filter in the Russian chapter.
    """
    end = max(first.raw["timeline"][-1]["end_s"], second.raw["timeline"][-1]["end_s"])
    covered = same = 0
    confusion: dict[str, int] = {}
    moment = 0.0
    while moment <= end:
        left, right = emotion_at(first, moment), emotion_at(second, moment)
        moment += step
        if left is None or right is None:
            continue
        covered += 1
        if left == right:
            same += 1
        else:
            key = f"{left} -> {right}"
            confusion[key] = confusion.get(key, 0) + 1
    grid = int(end / step) + 1
    return {
        "grid_points": grid,
        "covered": covered,
        "coverage": round(covered / grid, 4) if grid else 0.0,
        "same": same,
        "share": round(same / covered, 4) if covered else 0.0,
        "disagreements": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
    }


def check_consistency(report: Timeline, segments: Sequence[Segment] | None = None) -> list[str]:
    """Every relation the record has to satisfy before it is drawn.

    Pass ``segments`` -- the transcript the record names -- and the scene
    boundaries are checked against a fresh grouping at the record's own gap. That
    is the check that matters: the record carries no text, so without it a
    timeline could quietly describe a different transcript.
    """
    problems: list[str] = []
    scenes = report.scenes
    if not scenes:
        return ["the record carries no scenes"]

    if len(scenes) != int(report.raw["scenes"]):
        problems.append(f"header says {report.raw['scenes']} scenes, the list holds {len(scenes)}")
    counted = sum(int(scene["segments"]) for scene in scenes)
    if counted != int(report.raw["segments"]):
        problems.append(f"scenes hold {counted} segments, the header says {report.raw['segments']}")

    previous_end = float("-inf")
    for scene in scenes:
        name = f"scene {scene['scene']}"
        if float(scene["end_s"]) < float(scene["start_s"]):
            problems.append(f"{name} ends before it starts")
        if float(scene["start_s"]) < previous_end:
            problems.append(f"{name} starts before the scene before it ended")
        previous_end = float(scene["end_s"])
        for key in ("emotion", "second_opinion"):
            if scene[key] not in EMOTIONS:
                problems.append(f"{name}: {key} {scene[key]!r} is not one of the seven")
        for key in ("confidence", "second_confidence"):
            value = float(scene[key])
            # The top class of a seven-way distribution cannot be below 1/7.
            if not 1.0 / len(EMOTIONS) - 1e-9 <= value <= 1.0:
                problems.append(f"{name}: {key} {value} is not a top-class probability")
        if bool(scene["agreed"]) != (scene["emotion"] == scene["second_opinion"]):
            problems.append(f"{name}: the agreed flag contradicts the two predictions")

    agreed = sum(1 for scene in scenes if scene["agreed"])
    if agreed != int(report.agreement["scenes"]):
        problems.append(f"{agreed} scenes agree, the header says {report.agreement['scenes']}")
    if abs(float(report.agreement["share"]) - agreed / len(scenes)) > 5e-5:
        problems.append("the agreement share is not the agreeing scenes over all of them")

    span = float(scenes[-1]["end_s"]) - float(scenes[0]["start_s"])
    if abs(span - report.duration) > 0.01:
        problems.append(f"the scenes span {span:.1f}s, the header says {report.duration:.1f}s")

    if segments is not None:
        regrouped = group(segments, report.gap)
        if len(regrouped) != len(scenes):
            problems.append(
                f"the transcript regroups into {len(regrouped)} scenes at a "
                f"{report.gap}s gap, the record holds {len(scenes)}"
            )
        else:
            for scene, fresh in zip(scenes, regrouped, strict=True):
                if abs(float(scene["start_s"]) - fresh.start_s) > 1e-6:
                    problems.append(
                        f"scene {scene['scene']} does not start where the transcript does"
                    )
                    break

    return problems


def table(report: Timeline, segments: Sequence[Segment]) -> list[dict[str, Any]]:
    """The record joined back to the transcript, one row per scene."""
    scenes = group(segments, report.gap)
    if len(scenes) != len(report.scenes):
        raise ValueError(
            f"{len(scenes)} scenes in the transcript against {len(report.scenes)} in the record"
        )
    return [{**row, "text": scene.text} for row, scene in zip(report.scenes, scenes, strict=True)]


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """``timeline.csv``: both predictions, both calibrated confidences, both flags."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        # The repository is LF throughout; csv defaults to CRLF and the
        # pre-commit hook would rewrite the file behind the test that compares it.
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_COLUMNS})
    return target


def describe(report: Timeline) -> Iterator[str]:
    """The timeline, in the terms the chapter uses."""
    minutes = report.duration / 60
    yield (
        f"{report.raw['segments']:,} segments over {minutes:.1f} minutes, grouped into "
        f"{report.raw['scenes']} scenes at a {report.gap:g}s silence gap"
    )
    yield ""
    primary, second = report.raw["primary"], report.raw["second_opinion"]
    yield f"  emotion         {primary['name']}  (temperature {primary['temperature']})"
    yield f"  second opinion  {second['name']}  (temperature {second['temperature']})"
    yield ""
    counts = report.counts()
    widest = max(len(name) for name in counts)
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        share = count / len(report.scenes)
        bar = "#" * round(share * 40)
        yield f"  {name:<{widest}} {count:>3} {share:>6.1%}  {bar}"
    yield ""
    agreement = report.agreement
    yield (
        f"  the two models agree on {agreement['scenes']} of {report.raw['scenes']} scenes "
        f"({float(agreement['share']):.1%})"
    )
    if agreement.get("caveat"):
        yield f"  {agreement['caveat']}"
