"""Segments in, a scored timeline record out — the part the CLI and the web app share.

This was inline in `cmd_score_timeline` until the browser demo needed the same
thing, and two copies of the calibration step is exactly the sort of duplication
that lets a published number and a rendered one drift apart. The command and the
server now call one function.

Only :func:`score` needs a GPU. Everything it depends on — grouping, chunking,
aggregating, the record's own arithmetic — lives in `timeline.py` and is tested
without one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from emotion_timeline.pipeline import timeline as pipeline

#: Where the temperatures and the held-out accuracies come from. Refitting a
#: temperature here is impossible -- a documentary has no labels -- so the ones
#: stage 8 recorded are reused, which is the entire reason they were recorded.
DEFAULT_COMPARISON = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "russian" / "comparison.json"
)

DEFAULT_WEIGHTS = "models/distilbert-v1"
DEFAULT_RUBERT = "models/rubert-v1"

AGREEMENT_CAVEAT = (
    "agreement here is a consistency signal, not an accuracy: this recording has "
    "no labels, and two models can agree and both be wrong"
)


def plan(
    segments: Sequence[pipeline.Segment],
    gap: float = pipeline.GAP_SECONDS,
    chunk_chars: int = pipeline.CHUNK_CHARS,
) -> tuple[list[pipeline.Scene], list[pipeline.Chunk]]:
    """The scenes and the chunks, before any model is loaded.

    Split out so a caller can report what it is about to do -- how many scenes,
    how many chunks -- without waiting on a checkpoint to download.
    """
    scenes = pipeline.group(segments, gap)
    return scenes, pipeline.chunks(scenes, chunk_chars)


def score(  # pragma: no cover - loads two classifiers and a translator
    segments: Sequence[pipeline.Segment],
    *,
    source: str,
    gap: float = pipeline.GAP_SECONDS,
    chunk_chars: int = pipeline.CHUNK_CHARS,
    weights: str = DEFAULT_WEIGHTS,
    rubert: str = DEFAULT_RUBERT,
    comparison: str | Path = DEFAULT_COMPARISON,
    progress: object = None,
    valence: str | Path | None = None,
) -> dict[str, Any]:
    """Score every chunk with both models and build the timeline record.

    ``valence`` is a path to the valence-arousal checkpoint, or None. It is off
    by default and that is a measured decision rather than caution: the two
    dimensions add nothing to the label (p = 0.3634, see `russian/va.py`), so they
    are carried for a reader to look at and for nothing else.
    """
    import numpy as np

    from emotion_timeline.russian import baselines, translate
    from emotion_timeline.training import evaluate

    def say(message: str) -> None:
        if callable(progress):
            progress(message)

    scenes, pieces = plan(segments, gap, chunk_chars)
    texts = [piece.text for piece in pieces]
    say(
        f"{len(segments):,} segments -> {len(scenes)} scenes at a {gap:g}s gap "
        f"-> {len(pieces)} chunks of at most {chunk_chars} characters"
    )

    blocks = json.loads(Path(comparison).read_text(encoding="utf-8"))
    a_name, b_name = blocks["pair"]
    approaches = blocks["approaches"]

    say(f"classifying with {rubert}")
    b_raw, _ = baselines.predict(rubert, texts, multi_label=False)
    say(f"translating with {translate.MODEL}, then {weights}")
    english = translate.translate(texts, progress=progress)
    a_raw, _ = baselines.predict(weights, english, multi_label=False)

    def calibrate(raw: np.ndarray, block: dict[str, Any]) -> np.ndarray:
        scaled: np.ndarray = evaluate.softmax(
            np.log(np.clip(raw, 1e-12, None)), float(block["temperature"])
        )
        return scaled

    extra: dict[str, Any] = {}
    if valence is not None:
        from emotion_timeline.russian import va

        say(f"reading valence and arousal with {va.MODEL}")
        values, energy = va.predict(texts, valence, progress=progress)
        report = va.ValenceReport.load()
        extra = {
            "valence": values,
            "arousal": energy,
            "va_model": {
                "name": va.MODEL,
                "citation": va.CITATION,
                "valence_auc": report.auc_of("valence"),
                "arousal_auc": report.auc_of("arousal"),
                "improves_the_label": report.helps(),
                "caveat": (
                    "display only: measured on held-out Russian to add nothing to "
                    "the emotion label, and never validated on documentary speech"
                ),
            },
        }

    say("calibrating and building the record")
    return pipeline.build_record(
        scenes,
        pieces,
        calibrate(b_raw, approaches[b_name]),
        calibrate(a_raw, approaches[a_name]),
        source=source,
        gap=gap,
        primary_model={
            "name": b_name,
            "model": rubert,
            "temperature": approaches[b_name]["temperature"],
            "held_out_accuracy": approaches[b_name]["accuracy"],
        },
        second_model={
            "name": a_name,
            "model": f"{translate.MODEL} + {weights}",
            "temperature": approaches[a_name]["temperature"],
            "held_out_accuracy": approaches[a_name]["accuracy"],
        },
        agreement={
            "held_out_accuracy_where_they_agreed": blocks["combinations"]["agreement filter"][
                "accuracy"
            ],
            "measured_on": "the held-out ru-izard split, which is social-media register",
            "caveat": AGREEMENT_CAVEAT,
        },
        **extra,
    )
