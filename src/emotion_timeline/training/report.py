"""The three committed records of a fine-tune, read together and checked against each other.

``summarise`` writes these from a run's kept logits; this reads them back with
nothing else to hand. That distinction is the point. Reproducing the records needs
a 51 MB dataset rebuild and a release asset; reading them needs a clone, which is
what ``emotion-timeline training`` gives a reader who wants to see the numbers
before deciding whether to spend an hour reproducing them.

The comparison against ``docs/model.md`` is the reason the stage exists, and it
carries one rule enforced in code rather than in prose: **Disgust is not
comparable.** 9,151 of the card's 14,316 Disgust rows are a synthetic file that
did not survive, so nearly two thirds of that class is absent from what this model
trained on. :func:`compare_to_card` returns no number for it and says why, and a
test asserts the command never prints one. A caveat under a table gets read
second; a number that does not exist cannot be misread at all.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emotion_timeline.model.card import ClassScore
from emotion_timeline.training import evaluate, splits

BENCHMARK = splits.BENCHMARK
DEFAULT_RUN = BENCHMARK / "run-baseline.json"
DEFAULT_SUMMARY = BENCHMARK / "held-out-summary.json"

#: Nearly two thirds of the card's Disgust training rows are the synthetic file
#: that did not survive, so the two models did not see the same class.
INCOMPARABLE = {
    "Disgust": (
        "9,151 of the card's 14,316 Disgust rows are the synthetic file that did "
        "not survive, so 64% of that class is absent here"
    )
}


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What a run was, and what it scored, as committed."""

    run: dict[str, Any]
    summary: dict[str, Any]
    digest: str

    @classmethod
    def load(
        cls,
        run: str | Path = DEFAULT_RUN,
        summary: str | Path = DEFAULT_SUMMARY,
    ) -> TrainingReport:
        run_bytes = Path(run).read_bytes()
        summary_bytes = Path(summary).read_bytes()
        return cls(
            run=json.loads(run_bytes.decode("utf-8")),
            summary=json.loads(summary_bytes.decode("utf-8")),
            digest=hashlib.sha256(run_bytes + summary_bytes).hexdigest(),
        )

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.run["config"])

    @property
    def classes(self) -> list[str]:
        return list(self.config["classes"])

    @property
    def accuracy(self) -> float:
        return float(self.summary["accuracy"])

    @property
    def samples(self) -> int:
        return int(self.summary["total_samples"])

    @property
    def scores(self) -> list[ClassScore]:
        """Per-class precision, recall and F1, recomputed from the committed matrix."""
        names = self.classes
        matrix = [
            [int(self.summary["confusion"][row][column]) for column in names] for row in names
        ]
        return evaluate.class_scores(matrix, names)

    @property
    def calibration(self) -> dict[str, Any]:
        return dict(self.summary["calibration"])


def check_consistency(report: TrainingReport) -> list[str]:
    """Every relation the two records have to satisfy before anything is published."""
    problems: list[str] = []
    summary = report.summary

    matrix = summary.get("confusion")
    if not matrix:
        problems.append("no confusion matrix, so no per-class score can be recomputed")
        return problems

    # First, because everything below indexes the matrix by these names. A check
    # that raises on the inconsistency it exists to report is not a check.
    if report.classes != list(matrix):
        problems.append("the run's classes and the matrix's classes are not the same seven")
        return problems

    total = sum(sum(row.values()) for row in matrix.values())
    if total != report.samples:
        problems.append(f"the matrix holds {total} samples, the header says {report.samples}")

    right = sum(matrix[name][name] for name in matrix)
    if report.samples - right != summary["total_errors"]:
        problems.append(
            f"the matrix implies {report.samples - right} errors, "
            f"the header says {summary['total_errors']}"
        )

    for name, row in matrix.items():
        recorded = summary["classes"][name]["samples"]
        if sum(row.values()) != recorded:
            problems.append(f"{name}: matrix row is {sum(row.values())}, record says {recorded}")

    for score in report.scores:
        if abs(score.f1 - score.implied_f1) > 5e-4:
            problems.append(f"{score.name}: F1 is not the harmonic mean of its own row")

    calibration = report.calibration
    if calibration and calibration.get("temperature", 0) <= 0:
        problems.append("a temperature has to be positive")

    return problems


@dataclass(frozen=True, slots=True)
class Comparison:
    """One class, as the card scored it and as this model did -- or why not."""

    name: str
    card_f1: float
    retrain_f1: float | None
    reason: str | None

    @property
    def delta(self) -> float | None:
        return None if self.retrain_f1 is None else self.retrain_f1 - self.card_f1


def compare_to_card(
    report: TrainingReport,
    card_f1: dict[str, float],
    incomparable: dict[str, str] | None = None,
) -> list[Comparison]:
    """Class by class against the inherited table, with the missing class left out.

    Returning ``None`` rather than a number for Disgust is deliberate: a delta on a
    class whose training data is two thirds absent describes the missing file, not
    the model, and it is the figure most likely to be quoted forward by somebody
    reading quickly.
    """
    blocked = INCOMPARABLE if incomparable is None else incomparable
    scored = {score.name: score.f1 for score in report.scores}
    return [
        Comparison(
            name=name,
            card_f1=card_f1[name],
            retrain_f1=None if name in blocked else scored.get(name),
            reason=blocked.get(name),
        )
        for name in sorted(card_f1)
    ]


def describe(report: TrainingReport, card_f1: dict[str, float] | None = None) -> Iterator[str]:
    """What the run was and what it scored, in the terms the chapter uses."""
    config = report.config
    yield (
        f"{config['model_id']}, {config['epochs']} epochs, batch {config['batch_size']}, "
        f"seed {config['seed']}"
    )
    yield f"  trained on {report.run['device']} in {report.run['seconds'] / 60:.1f} minutes"
    for entry in report.run["epochs"]:
        yield (
            f"    epoch {int(entry['epoch'])}  train loss {entry['train_loss']:.4f}  "
            f"validation accuracy {entry['val_accuracy']:.4f}"
        )

    yield ""
    yield f"{report.samples:,} held out, accuracy {report.accuracy:.4f}"
    scores = report.scores
    macro = evaluate.macro(scores, "f1")
    weighted = evaluate.weighted(scores, "f1")
    yield f"  macro F1 {macro:.4f}   weighted F1 {weighted:.4f}"
    yield ""
    for score in sorted(scores, key=lambda item: -item.support):
        yield (
            f"    {score.name:<9} {score.support:>6,}   "
            f"P {score.precision:.4f}  R {score.recall:.4f}  F1 {score.f1:.4f}"
        )

    if card_f1:
        yield ""
        yield "  against the card's held-out table"
        for row in compare_to_card(report, card_f1):
            if row.retrain_f1 is None:
                yield f"    {row.name:<9} not comparable -- {row.reason}"
            else:
                yield (
                    f"    {row.name:<9} card {row.card_f1:.4f}   here {row.retrain_f1:.4f}   "
                    f"{row.delta:+.4f}"
                )

    calibration = report.calibration
    if calibration:
        yield ""
        yield f"  temperature {calibration['temperature']:.3f}, fitted on validation"
        for stage in ("before", "after"):
            part = calibration[stage]
            yield (
                f"    {stage:<6} calibration error {part['expected_calibration_error']:.4f}   "
                f"confidence {part['correct']:.4f} right against {part['incorrect']:.4f} wrong"
            )

    bug = report.summary.get("url_bug", {})
    mangled = bug.get("mangled")
    if mangled and mangled["samples"]:
        yield ""
        yield (
            f"  the {mangled['samples']} held-out rows carrying a mangled URL are wrong "
            f"{mangled['error_rate']:.2%} of the time, against "
            f"{mangled['error_rate_elsewhere']:.2%} elsewhere"
        )
        masked = bug.get("masked", {})
        if masked.get("samples"):
            yield (
                f"    only {masked['samples']} rows reached [URL] masking intact, too few "
                "to say whether the mangling is what costs the accuracy"
            )


def card_f1_from(path: str | Path) -> dict[str, float]:
    """The card's held-out per-class F1, read from the transcription already committed."""
    from emotion_timeline.model.card import DEFAULT_CARD, ModelReport

    report = ModelReport.load(path or DEFAULT_CARD)
    held_out = report.evaluation("held_out")
    return {name: score.f1 for name, score in held_out.classes.items()}
