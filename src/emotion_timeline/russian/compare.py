"""Four approaches to Russian on one held-out set, and whether combining two helps.

The question `Task12/emotion_classifier_ru.py` calls *"Next stage: Cross-validation
between multiple emotion models (future)"* and never answers. It could not: it had
no Russian ground truth to score against. With ru-izard held out, it becomes
arithmetic.

**Only A and B can be combined.** Both answer in our seven classes and both are
ours, so their probability vectors are comparable once calibrated. C and D answer
in their own vocabularies and are baselines, not ensemble members.

**Calibration first, always.** Averaging two models' raw probabilities measures
which of them is more strident, not which is more often right. Each is scaled by
its own temperature, fitted on its own validation rows, before any rule below
touches it.

**Coverage travels with the agreement filter.** An accuracy over the rows two
models happened to agree on is not comparable to an accuracy over all of them,
and quoting it alone is the same mistake as the 240-row word error rate this
repository opens with. :func:`agreement_filter` returns both and
`check_consistency` refuses a record that carries one without the other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.model.card import harmonic_mean
from emotion_timeline.training import evaluate

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "russian"
DEFAULT_COMPARISON = BENCHMARK / "comparison.json"


def soft_vote(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Average the two calibrated distributions and take the argmax."""
    averaged: np.ndarray = (np.asarray(first) + np.asarray(second)) / 2
    chosen: np.ndarray = averaged.argmax(axis=1)
    return chosen


def confidence_pick(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Take whichever model was more confident about its own answer."""
    first, second = np.asarray(first), np.asarray(second)
    use_first = first.max(axis=1) >= second.max(axis=1)
    return np.where(use_first, first.argmax(axis=1), second.argmax(axis=1))


def agreement_filter(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the two models pick the same class, and what they picked.

    Returns the mask and the shared prediction. The mask is the point: an
    accuracy over it means nothing without the share of rows it covers.
    """
    first, second = np.asarray(first).argmax(axis=1), np.asarray(second).argmax(axis=1)
    return first == second, first


def measure(
    true: Sequence[str],
    predicted: Sequence[str],
    classes: Sequence[str] = EMOTIONS,
) -> dict[str, Any]:
    """Accuracy, the two averages, and the per-class table."""
    matrix = evaluate.confusion(true, predicted, classes)
    scores = evaluate.class_scores(matrix, classes)
    return {
        "samples": len(true),
        "accuracy": round(evaluate.accuracy_of(matrix), 4),
        "macro_f1": round(evaluate.macro(scores, "f1"), 4),
        "weighted_f1": round(evaluate.weighted(scores, "f1"), 4),
        "classes": {
            score.name: {
                "support": score.support,
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "f1": round(score.f1, 4),
            }
            for score in scores
        },
    }


def measure_filtered(
    true: Sequence[str],
    predicted: Sequence[str],
    covered: Sequence[bool],
) -> dict[str, Any]:
    """The agreement filter: what it scores, and over how much of the set.

    ``coverage`` is not a footnote. A rule that answers a third of the rows very
    well is a different product from one that answers all of them adequately, and
    the accuracy alone cannot tell them apart.
    """
    mask = np.asarray(covered, dtype=bool)
    kept_true = [value for value, keep in zip(true, mask, strict=True) if keep]
    kept_predicted = [value for value, keep in zip(predicted, mask, strict=True) if keep]
    if not kept_true:
        return {"coverage": 0.0, "samples": 0, "accuracy": 0.0, "macro_f1": 0.0}
    result = measure(kept_true, kept_predicted)
    result["coverage"] = round(float(mask.mean()), 4)
    return result


@dataclass(frozen=True, slots=True)
class Comparison:
    """The committed record of the four approaches and the three combinations."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_COMPARISON) -> Comparison:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def approaches(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw["approaches"])

    @property
    def combinations(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw.get("combinations", {}))

    @property
    def samples(self) -> int:
        return int(self.raw["held_out_rows"])

    def best(self) -> tuple[str, float]:
        """The highest accuracy, and what produced it."""
        everything = {**self.approaches, **self.combinations}
        name = max(everything, key=lambda key: float(everything[key]["accuracy"]))
        return name, float(everything[name]["accuracy"])


def check_consistency(report: Comparison) -> list[str]:
    """Every relation the record has to satisfy before it is published."""
    problems: list[str] = []
    everything = {**report.approaches, **report.combinations}

    for name, block in everything.items():
        if not 0.0 <= float(block["accuracy"]) <= 1.0:
            problems.append(f"{name}: accuracy {block['accuracy']} is not a share")
        classes = block.get("classes")
        if classes:
            support = sum(int(entry["support"]) for entry in classes.values())
            if support != int(block["samples"]):
                problems.append(
                    f"{name}: supports sum to {support}, the header says {block['samples']}"
                )
            for label, entry in classes.items():
                implied = harmonic_mean(entry["precision"], entry["recall"])
                if abs(implied - entry["f1"]) > 5e-4:
                    problems.append(f"{name}/{label}: F1 is not the harmonic mean of its own row")

    for name, block in report.approaches.items():
        if int(block["samples"]) != report.samples:
            problems.append(
                f"{name} scored {block['samples']} rows, the set holds {report.samples}"
            )

    for name, block in report.combinations.items():
        if "coverage" in block:
            if not 0.0 <= float(block["coverage"]) <= 1.0:
                problems.append(f"{name}: coverage {block['coverage']} is not a share")
        elif int(block["samples"]) != report.samples:
            problems.append(f"{name} answers {block['samples']} rows but reports no coverage")

    return problems


def describe(report: Comparison) -> Iterator[str]:
    """The comparison, in the terms the chapter uses."""
    yield f"{report.samples:,} held-out Russian rows"
    yield ""
    yield f"  {'':<34} {'accuracy':>9} {'macro F1':>9} {'coverage':>9}"
    for name, block in report.approaches.items():
        label = f"{name} {block.get('what', '')}".strip()
        yield (
            f"  {label:<34} {float(block['accuracy']):>9.4f} "
            f"{float(block['macro_f1']):>9.4f} {'':>9}"
        )
    if report.combinations:
        yield ""
        yield "  combining the two that answer in our seven classes"
        for name, block in report.combinations.items():
            coverage = f"{float(block['coverage']):>9.1%}" if "coverage" in block else f"{'all':>9}"
            yield (
                f"  {name:<34} {float(block['accuracy']):>9.4f} "
                f"{float(block['macro_f1']):>9.4f} {coverage}"
            )

    for name, block in report.approaches.items():
        if block.get("caveat"):
            yield ""
            yield f"  {name}: {block['caveat']}"
        if block.get("approximate_share"):
            yield (
                f"  {name}: {float(block['approximate_share']):.1%} of its top answers were a "
                "class our label map could only approximate"
            )


def build_record(
    true: Sequence[str],
    approaches: dict[str, dict[str, Any]],
    pair: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """The whole committed record, from each approach's calibrated probabilities.

    ``approaches`` maps a name to ``{"what", "probabilities", ...}``; anything
    else in the block -- a caveat, an approximate share -- is carried through
    untouched so the reasons a number should be discounted travel with it.

    ``pair`` names the two that can be combined. Only models answering in our
    seven classes qualify, which is why it is passed rather than inferred.
    """
    record: dict[str, Any] = {
        "_comment": [
            "Four approaches to Russian emotion classification on one held-out",
            "split of ru-izard-emotions, and three ways of combining the two that",
            "answer in this project's seven classes. Written by",
            "`emotion-timeline compare-russian`. Every approach is scored on the",
            "same rows; where a number should be discounted, the reason is in the",
            "block beside it rather than in the prose.",
        ],
        "held_out_rows": len(true),
        "approaches": {},
    }

    for name, block in approaches.items():
        probabilities = np.asarray(block["probabilities"])
        predicted = [EMOTIONS[index] for index in probabilities.argmax(axis=1)]
        measured = measure(true, predicted)
        carried = {
            key: value
            for key, value in block.items()
            if key not in {"probabilities", "validation_probabilities"}
        }
        record["approaches"][name] = {**carried, **measured}

    if pair:
        first, second = (np.asarray(approaches[name]["probabilities"]) for name in pair)
        mask, shared = agreement_filter(first, second)
        record["pair"] = list(pair)
        record["combinations"] = {
            "soft vote": measure(true, [EMOTIONS[i] for i in soft_vote(first, second)]),
            "confidence pick": measure(true, [EMOTIONS[i] for i in confidence_pick(first, second)]),
            "agreement filter": measure_filtered(
                true, [EMOTIONS[i] for i in shared], mask.tolist()
            ),
        }
    return record
