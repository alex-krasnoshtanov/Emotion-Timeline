"""Auditing a model card against its own arithmetic, and against the training script.

Two records of the trained classifier survive and they describe different models.
The card claims DeBERTa-V2-Base evaluated single-label; the committed training
script is DistilBERT evaluated multi-label. Neither can be rerun -- the weights
are gone from both university repositories -- so this stage establishes what each
record can support using only the numbers in it.

**Why that is worth doing rather than retraining.** A fresh fine-tune would
produce a new model with new numbers and would say nothing about the one the
error analysis describes, which is the model this study actually reports on. The
question a reader has is whether the published evaluation means what it says, and
that is answerable from the committed tables.

Three kinds of check live here, in descending order of strength.

**Recomputation.** Every per-class F1 is the harmonic mean of its own precision
and recall; every macro average is the plain mean of its column; weighted F1 and
accuracy follow from the support column. All three of the card's evaluation
tables satisfy all of it, to the four decimal places they were written with. That
is worth stating plainly: the metrics were computed, not typed.

**Identities that separate the two records.** For single-label classification
micro F1 and accuracy are the same number by construction. The card reports them
equal; the script reports a micro F1 of 0.8724 against a subset accuracy of
0.8279, which single-label evaluation cannot produce. And the script's hamming
loss, read against its own micro precision and recall, gives the average number
of true labels per sample: :func:`labels_per_sample` recovers 1.025, so about one
row in forty carried more than one label. The card's dataset is single-label by
construction. The two records are not the same evaluation.

**Contradictions inside one record.** The card's dataset table attaches its class
labels to the wrong counts -- Joy's 149,321 rows are called Neutral, Neutral's
13,401 are called Fear -- and the card's own support column, fifteen pages later,
proves it in all seven classes.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "model"
DEFAULT_CARD = BENCHMARK / "card-metrics.json"

# The card printed four decimal places, so agreement to three is agreement, and
# anything looser would not be evidence of anything.
TOLERANCE = 5e-4

# The held-out fraction the training script configures, and the fraction that
# turns the dataset's class counts into the card's support column.
HELD_OUT_FRACTION = 0.15

LABEL_SLOTS = 7


def harmonic_mean(precision: float, recall: float) -> float:
    """F1 from precision and recall. Zero when both are, rather than dividing by it."""
    total = precision + recall
    return 0.0 if total == 0 else 2 * precision * recall / total


def labels_per_sample(
    recall_micro: float, precision_micro: float, hamming_loss: float, slots: int = LABEL_SLOTS
) -> float:
    """Average true labels per sample, recovered from three multi-label metrics.

    The one number here that is derived rather than checked, and the one that
    settles which dataset the script trained on.

    Over ``t`` true labels per sample, micro recall fixes the true positives at
    ``t * R`` and micro precision fixes the predictions at ``t * R / P``, so the
    wrong label slots per sample come to ``t[(1 - R) + R(1 - P)/P]``. Hamming
    loss is that divided by the number of slots, which leaves ``t`` as the only
    unknown.

    For the script's figures this gives 1.025. A single-label dataset gives
    exactly 1, so the file the script trained on was not the collapsed one the
    card names.
    """
    per_label = (1 - recall_micro) + recall_micro * (1 - precision_micro) / precision_micro
    return hamming_loss * slots / per_label


@dataclass(frozen=True, slots=True)
class ClassScore:
    name: str
    precision: float
    recall: float
    f1: float
    support: int

    @property
    def implied_f1(self) -> float:
        """What F1 has to be, given this row's own precision and recall."""
        return harmonic_mean(self.precision, self.recall)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One of the card's three evaluation tables."""

    name: str
    samples: int
    accuracy: float
    macro_f1: float
    classes: dict[str, ClassScore]
    raw: dict[str, Any]

    @property
    def scored_classes(self) -> int:
        """Classes this table actually scored, which is not always seven."""
        return len(self.classes)

    @property
    def implied_macro_f1(self) -> float:
        return statistics.fmean(c.f1 for c in self.classes.values())

    @property
    def implied_accuracy(self) -> float:
        """Support-weighted recall, which is the share of all samples got right."""
        return sum(c.recall * c.support for c in self.classes.values()) / self.samples

    @property
    def implied_weighted_f1(self) -> float:
        return sum(c.f1 * c.support for c in self.classes.values()) / self.samples

    def macro_f1_over(self, classes: int) -> float:
        """The macro average this table would report over a given class count."""
        return sum(c.f1 for c in self.classes.values()) / classes

    @property
    def confidence_gap(self) -> float:
        return float(self.raw["confidence_when_correct"] - self.raw["confidence_when_incorrect"])


@dataclass(frozen=True, slots=True)
class OutlierType:
    """One row of the stress test's by-category breakdown."""

    name: str
    accuracy: float
    f1: float
    samples: int

    @property
    def is_control(self) -> bool:
        return "control" in self.name.lower()

    @property
    def is_total_failure(self) -> bool:
        """Scored exactly zero, which is not what a merely bad model does."""
        return self.accuracy == 0.0


@dataclass(frozen=True)
class ModelReport:
    """Both records of the trained classifier, read back for checking and drawing."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CARD) -> ModelReport:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    # --- the card ------------------------------------------------------------

    @property
    def card(self) -> dict[str, Any]:
        return dict(self.raw["card"])

    def evaluation(self, name: str) -> Evaluation:
        body = self.raw["card"]["evaluations"][name]
        return Evaluation(
            name=name,
            samples=int(body["samples"]),
            accuracy=float(body["accuracy"]),
            macro_f1=float(body["macro_f1"]),
            classes={
                label: ClassScore(
                    name=label,
                    precision=float(scores["precision"]),
                    recall=float(scores["recall"]),
                    f1=float(scores["f1"]),
                    support=int(scores["support"]),
                )
                for label, scores in body["classes"].items()
            },
            raw=body,
        )

    @property
    def evaluations(self) -> list[Evaluation]:
        return [self.evaluation(name) for name in self.raw["card"]["evaluations"]]

    @property
    def outlier_types(self) -> list[OutlierType]:
        rows = self.raw["card"]["evaluations"]["stress"]["outlier_types"]
        return [
            OutlierType(r["name"], float(r["accuracy"]), float(r["f1"]), int(r["samples"]))
            for r in rows
        ]

    @property
    def control(self) -> OutlierType:
        return next(row for row in self.outlier_types if row.is_control)

    def outliers_beating_the_control(self) -> list[OutlierType]:
        """Categories built to be harder that the model handles better than the control.

        A control condition the deliberately corrupted inputs outscore is not
        measuring what the stress test set out to measure.
        """
        control = self.control
        return [
            row
            for row in self.outlier_types
            if not row.is_control and row.accuracy > control.accuracy
        ]

    def total_failures(self) -> list[OutlierType]:
        return [row for row in self.outlier_types if row.is_total_failure]

    # --- the script ----------------------------------------------------------

    @property
    def script(self) -> dict[str, Any]:
        return dict(self.raw["script"])

    @property
    def script_metrics(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["script"]["test_metrics"].items()}

    @property
    def script_labels_per_sample(self) -> float:
        m = self.script_metrics
        return labels_per_sample(m["recall_micro"], m["precision_micro"], m["hamming_loss"])

    # --- the mislabelled table -----------------------------------------------

    @property
    def card_dataset_table(self) -> list[tuple[str, int]]:
        rows = self.raw["card"]["dataset_table"]["rows"]
        return [(r["label"], int(r["samples"])) for r in rows]

    @property
    def corrected_labels(self) -> dict[int, str]:
        return {int(k): v for k, v in self.raw["mislabelled_dataset_table"]["corrected"].items()}

    def mislabelled(self) -> list[tuple[int, str, str]]:
        """Each count, the label the card gave it, and the label it belongs to.

        Only the rows where those differ. "Happiness" and "Joy" are treated as
        the same class: the card uses both spellings for it, which is itself a
        sign the table came from somewhere else.
        """
        corrected = self.corrected_labels
        found: list[tuple[int, str, str]] = []
        for label, count in self.card_dataset_table:
            truth = corrected.get(count)
            if truth is None:  # the synthetic Disgust rows, which are not misplaced
                continue
            if label != truth and not (label == "Happiness" and truth == "Joy"):
                found.append((count, label, truth))
        return found


def check_consistency(report: ModelReport) -> list[str]:
    """Every arithmetic relation the two records have to satisfy internally.

    This is what makes the audit safe to publish: a reader who cannot rerun
    either record can see that each one's numbers hold together, and exactly
    where one of them does not.
    """
    problems: list[str] = []

    for evaluation in report.evaluations:
        listed = sum(c.support for c in evaluation.classes.values())
        if listed != evaluation.samples:
            problems.append(
                f"{evaluation.name}: supports sum to {listed}, header says {evaluation.samples}"
            )

        for score in evaluation.classes.values():
            if abs(score.implied_f1 - score.f1) > TOLERANCE:
                problems.append(
                    f"{evaluation.name}/{score.name}: F1 {score.f1} but precision and "
                    f"recall imply {score.implied_f1:.4f}"
                )

        # The macro average is over the classes the table scored, which is six
        # for the stress test. Checking it against seven would report a problem
        # that is really a finding, and the finding belongs in the docs.
        implied = evaluation.macro_f1_over(evaluation.scored_classes)
        if abs(implied - evaluation.macro_f1) > TOLERANCE:
            problems.append(
                f"{evaluation.name}: macro F1 {evaluation.macro_f1} but the column "
                f"means to {implied:.4f} over {evaluation.scored_classes} classes"
            )

        if abs(evaluation.implied_accuracy - evaluation.accuracy) > TOLERANCE:
            problems.append(
                f"{evaluation.name}: accuracy {evaluation.accuracy} but "
                f"support-weighted recall is {evaluation.implied_accuracy:.4f}"
            )

        for key, implied_value in (
            ("macro_precision", statistics.fmean(c.precision for c in evaluation.classes.values())),
            ("macro_recall", statistics.fmean(c.recall for c in evaluation.classes.values())),
            ("weighted_f1", evaluation.implied_weighted_f1),
        ):
            if (
                key in evaluation.raw
                and abs(float(evaluation.raw[key]) - implied_value) > TOLERANCE
            ):
                problems.append(
                    f"{evaluation.name}: {key} {evaluation.raw[key]} but the table "
                    f"implies {implied_value:.4f}"
                )

    # The card's dataset counts must be the ones this repository's own build
    # produces, whatever labels the card attached to them.
    counts = {count for _, count in report.card_dataset_table}
    synthetic = int(report.raw["mislabelled_dataset_table"]["synthetic_disgust_rows"])
    expected = set(report.corrected_labels) | {synthetic}
    if counts != expected:
        problems.append(f"dataset table holds {sorted(counts)}, expected {sorted(expected)}")

    # The script's micro F1 has to follow from its own micro precision and recall.
    m = report.script_metrics
    implied_micro = harmonic_mean(m["precision_micro"], m["recall_micro"])
    if abs(implied_micro - m["f1_micro"]) > TOLERANCE:
        problems.append(
            f"script: f1_micro {m['f1_micro']} but micro precision and recall "
            f"imply {implied_micro:.4f}"
        )

    return problems


def single_label_identity_holds(evaluation: Evaluation) -> bool:
    """Whether micro F1 and accuracy agree, which for single-label they must.

    Every prediction is exactly one label, so a false positive for one class is
    the false negative of another and micro precision, micro recall and accuracy
    are the same quantity. A thresholded multi-label head has no such guarantee.
    """
    micro = evaluation.raw.get("micro_f1")
    return micro is not None and abs(float(micro) - evaluation.accuracy) <= TOLERANCE


def disgust_false_positive_bounds(evaluation: Evaluation) -> tuple[int, int]:
    """What the Disgust false-positive count can be, given the printed precision.

    The card's narrative says 2,753. Precision printed as 0.0763 stands for
    anything in [0.07625, 0.07635), and with 227 true Disgust samples that pins
    the predictions, and therefore the false positives, into a four-wide range.
    2,753 is outside it.
    """
    score = evaluation.classes["Disgust"]
    printed = score.precision
    half = 0.5 * 10 ** -len(str(printed).partition(".")[2])
    low = round(score.support / (printed + half)) - score.support
    high = round(score.support / (printed - half)) - score.support
    return low, high
