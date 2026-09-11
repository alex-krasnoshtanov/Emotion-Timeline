"""Auditing a nine-family model comparison that turns out not to compare anything.

Two records of the same coursework benchmark survive. The submitted log is eight
hand-curated rows; the automatic log is 101 runs written by the training script
as it went. They disagree about which family won, and the audit is about why.

**Why this stage recomputes an audit rather than the benchmark.** The obvious
move is to rerun the nine families on one feature pipeline and publish that
table. It would not settle anything about the original result, and the original
result is what the docs and a reader's questions are about. What can be settled
from the committed logs is whether the reported ranking was ever a ranking, and
the answer is no. That is a finding, it needs no GPU, and it is checkable to the
row.

**The load-bearing arithmetic is denominator recovery.** An accuracy is a whole
number of correct predictions over a whole number of samples, so the fraction in
lowest terms carries a divisor of the evaluation-set size. Recover that divisor
for each run and the runs sort themselves into groups that cannot have been
scored on the same data: the best run overall and the best transformer run need
evaluation sets whose sizes share a common multiple of 6,044,800, which is
fourteen times the whole training set. Nothing else in either log records the
evaluation size, so this is the only way to see it.

The recovery is deliberately conservative. It reports a divisor, never a size --
0.5 is consistent with 2 samples and with 3,200 -- and it declines entirely on
the 22 rows the log rounded to fewer than seven decimal places, where the
fraction is unrecoverable rather than merely imprecise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "model-selection"
DEFAULT_RUN_LOG = BENCHMARK / "run-log.csv"
DEFAULT_SUBMITTED_LOG = BENCHMARK / "submitted-log.json"

# Below seven decimal places a recorded accuracy cannot be resolved back into
# the fraction it came from: 0.2796 is 699/2500 in lowest terms and 2,500 is not
# an evaluation-set size anybody used. Seven places pins denominators into the
# low thousands, which is the range the runs actually sit in.
MIN_DECIMALS = 7

# No coursework run scored more than a few thousand held-out samples. Searching
# past this would start fitting noise in the last decimal place rather than
# recovering anything.
MAX_DENOMINATOR = 20_000

# 95%, two-sided.
Z = 1.959963985

# The seven-class problem the whole study is about, and the class count the
# benchmark's own dataset string names. It is what turns a collapsed model's
# accuracy into its F1 scores below.
CLASSES = 7

# The logs wrote nine decimal places, so a reconstruction that agrees to eight
# is agreeing exactly rather than approximately.
EXACT = 1e-8


def _decimals(recorded: str) -> int:
    """Decimal places the log actually wrote, which is what limits recovery.

    Read off the recorded text rather than the parsed float: the precision is a
    property of what was written down, and ``repr`` of a float would not
    distinguish a value rounded to four places from one that landed there.
    """
    _, _, fraction = recorded.strip().partition(".")
    return len(fraction)


def evaluation_divisor(recorded: str) -> int | None:
    """A divisor of the evaluation-set size this accuracy was computed over.

    ``None`` where the log rounded the value too far to tell. Note that this is
    a divisor and not the size: an accuracy of 69/80 is equally consistent with
    2,760 correct out of 3,200. Everything downstream is phrased in terms of
    divisors for that reason -- which costs nothing, because two runs whose
    divisors have no plausible common multiple cannot share an evaluation set
    whatever the true sizes were.
    """
    if _decimals(recorded) < MIN_DECIMALS:
        return None
    value = float(recorded)
    fraction = Fraction(value).limit_denominator(MAX_DENOMINATOR)
    return fraction.denominator if abs(float(fraction) - value) < 1e-9 else None


def shared_evaluation_size(divisors: Iterable[int]) -> int:
    """Smallest evaluation set every one of these runs could have been scored on."""
    return math.lcm(*divisors)


def single_class_scores(correct: int, total: int, classes: int = CLASSES) -> tuple[float, float]:
    """Macro and weighted F1 of a model that answers with one class for every input.

    This is the one thing in the whole chapter that is recomputed rather than
    cross-checked, and it works because nothing about the model enters into it.
    A run that predicts the majority class every time gets that class's recall
    exactly right and its precision exactly ``k/n``, so the class scores
    ``2k/(n+k)`` and the other six score zero. Macro averaging then divides by
    seven and support weighting multiplies by ``k/n``.

    So the accuracy alone determines both F1 scores, and any logged run whose
    two F1 scores match these to nine decimal places was a model that had
    learned nothing. Seventeen of the 101 were.
    """
    majority = 2 * correct / (total + correct)
    return majority / classes, majority * correct / total


def placed(pool: Iterable[Run]) -> list[tuple[Run, int]]:
    """The runs that can be placed on an evaluation-set axis, with their divisor.

    Runs whose accuracy was logged rounded are dropped: there is no honest x
    coordinate for them, and inventing one would put a made-up number in a
    picture.
    """
    found: list[tuple[Run, int]] = []
    for run in pool:
        divisor = run.divisor
        if divisor is not None:
            found.append((run, divisor))
    return found


def wilson(correct: int, total: int) -> tuple[float, float]:
    """A 95% interval on an accuracy, so "the gap is noise" can be ruled out.

    Wilson rather than the textbook normal interval because it stays inside
    [0, 1] and behaves at the small evaluation sizes several of these runs used.
    """
    p = correct / total
    denominator = 1 + Z * Z / total
    centre = p + Z * Z / (2 * total)
    spread = Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total))
    return (centre - spread) / denominator, (centre + spread) / denominator


@dataclass(frozen=True, slots=True)
class Run:
    """One row of the automatic log: a run, and nothing about what it was scored on."""

    iteration: int
    model_type: str
    model_name: str
    f1_macro: float
    f1_weighted: float
    accuracy: float
    accuracy_recorded: str
    training_time: float
    notes: str
    date_run: str

    @property
    def divisor(self) -> int | None:
        """A divisor of this run's evaluation-set size, if the log kept the precision."""
        return evaluation_divisor(self.accuracy_recorded)

    @property
    def summary(self) -> str:
        """The note without the source link some rows append to it."""
        head, _, _ = self.notes.partition("http")
        return head.rstrip(" -|")

    @property
    def interval(self) -> tuple[float, float] | None:
        """A 95% interval on the accuracy, where the evaluation size is recoverable."""
        divisor = self.divisor
        if divisor is None:
            return None
        return wilson(round(self.accuracy * divisor), divisor)


@dataclass(frozen=True, slots=True)
class Submitted:
    """One row of the hand-curated log, which does record what it was scored on."""

    nr: int
    model_type: str
    model_name: str
    framework: str
    preprocessing: str
    neutral_cap: int
    features: str
    augmentation: str | None
    hyperparameters: str
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    f1_macro: float
    training_time_s: float

    @property
    def label(self) -> str:
        """Row number included: logistic regression appears twice."""
        return f"{self.model_name} #{self.nr}"


@dataclass(frozen=True, slots=True)
class RankSwing:
    """Where a model lands under each average, when the two disagree."""

    label: str
    weighted_rank: int
    macro_rank: int

    @property
    def places(self) -> int:
        """Positive when macro averaging moves the model up the table."""
        return self.weighted_rank - self.macro_rank


def model_type_of(recorded: str) -> str:
    """The recorded model type, in one spelling.

    The two logs write the same three categories differently -- ``traditional_ml``
    against ``Traditional ML``, and the curated log qualifies one of them as
    ``Deep Learning - RNN``. Comparing them needs normalising, and normalising
    the recorded field is the whole of it: no model is reclassified here, which
    matters because the log's *names* cannot be trusted to identify a family
    (``lr_all_optimized`` and ``logistic_regression`` are the same estimator).
    """
    head, _, _ = recorded.partition(" - ")
    return head.strip().lower().replace(" ", "_")


@dataclass(frozen=True, slots=True)
class NearestRun:
    """The closest thing in the automatic log to a submitted score."""

    submitted: Submitted
    run: Run
    gap: float

    @property
    def same_model_type(self) -> bool:
        """Whether the closest score even came from the same kind of model."""
        return model_type_of(self.run.model_type) == model_type_of(self.submitted.model_type)


@dataclass(frozen=True)
class SelectionReport:
    """Both records of the benchmark, read back for checking and drawing."""

    runs: list[Run]
    submitted: list[Submitted]
    dataset: str
    split: str
    sources: tuple[Path, Path]
    digest: str

    @classmethod
    def load(
        cls,
        run_log: str | Path = DEFAULT_RUN_LOG,
        submitted_log: str | Path = DEFAULT_SUBMITTED_LOG,
    ) -> SelectionReport:
        run_path, submitted_path = Path(run_log), Path(submitted_log)
        run_payload = run_path.read_bytes()
        submitted_payload = submitted_path.read_bytes()

        rows = list(csv.DictReader(run_payload.decode("utf-8").splitlines()))
        runs = [
            Run(
                iteration=int(row["iteration_id"]),
                model_type=row["model_type"],
                model_name=row["model_name"],
                f1_macro=float(row["f1_macro"]),
                f1_weighted=float(row["f1_weighted"]),
                accuracy=float(row["accuracy"]),
                accuracy_recorded=row["accuracy"],
                training_time=float(row["training_time"]),
                notes=row["notes"].strip(),
                date_run=row["date_run"],
            )
            for row in rows
        ]

        raw: dict[str, Any] = json.loads(submitted_payload.decode("utf-8"))
        submitted = [
            Submitted(
                nr=int(body["nr"]),
                model_type=body["model_type"],
                model_name=body["model_name"],
                framework=body["framework"],
                preprocessing=body["preprocessing"],
                neutral_cap=int(body["neutral_cap"]),
                features=body["features"],
                augmentation=body["augmentation"],
                hyperparameters=body["hyperparameters"],
                accuracy=float(body["accuracy"]),
                precision_weighted=float(body["precision_weighted"]),
                recall_weighted=float(body["recall_weighted"]),
                f1_weighted=float(body["f1_weighted"]),
                f1_macro=float(body["f1_macro"]),
                training_time_s=float(body["training_time_s"]),
            )
            for body in raw["rows"]
        ]

        return cls(
            runs=runs,
            submitted=submitted,
            dataset=raw["dataset"],
            split=raw["split"],
            sources=(run_path, submitted_path),
            digest=hashlib.sha256(run_payload + submitted_payload).hexdigest(),
        )

    # --- the automatic log ---------------------------------------------------

    @property
    def measurable(self) -> list[Run]:
        """Runs whose accuracy kept enough precision to say anything about its set."""
        return [run for run, _ in placed(self.runs)]

    @property
    def divisors(self) -> list[int]:
        return sorted({divisor for _, divisor in placed(self.runs)})

    def collapsed(self) -> list[Run]:
        """Runs that provably answered with one class for every input.

        Recomputed from each run's own accuracy: see :func:`single_class_scores`.
        Not a heuristic -- a run is in this list only if both of its F1 scores
        match the closed form to eight decimal places, which no model that had
        learned anything would do.
        """
        found: list[Run] = []
        for run, divisor in placed(self.runs):
            macro, weighted = single_class_scores(round(run.accuracy * divisor), divisor)
            if abs(macro - run.f1_macro) < EXACT and abs(weighted - run.f1_weighted) < EXACT:
                found.append(run)
        return found

    def best_run(self, model_type: str | None = None) -> Run:
        """Highest macro F1, over one family of models or over all of them."""
        pool = [r for r in self.runs if model_type is None or r.model_type == model_type]
        return max(pool, key=lambda run: run.f1_macro)

    def headline_pair(self) -> tuple[Run, Run]:
        """The two runs the coursework's conclusion rested on.

        The best run in the log, and the best transformer run. The claim drawn
        from them was that a feature-engineered network beat a transformer.
        """
        return self.best_run(), self.best_run("transformer")

    # --- the submitted log ---------------------------------------------------

    def ranked(self, macro: bool) -> list[Submitted]:
        key = (lambda row: row.f1_macro) if macro else (lambda row: row.f1_weighted)
        return sorted(self.submitted, key=key, reverse=True)

    def swings(self) -> list[RankSwing]:
        """Models whose position depends on which average is read, worst first."""
        weighted = {row.label: i + 1 for i, row in enumerate(self.ranked(macro=False))}
        macro = {row.label: i + 1 for i, row in enumerate(self.ranked(macro=True))}
        swings = [RankSwing(label, weighted[label], macro[label]) for label in weighted]
        moved = [swing for swing in swings if swing.places]
        return sorted(moved, key=lambda swing: abs(swing.places), reverse=True)

    def neutral_caps(self) -> dict[int, list[Submitted]]:
        """Submitted rows grouped by the Neutral ceiling their dataset was built to."""
        groups: dict[int, list[Submitted]] = {}
        for row in self.submitted:
            groups.setdefault(row.neutral_cap, []).append(row)
        return dict(sorted(groups.items(), reverse=True))

    def nearest_logged(self) -> list[NearestRun]:
        """For each submitted score, the closest macro F1 in the automatic log."""
        nearest = []
        for row in self.submitted:
            run = min(self.runs, key=lambda r: abs(r.f1_macro - row.f1_macro))
            nearest.append(NearestRun(row, run, abs(run.f1_macro - row.f1_macro)))
        return nearest


def check_consistency(report: SelectionReport) -> list[str]:
    """Every arithmetic relation the two records have to satisfy.

    Neither log can be recomputed -- the runs are gone and the client dataset
    behind the earliest of them cannot be republished -- so this is what stands
    in for reproducing them. It is stricter than it looks: the weighted-average
    identity below is what identifies the submitted log's unlabelled Precision,
    Recall and F1 columns as weighted rather than macro, and the whole chapter
    depends on knowing which they are.
    """
    problems: list[str] = []

    iterations = [run.iteration for run in report.runs]
    if iterations != list(range(1, len(iterations) + 1)):
        problems.append(f"run log iteration ids are not 1..{len(iterations)} in order")

    for run in report.runs:
        for name, value in (
            ("f1_macro", run.f1_macro),
            ("f1_weighted", run.f1_weighted),
            ("accuracy", run.accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                problems.append(f"run {run.iteration}: {name} is {value}, not a rate")
        if run.training_time <= 0:
            problems.append(f"run {run.iteration}: training time is {run.training_time}")

    numbers = [row.nr for row in report.submitted]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"submitted log rows are not 1..{len(numbers)} in order")

    for row in report.submitted:
        # Recall averaged over classes weighted by their support is the share of
        # all samples got right, which is accuracy. It holds for weighted
        # averaging and not for macro, so this is the check that tells us which
        # average the log's bare "Precision"/"Recall"/"F1 score" columns used.
        if abs(row.recall_weighted - row.accuracy) > 5e-5:
            problems.append(
                f"submitted row {row.nr}: recall {row.recall_weighted} and accuracy "
                f"{row.accuracy} differ, so these are not weighted averages"
            )
        # Macro averaging gives the rare classes equal weight, and this model
        # is worse on its rare classes -- which the error analysis measures
        # directly. Macro above weighted would contradict that.
        if row.f1_macro > row.f1_weighted:
            problems.append(
                f"submitted row {row.nr}: macro F1 {row.f1_macro:.4f} exceeds weighted "
                f"{row.f1_weighted:.4f}, so the rare classes beat the common ones"
            )
        for name, value in (
            ("accuracy", row.accuracy),
            ("precision", row.precision_weighted),
            ("f1_weighted", row.f1_weighted),
            ("f1_macro", row.f1_macro),
        ):
            if not 0.0 <= value <= 1.0:
                problems.append(f"submitted row {row.nr}: {name} is {value}, not a rate")
        if row.training_time_s <= 0:
            problems.append(f"submitted row {row.nr}: training time is {row.training_time_s}")
        # The Neutral ceiling is read out of the row's own preprocessing text,
        # so the two must agree or the transcription moved a number.
        if f"{row.neutral_cap // 1000}k neutral" not in row.preprocessing.lower():
            problems.append(
                f"submitted row {row.nr}: neutral cap {row.neutral_cap:,} is not what "
                f"its preprocessing says ({row.preprocessing!r})"
            )

    return problems


def _evaluation_phrase(run: Run) -> str:
    """What can be said about this run's evaluation set, which is sometimes nothing.

    A rounded accuracy is not an inconsistency, so ``check_consistency`` lets it
    through and the description has to cope with it rather than assume a
    divisor was recovered.
    """
    divisor = run.divisor
    if divisor is None:
        return "evaluation set unrecoverable: this accuracy was logged rounded"
    return f"evaluation set divisible by {divisor:,}"


def describe(report: SelectionReport) -> list[str]:
    """The audit, as the lines the command line prints."""
    lines: list[str] = []
    submitted, runs = report.submitted, report.runs

    lines.append(
        f"two records of one benchmark: {len(submitted)} submitted rows, {len(runs)} logged runs"
    )
    lines.append("")

    lines.append(f"  the submitted log -- {report.dataset}, {report.split}")
    by_weighted = report.ranked(macro=False)
    by_macro = report.ranked(macro=True)
    width = max(len(row.label) for row in submitted)
    lines.append(f"    {'ranked by weighted F1':<{width + 9}}  ranked by macro F1")
    for left, right in zip(by_weighted, by_macro, strict=True):
        lines.append(
            f"    {left.label:<{width}} {left.f1_weighted:.4f}  "
            f"{right.label:<{width}} {right.f1_macro:.4f}"
        )
    lines.append("")
    lines.append("    the same eight runs, and below the top two the order changes:")
    for swing in report.swings():
        direction = "up" if swing.places > 0 else "down"
        moved = abs(swing.places)
        lines.append(
            f"      {swing.label:<{width}} {swing.weighted_rank} by weighted F1, "
            f"{swing.macro_rank} by macro -- {moved} place{'s' if moved > 1 else ''} {direction}"
        )
    lines.append("")

    caps = report.neutral_caps()
    lines.append("    and the eight rows are not one dataset either:")
    for cap, rows in caps.items():
        names = ", ".join(row.model_name for row in rows)
        lines.append(f"      Neutral capped at {cap:>6,}   {names}")
    lines.append("")

    best, best_transformer = report.headline_pair()
    lines.append(
        f"  the automatic log -- {len(runs)} runs, {runs[0].date_run} to {runs[-1].date_run}"
    )
    lines.append(
        f"    {len(report.measurable)} of {len(runs)} accuracies kept enough precision to "
        f"recover a divisor of their evaluation set"
    )
    lines.append(f"    those recover {len(report.divisors)} distinct divisors, not one")
    collapsed = report.collapsed()
    families = sorted({run.model_name for run in collapsed})
    lines.append(
        f"    {len(collapsed)} of the {len(runs)} runs are provably a model that answered "
        f"with one class every time"
    )
    lines.append(f"    {'':<4}({', '.join(families)} -- recomputed from each run's own accuracy)")
    lines.append("")
    for label, run in (("best overall", best), ("best transformer", best_transformer)):
        interval = run.interval
        span = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "unrecoverable"
        lines.append(
            f"    {label:<17} {run.model_name:<22} macro F1 {run.f1_macro:.4f}   "
            f"accuracy {run.accuracy:.4f} 95% CI {span}"
        )
        lines.append(f"    {'':<17} {_evaluation_phrase(run)}   {run.summary}")

    divisors = [d for d in (best.divisor, best_transformer.divisor) if d is not None]
    lines.append("")
    if len(divisors) == 2:
        shared = shared_evaluation_size(divisors)
        lines.append(
            f"    one evaluation set behind both would need {shared:,} samples, and the two "
            "intervals do not overlap, so the gap between them is the data and not noise"
        )
    else:
        lines.append(
            "    one of the two was logged too coarsely to place, so the comparison "
            "cannot even be shown to be unfair -- only that it is unsupported"
        )
    lines.append("")

    nearest = report.nearest_logged()
    gaps = [match.gap for match in nearest]
    off_type = [match for match in nearest if not match.same_model_type]
    lines.append("  and the two records do not meet")
    lines.append(
        f"    no submitted score appears in the log: the nearest logged macro F1 is "
        f"{min(gaps):.4f} to {max(gaps):.4f} away, never equal"
    )
    lines.append(
        f"    {len(off_type)} of {len(nearest)} of those nearest scores were not even set by "
        "the same kind of model, so matching by value identifies nothing"
    )
    return lines


def audit(report: SelectionReport) -> dict[str, Any]:
    """The published numbers, as a dict the tests can assert against."""
    best, best_transformer = report.headline_pair()
    divisors = [d for d in (best.divisor, best_transformer.divisor) if d is not None]
    return {
        "submitted_rows": len(report.submitted),
        "logged_runs": len(report.runs),
        "measurable_runs": len(report.measurable),
        "distinct_divisors": len(report.divisors),
        "collapsed_runs": len(report.collapsed()),
        "best_overall": best.model_name,
        "best_overall_f1_macro": best.f1_macro,
        "best_transformer": best_transformer.model_name,
        "best_transformer_f1_macro": best_transformer.f1_macro,
        "shared_evaluation_size": shared_evaluation_size(divisors),
        "submitted_winner_weighted": report.ranked(macro=False)[0].model_name,
        "submitted_winner_macro": report.ranked(macro=True)[0].model_name,
        "largest_swing_places": max(abs(swing.places) for swing in report.swings()),
        "smallest_gap_to_a_logged_run": min(match.gap for match in report.nearest_logged()),
        "nearest_of_another_model_type": sum(
            not match.same_model_type for match in report.nearest_logged()
        ),
        "neutral_caps": sorted(report.neutral_caps()),
    }
