"""Rebuilding the training set from its public source, one accounted-for step at a time.

Every stage records how many rows went in and how many came out, so the whole
funnel from 552,821 to 419,180 can be checked against
``benchmarks/dataset/build-record.json`` rather than taken on trust. That record
is what CI verifies; this module is what a reader runs to reproduce it.

**What cannot be reproduced.** The published dataset is 428,331 rows: the
419,180 built here plus 9,151 synthetic Disgust examples generated for the
original project. Disgust is not a target class in the upstream corpus, so
those rows were written to give the class a floor. The generated file did not
survive, and no committed copy exists, so ``build`` reaches 419,180 and stops.
The docs say so, the record stores both numbers, and nothing in the README
quotes a figure that depends on the missing 9,151 without saying which it is.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from emotion_timeline.data import clean
from emotion_timeline.data import labels as lbl

if TYPE_CHECKING:
    import pandas as pd

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "dataset"
DEFAULT_RECORD = BENCHMARK / "build-record.json"

SOURCE_DATASET = "cirimus/super-emotion"
MIN_TOKENS = 3
MAX_TOKENS = 512

# The upstream corpus bundles six datasets. GoEmotions is kept only where it
# contributes disgust: its 27 fine-grained labels map onto the seven classes so
# unevenly that including it whole would have swamped every other source with
# Reddit comments, 52,534 of them against 13,708 from MELD.
KEEP_ONLY_FOR = {"GoEmotions": "disgust"}


@dataclass
class Step:
    """One stage of the funnel: what it did, and what it cost in rows."""

    name: str
    rows_in: int
    rows_out: int
    note: str = ""

    @property
    def removed(self) -> int:
        return self.rows_in - self.rows_out


@dataclass
class BuildRecord:
    """The result of a build, comparable against the committed record."""

    source: str
    steps: list[Step] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return self.steps[-1].rows_out if self.steps else 0

    def add(self, name: str, rows_in: int, rows_out: int, note: str = "") -> None:
        self.steps.append(Step(name, rows_in, rows_out, note))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows": self.rows,
            "steps": [
                {"name": s.name, "rows_in": s.rows_in, "rows_out": s.rows_out, "note": s.note}
                for s in self.steps
            ],
            "class_counts": self.counts,
        }


@dataclass(frozen=True)
class DatasetReport:
    """The committed record of a build, as read back for checking and drawing."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RECORD) -> DatasetReport:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self.raw["steps"])

    @property
    def rows(self) -> int:
        """Rows the public build produces. Not the size of the published set."""
        return int(self.raw["rows"])

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(self.raw["class_counts"])

    @property
    def published(self) -> dict[str, Any]:
        """The set the model was trained on: this build plus the synthetic rows."""
        return dict(self.raw["published"])

    @property
    def composition(self) -> dict[str, dict[str, int]]:
        return dict(self.raw["source_composition"])


def check_consistency(report: DatasetReport) -> list[str]:
    """Every arithmetic relation the recorded build has to satisfy.

    The same idea as the error report's checks, for the same reason: a reader
    who cannot rerun the build (it needs a 550,000-row download) can at least
    see that the numbers hold together.
    """
    record = report.raw
    problems: list[str] = []
    steps = record["steps"]

    for earlier, later in itertools.pairwise(steps):
        if earlier["rows_out"] != later["rows_in"]:
            problems.append(
                f"{later['name']}: takes {later['rows_in']} rows but "
                f"{earlier['name']} produced {earlier['rows_out']}"
            )

    if steps and steps[-1]["rows_out"] != record["rows"]:
        problems.append(
            f"final step gives {steps[-1]['rows_out']} rows, header says {record['rows']}"
        )

    total = sum(record["class_counts"].values())
    if total != record["rows"]:
        problems.append(f"class counts sum to {total}, build produced {record['rows']}")

    published = record["published"]
    reconstructed = record["rows"] + published["synthetic_disgust_rows"]
    if reconstructed != published["rows"]:
        problems.append(
            f"{record['rows']} reproducible + {published['synthetic_disgust_rows']} synthetic "
            f"= {reconstructed}, but the published set is {published['rows']}"
        )

    published_total = sum(published["class_counts"].values())
    if published_total != published["rows"]:
        problems.append(
            f"published class counts sum to {published_total}, header says {published['rows']}"
        )

    for name, count in record["class_counts"].items():
        synthetic = published["synthetic_disgust_rows"] if name == "Disgust" else 0
        if count + synthetic != published["class_counts"][name]:
            problems.append(
                f"{name}: {count} reproducible + {synthetic} synthetic does not reach "
                f"the published {published['class_counts'][name]}"
            )

    return problems


def load_source(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Every split of the upstream corpus, concatenated.

    The published splits are not kept: the project made its own stratified
    split afterwards, so preserving the upstream one would only constrain it.
    """
    import pandas as pd
    from datasets import load_dataset

    dataset = load_dataset(SOURCE_DATASET, cache_dir=str(cache_dir) if cache_dir else None)
    frames = [pd.DataFrame(dataset[split]) for split in ("train", "validation", "test")]
    return pd.concat(frames, ignore_index=True)


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def filter_sources(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop GoEmotions rows that carry no disgust annotation."""

    def keep(row: Any) -> bool:
        wanted = KEEP_ONLY_FOR.get(row["source"])
        return True if wanted is None else wanted in _as_list(row["labels_source"])

    return frame[frame.apply(keep, axis=1)].reset_index(drop=True)


def clean_early(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalisation up to the point duplicates were removed."""
    frame = frame.copy()
    frame["text"] = frame["text"].astype(str).map(clean.clean_early)
    return frame


def clean_late(frame: pd.DataFrame) -> pd.DataFrame:
    """Case, punctuation and repeated characters, applied to what survived."""
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean.clean_late)
    return frame


def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """The final pass, once short rows are already gone."""
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean.tidy)
    return frame


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """First occurrence wins. 9% of the corpus is repeated text."""
    return frame.drop_duplicates(subset="text", keep="first").reset_index(drop=True)


def filter_by_length(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep 3 to 512 whitespace tokens.

    Under three tokens is mostly acknowledgements -- "ok!", "hey!" -- that carry
    no emotion a reader could recover either. Over 512 is past what the model
    reads anyway.
    """
    frame = frame.copy()
    frame["token_count"] = frame["text"].map(lambda text: len(str(text).split()))
    keep = (frame["token_count"] >= MIN_TOKENS) & (frame["token_count"] <= MAX_TOKENS)
    return frame[keep].reset_index(drop=True)


def drop_love(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the rows labelled Love, which has no seven-class equivalent.

    The original build tried to relabel these to the emotion their source
    annotation named, and relabelled exactly zero of them: the lookup was keyed
    by the capitalised seven-class names while the source annotations are
    lowercase and fine-grained, so no candidate ever matched. Fixing the case
    would not have saved them either -- ``love`` and ``admiration`` have no
    seven-class equivalent to be relabelled *to*, which is why the class was
    being removed in the first place. The rows were dropped, and this reproduces
    that, because it is what the trained model saw.
    """
    keep = ~frame["labels_str"].map(lambda v: lbl.is_dropped(_as_list(v)))
    return frame[keep].reset_index(drop=True)


def assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one label per row, then restore the annotated classes."""
    frame = frame.copy()
    collapsed = frame["labels_str"].map(lambda v: lbl.collapse(_as_list(v)))
    frame["label"] = [
        lbl.restore(label, _as_list(sources))
        for label, sources in zip(collapsed, frame["labels_source"], strict=True)
    ]
    return frame


def build(cache_dir: str | Path | None = None) -> tuple[pd.DataFrame, BuildRecord]:
    """The whole funnel, from the public corpus to the labelled training set."""
    record = BuildRecord(source=SOURCE_DATASET)

    frame = load_source(cache_dir)
    record.add("load", len(frame), len(frame), f"every split of {SOURCE_DATASET}")

    stages: tuple[tuple[str, Any, str], ...] = (
        ("filter-sources", filter_sources, "GoEmotions kept only where it annotates disgust"),
        ("normalise", clean_early, "quotes, emoji, slang, entity masking"),
        ("deduplicate", deduplicate, "identical text, before case is folded"),
        ("fold-case", clean_late, "lowercase with [CAPS], punctuation, repeats"),
        ("filter-length", filter_by_length, f"{MIN_TOKENS} to {MAX_TOKENS} tokens"),
        ("tidy", tidy, "collapse runs of identical placeholders"),
        ("drop-love", drop_love, "no seven-class equivalent"),
        ("assign-labels", assign_labels, "collapse by priority, restore from source"),
    )
    for name, step, note in stages:
        before = len(frame)
        frame = step(frame)
        record.add(name, before, len(frame), note)

    frame = frame[["text", "label", "source", "labels_source", "token_count"]]
    record.counts = {
        str(name): int(count) for name, count in frame["label"].value_counts().sort_index().items()
    }
    return frame, record


def iter_progress(record: BuildRecord) -> Iterator[str]:
    """One line per step, for the command line."""
    width = max(len(s.name) for s in record.steps)
    for step in record.steps:
        removed = f"-{step.removed:,}" if step.removed else ""
        yield f"  {step.name:<{width}}  {step.rows_out:>7,} rows  {removed:>9}  {step.note}"


def compare(record: BuildRecord, expected: DatasetReport) -> list[str]:
    """Every way a fresh build can disagree with the committed one.

    Row counts and class counts are exact, not approximate. The source corpus is
    a fixed published artefact and every step here is deterministic, so a
    difference means the code changed, not that the data drifted.
    """
    problems: list[str] = []

    if record.rows != expected.rows:
        problems.append(f"{record.rows:,} rows, record says {expected.rows:,}")

    built = {step.name: step for step in record.steps}
    for step in expected.steps:
        name = step["name"]
        if name not in built:
            problems.append(f"step '{name}' is in the record but was not run")
        elif built[name].rows_out != step["rows_out"]:
            problems.append(
                f"{name}: {built[name].rows_out:,} rows out, record says {step['rows_out']:,}"
            )
    for name in built:
        if all(step["name"] != name for step in expected.steps):
            problems.append(f"step '{name}' was run but is not in the record")

    for name, count in expected.class_counts.items():
        found = record.counts.get(name, 0)
        if found != count:
            problems.append(f"{name}: {found:,} rows, record says {count:,}")
    for name in record.counts:
        if name not in expected.class_counts:
            problems.append(f"class '{name}' is not in the record")

    return problems
