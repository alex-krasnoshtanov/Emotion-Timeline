"""Rebuilding the Russian evaluation set, and saying what mapping it down costs.

`Djacon/ru-izard-emotions` is the only labelled Russian emotion corpus this
project has, and the original coursework trained a ruBERT on it once and then ran
a different, off-the-shelf model on untranslated transcripts with nothing to score
against. Used the other way round -- as a **held-out test set** -- it turns "we
decided there was no good Russian option" into a measurement.

Getting from its ten columns to our seven costs three things, and each is
recorded rather than assumed.

**Shame and guilt go.** Neither has a seven-class equivalent and neither is in
the client's set. The original dropped them too, which is the only reason the two
builds can be compared at all. Rows left with nothing after the drop go with them.

**Enthusiasm merges into Joy.** It is the one label with a near neighbour rather
than no home, so merging keeps rows a drop would lose from an already small set.
That is a judgement, so :func:`build` records how many rows it moves and
``--drop-enthusiasm`` builds the other version for comparison. This is the same
treatment `dataset.md` gives the 34,940 Love rows it could not place.

**Multi-label collapses by priority**, reusing `labels.collapse` and the same
``PRIORITY`` order the English build uses, so a Russian row and an English row
carrying the same pair of labels resolve the same way.

The register is wrong for the job and that has to be said out loud: this is
translated social-media text, and the pipeline it informs runs on documentary
speech. It is the only Russian ground truth available, not a representative one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from emotion_timeline.data.build import BuildRecord
from emotion_timeline.data.labels import EMOTIONS, collapse

if TYPE_CHECKING:
    import pandas as pd

SOURCE_DATASET = "Djacon/ru-izard-emotions"

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "russian"
DEFAULT_RECORD = BENCHMARK / "build-record.json"

#: The corpus's own ten columns, in its own order.
SOURCE_COLUMNS = (
    "neutral",
    "joy",
    "sadness",
    "anger",
    "enthusiasm",
    "surprise",
    "disgust",
    "fear",
    "guilt",
    "shame",
)

#: No seven-class equivalent, and not in the client's set. The original dropped
#: these two as well, which is what makes the two builds comparable.
DROPPED_COLUMNS = ("guilt", "shame")

#: The one column with a near neighbour rather than no home at all.
MERGED_INTO = {"enthusiasm": "Joy"}


def kept_columns(drop_enthusiasm: bool = False) -> tuple[str, ...]:
    """The source columns this build carries forward."""
    dropped = set(DROPPED_COLUMNS) | (set(MERGED_INTO) if drop_enthusiasm else set())
    return tuple(name for name in SOURCE_COLUMNS if name not in dropped)


def to_seven(active: list[str], drop_enthusiasm: bool = False) -> str | None:
    """One of our seven classes from a row's active columns, or ``None``.

    Enthusiasm becomes Joy before the collapse rather than after, so a row
    labelled both enthusiasm and joy is one Joy and not two votes for it.
    """
    names: list[str] = []
    for column in active:
        if column in DROPPED_COLUMNS:
            continue
        if column in MERGED_INTO:
            if drop_enthusiasm:
                continue
            names.append(MERGED_INTO[column])
        else:
            names.append(column.capitalize())
    return collapse(sorted(set(names), key=lambda name: EMOTIONS.index(name)))


def load_source(
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:  # pragma: no cover - downloads the corpus
    """Every split of the corpus, concatenated.

    All three, because the corpus's own train/validation/test split is not the
    one this project uses -- `training/splits.py` derives a split from row content
    so it can be pinned by digest, and that needs the whole set.
    """
    import pandas as pd
    from datasets import load_dataset

    dataset = load_dataset(SOURCE_DATASET, cache_dir=str(cache_dir) if cache_dir else None)
    frames = [split.to_pandas() for split in dataset.values()]
    combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return combined


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Identical text, keeping the first. The original deduplicated here too."""
    return frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


def drop_unlabelled(frame: pd.DataFrame, drop_enthusiasm: bool = False) -> pd.DataFrame:
    """Rows carrying nothing once the classes with no home are removed."""
    columns = list(kept_columns(drop_enthusiasm))
    return frame[frame[columns].sum(axis=1) > 0].reset_index(drop=True)


def active_columns(row: Any, drop_enthusiasm: bool = False) -> list[str]:
    """Which source columns a row has set."""
    return [name for name in kept_columns(drop_enthusiasm) if int(row[name]) == 1]


def assign_labels(frame: pd.DataFrame, drop_enthusiasm: bool = False) -> pd.DataFrame:
    """Collapse each row's columns to one of our seven."""
    labels: list[str | None] = [
        to_seven(active_columns(row, drop_enthusiasm), drop_enthusiasm)
        for _, row in frame.iterrows()
    ]
    out = frame.copy()
    out["label"] = labels
    return out[out["label"].notna()].reset_index(drop=True)


def build(
    cache_dir: str | Path | None = None,
    drop_enthusiasm: bool = False,
) -> tuple[pd.DataFrame, BuildRecord, int]:
    """The whole funnel, with every step's row count recorded.

    Returns the rows, the record, and how many rows carried ``enthusiasm`` --
    the last because it is the cost of the one mapping decision here, and a
    number nobody can check afterwards from the output alone.
    """
    record = BuildRecord(source=SOURCE_DATASET)
    frame = load_source(cache_dir)
    record.add("load", len(frame), len(frame), f"every split of {SOURCE_DATASET}")

    before = len(frame)
    frame = deduplicate(frame)
    record.add("deduplicate", before, len(frame), "identical text, keeping the first")

    enthusiasm = int(frame["enthusiasm"].sum())
    before = len(frame)
    frame = drop_unlabelled(frame, drop_enthusiasm)
    note = "no label left once guilt and shame go"
    if drop_enthusiasm:
        note += ", enthusiasm dropped as well"
    record.add("drop-unlabelled", before, len(frame), note)

    before = len(frame)
    frame = assign_labels(frame, drop_enthusiasm)
    record.add(
        "assign-labels",
        before,
        len(frame),
        "collapse by priority; enthusiasm becomes Joy"
        if not drop_enthusiasm
        else "collapse by priority; enthusiasm dropped",
    )
    # A source column so `training/splits.py` can key rows the same way it keys
    # the English build. One corpus, so one value.
    out = frame[["text", "label"]].assign(source=SOURCE_DATASET)
    record.counts.update(
        {str(name): int(count) for name, count in out["label"].value_counts().items()}
    )
    return out, record, enthusiasm


def as_record(
    record: BuildRecord,
    enthusiasm_rows: int,
    rows_moved: int,
    drop_enthusiasm: bool = False,
) -> dict[str, Any]:
    """The committed record, in the shape `data/build.py`'s already takes."""
    return {
        "_comment": [
            "The Russian evaluation set, rebuilt from Djacon/ru-izard-emotions.",
            "Written by `emotion-timeline build-russian`. The corpus is public and",
            "ungated; this file is the funnel, so `emotion-timeline russian` can",
            "report it without downloading anything. Register is translated social",
            "media, which is not the documentary speech the pipeline runs on -- it",
            "is the only Russian ground truth available, not a representative one.",
        ],
        "source": SOURCE_DATASET,
        "enthusiasm": {
            # The raw column count overstates the decision by about four and a
            # half times: most enthusiasm rows carry another label that wins the
            # priority collapse anyway, so only rows_moved actually change class.
            "rows": enthusiasm_rows,
            "rows_moved": rows_moved,
            "mapped_to": None if drop_enthusiasm else MERGED_INTO["enthusiasm"],
        },
        "dropped_columns": list(DROPPED_COLUMNS),
        "rows": record.rows,
        "steps": record.as_dict()["steps"],
        "class_counts": dict(sorted(record.counts.items())),
    }


@dataclass(frozen=True, slots=True)
class RussianReport:
    """The committed record of the Russian build, read back for checking."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RECORD) -> RussianReport:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def rows(self) -> int:
        return int(self.raw["rows"])

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self.raw["steps"])

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(self.raw["class_counts"])

    @property
    def enthusiasm(self) -> dict[str, Any]:
        return dict(self.raw["enthusiasm"])


def check_consistency(report: RussianReport) -> list[str]:
    """Every relation the record has to satisfy before anything is published."""
    problems: list[str] = []

    steps = report.steps
    for earlier, later in pairwise(steps):
        if earlier["rows_out"] != later["rows_in"]:
            problems.append(
                f"{earlier['name']} leaves {earlier['rows_out']} rows but "
                f"{later['name']} starts from {later['rows_in']}"
            )
    if steps and steps[-1]["rows_out"] != report.rows:
        problems.append(
            f"the funnel ends at {steps[-1]['rows_out']}, the header says {report.rows}"
        )

    total = sum(report.class_counts.values())
    if total != report.rows:
        problems.append(f"class counts sum to {total}, the build produced {report.rows}")

    unknown = set(report.class_counts) - set(EMOTIONS)
    if unknown:
        problems.append(f"classes that are not ours: {sorted(unknown)}")

    enthusiasm = report.enthusiasm
    if enthusiasm.get("rows_moved", 0) > enthusiasm.get("rows", 0):
        problems.append("more rows changed class than carried enthusiasm at all")

    mapped = enthusiasm.get("mapped_to")
    if mapped is not None and mapped not in EMOTIONS:
        problems.append(f"enthusiasm is mapped to {mapped}, which is not one of the seven")

    return problems


def compare(record: BuildRecord, expected: RussianReport) -> list[str]:
    """Exact, not approximate, for the reason `build.compare` is."""
    problems: list[str] = []
    if record.rows != expected.rows:
        problems.append(f"build produced {record.rows} rows, the record says {expected.rows}")
    for name, count in sorted(record.counts.items()):
        recorded = expected.class_counts.get(name)
        if recorded != count:
            problems.append(f"class '{name}': built {count}, record says {recorded}")
    for name in sorted(set(expected.class_counts) - set(record.counts)):
        problems.append(f"class '{name}' is in the record but not in the build")
    return problems


def iter_progress(record: BuildRecord) -> Iterator[str]:
    """One line per step, the way `build-dataset` prints it."""
    for step in record.as_dict()["steps"]:
        removed = step["rows_in"] - step["rows_out"]
        change = f"{-removed:>8,}" if removed else " " * 8
        yield f"  {step['name']:<16}{step['rows_out']:>8,} rows {change}  {step['note']}"
