"""Splitting the rebuilt dataset reproducibly, without committing any of it.

A split is a claim about which rows a number was computed over, so it has to be
checkable by someone who has only this repository. Three decisions follow.

**Rows are keyed by their own content, never by position**, so the split cannot
depend on the order the build happens to emit rows in. Rerunning ``build-dataset``
on another machine and splitting again gives the same three sets even if pandas
hands the rows back differently.

The key covers text, label and source together, because **text alone is not
unique**. Deduplication runs before case is folded -- `docs/dataset.md` explains
why, and why moving it would stop the build reproducing -- so 897 rows reach the
training set sharing their text with another row. Where even text, label and
source coincide the rows are identical in every column, and those are numbered by
occurrence: identical things are interchangeable, so numbering them gives the same
set of keys whatever order they arrived in.

**What gets committed is counts and digests, not rows.** Each split carries a
SHA-256 over its sorted keys: 64 characters that pin membership exactly while the
50 MB of text stays out of git, where ``CLAUDE.md`` says it belongs.

The quota is **largest remainder** rather than per-class rounding, because
rounding each class on its own misses. 15% of 419,180 is exactly 62,877 and the
seven independently rounded shares sum to 62,876; the spare row goes to the class
with the largest fraction left over, ties broken by class name so the result never
depends on dictionary order. That tie-break is one of two reasons this held-out
set is not the one the model card reports on, and the smaller one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emotion_timeline.model.card import HELD_OUT_FRACTION

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "training"
DEFAULT_MANIFEST = BENCHMARK / "split-manifest.json"

#: Fixed, so the split is a property of the data rather than of the run.
SEED = 20251115

#: Separator inside a row key. A control character, so no field value contains it.
FIELD = "\x1f"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLITS = (TRAIN, VALIDATION, TEST)


def row_key(text: str, label: str, source: str, occurrence: int, seed: int = SEED) -> str:
    """One row's identity, derived from the row itself."""
    parts = (str(seed), text, label, source, str(occurrence))
    return hashlib.sha256(FIELD.join(parts).encode()).hexdigest()


def row_keys(
    texts: Sequence[str],
    labels: Sequence[str],
    sources: Sequence[str],
    seed: int = SEED,
) -> list[str]:
    """Key every row, numbering rows that are identical in all three fields.

    The numbering is what makes the result independent of order: a group of *k*
    identical rows always produces occurrences 0 to *k*-1, whichever of them the
    build emitted first.
    """
    seen: dict[tuple[str, str, str], int] = {}
    keys: list[str] = []
    for row in zip(texts, labels, sources, strict=True):
        occurrence = seen.get(row, 0)
        seen[row] = occurrence + 1
        keys.append(row_key(*row, occurrence, seed=seed))
    return keys


def counts_of(labels: Iterable[str]) -> dict[str, int]:
    """How many rows each class holds."""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def quota(counts: Mapping[str, int], fraction: float) -> dict[str, int]:
    """Split ``fraction`` of the total across classes, losing no row to rounding.

    Every class takes its floor, then the classes with the largest fractions left
    over take one more each until the total is reached. Ties go to the class whose
    name sorts first, which matters only in that it has to be decided somewhere.
    """
    total = sum(counts.values())
    target = round(total * fraction)
    exact = {name: count * fraction for name, count in counts.items()}
    taken = {name: int(value) for name, value in exact.items()}
    short = target - sum(taken.values())
    by_remainder = sorted(counts, key=lambda name: (-(exact[name] - taken[name]), name))
    for name in by_remainder[:short]:
        taken[name] += 1
    return taken


def assign(
    keys: Sequence[str],
    labels: Sequence[str],
    fraction: float = HELD_OUT_FRACTION,
) -> list[str]:
    """Put every row in a split, stratified by class and fixed by its key.

    Within a class the rows are ordered by key: the first ``fraction`` of them are
    the test set, the next ``fraction`` the validation set, the rest train. Test
    and validation therefore hold the same number of rows.

    Rows are placed one at a time, as the original did, rather than grouping rows
    that share a text. See :func:`text_overlap` for what that costs and why it is
    measured rather than avoided.
    """
    if len(keys) != len(labels):
        raise ValueError(f"{len(keys)} keys against {len(labels)} labels")
    held = quota(counts_of(labels), fraction)
    rows: dict[str, list[tuple[str, int]]] = {}
    for index, (key, label) in enumerate(zip(keys, labels, strict=True)):
        rows.setdefault(label, []).append((key, index))

    out = [TRAIN] * len(keys)
    for label, ordered in rows.items():
        ordered.sort()
        take = held[label]
        for position, (_, index) in enumerate(ordered):
            if position < take:
                out[index] = TEST
            elif position < 2 * take:
                out[index] = VALIDATION
    return out


def text_overlap(
    texts: Sequence[str],
    labels: Sequence[str],
    assignment: Sequence[str],
) -> dict[str, int]:
    """Held-out rows whose exact text also appears in train, and how many disagree.

    Splitting row by row rather than by text is what the original did, and it is
    kept so the two evaluations stay comparable. It has a price: deduplication
    runs before case folding, so a text can reach both halves of the split, and
    where the two copies carry different labels one of them is a guaranteed error
    rather than a gift. Both numbers are small. They are measured and published
    rather than assumed away, which is the same treatment `docs/dataset.md` gives
    the URL bug.
    """
    trained: dict[str, set[str]] = {}
    for text, label, split in zip(texts, labels, assignment, strict=True):
        if split == TRAIN:
            trained.setdefault(text, set()).add(label)

    shared = 0
    contradicted = 0
    for text, label, split in zip(texts, labels, assignment, strict=True):
        if split == TRAIN:
            continue
        trained_labels = trained.get(text)
        if trained_labels is None:
            continue
        shared += 1
        if label not in trained_labels:
            contradicted += 1
    return {
        "held_out_rows_sharing_a_training_text": shared,
        "of_those_labelled_differently": contradicted,
    }


def membership_digest(keys: Iterable[str]) -> str:
    """Pin exactly which rows a split holds, in 64 characters.

    Sorted before hashing, so what is pinned is the set rather than an order no
    reader could reproduce.
    """
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def manifest(
    keys: Sequence[str],
    labels: Sequence[str],
    fraction: float = HELD_OUT_FRACTION,
    seed: int = SEED,
) -> dict[str, object]:
    """The committed record: counts anyone can check, digests anyone can reproduce."""
    assignment = assign(keys, labels, fraction)
    splits: dict[str, object] = {}
    for name in SPLITS:
        members = [key for key, split in zip(keys, assignment, strict=True) if split == name]
        chosen = [label for label, split in zip(labels, assignment, strict=True) if split == name]
        splits[name] = {
            "rows": len(members),
            "class_counts": dict(sorted(counts_of(chosen).items())),
            "membership_sha256": membership_digest(members),
        }
    return {
        "seed": seed,
        "held_out_fraction": fraction,
        "allocation": "largest remainder, ties broken by class name",
        "source_rows": len(keys),
        "class_counts": dict(sorted(counts_of(labels).items())),
        "splits": splits,
    }


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """The committed record of a split, as read back for checking and verifying."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MANIFEST) -> SplitManifest:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def fraction(self) -> float:
        return float(self.raw["held_out_fraction"])

    @property
    def source_rows(self) -> int:
        return int(self.raw["source_rows"])

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(self.raw["class_counts"])

    @property
    def splits(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw["splits"])

    @property
    def overlap(self) -> dict[str, int]:
        return dict(self.raw["text_overlap"])

    def part(self, name: str) -> dict[str, Any]:
        return dict(self.splits[name])


def check_consistency(record: SplitManifest) -> list[str]:
    """Every relation the manifest has to satisfy for its numbers to mean anything.

    The same job `check_consistency` does in every other stage: the record has to
    hold together before anything is drawn or published from it. A split whose
    parts do not sum to the whole is not a split.
    """
    problems: list[str] = []

    if set(record.splits) != set(SPLITS):
        problems.append(f"splits are {sorted(record.splits)}, expected {sorted(SPLITS)}")
        return problems

    totalled: dict[str, int] = {}
    for name in SPLITS:
        part = record.part(name)
        counts = dict(part["class_counts"])
        if sum(counts.values()) != part["rows"]:
            problems.append(
                f"{name}: class counts sum to {sum(counts.values())}, header says {part['rows']}"
            )
        digest = str(part["membership_sha256"])
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            problems.append(f"{name}: membership digest is not a sha256")
        for label, count in counts.items():
            totalled[label] = totalled.get(label, 0) + count

    if totalled != record.class_counts:
        problems.append("the three splits do not add back up to the class counts they came from")

    rows = sum(int(record.part(name)["rows"]) for name in SPLITS)
    if rows != record.source_rows:
        problems.append(f"splits hold {rows} rows, the record says {record.source_rows}")

    expected = quota(record.class_counts, record.fraction)
    for name in (TEST, VALIDATION):
        if dict(record.part(name)["class_counts"]) != expected:
            problems.append(f"{name} is not the {record.fraction:.0%} quota of the class counts")

    shared = record.overlap["held_out_rows_sharing_a_training_text"]
    if record.overlap["of_those_labelled_differently"] > shared:
        problems.append("more held-out rows contradict a training label than share a text at all")

    return problems


def verify(
    record: SplitManifest,
    texts: Sequence[str],
    labels: Sequence[str],
    sources: Sequence[str],
) -> list[str]:
    """Recompute the split from a rebuilt dataset and refuse to agree if it differs.

    Exact, not approximate, for the reason `build.compare` is: the build is
    deterministic and so is the split, so a difference means the code changed.
    """
    problems: list[str] = []
    if len(texts) != record.source_rows:
        problems.append(f"{len(texts)} rows given, the record was built from {record.source_rows}")
        return problems

    keys = row_keys(texts, labels, sources, seed=record.seed)
    assignment = assign(keys, labels, record.fraction)
    for name in SPLITS:
        members = [key for key, split in zip(keys, assignment, strict=True) if split == name]
        digest = membership_digest(members)
        recorded = str(record.part(name)["membership_sha256"])
        if digest != recorded:
            problems.append(
                f"{name}: membership is {digest[:16]}..., record says {recorded[:16]}..."
            )

    measured = text_overlap(texts, labels, assignment)
    if measured != record.overlap:
        problems.append(f"text overlap is {measured}, record says {record.overlap}")
    return problems


def describe(record: SplitManifest) -> Iterator[str]:
    """What the split is, in the terms the documentation uses."""
    held = record.fraction
    yield (
        f"{record.source_rows:,} rows, split "
        f"{1 - 2 * held:.0%} / {held:.0%} / {held:.0%}, seed {record.seed}"
    )
    yield ""
    for name in SPLITS:
        part = record.part(name)
        yield f"  {name:<11} {int(part['rows']):>7,}   {str(part['membership_sha256'])[:16]}..."
    yield ""
    yield "  held out, class by class"
    for label, count in sorted(record.part(TEST)["class_counts"].items(), key=lambda kv: -kv[1]):
        yield f"    {label:<9} {count:>6,}"
    yield ""
    shared = record.overlap["held_out_rows_sharing_a_training_text"]
    different = record.overlap["of_those_labelled_differently"]
    yield f"  {shared} held-out rows share their text with a training row"
    yield f"    {different} of those are labelled differently, so they cannot be answered correctly"


#: Written into the record, because a reader opening the JSON first needs to know
#: what it is and how to check it.
MANIFEST_COMMENT = [
    "How the rebuilt dataset is divided for training, and how anyone can check it.",
    "Written by `emotion-timeline split --write` over the rows that",
    "`build-dataset` reproduces. No row is recorded here: each split carries a",
    "SHA-256 over its sorted row keys instead, and a row key is the SHA-256 of the",
    "row's own text, label and source under the seed below. `split --verify`",
    "recomputes both from a rebuilt CSV and refuses to agree if either differs.",
    "The held-out set is 62,877 rows, which is not the 64,250 the model card",
    "reports on: the difference is the 15% share of the 9,151 synthetic Disgust",
    "rows that did not survive. Five of the seven per-class supports match the",
    "card's exactly all the same. See docs/dataset.md and docs/fine-tune.md.",
]


def build_manifest(
    texts: Sequence[str],
    labels: Sequence[str],
    sources: Sequence[str],
    fraction: float = HELD_OUT_FRACTION,
    seed: int = SEED,
) -> dict[str, object]:
    """The whole committed record, from a rebuilt dataset.

    One function rather than several so that what `split --write` produces and
    what `split --verify` checks cannot drift apart.
    """
    keys = row_keys(texts, labels, sources, seed=seed)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{len(keys) - len(set(keys))} rows could not be told apart")
    assignment = assign(keys, labels, fraction)
    return {
        "_comment": MANIFEST_COMMENT,
        **manifest(keys, labels, fraction, seed),
        "text_overlap": text_overlap(texts, labels, assignment),
    }


def write_manifest(record: dict[str, object], path: str | Path) -> Path:
    """Write a manifest to disk as the committed file, LF-terminated."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out
