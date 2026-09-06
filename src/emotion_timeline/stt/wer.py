"""Word error rate for the speech-to-text comparison.

The annotations these numbers come from are hand-counted: for each transcript
segment a human marked how many substitutions, insertions and deletions it
contained against the audio. There is no reference transcript in the data, so
WER cannot be recomputed by aligning two strings -- only recombined from the
counts.

That constrains the arithmetic. The token count available per row is the
*hypothesis* length, and WER is defined over the *reference* length, so the
reference is recovered from the edits::

    N_ref = N_hyp + deletions - insertions
    WER   = (S + I + D) / N_ref

A deletion means the reference had a word the system dropped, so it is missing
from the hypothesis and has to be added back. An insertion is the reverse.

The one thing that matters more than the arithmetic is :func:`in_window`. Two
systems segment the same audio differently -- in this benchmark Whisper emits
797 segments where AssemblyAI emits 316 -- so "the first 240 rows of each" is
not a comparison of the same audio. It is the mistake the original analysis
made, and it changed the answer by a factor of four. Always compare a time
window, never a row count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("start_s", "end_s", "hypothesis", "substitutions", "insertions", "deletions")


@dataclass(frozen=True, slots=True)
class WerResult:
    """One system's error rate over one window of audio."""

    system: str
    segments: int
    hypothesis_tokens: int
    substitutions: int
    insertions: int
    deletions: int
    window_start_s: float
    window_end_s: float

    @property
    def reference_tokens(self) -> int:
        """Reference length, recovered from the hypothesis length and the edits."""
        return self.hypothesis_tokens + self.deletions - self.insertions

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float:
        """Word error rate as a percentage. Zero-length reference scores 0.0."""
        return 100.0 * self.errors / self.reference_tokens if self.reference_tokens else 0.0

    @property
    def window_minutes(self) -> float:
        return (self.window_end_s - self.window_start_s) / 60.0

    def __str__(self) -> str:
        return (
            f"{self.system:<18} {self.wer:6.2f}%  "
            f"({self.errors} errors / {self.reference_tokens} reference tokens, "
            f"{self.segments} segments, {self.window_minutes:.1f} min)"
        )


def load(path: str | Path, system: str | None = None) -> pd.DataFrame:
    """Read one annotated transcript. The system name defaults to the filename."""
    frame = pd.read_csv(path, encoding="utf-8")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
    frame = frame.copy()
    frame["hypothesis"] = frame["hypothesis"].fillna("").astype(str)
    for column in ("substitutions", "insertions", "deletions"):
        frame[column] = frame[column].fillna(0).astype(int)
    frame.attrs["system"] = system or Path(path).stem
    return frame


def in_window(
    frame: pd.DataFrame, start_s: float = 0.0, end_s: float | None = None
) -> pd.DataFrame:
    """Segments that begin inside ``[start_s, end_s]``.

    Filtering on the start time rather than on overlap keeps each segment whole,
    so its hand-counted edits stay attached to the text they were counted
    against. A segment straddling the boundary is included or excluded entire.
    """
    selected = frame[frame["start_s"] >= start_s]
    if end_s is not None:
        selected = selected[selected["start_s"] <= end_s]
    result = selected.copy()
    result.attrs.update(frame.attrs)
    return result


def score(frame: pd.DataFrame, system: str | None = None) -> WerResult:
    """Total one system's annotations into a single error rate."""
    name = system or frame.attrs.get("system", "unnamed")
    if frame.empty:
        return WerResult(name, 0, 0, 0, 0, 0, 0.0, 0.0)
    tokens = sum(len(str(text).split()) for text in frame["hypothesis"])
    return WerResult(
        system=name,
        segments=len(frame),
        hypothesis_tokens=tokens,
        substitutions=int(frame["substitutions"].sum()),
        insertions=int(frame["insertions"].sum()),
        deletions=int(frame["deletions"].sum()),
        window_start_s=float(frame["start_s"].min()),
        window_end_s=float(frame["end_s"].max()),
    )


def compare(
    systems: dict[str, pd.DataFrame],
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[WerResult]:
    """Score every system over the same window, best first.

    Passing the same window to each is the whole point of this function; scoring
    systems separately is how the row-count mistake happens.
    """
    results = [score(in_window(f, start_s, end_s), name) for name, f in systems.items()]
    return sorted(results, key=lambda r: r.wer)


def annotated_extent(frame: pd.DataFrame) -> float:
    """Last start time carrying a marked error.

    Annotation usually thins out towards the end of a long file, and comparing
    a densely checked window against a sparsely checked one flatters whichever
    system was checked less. This is the honest right-hand edge of a comparison.
    """
    marked = frame[(frame["substitutions"] + frame["insertions"] + frame["deletions"]) > 0]
    return float(marked["start_s"].max()) if not marked.empty else 0.0
