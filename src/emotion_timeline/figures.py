"""Shared figure machinery: one palette, one styling pass, one staleness check.

Each stage of the study owns its own figures but draws them the same way, so a
reader moving between chapters reads the colours the same way in each. Red is
always the thing that went wrong, green always the thing that went right, teal
everything else.

Figures are checked by **what they were drawn from**, not by their bytes.
Byte-comparing a regenerated PNG against the committed one cannot work:
matplotlib lays out text with whatever fonts the machine has, so the same figure
differs between a Windows laptop and a Linux runner, which is exactly how this
repository's first CI run failed. Each PNG instead carries the SHA-256 of its
source data in a tEXt chunk, and the check compares that. Same answer
everywhere, and it fails for the right reason: the picture being older than the
numbers.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # matplotlib is imported inside the figure functions only
    from matplotlib.axes import Axes

# One colour per role, used the same way in every figure.
INK = "#1d2427"
MUTED = "#6b7a7d"
GRID = "#dde4e3"
WRONG = "#ab2f27"
RIGHT = "#2a6a46"
ACCENT = "#0d6a70"

# One colour per emotion, so a reader who learns the timeline's key can read it
# anywhere else the seven classes are drawn. Neutral borrows MUTED and Surprise
# ACCENT deliberately: the two are the background against which the rest reads.
EMOTION_COLOURS = {
    "Anger": "#ab2f27",
    "Disgust": "#7a6a1f",
    "Fear": "#5b4b8a",
    "Joy": "#c98a1e",
    "Neutral": "#6b7a7d",
    "Sadness": "#2f5d8a",
    "Surprise": "#0d6a70",
}

STAMP_KEY = "Source-SHA256"

# A figure function takes the stage's report and a destination, and returns the
# path it wrote.
Renderer = Callable[[Any, Path], Path]


def digest_of(path: str | Path) -> str:
    """SHA-256 of a source file, which is what a figure records having read."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(digest: str) -> dict[str, str]:
    """PNG text chunks recording which data the figure was drawn from."""
    return {"Software": "emotion-timeline", STAMP_KEY: digest}


def read_stamp(path: str | Path) -> str | None:
    """The Source-SHA256 recorded in a PNG, or None if it carries no stamp.

    Walks tEXt chunks directly rather than adding an image library to a
    dependency list that is deliberately four packages long.
    """
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    offset = 8
    while offset + 12 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if kind == b"tEXt" and b"\x00" in body:
            key, _, value = body.partition(b"\x00")
            if key.decode("latin-1") == STAMP_KEY:
                return value.decode("latin-1")
        offset += 12 + length
    return None


def check_current(names: Iterable[str], digest: str, out_dir: str | Path) -> list[str]:
    """Figures that are missing, unstamped, or drawn from older data."""
    directory = Path(out_dir)
    problems: list[str] = []
    for name in names:
        path = directory / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        found = read_stamp(path)
        if found is None:
            problems.append(f"{name}: carries no {STAMP_KEY} stamp")
        elif found != digest:
            problems.append(f"{name}: drawn from {found[:12]}, source is {digest[:12]}")
    return problems


def render_all(renderers: Mapping[str, Renderer], report: Any, out_dir: str | Path) -> list[Path]:
    """Write every figure a stage owns. Deterministic, so the result is diffable."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [render(report, directory / name) for name, render in renderers.items()]


def style(ax: Axes, axis: Literal["x", "y", "both"] = "x") -> None:
    """The house chart style: gridlines behind the data, no box around it."""
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    hide = ("top", "right", "left") if axis == "x" else ("top", "right")
    for side in hide:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
