"""Where the emotion classifier goes wrong, and what that predicts.

Reads the recorded statistics in ``benchmarks/error-analysis/`` and renders the
figures the README shows. The raw per-sample predictions were not kept, so this
renders from summary statistics rather than recomputing them -- which is why
:func:`check_consistency` exists. It is the substitute for being able to re-derive
the numbers: everything that can be cross-checked, is.

The headline is not the accuracy. It is that three surface features nobody
trains on -- an exclamation mark, a question mark, a shouted word -- each raise
the error rate from under 10% to over 55%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "error-analysis"
DEFAULT_REPORT = BENCHMARK / "held-out-64250.json"

# One colour per role, used the same way in every figure: the thing that went
# wrong is always this red, the thing that went right is always this green.
INK = "#1d2427"
MUTED = "#6b7a7d"
GRID = "#dde4e3"
WRONG = "#ab2f27"
RIGHT = "#2a6a46"
ACCENT = "#0d6a70"


@dataclass(frozen=True, slots=True)
class ClassResult:
    name: str
    samples: int
    errors: int
    error_rate: float
    confused_with: dict[str, int]

    @property
    def recall(self) -> float:
        """Share of this class the model got right. 1 - error rate, by definition."""
        return 1.0 - self.error_rate


@dataclass(frozen=True, slots=True)
class ErrorReport:
    """One evaluation's worth of recorded error statistics."""

    split: str
    total_samples: int
    total_errors: int
    accuracy: float
    classes: dict[str, ClassResult]
    length: dict
    textual_features: dict[str, dict]
    vocabulary: dict[str, list[str]]
    confidence: dict

    @classmethod
    def load(cls, path: str | Path = DEFAULT_REPORT) -> ErrorReport:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        classes = {
            name: ClassResult(
                name=name,
                samples=body["samples"],
                errors=body["errors"],
                error_rate=body["error_rate"],
                confused_with=body["confused_with"],
            )
            for name, body in raw["classes"].items()
        }
        return cls(
            split=raw["split"],
            total_samples=raw["total_samples"],
            total_errors=raw["total_errors"],
            accuracy=raw["accuracy"],
            classes=classes,
            length=raw["length"],
            textual_features={
                k: v for k, v in raw["textual_features"].items() if not k.startswith("_")
            },
            vocabulary={k: v for k, v in raw["vocabulary"].items() if not k.startswith("_")},
            confidence=raw["confidence"],
        )

    def hardest(self, n: int = 3) -> list[ClassResult]:
        return sorted(self.classes.values(), key=lambda c: c.error_rate, reverse=True)[:n]

    def worst_features(self) -> list[tuple[str, float]]:
        """Surface markers ordered by how much they multiply the error rate."""
        ratios = [
            (name, body["error_rate_present"] / body["error_rate_absent"])
            for name, body in self.textual_features.items()
            if body["error_rate_absent"]
        ]
        return sorted(ratios, key=lambda pair: pair[1], reverse=True)


def check_consistency(report: ErrorReport) -> list[str]:
    """Every arithmetic relation the recorded numbers have to satisfy.

    Returns the problems found, empty if the report holds together. Rendering a
    figure from statistics nobody can recompute is only defensible if the
    statistics are at least self-consistent.
    """
    problems: list[str] = []

    sample_total = sum(c.samples for c in report.classes.values())
    if sample_total != report.total_samples:
        problems.append(f"class samples sum to {sample_total}, header says {report.total_samples}")

    error_total = sum(c.errors for c in report.classes.values())
    if error_total != report.total_errors:
        problems.append(f"class errors sum to {error_total}, header says {report.total_errors}")

    implied = 1.0 - report.total_errors / report.total_samples
    if abs(implied - report.accuracy) > 5e-4:
        problems.append(f"accuracy {report.accuracy} but errors/samples imply {implied:.4f}")

    for c in report.classes.values():
        implied_rate = c.errors / c.samples
        if abs(implied_rate - c.error_rate) > 5e-4:
            problems.append(
                f"{c.name}: error rate {c.error_rate} but "
                f"{c.errors}/{c.samples} = {implied_rate:.4f}"
            )
        listed = sum(c.confused_with.values())
        if listed > c.errors:
            problems.append(
                f"{c.name}: listed confusions ({listed}) exceed its errors ({c.errors})"
            )

    share = report.confidence["high_confidence_errors"] / report.total_errors
    if abs(share - report.confidence["high_confidence_error_share_of_errors"]) > 5e-4:
        problems.append(f"high-confidence error share does not match: implied {share:.4f}")

    return problems


# --- figures ----------------------------------------------------------------


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def figure_textual_features(report: ErrorReport, path: Path) -> Path:
    """The finding worth leading with: punctuation predicts failure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(n, b) for n, b in report.textual_features.items() if b["present_samples"] >= 100]
    items.sort(key=lambda kv: kv[1]["error_rate_present"])
    labels = [f"{n}\n({b['present_samples']:,} samples)" for n, b in items]
    present = [b["error_rate_present"] * 100 for _, b in items]
    absent = [b["error_rate_absent"] * 100 for _, b in items]

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=160)
    y = range(len(items))
    ax.barh([i + 0.19 for i in y], present, height=0.36, color=WRONG, label="marker present")
    ax.barh([i - 0.19 for i in y], absent, height=0.36, color=RIGHT, label="marker absent")
    for i, (p, a) in enumerate(zip(present, absent, strict=True)):
        ax.text(p + 1, i + 0.19, f"{p:.1f}%", va="center", fontsize=9, color=WRONG, weight="bold")
        ax.text(a + 1, i - 0.19, f"{a:.1f}%", va="center", fontsize=9, color=RIGHT)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(present) * 1.18)
    ax.set_xlabel("error rate (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "Three surface markers each take the error rate past 55%",
        color=INK,
        fontsize=12,
        weight="bold",
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figure_class_difficulty(report: ErrorReport, path: Path) -> Path:
    """Per-class error rate against support: the rare classes are the hard ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(report.classes.values(), key=lambda c: c.error_rate)
    names = [c.name for c in ordered]
    rates = [c.error_rate * 100 for c in ordered]
    supports = [c.samples for c in ordered]

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=160)
    colours = [WRONG if r > report.accuracy * 0 + 15 else ACCENT for r in rates]
    bars = ax.barh(names, rates, color=colours, height=0.62)
    for bar, rate, support in zip(bars, rates, supports, strict=True):
        ax.text(
            rate + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%   n={support:,}",
            va="center",
            fontsize=9,
            color=MUTED,
        )
    ax.set_xlim(0, max(rates) * 1.35)
    ax.set_xlabel("error rate (%)", color=MUTED, fontsize=9)
    ax.set_title(
        f"Class difficulty is not uniform — {report.total_errors:,} errors "
        f"in {report.total_samples:,} samples",
        color=INK,
        fontsize=12,
        weight="bold",
        loc="left",
        pad=12,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figure_confidence(report: ErrorReport, path: Path) -> Path:
    """Confidence separates well on average, and 625 errors ignore that."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=160)
    correct = report.confidence["mean_when_correct"]
    incorrect = report.confidence["mean_when_incorrect"]
    ax.barh(["correct"], [correct], color=RIGHT, height=0.5)
    ax.barh(["incorrect"], [incorrect], color=WRONG, height=0.5)
    ax.text(
        correct + 0.012, 0, f"{correct:.3f}", va="center", fontsize=10, color=RIGHT, weight="bold"
    )
    ax.text(
        incorrect + 0.012,
        1,
        f"{incorrect:.3f}",
        va="center",
        fontsize=10,
        color=WRONG,
        weight="bold",
    )
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("mean predicted confidence", color=MUTED, fontsize=9)
    n = report.confidence["high_confidence_errors"]
    share = report.confidence["high_confidence_error_share_of_errors"] * 100
    ax.set_title(
        f"A wide confidence gap — but {n} errors ({share:.1f}%) are confident anyway",
        color=INK,
        fontsize=12,
        weight="bold",
        loc="left",
        pad=12,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figure_length(report: ErrorReport, path: Path) -> Path:
    """Short inputs fail more often; the gap is small but overwhelming in n."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=160)
    metrics = ["mean_characters", "mean_words"]
    labels = ["characters", "words"]
    x = range(len(metrics))
    right = [report.length["correct"][m] for m in metrics]
    wrong = [report.length["incorrect"][m] for m in metrics]
    ax.barh([i + 0.18 for i in x], right, height=0.34, color=RIGHT, label="correct")
    ax.barh([i - 0.18 for i in x], wrong, height=0.34, color=WRONG, label="incorrect")
    for i, (r, w) in enumerate(zip(right, wrong, strict=True)):
        ax.text(r + 1.2, i + 0.18, f"{r:.2f}", va="center", fontsize=9, color=RIGHT, weight="bold")
        ax.text(w + 1.2, i - 0.18, f"{w:.2f}", va="center", fontsize=9, color=WRONG)
    ax.set_yticks(list(x))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(right) * 1.2)
    ax.set_xlabel("mean length", color=MUTED, fontsize=9)
    p = report.length["p_value"]
    ax.set_title(
        f"Correct predictions are the longer inputs (Mann-Whitney p ≈ {p:.2e})",
        color=INK,
        fontsize=12,
        weight="bold",
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


FIGURES = {
    "error-by-textual-feature.png": figure_textual_features,
    "error-by-class.png": figure_class_difficulty,
    "confidence-gap.png": figure_confidence,
    "error-by-length.png": figure_length,
}


def render_all(report: ErrorReport, out_dir: str | Path) -> list[Path]:
    """Write every figure. Deterministic, so CI can diff the result."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [render(report, directory / name) for name, render in FIGURES.items()]
