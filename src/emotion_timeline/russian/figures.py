"""One picture of the Russian comparison, with the coverage kept next to the accuracy.

The chart has to make one thing impossible to miss: the tallest bar is not the
best classifier. The agreement filter scores highest and answers fewer than half
the rows, so its bar carries its coverage as a label and is drawn hatched, while
every full-coverage bar is solid. A reader who takes only the shape away should
still take away the right shape.
"""

from __future__ import annotations

from pathlib import Path

from emotion_timeline import figures
from emotion_timeline.russian.compare import Comparison


def figure_approaches(report: Comparison, path: Path) -> Path:
    """Accuracy per approach, with the partial-coverage rule marked as partial."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, float, float | None, bool]] = []
    for name, block in report.approaches.items():
        rows.append((name, float(block["accuracy"]), None, False))
    for name, block in report.combinations.items():
        coverage = float(block["coverage"]) if "coverage" in block else None
        rows.append((name, float(block["accuracy"]), coverage, True))
    rows.sort(key=lambda row: row[1])

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    positions = range(len(rows))
    for position, (_, accuracy, coverage, combined) in zip(positions, rows, strict=True):
        partial = coverage is not None
        ax.barh(
            position,
            accuracy,
            height=0.6,
            color=figures.ACCENT if combined else figures.MUTED,
            hatch="///" if partial else None,
            edgecolor="white" if partial else "none",
        )
        label = f"{accuracy:.3f}"
        if partial:
            label += f"   on {coverage:.0%} of the rows"
        ax.text(
            accuracy + 0.008,
            position,
            label,
            va="center",
            fontsize=9,
            color=figures.WRONG if partial else figures.INK,
            weight="bold" if partial else "normal",
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels([name for name, _, _, _ in rows])
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("accuracy on the held-out Russian split", color=figures.MUTED, fontsize=9)
    best_full = max(accuracy for _, accuracy, coverage, _ in rows if coverage is None)
    winner = next(
        name for name, accuracy, coverage, _ in rows if coverage is None and accuracy == best_full
    )
    ax.set_title(
        f"{winner} wins outright; combining only helps where both models agree",
        color=figures.INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    figures.style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


FIGURES: dict[str, figures.Renderer] = {
    "russian-approaches.png": figure_approaches,
}


def render_all(report: Comparison, out_dir: str | Path) -> list[Path]:
    return figures.render_all(FIGURES, report, out_dir)


def check_figures_current(report: Comparison, out_dir: str | Path) -> list[str]:
    return figures.check_current(FIGURES, report.digest, out_dir)
