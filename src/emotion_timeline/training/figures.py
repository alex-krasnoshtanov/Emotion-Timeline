"""One picture of the retrain: how it sits against the record it was measured for.

Drawn from the two committed records together and stamped with their combined
digest, so changing either makes it stale and CI says so.

The class that is missing from the chart is the point of it. Disgust gets a bar
on the card's side and nothing on ours, labelled with why, because 64% of its
training rows are a file that did not survive. Drawing the bar anyway would put a
number on the page that the text then has to spend a paragraph withdrawing, and
the number is what gets remembered.
"""

from __future__ import annotations

from pathlib import Path

from emotion_timeline import figures
from emotion_timeline.training.report import TrainingReport, card_f1_from, compare_to_card


def figure_against_card(report: TrainingReport, path: Path) -> Path:
    """Per-class F1, the card's against this run's, with the missing class left blank."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = compare_to_card(report, card_f1_from(""))
    rows.sort(key=lambda row: row.card_f1)
    positions = range(len(rows))
    height = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.barh(
        [p - height / 2 for p in positions],
        [row.card_f1 for row in rows],
        height=height,
        color=figures.MUTED,
        label="the card's held-out F1",
    )
    ax.barh(
        [p + height / 2 for p in positions],
        [row.retrain_f1 or 0.0 for row in rows],
        height=height,
        color=figures.ACCENT,
        label="this fine-tune",
    )

    for position, row in enumerate(rows):
        ax.text(
            row.card_f1 + 0.006,
            position - height / 2,
            f"{row.card_f1:.3f}",
            va="center",
            fontsize=8,
            color=figures.MUTED,
        )
        if row.retrain_f1 is None:
            ax.text(
                0.012,
                position + height / 2,
                "not comparable: 64% of its training rows did not survive",
                va="center",
                fontsize=8,
                color=figures.WRONG,
                style="italic",
            )
            continue
        colour = figures.RIGHT if row.delta and row.delta > 0 else figures.WRONG
        ax.text(
            row.retrain_f1 + 0.006,
            position + height / 2,
            f"{row.retrain_f1:.3f}  ({row.delta:+.3f})",
            va="center",
            fontsize=8,
            color=colour,
            weight="bold",
        )

    ahead = sum(1 for row in rows if row.delta is not None and row.delta > 0)
    comparable = sum(1 for row in rows if row.delta is not None)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([row.name for row in rows])
    ax.set_xlim(0, 1.24)
    ax.set_xlabel("F1", color=figures.MUTED, fontsize=9)
    ax.set_title(
        f"{ahead} of {comparable} comparable classes beat the record they were trained to match",
        color=figures.INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    figures.style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


FIGURES: dict[str, figures.Renderer] = {
    "fine-tune-against-card.png": figure_against_card,
}


def render_all(report: TrainingReport, out_dir: str | Path) -> list[Path]:
    return figures.render_all(FIGURES, report, out_dir)


def check_figures_current(report: TrainingReport, out_dir: str | Path) -> list[str]:
    return figures.check_current(FIGURES, report.digest, out_dir)
