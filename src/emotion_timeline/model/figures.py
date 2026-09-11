"""Two pictures of the model: where it fails by class, and where the stress test broke.

Both drawn from ``benchmarks/model/card-metrics.json`` and stamped with its
digest, so a changed record makes them stale and CI says so.
"""

from __future__ import annotations

from pathlib import Path

from emotion_timeline import figures
from emotion_timeline.model.card import ModelReport


def figure_class_scores(report: ModelReport, path: Path) -> Path:
    """Precision against recall per class, which is where Disgust gives itself away."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    held = report.evaluation("held_out")
    order = sorted(held.classes.values(), key=lambda c: c.f1, reverse=True)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = range(len(order))
    height = 0.38
    ax.barh(
        [p + height / 2 for p in positions],
        [c.recall for c in order],
        height=height,
        color=figures.ACCENT,
        label="recall (of this class, how much was found)",
    )
    ax.barh(
        [p - height / 2 for p in positions],
        [c.precision for c in order],
        height=height,
        color=figures.MUTED,
        label="precision (of what was called this, how much was right)",
    )

    for y, score in zip(positions, order, strict=True):
        ax.text(
            score.recall + 0.012, y + height / 2, f"{score.recall:.2f}", va="center", fontsize=8.5
        )
        ax.text(
            score.precision + 0.012,
            y - height / 2,
            f"{score.precision:.2f}",
            va="center",
            fontsize=8.5,
            color=figures.MUTED,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels([f"{c.name}\nF1 {c.f1:.2f}" for c in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(
        "Disgust is found nine times in ten and wrong four times in ten",
        color=figures.INK,
        fontsize=12.5,
        loc="left",
        pad=16,
    )
    ax.text(
        0,
        1.03,
        f"held-out set, {held.samples:,} samples, accuracy {held.accuracy:.1%} "
        f"-- a gap this wide means the class is over-predicted",
        transform=ax.transAxes,
        color=figures.MUTED,
        fontsize=9.5,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=figures.MUTED)
    figures.style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


def figure_stress_control(report: ModelReport, path: Path) -> Path:
    """The stress test's control scoring worse than the inputs built to break it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(report.outlier_types, key=lambda r: r.accuracy, reverse=True)
    control = report.control
    beating = {row.name for row in report.outliers_beating_the_control()}

    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = range(len(rows))
    colours = [
        figures.INK if row.is_control else (figures.RIGHT if row.name in beating else figures.WRONG)
        for row in rows
    ]
    ax.barh(positions, [row.accuracy for row in rows], color=colours, height=0.62)

    for y, row in zip(positions, rows, strict=True):
        ax.text(
            row.accuracy + 0.012,
            y,
            f"{row.accuracy:.2f}   n={row.samples:,}",
            va="center",
            fontsize=8.5,
            color=figures.INK if row.accuracy > 0 else figures.WRONG,
        )

    ax.axvline(control.accuracy, color=figures.INK, linewidth=1, linestyle=":")
    ax.set_yticks(list(positions))
    ax.set_yticklabels([row.name for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.92)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_title(
        f"{len(beating)} categories built to be harder beat the control",
        color=figures.INK,
        fontsize=12.5,
        loc="left",
        pad=16,
    )
    ax.text(
        0,
        1.03,
        f"the control scores {control.accuracy:.2f} on {control.samples:,} of the 5,000 samples, "
        "so it sets the headline rather than a baseline",
        transform=ax.transAxes,
        color=figures.MUTED,
        fontsize=9.5,
    )
    figures.style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


FIGURES: dict[str, figures.Renderer] = {
    "model-class-scores.png": figure_class_scores,
    "model-stress-control.png": figure_stress_control,
}


def render_all(report: ModelReport, out_dir: str | Path) -> list[Path]:
    return figures.render_all(FIGURES, report, out_dir)


def check_figures_current(report: ModelReport, out_dir: str | Path) -> list[str]:
    return figures.check_current(FIGURES, report.digest, out_dir)
