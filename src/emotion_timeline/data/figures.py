"""Two pictures of the dataset: what building it discarded, and what it left.

Both are drawn from ``benchmarks/dataset/build-record.json`` and stamped with
its digest, so a changed record makes them stale and CI says so.
"""

from __future__ import annotations

from pathlib import Path

from emotion_timeline import figures
from emotion_timeline.data.build import DatasetReport

# Steps that only rewrite text are not drawn: a funnel bar of the same height as
# the one before it says nothing, and the caption would have to explain why.
INVISIBLE = {"normalise", "fold-case", "tidy", "assign-labels"}


def figure_funnel(report: DatasetReport, path: Path) -> Path:
    """Where a quarter of the source corpus went."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [s for s in report.steps if s["name"] not in INVISIBLE]
    names = [s["name"] for s in steps]
    rows = [s["rows_out"] for s in steps]
    lost = [s["rows_in"] - s["rows_out"] for s in steps]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    positions = range(len(steps))
    ax.barh(positions, rows, color=figures.ACCENT, height=0.62)
    ax.barh(positions, lost, left=rows, color=figures.WRONG, height=0.62, alpha=0.85)

    for y, (kept, dropped) in enumerate(zip(rows, lost, strict=True)):
        ax.text(kept - 6000, y, f"{kept:,}", va="center", ha="right", color="white", fontsize=9)
        if dropped:
            ax.text(
                kept + dropped + 6000,
                y,
                f"-{dropped:,}",
                va="center",
                fontsize=9,
                color=figures.WRONG,
            )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 600_000)
    ax.set_xticks([0, 200_000, 400_000, 600_000])
    ax.set_xticklabels(["0", "200k", "400k", "600k"])
    ax.set_title(
        "One in four source rows never reaches the training set",
        color=figures.INK,
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    ax.text(
        0,
        1.02,
        f"{report.steps[0]['rows_out']:,} rows in {report.raw['source']}, {report.rows:,} out",
        transform=ax.transAxes,
        color=figures.MUTED,
        fontsize=9.5,
    )
    figures.style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


def figure_class_balance(report: DatasetReport, path: Path) -> Path:
    """The imbalance the error analysis later runs into."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    published = report.published["class_counts"]
    synthetic = report.published["synthetic_disgust_rows"]
    order = sorted(published, key=lambda name: published[name], reverse=True)
    real = [published[name] - (synthetic if name == "Disgust" else 0) for name in order]
    made = [synthetic if name == "Disgust" else 0 for name in order]
    total = report.published["rows"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    positions = range(len(order))
    ax.barh(positions, real, color=figures.ACCENT, height=0.62, label="from the source corpora")
    ax.barh(
        positions,
        made,
        left=real,
        color=figures.MUTED,
        height=0.62,
        label="synthetic, generated for Disgust",
    )

    for y, name in enumerate(order):
        count = published[name]
        ax.text(
            count + 2500,
            y,
            f"{count:,}  ({count / total * 100:.1f}%)",
            va="center",
            fontsize=9,
            color=figures.INK,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(order, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 185_000)
    ax.set_xticks([0, 50_000, 100_000, 150_000])
    ax.set_xticklabels(["0", "50k", "100k", "150k"])
    ax.set_title(
        "Joy outnumbers Neutral eleven to one",
        color=figures.INK,
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    ax.text(
        0,
        1.02,
        "and Neutral, Fear and Surprise are exactly the classes the model gets wrong",
        transform=ax.transAxes,
        color=figures.MUTED,
        fontsize=9.5,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=figures.MUTED)
    figures.style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


FIGURES: dict[str, figures.Renderer] = {
    "dataset-funnel.png": figure_funnel,
    "dataset-classes.png": figure_class_balance,
}


def render_all(report: DatasetReport, out_dir: str | Path) -> list[Path]:
    return figures.render_all(FIGURES, report, out_dir)


def check_figures_current(report: DatasetReport, out_dir: str | Path) -> list[str]:
    return figures.check_current(FIGURES, report.digest, out_dir)
