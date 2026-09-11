"""Two pictures of why the benchmark does not rank anything.

The first stays inside the submitted log, where the eight runs at least share a
corpus, and shows that the order below the top two depends on which average is
read. The second leaves it, and plots every logged run against the evaluation
set its accuracy proves it was scored on -- which is where a table of 101
numbers stops being a comparison.

Both are drawn from both committed logs and stamped with the digest of the pair,
so touching either makes both figures stale.
"""

from __future__ import annotations

from pathlib import Path

from emotion_timeline import figures
from emotion_timeline.selection.runs import SelectionReport, placed, shared_evaluation_size


def figure_averaging(report: SelectionReport, path: Path) -> Path:
    """The ranking below the top two is a choice of average, not a result."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_macro = report.ranked(macro=True)
    swings = {swing.label: swing for swing in report.swings()}
    positions = range(len(by_macro))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    weighted = [row.f1_weighted for row in by_macro]
    macro = [row.f1_macro for row in by_macro]

    ax.barh(
        [p + 0.19 for p in positions],
        weighted,
        height=0.36,
        color=figures.MUTED,
        label="weighted F1",
    )
    ax.barh(
        [p - 0.19 for p in positions],
        macro,
        height=0.36,
        color=figures.ACCENT,
        label="macro F1",
    )

    for p, (row, w, m) in enumerate(zip(by_macro, weighted, macro, strict=True)):
        ax.text(w + 0.008, p + 0.19, f"{w:.3f}", va="center", fontsize=8.5, color=figures.MUTED)
        moved = swings.get(row.label)
        colour = figures.WRONG if moved else figures.ACCENT
        weight = "bold" if moved else "normal"
        ax.text(
            m + 0.008, p - 0.19, f"{m:.3f}", va="center", fontsize=8.5, color=colour, weight=weight
        )

    labels = []
    for row in by_macro:
        moved = swings.get(row.label)
        suffix = f"   {abs(moved.places)}▲" if moved and moved.places > 0 else ""
        if moved and moved.places < 0:
            suffix = f"   {abs(moved.places)}▼"
        labels.append(f"{row.label}{suffix}")

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    # Past 1.0 so the legend has somewhere to sit that is not on top of a bar.
    # The ticks stop at 1.0, so the extra room reads as margin rather than range.
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("F1", color=figures.MUTED, fontsize=9)
    biggest = report.swings()[0]
    ax.set_title(
        f"Below the top two, the ranking is a choice of average — "
        f"{biggest.label.split()[0]} moves {abs(biggest.places)} places",
        color=figures.INK,
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    ax.text(
        0,
        1.02,
        f"{report.dataset}, {report.split} — ordered by the macro F1 the log did not report",
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


def figure_evaluation_sets(report: SelectionReport, path: Path) -> Path:
    """101 runs scored on 27 different things is not one benchmark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best, best_transformer = report.headline_pair()
    headline = {best.iteration, best_transformer.iteration}

    fig, ax = plt.subplots(figsize=(9, 4.6))

    plain = [
        (run, divisor) for run, divisor in placed(report.runs) if run.iteration not in headline
    ]
    ax.scatter(
        [divisor for _, divisor in plain],
        [run.f1_macro for run, _ in plain],
        s=34,
        color=figures.ACCENT,
        alpha=0.55,
        linewidths=0,
    )

    # The two sit close enough together that one offset would put the labels on
    # top of each other. Both go up -- the band above them is empty -- and they
    # part sideways, one reading right of its marker and the other left.
    offsets = {
        best.iteration: ((12, 16), "left"),
        best_transformer.iteration: ((-14, 14), "right"),
    }
    marked = placed([best, best_transformer])
    for run, divisor in marked:
        offset, align = offsets[run.iteration]
        ax.scatter(
            [divisor],
            [run.f1_macro],
            s=110,
            color=figures.WRONG,
            zorder=3,
            marker="D",
            linewidths=0,
        )
        ax.annotate(
            f"{run.model_name}\nmacro F1 {run.f1_macro:.3f}, n divisible by {divisor:,}",
            (divisor, run.f1_macro),
            textcoords="offset points",
            xytext=offset,
            ha=align,
            fontsize=9,
            color=figures.WRONG,
            weight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "divisor of the evaluation-set size, recovered from the accuracy (log scale)",
        color=figures.MUTED,
        fontsize=9,
    )
    ax.set_ylabel("macro F1", color=figures.MUTED, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "The two runs the conclusion rested on were never scored on the same data",
        color=figures.INK,
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    counted = (
        f"{len(report.measurable)} of {len(report.runs)} logged runs over "
        f"{len(report.divisors)} distinct evaluation sets"
    )
    if len(marked) == 2:
        shared = shared_evaluation_size([divisor for _, divisor in marked])
        counted += f" — one set holding both marked runs would need {shared:,} samples"
    ax.text(
        0,
        1.02,
        counted,
        transform=ax.transAxes,
        color=figures.MUTED,
        fontsize=9.5,
    )
    # No legend: the subtitle counts the plain markers and the two that matter
    # are labelled where they sit, so a key would only cover data.
    figures.style(ax, axis="both")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


FIGURES: dict[str, figures.Renderer] = {
    "model-selection-averaging.png": figure_averaging,
    "model-selection-evaluation-sets.png": figure_evaluation_sets,
}


def render_all(report: SelectionReport, out_dir: str | Path) -> list[Path]:
    return figures.render_all(FIGURES, report, out_dir)


def check_figures_current(report: SelectionReport, out_dir: str | Path) -> list[str]:
    return figures.check_current(FIGURES, report.digest, out_dir)
