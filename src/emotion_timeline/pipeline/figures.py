"""The picture the repository is named for: emotion against wall clock.

One rule shapes it. The band is drawn **solid where the two models picked the
same class and hatched where they split**, because the one thing a reader should
take away from a timeline with no ground truth under it is which parts of it to
believe. A solid stretch is two models trained on different languages and
different data reaching the same answer; a hatched one is where a human should
look. Neither is an accuracy, and the caption says so.

The confidence strip underneath is the calibrated top-class probability, which is
what replaced the original pipeline's separate intensity model. It is drawn as a
step rather than a curve because a scene has one value for its whole length, and
a smooth line would invent a reading between scenes that nothing measured.
"""

from __future__ import annotations

from pathlib import Path

from emotion_timeline import figures
from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.pipeline.timeline import Timeline


def figure_timeline(report: Timeline, path: Path) -> Path:
    """Emotion per scene across the whole recording, marked where the models split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    scenes = report.scenes
    rows = {name: position for position, name in enumerate(EMOTIONS)}

    fig, (band, strip) = plt.subplots(
        2,
        1,
        figsize=(11, 5.4),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.14},
    )

    for scene in scenes:
        start = float(scene["start_s"]) / 60
        width = (float(scene["end_s"]) - float(scene["start_s"])) / 60
        agreed = bool(scene["agreed"])
        band.broken_barh(
            [(start, width)],
            (rows[str(scene["emotion"])] - 0.34, 0.68),
            facecolors=figures.EMOTION_COLOURS[str(scene["emotion"])],
            hatch=None if agreed else "///",
            edgecolor="none" if agreed else "white",
            linewidth=0.0 if agreed else 0.6,
        )

    band.set_yticks(range(len(EMOTIONS)))
    band.set_yticklabels(EMOTIONS)
    band.set_ylim(-0.8, len(EMOTIONS) - 0.2)
    band.invert_yaxis()
    agreement = report.agreement
    band.set_title(
        f"{report.raw['scenes']} scenes over {report.duration / 60:.0f} minutes; "
        f"the two models agree on {float(agreement['share']):.0%} of them",
        color=figures.INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    band.legend(
        handles=[
            Patch(facecolor=figures.MUTED, label="both models agree"),
            Patch(facecolor=figures.MUTED, hatch="///", edgecolor="white", label="they disagree"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    figures.style(band, axis="both")

    edges = [float(scenes[0]["start_s"]) / 60]
    values = []
    for scene in scenes:
        edges.append(float(scene["end_s"]) / 60)
        values.append(float(scene["confidence"]))
    strip.stairs(values, edges, fill=True, color=figures.ACCENT, alpha=0.75, linewidth=0)
    strip.set_ylim(0, 1)
    strip.set_ylabel("confidence", color=figures.MUTED, fontsize=9)
    strip.set_xlabel("minutes into the recording", color=figures.MUTED, fontsize=9)
    strip.set_xlim(edges[0], edges[-1])
    figures.style(strip, axis="both")

    # `tight_layout` cannot handle the shared axis and the ratio together.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.115)
    fig.savefig(path, facecolor="white", metadata=figures.stamp(report.digest))
    plt.close(fig)
    return path


FIGURES: dict[str, figures.Renderer] = {
    "emotion-timeline.png": figure_timeline,
}


def render_all(report: Timeline, out_dir: str | Path) -> list[Path]:
    return figures.render_all(FIGURES, report, out_dir)


def check_figures_current(report: Timeline, out_dir: str | Path) -> list[str]:
    return figures.check_current(FIGURES, report.digest, out_dir)
