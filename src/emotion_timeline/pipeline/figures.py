"""The picture the repository is named for: emotion against wall clock.

One rule shapes it. The band is drawn **solid where the two models picked the
same class and hatched where they split**, because the one thing a reader should
take away from a timeline with no ground truth under it is which parts of it to
believe. A solid stretch is two models trained on different languages and
different data reaching the same answer; a hatched one is where a human should
look. Neither is an accuracy, and the caption says so.

The confidence strip underneath is the calibrated top-class probability. It is
drawn as a step rather than a curve because a scene has one value for its whole
length, and a smooth line would invent a reading between scenes that nothing
measured.

**The valence band appears only when the record carries one**, because valence is
optional and off by default -- it is measured not to improve the emotion label
(see `russian/va.py`). What it does do is separate scenes the label cannot: on the
committed recording the 26 scenes called Neutral span 91% of the episode's whole
valence range, so the band is carrying most of the information the label throws
away. Diverging colour, red through grey to green, with the midpoint pinned at
0.5 so the neutral point is the same in every figure rather than the middle of
whatever this recording happened to contain.
"""

from __future__ import annotations

from pathlib import Path

from emotion_timeline import figures
from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.pipeline import timeline as pipeline_timeline
from emotion_timeline.pipeline.timeline import Timeline


def figure_timeline(report: Timeline, path: Path) -> Path:
    """Emotion per scene across the whole recording, marked where the models split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    scenes = report.scenes
    rows = {name: position for position, name in enumerate(EMOTIONS)}

    has_valence = pipeline_timeline.carries_dimensions(scenes)
    if has_valence:
        fig, (band, mood, strip) = plt.subplots(
            3,
            1,
            figsize=(11, 6.4),
            dpi=160,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 0.7, 1], "hspace": 0.16},
        )
    else:
        fig, (band, strip) = plt.subplots(
            2,
            1,
            figsize=(11, 5.4),
            dpi=160,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.14},
        )
        mood = None

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

    if mood is not None:
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        # Pinned at 0.5 rather than the data's own middle: a band whose midpoint
        # moved with the recording would make two timelines incomparable.
        diverging = LinearSegmentedColormap.from_list(
            "valence", [figures.WRONG, "#e8e4dc", figures.RIGHT]
        )
        scale = Normalize(vmin=0.0, vmax=1.0)
        for scene in scenes:
            start = float(scene["start_s"]) / 60
            mood.broken_barh(
                [(start, max((float(scene["end_s"]) - float(scene["start_s"])) / 60, 0.02))],
                (0, 1),
                facecolors=diverging(scale(float(scene["valence"]))),
                edgecolor="none",
            )
        mood.set_yticks([])
        mood.set_ylim(0, 1)
        mood.set_ylabel(
            "valence",
            color=figures.MUTED,
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
        )
        for side in ("top", "right", "left", "bottom"):
            mood.spines[side].set_visible(False)
        mood.tick_params(labelbottom=False, length=0)

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
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.9 if mood is None else 0.915,
        bottom=0.115 if mood is None else 0.1,
    )
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
