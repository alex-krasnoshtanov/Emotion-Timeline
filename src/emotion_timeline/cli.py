"""Command line entry point.

One subcommand per stage of the study. Subcommands appear here as the stage
they belong to lands, so `--help` is an honest statement of what works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def parse_timestamp(text: str) -> float:
    """Seconds from ``SS``, ``MM:SS`` or ``HH:MM:SS``, decimals allowed."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError(f"not a timestamp: {text!r}")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a timestamp: {text!r}") from None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def parse_window(text: str) -> tuple[float, float | None]:
    """``START-END``, either side optional: ``0:00-18:09``, ``-18:09``, ``5:00-``."""
    if "-" not in text:
        raise argparse.ArgumentTypeError("window must look like 0:00-18:09")
    start_text, _, end_text = text.partition("-")
    start = parse_timestamp(start_text) if start_text.strip() else 0.0
    end = parse_timestamp(end_text) if end_text.strip() else None
    if end is not None and end <= start:
        raise argparse.ArgumentTypeError(f"window ends before it starts: {text!r}")
    return start, end


def cmd_wer(args: argparse.Namespace) -> int:
    from emotion_timeline.stt import wer

    directory = Path(args.benchmarks)
    annotations = sorted(directory.glob("*.csv"))
    if not annotations:
        print(f"no annotated transcripts in {directory}", file=sys.stderr)
        return 1

    systems = {path.stem: wer.load(path) for path in annotations}
    start, end = args.window

    # Scoring past the point either system stopped being annotated compares a
    # checked window against an unchecked one, which is how the published 0.61%
    # happened. Warn rather than refuse: a deliberate wider look is still useful.
    extents = {name: wer.annotated_extent(frame) for name, frame in systems.items()}
    limit = min(extents.values())
    if end is None or end > limit:
        print(
            f"note: annotation for {min(extents, key=extents.get)} stops at "
            f"{limit / 60:.1f} min; beyond that the comparison is uneven.",
            file=sys.stderr,
        )

    results = wer.compare(systems, start, end)
    width = max(len(r.system) for r in results)
    if end is None:
        print(f"from {start / 60:.1f} min")
    else:
        print(f"window {start / 60:.1f}-{end / 60:.1f} min")
    for result in results:
        print(
            f"  {result.system:<{width}}  {result.wer:6.2f}%   "
            f"{result.errors:>4} errors / {result.reference_tokens:>5} reference tokens   "
            f"S={result.substitutions} I={result.insertions} D={result.deletions}   "
            f"{result.segments} segments"
        )
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from emotion_timeline.analysis import error_analysis

    report = error_analysis.ErrorReport.load(args.report)
    problems = error_analysis.check_consistency(report)
    if problems:
        # Rendering a chart from statistics that contradict each other would
        # publish the contradiction as a picture. Refuse instead.
        for problem in problems:
            print(f"inconsistent report: {problem}", file=sys.stderr)
        return 1

    if args.check:
        stale = error_analysis.check_figures_current(report, args.out)
        for problem in stale:
            print(f"stale figure: {problem}", file=sys.stderr)
        if stale:
            print(
                "run 'emotion-timeline figures --out assets' and commit the result",
                file=sys.stderr,
            )
            return 1
        print(f"{len(error_analysis.FIGURES)} figures current for report {report.digest[:12]}")
        return 0

    for path in error_analysis.render_all(report, args.out):
        print(f"wrote {path}")
    return 0


def cmd_errors(args: argparse.Namespace) -> int:
    from emotion_timeline.analysis import error_analysis

    report = error_analysis.ErrorReport.load(args.report)
    print(
        f"{report.split}: {report.total_samples:,} samples, "
        f"{report.total_errors:,} errors, accuracy {report.accuracy:.4f}"
    )
    print()
    print("  hardest classes")
    for c in report.hardest(3):
        print(f"    {c.name:<9} {c.error_rate * 100:5.2f}% of {c.samples:>6,} samples")
    print()
    print("  surface markers, by how much they multiply the error rate")
    for name, ratio in report.worst_features():
        body = report.textual_features[name]
        print(
            f"    {name:<17} x{ratio:5.1f}   "
            f"{body['error_rate_present'] * 100:5.2f}% with, "
            f"{body['error_rate_absent'] * 100:5.2f}% without "
            f"({body['present_samples']:,} samples)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emotion-timeline",
        description="Per-scene emotion analysis: dataset, model and pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    wer_parser = sub.add_parser(
        "wer",
        help="score the speech-to-text systems over one window of audio",
        description=(
            "Compare transcribers over the same stretch of audio. The window is "
            "a time range, never a row count: the systems segment differently, "
            "so equal row counts cover unequal amounts of speech."
        ),
    )
    wer_parser.add_argument(
        "--window",
        type=parse_window,
        default=(0.0, 1089.0),
        metavar="START-END",
        help="time range to score, e.g. 0:00-18:09 (default: %(default)s seconds)",
    )
    wer_parser.add_argument(
        "--benchmarks",
        default=str(BENCHMARKS / "stt"),
        help="directory of annotated transcripts (default: benchmarks/stt)",
    )
    wer_parser.set_defaults(func=cmd_wer)

    report_default = str(BENCHMARKS / "error-analysis" / "held-out-64250.json")

    errors_parser = sub.add_parser(
        "errors",
        help="summarise where the emotion classifier goes wrong",
    )
    errors_parser.add_argument("--report", default=report_default)
    errors_parser.set_defaults(func=cmd_errors)

    figures_parser = sub.add_parser(
        "figures",
        help="render the README figures from the recorded statistics",
    )
    figures_parser.add_argument("--report", default=report_default)
    figures_parser.add_argument("--out", default="assets", help="output directory")
    figures_parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed figures were drawn from the current report, and do not redraw",
    )
    figures_parser.set_defaults(func=cmd_figures)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
