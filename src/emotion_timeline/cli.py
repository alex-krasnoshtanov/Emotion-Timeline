"""Command line entry point.

One subcommand per stage of the study. Subcommands appear here as the stage
they belong to lands, so `--help` is an honest statement of what works.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
            f"note: annotation for {min(extents, key=extents.__getitem__)} stops at "
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


def _stages(args: argparse.Namespace) -> list[tuple[str, Any, Any, Any]]:
    """Every stage that owns figures: its report, renderers and consistency check.

    Each stage stamps its figures with the digest of its own source file, so a
    changed error report does not make the dataset figures stale and the other
    way round.
    """
    from emotion_timeline.analysis import error_analysis
    from emotion_timeline.data import build as dataset
    from emotion_timeline.data import figures as dataset_figures
    from emotion_timeline.model import card as model_card
    from emotion_timeline.model import figures as model_figures
    from emotion_timeline.selection import figures as selection_figures
    from emotion_timeline.selection import runs as selection

    return [
        (
            "error-analysis",
            error_analysis.ErrorReport.load(args.report),
            error_analysis.FIGURES,
            error_analysis.check_consistency,
        ),
        (
            "dataset",
            dataset.DatasetReport.load(args.build_record),
            dataset_figures.FIGURES,
            dataset.check_consistency,
        ),
        (
            "model-selection",
            selection.SelectionReport.load(args.run_log, args.submitted_log),
            selection_figures.FIGURES,
            selection.check_consistency,
        ),
        (
            "model",
            model_card.ModelReport.load(args.card_metrics),
            model_figures.FIGURES,
            model_card.check_consistency,
        ),
    ]


def cmd_figures(args: argparse.Namespace) -> int:
    from emotion_timeline import figures as shared

    stages = _stages(args)

    problems = [f"{name}: {p}" for name, report, _, check in stages for p in check(report)]
    if problems:
        # Rendering a chart from statistics that contradict each other would
        # publish the contradiction as a picture. Refuse instead.
        for problem in problems:
            print(f"inconsistent report: {problem}", file=sys.stderr)
        return 1

    if args.check:
        stale = [
            f"{name}: {p}"
            for name, report, renderers, _ in stages
            for p in shared.check_current(renderers, report.digest, args.out)
        ]
        for problem in stale:
            print(f"stale figure: {problem}", file=sys.stderr)
        if stale:
            print(
                "run 'emotion-timeline figures --out assets' and commit the result",
                file=sys.stderr,
            )
            return 1
        total = sum(len(renderers) for _, _, renderers, _ in stages)
        print(f"{total} figures current across {len(stages)} stages")
        return 0

    for _, report, renderers, _ in stages:
        for path in shared.render_all(renderers, report, args.out):
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


def cmd_models(args: argparse.Namespace) -> int:
    """Audit the nine-family benchmark, from both of its surviving records."""
    from emotion_timeline.selection import runs as selection

    report = selection.SelectionReport.load(args.run_log, args.submitted_log)
    problems = selection.check_consistency(report)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    for line in selection.describe(report):
        print(line)
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    """What the training set is made of, from the committed record."""
    from emotion_timeline.data.build import DatasetReport, check_consistency

    report = DatasetReport.load(args.build_record)
    problems = check_consistency(report)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    published = report.published
    print(f"{report.raw['source']}: {report.steps[0]['rows_out']:,} rows across six corpora")
    width = max(len(name) for name in report.composition)
    for name, body in sorted(report.composition.items(), key=lambda kv: -kv[1]["rows"]):
        dropped = body["rows"] - body["kept"]
        note = f"  ({dropped:,} dropped, kept only where it annotates disgust)" if dropped else ""
        print(f"  {name:<{width}} {body['rows']:>7,}{note}")

    print()
    print("  the build")
    for line in iter_progress_from(report):
        print(line)

    print()
    print(f"  published training set: {published['rows']:,} rows")
    print(
        f"    {report.rows:,} reproducible + {published['synthetic_disgust_rows']:,} "
        "synthetic Disgust rows that no longer exist"
    )
    total = published["rows"]
    for name, count in sorted(published["class_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:<9} {count:>7,}  {count / total * 100:5.2f}%")
    return 0


def iter_progress_from(report: Any) -> Iterator[str]:
    width = max(len(s["name"]) for s in report.steps)
    for step in report.steps:
        removed = step["rows_in"] - step["rows_out"]
        marker = f"-{removed:,}" if removed else ""
        yield f"    {step['name']:<{width}}  {step['rows_out']:>7,}  {marker:>9}  {step['note']}"


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Rebuild the training set from the public corpus and check the funnel."""
    from emotion_timeline.data import build as dataset

    try:
        frame, record = dataset.build(args.cache)
    except ImportError:
        print(
            "building needs the source corpus: install with --extra data (uv sync --extra data)",
            file=sys.stderr,
        )
        return 1

    for line in dataset.iter_progress(record):
        print(line)
    print()
    print(f"{record.rows:,} rows")
    for name, count in sorted(record.counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<9} {count:>7,}")

    expected = dataset.DatasetReport.load(args.build_record)
    differences = dataset.compare(record, expected)
    for difference in differences:
        print(f"differs from the record: {difference}", file=sys.stderr)
    if differences:
        print(
            "the build no longer reproduces benchmarks/dataset/build-record.json",
            file=sys.stderr,
        )
        return 1
    print()
    print(f"matches {expected.source.name} exactly")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


def _read_dataset(path: str) -> tuple[list[str], list[str], list[str]]:
    """Read a rebuilt dataset CSV into the three columns the split is keyed on."""
    import pandas as pd

    frame = pd.read_csv(path)
    missing = {"text", "label", "source"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)}")
    return (
        [str(value) for value in frame["text"]],
        [str(value) for value in frame["label"]],
        [str(value) for value in frame["source"]],
    )


def cmd_preflight(args: argparse.Namespace) -> int:
    """Whether this machine can train, and what to fix if it cannot."""
    from emotion_timeline.training import preflight

    device = preflight.probe()
    if device is not None:
        for line in preflight.describe(device):
            print(line)
        print()

    problems = preflight.assess(device, preflight.smoke(), args.need_mib)
    for problem in problems:
        print(f"cannot train here: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("ready to train")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """The committed split: what it holds, and whether a rebuilt dataset still gives it."""
    from emotion_timeline.training import splits

    if args.write:
        texts, labels, sources = _read_dataset(args.write)
        written = splits.write_manifest(
            splits.build_manifest(texts, labels, sources), args.manifest
        )
        print(f"wrote {written}")

    manifest = splits.SplitManifest.load(args.manifest)
    problems = splits.check_consistency(manifest)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    for line in splits.describe(manifest):
        print(line)

    if args.verify:
        texts, labels, sources = _read_dataset(args.verify)
        differences = splits.verify(manifest, texts, labels, sources)
        print()
        for difference in differences:
            print(f"differs from the record: {difference}", file=sys.stderr)
        if differences:
            print(f"{args.verify} does not give the committed split", file=sys.stderr)
            return 1
        print(f"  {args.verify} reproduces every split exactly")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    """What the two surviving records of the trained classifier can support."""
    from emotion_timeline.model.card import (
        ModelReport,
        check_consistency,
        disgust_false_positive_bounds,
        single_label_identity_holds,
    )

    report = ModelReport.load(args.card_metrics)
    problems = check_consistency(report)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    print("the card's three evaluations, every figure recomputed from its own table")
    for evaluation in report.evaluations:
        note = (
            ""
            if evaluation.scored_classes == 7
            else f"  ({evaluation.scored_classes} of 7 classes)"
        )
        print(
            f"  {evaluation.name:<9} {evaluation.samples:>6,} samples   "
            f"accuracy {evaluation.accuracy:.4f}   macro F1 {evaluation.macro_f1:.4f}{note}"
        )

    stress = report.evaluation("stress")
    print(f"    stress macro F1 over all seven classes would be {stress.macro_f1_over(7):.4f}")

    print()
    print("  which record is which")
    for evaluation in report.evaluations:
        if "micro_f1" not in evaluation.raw:
            verdict = "no micro F1 reported, so silent either way"
        elif single_label_identity_holds(evaluation):
            verdict = "holds, so this table is single-label"
        else:
            verdict = "does not hold"
        print(f"    {evaluation.name:<9} micro F1 == accuracy: {verdict}")
    script = report.script_metrics
    print(
        f"    script    micro F1 {script['f1_micro']:.4f} against subset accuracy "
        f"{script['subset_accuracy']:.4f}, so multi-label"
    )
    print(
        f"    script    hamming loss implies {report.script_labels_per_sample:.3f} true labels "
        "per sample, so not the collapsed dataset"
    )
    print(
        f"    script    surviving tokenizer holds "
        f"{report.script['surviving_tokenizer_tokens']:,} tokens; the card claims "
        f"{report.card['claimed_vocabulary_size']:,}"
    )

    print()
    print("  the card's dataset table, against the counts this repository builds")
    for count, said, truth in report.mislabelled():
        print(f"    {count:>7,} rows called {said:<10} are {truth}")

    print()
    print("  the stress test")
    control = report.control
    print(f"    control scores {control.accuracy:.4f} on {control.samples:,} of 5,000 samples")
    for row in report.outliers_beating_the_control():
        print(f"      {row.name:<18} {row.accuracy:.4f}  beats the control")
    failures = report.total_failures()
    print(
        f"    {len(failures)} categories score exactly zero over "
        f"{sum(row.samples for row in failures):,} samples: "
        + ", ".join(row.name.lower() for row in failures)
    )
    low, high = disgust_false_positive_bounds(stress)
    reported = stress.raw["reported_disgust_false_positives"]
    print(
        f"    the card's {reported:,} Disgust false positives sit outside the "
        f"{low:,}-{high:,} its own precision allows"
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
    record_default = str(BENCHMARKS / "dataset" / "build-record.json")
    run_log_default = str(BENCHMARKS / "model-selection" / "run-log.csv")
    submitted_log_default = str(BENCHMARKS / "model-selection" / "submitted-log.json")
    card_default = str(BENCHMARKS / "model" / "card-metrics.json")

    def add_selection_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--run-log", default=run_log_default)
        target.add_argument("--submitted-log", default=submitted_log_default)

    model_parser = sub.add_parser(
        "model",
        help="audit the trained classifier against both surviving records of it",
        description=(
            "Two records of the classifier survive and they describe different "
            "models: the group's model card claims DeBERTa-V2 evaluated "
            "single-label, and the committed training script is DistilBERT "
            "evaluated multi-label. The weights are gone from both university "
            "repositories, so nothing here is rerun. What is checked is each "
            "record's own arithmetic, and the identities that tell the two apart."
        ),
    )
    model_parser.add_argument("--card-metrics", default=card_default)
    model_parser.set_defaults(func=cmd_model)

    dataset_parser = sub.add_parser(
        "dataset",
        help="what the training set is made of, and what building it discarded",
        description=(
            "Reads the committed build record. Every number it prints was produced "
            "by 'build-dataset' on this machine, not copied from the original "
            "notebook -- except the 9,151 synthetic Disgust rows, which are named "
            "as such because their source file no longer exists."
        ),
    )
    dataset_parser.add_argument("--build-record", default=record_default)
    dataset_parser.set_defaults(func=cmd_dataset)

    build_dataset_parser = sub.add_parser(
        "build-dataset",
        help="rebuild the training set from the public corpus (needs --extra data)",
        description=(
            "Downloads the source corpus and reruns the whole funnel, then checks "
            "the result against the committed record. Takes a few minutes and about "
            "a gigabyte of cache."
        ),
    )
    build_dataset_parser.add_argument("--build-record", default=record_default)
    build_dataset_parser.add_argument(
        "--out", help="write the rebuilt dataset here as CSV (default: do not write it)"
    )
    build_dataset_parser.add_argument("--cache", help="dataset download cache directory")
    build_dataset_parser.set_defaults(func=cmd_build_dataset)

    preflight_parser = sub.add_parser(
        "preflight",
        help="check the GPU can actually run a kernel before training on it",
        description=(
            "Reports the card, the torch build and the architectures it carries "
            "kernels for, then runs one real matrix multiply. Both ways this goes "
            "wrong are quiet, so it is worth a second before an hour of training."
        ),
    )
    preflight_parser.add_argument(
        "--need-mib",
        type=int,
        default=6000,
        help="free video memory the run needs (default: enough for batch 64 at length 128)",
    )
    preflight_parser.set_defaults(func=cmd_preflight)

    split_parser = sub.add_parser(
        "split",
        help="the committed train/validation/test split, and how to check it",
        description=(
            "Prints the split the fine-tune trains on. No rows are committed, so "
            "membership is pinned by a digest per split: --verify recomputes those "
            "from a rebuilt dataset, and --write regenerates the record from one."
        ),
    )
    split_parser.add_argument(
        "--manifest", default=str(BENCHMARKS / "training" / "split-manifest.json")
    )
    split_parser.add_argument("--verify", help="recompute the split from this rebuilt CSV")
    split_parser.add_argument("--write", help="regenerate the manifest from this rebuilt CSV")
    split_parser.set_defaults(func=cmd_split)

    errors_parser = sub.add_parser(
        "errors",
        help="summarise where the emotion classifier goes wrong",
    )
    errors_parser.add_argument("--report", default=report_default)
    errors_parser.set_defaults(func=cmd_errors)

    models_parser = sub.add_parser(
        "models",
        help="audit the nine-family model comparison the project inherited",
        description=(
            "Reads both surviving records of the benchmark and reports what they "
            "establish, which is less than the coursework claimed. Neither log can "
            "be recomputed -- the runs are gone and the dataset behind the earliest "
            "of them is client material -- so the arithmetic here is a cross-check "
            "between the two, plus what each run's own accuracy proves about the "
            "evaluation set it was scored over."
        ),
    )
    add_selection_arguments(models_parser)
    models_parser.set_defaults(func=cmd_models)

    figures_parser = sub.add_parser(
        "figures",
        help="render the README figures from the recorded statistics",
    )
    figures_parser.add_argument("--report", default=report_default)
    figures_parser.add_argument("--build-record", default=record_default)
    figures_parser.add_argument("--card-metrics", default=card_default)
    add_selection_arguments(figures_parser)
    figures_parser.add_argument("--out", default="assets", help="output directory")
    figures_parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed figures were drawn from the current report, and do not redraw",
    )
    figures_parser.set_defaults(func=cmd_figures)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Loaded before dispatch so a subcommand never has to think about it. No
    # command committed so far needs a key -- everything reads from benchmarks/
    # -- but the stages that call a hosted service will, and this is the one
    # place they get it from. An exported variable always wins over the file.
    from emotion_timeline import credentials

    credentials.load_env_file()

    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
