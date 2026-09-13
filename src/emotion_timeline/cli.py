"""Command line entry point.

One subcommand per stage of the study. Subcommands appear here as the stage
they belong to lands, so `--help` is an honest statement of what works.
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import sys
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pandas is imported inside the functions that need it
    import pandas as pd

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
RU_RECORD = BENCHMARKS / "russian" / "build-record.json"
TIMELINE = BENCHMARKS / "pipeline" / "timeline.json"
SEGMENTS = BENCHMARKS / "stt" / "assemblyai-best.csv"

#: Re-exported so `--help` shows the default the module documents.
GAP_SECONDS = 1.0
CHUNK_CHARS = 400
TURBO = "large-v3-turbo"

#: Every subcommand, grouped by the chapter it belongs to and ordered so the
#: pipeline -- the thing a visitor came for -- is first. This is the only source
#: of the listing `--help` prints, and `test_cli.py` asserts that every
#: registered command appears here exactly once, so a new command cannot quietly
#: go missing from it.
CHAPTERS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "The timeline, on a real recording",
        (
            ("timeline", "the per-scene emotion timeline over the committed transcript"),
            (
                "serve",
                "open the pipeline in a browser (needs --extra web, and stt+model to run one)",
            ),
            ("transcribe", "a video URL or an audio file into a segment CSV (needs --extra stt)"),
            (
                "score-timeline",
                "run both models over a transcript and write the record (needs --extra model)",
            ),
        ),
    ),
    (
        "Which speech-to-text system",
        (("wer", "score the speech-to-text systems over one window of audio"),),
    ),
    (
        "What it was trained on",
        (
            ("dataset", "what the training set is made of, and what building it discarded"),
            (
                "build-dataset",
                "rebuild the training set from the public corpus (needs --extra data)",
            ),
        ),
    ),
    (
        "Which model family, and the trained classifier",
        (
            ("models", "audit the nine-family model comparison the project inherited"),
            ("model", "audit the trained classifier against both surviving records of it"),
            ("errors", "summarise where the emotion classifier goes wrong"),
        ),
    ),
    (
        "A model that exists",
        (
            ("preflight", "check the GPU can actually run a kernel before training on it"),
            ("split", "the committed train/validation/test split, and how to check it"),
            (
                "fine-tune",
                "train the classifier on the rebuilt dataset (needs --extra model and a GPU)",
            ),
            ("summarise", "turn a run's kept predictions into the held-out record"),
            ("training", "the fine-tune: what it was, what it scored, and how it compares"),
        ),
    ),
    (
        "Russian, and what translation costs",
        (
            ("russian", "the Russian evaluation set, and what mapping it to seven classes cost"),
            (
                "build-russian",
                "rebuild the Russian evaluation set from the public corpus (needs --extra data)",
            ),
            (
                "compare-russian",
                "score every approach to Russian on the held-out split (needs --extra model)",
            ),
            (
                "translation-cost",
                "price what translation costs, on the model's own rows (needs --extra model)",
            ),
        ),
    ),
    (
        "Valence and arousal, scored",
        (("valence", "what the valence-arousal model separates, and what it does not add"),),
    ),
    ("The figures", (("figures", "render the README figures from the recorded statistics"),)),
)

#: Flattened, in the order `--help` shows them. Used to tell a typo from a
#: command before argparse gets the chance to reprint all twenty-one names.
COMMANDS: tuple[str, ...] = tuple(name for _, entries in CHAPTERS for name, _ in entries)

#: What each command looks like when somebody actually runs it. clig.dev's
#: first rule for help text is to lead with examples, and the bare commands that
#: take no interesting flag are their own example, so only these carry one.
EXAMPLES: dict[str, str] = {
    "timeline": """examples:
  emotion-timeline timeline
  emotion-timeline timeline --against benchmarks/pipeline/timeline-whisper.json
  emotion-timeline timeline --record mine.json --segments data/segments.csv""",
    "serve": """examples:
  emotion-timeline serve
  emotion-timeline serve --port 8080""",
    "transcribe": """examples:
  emotion-timeline transcribe https://www.youtube.com/watch?v=SOME_ID
  emotion-timeline transcribe episode.mp4 --out data/segments.csv
  emotion-timeline transcribe episode.mp4 --no-vad   # if the transcript has holes""",
    "score-timeline": """examples:
  emotion-timeline score-timeline --segments data/segments.csv --out mine.json
  emotion-timeline score-timeline --segments data/segments.csv --valence""",
    "wer": """examples:
  emotion-timeline wer
  emotion-timeline wer --window 0:00-5:00""",
    "build-dataset": """examples:
  emotion-timeline build-dataset
  emotion-timeline build-dataset --out data/dataset.csv""",
    "preflight": """examples:
  emotion-timeline preflight
  emotion-timeline preflight --need-mib 8000""",
    "split": """examples:
  emotion-timeline split
  emotion-timeline split --verify data/dataset.csv""",
    "fine-tune": """examples:
  emotion-timeline fine-tune
  emotion-timeline fine-tune --epochs 3 --batch-size 32""",
    "build-russian": """examples:
  emotion-timeline build-russian
  emotion-timeline build-russian --out data/russian.csv""",
    "compare-russian": """examples:
  emotion-timeline compare-russian             # read the committed comparison
  emotion-timeline compare-russian --rescore   # rerun all four, the better part of an hour""",
    "translation-cost": """examples:
  emotion-timeline translation-cost
  emotion-timeline translation-cost --rows 500""",
    "valence": """examples:
  emotion-timeline valence
  emotion-timeline valence --rescore""",
    "figures": """examples:
  emotion-timeline figures
  emotion-timeline figures --check   # verify the committed figures match the records""",
}

#: The optional dependency group each command needs. One dict rather than a
#: handler per command, because the failure is always the same shape and the
#: only thing that varies is the name to install. `test_cli.py` checks it against
#: what each command's own help text promises, so the two cannot drift.
EXTRAS: dict[str, str] = {
    "build-dataset": "data",
    "build-russian": "data",
    "fine-tune": "model",
    "compare-russian": "model",
    "score-timeline": "model",
    "translation-cost": "model",
    "transcribe": "stt",
    "serve": "web",
}

#: Set by --debug or this variable; both make an unexpected failure print its
#: traceback instead of one line.
DEBUG_ENV = "EMOTION_TIMELINE_DEBUG"


def head_commit(git: Path | None = None) -> str | None:
    """The checked-out commit, read from .git rather than shelled out for.

    `--version` should say which commit produced the records, and a user who
    installed the wheel has no .git at all -- so this returns None rather than
    depending on git being on PATH.
    """
    git = git if git is not None else Path(__file__).resolve().parents[2] / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            head = (git / head[5:]).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return head[:12] if len(head) >= 12 else None


def version_string() -> str:
    """What `--version` prints: the package, and the commit it was run from."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("emotion-timeline")
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        installed = "unknown"
    commit = head_commit()
    return f"emotion-timeline {installed}" + (f" ({commit})" if commit else "")


def wrapped_help(entries: Sequence[tuple[str, str]], pad: int, width: int) -> Iterator[str]:
    """One `name  help` line per command, wrapped to the terminal."""
    for name, summary in entries:
        body = textwrap.wrap(summary, max(width - pad - 4, 28)) or [""]
        yield f"    {name:<{pad}}{body[0]}"
        for line in body[1:]:
            yield f"    {'':<{pad}}{line}"


def command_listing(width: int | None = None) -> str:
    """The grouped listing `--help` ends with.

    argparse cannot put headings in its own subcommand list, and twenty-one
    names in one flat block is not a list anybody reads -- so this is built from
    :data:`CHAPTERS`, and argparse is never given the one-line help at all.
    """
    if width is None:  # pragma: no cover - depends on the terminal
        width = min(max(shutil.get_terminal_size((88, 24)).columns, 60), 100)
    pad = max(len(name) for name in COMMANDS) + 3
    lines = ["commands:"]
    for title, entries in CHAPTERS:
        lines.append("")
        lines.append(f"  {title}")
        lines.extend(wrapped_help(entries, pad, width))
    lines.append("")
    lines.append("  emotion-timeline <command> --help   what it does, and every default it uses")
    return "\n".join(lines)


def registered_commands(parser: argparse.ArgumentParser) -> frozenset[str]:
    """Every subcommand the parser actually accepts.

    argparse keeps this behind a private attribute. It is dug out here rather
    than in the test that uses it, because that test exists to check that
    :data:`CHAPTERS` lists every command -- and a test that asked CHAPTERS what
    the commands are would check nothing at all.
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return frozenset(action.choices)
    return frozenset()  # pragma: no cover - the parser always has subcommands


def suggest_command(argv: Sequence[str], known: Sequence[str]) -> str | None:
    """A message for a command that does not exist, or None if one does.

    argparse's own invalid-choice error reprints all twenty-one names and
    suggests nothing. The first token that is not a flag is the command, because
    the only options that can precede it are --debug, --version and -h.
    """
    for token in argv:
        if token.startswith("-"):
            continue
        if token in known:
            return None
        close = difflib.get_close_matches(token, list(known), n=3, cutoff=0.5)
        hint = "\n  did you mean:  " + ", ".join(close) if close else ""
        return (
            f"emotion-timeline: there is no {token!r} command{hint}"
            "\n  emotion-timeline --help lists all of them"
        )
    return None


def missing_extra(command: str, error: ImportError) -> str:
    """The one message a user sees when an optional dependency is not there."""
    extra = EXTRAS.get(command)
    if extra is None:  # pragma: no cover - a genuine bug, not a missing extra
        return f"emotion-timeline {command}: {type(error).__name__}: {error}"
    missing = error.name or "a dependency"
    return (
        f"emotion-timeline {command} needs the optional {extra!r} dependencies, "
        f"and {missing} is not installed."
        f"\n  uv sync --extra {extra}"
        f"\n  or just this once:  uv run --extra {extra} emotion-timeline {command} ..."
    )


def missing_file(command: str, error: OSError) -> str:
    """A missing path, said once, instead of a traceback through pathlib."""
    name = error.filename or "a file it needs"
    return (
        f"emotion-timeline {command}: cannot find {name}"
        "\n  every record a read-only command needs is committed, so this usually"
        "\n  means the working directory is not the repository root"
    )


def unexpected(command: str, error: BaseException) -> str:
    """Anything not recognised: one line, and how to see the whole thing."""
    return (
        f"emotion-timeline {command}: {type(error).__name__}: {error}"
        "\n  run it again with --debug for the traceback, and please report it at"
        "\n  https://github.com/alex-krasnoshtanov/Emotion-Timeline/issues"
    )


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
    from emotion_timeline.pipeline import figures as pipeline_figures
    from emotion_timeline.pipeline import timeline as pipeline_timeline
    from emotion_timeline.russian import compare as russian_compare
    from emotion_timeline.russian import figures as russian_figures
    from emotion_timeline.selection import figures as selection_figures
    from emotion_timeline.selection import runs as selection
    from emotion_timeline.training import figures as training_figures
    from emotion_timeline.training import report as training

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
        (
            "fine-tune",
            training.TrainingReport.load(args.run, args.summary),
            training_figures.FIGURES,
            training.check_consistency,
        ),
        (
            "russian",
            russian_compare.Comparison.load(args.comparison),
            russian_figures.FIGURES,
            russian_compare.check_consistency,
        ),
        (
            "pipeline",
            pipeline_timeline.Timeline.load(args.timeline),
            pipeline_figures.FIGURES,
            pipeline_timeline.check_consistency,
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
    """The three columns the split is keyed on, as lists."""
    frame = _read_frame(path)
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


def cmd_fine_tune(args: argparse.Namespace) -> int:
    """Fine-tune the classifier on the rebuilt dataset."""
    from emotion_timeline.training import preflight, run, splits

    try:
        device = preflight.require()
    except RuntimeError as error:
        print(f"cannot train here: {error}", file=sys.stderr)
        return 1
    except ImportError:
        print("training needs --extra model (uv sync --extra model)", file=sys.stderr)
        return 1

    manifest = splits.SplitManifest.load(args.manifest)
    problems = splits.check_consistency(manifest)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    config = run.build_config(
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        weighted_loss=args.class_weights,
    )
    print(f"{config.model_id}, {config.epochs} epochs, batch {config.batch_size}, on {device.name}")
    frames = run.load_split_frames(args.dataset, manifest)
    for name in splits.SPLITS:
        print(f"  {name:<11} {len(frames[name]):>7,}")
    print()

    history, logits = run.train(config, frames, args.weights)
    tail = history.pop()

    record = run.run_record(
        config,
        manifest,
        history,
        seconds=tail["seconds"],
        peak_mib=int(tail["peak_mib"]),
        device_name=device.name,
    )
    written = run.write_record(record, args.out)
    predictions = run.save_predictions(
        args.predictions,
        {name: run.predictions_table(frames[name], logits[name], config) for name in logits},
    )
    print()
    print(f"wrote {written}")
    print(f"wrote {predictions}")
    return 0


def _read_frame(path: str) -> pd.DataFrame:
    """A rebuilt dataset, with the three columns the split is keyed on."""
    import pandas as pd

    frame = pd.read_csv(path)
    missing = {"text", "label", "source"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)}")
    return frame


def cmd_summarise(args: argparse.Namespace) -> int:
    """Turn the predictions a run kept into the records the chapters read."""
    import numpy as np

    from emotion_timeline.data.labels import EMOTIONS
    from emotion_timeline.training import run, splits, summary

    manifest = splits.SplitManifest.load(args.manifest)
    problems = splits.check_consistency(manifest)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    archive = np.load(args.predictions)
    frame = _read_frame(args.dataset)
    keys = splits.row_keys(
        [str(v) for v in frame["text"]],
        [str(v) for v in frame["label"]],
        [str(v) for v in frame["source"]],
        seed=manifest.seed,
    )
    text_of = dict(zip(keys, (str(v) for v in frame["text"]), strict=True))

    classes = list(EMOTIONS)
    held_keys = [str(key) for key in archive["test_row_key"]]
    missing = [key for key in held_keys if key not in text_of]
    if missing:
        print(
            f"{len(missing):,} held-out rows are not in {args.dataset}; "
            "the predictions and the dataset are not the same build",
            file=sys.stderr,
        )
        return 1

    texts = [text_of[key] for key in held_keys]
    true_index = archive["test_true"]
    true = [classes[position] for position in true_index]
    predicted, confidence = summary.decode(archive["test_logits"], classes)

    record = summary.held_out_summary(texts, true, predicted, confidence, classes)
    record["calibration"] = summary.calibration_report(
        archive["validation_logits"],
        archive["validation_true"],
        archive["test_logits"],
        true_index,
    )
    hits = [a == b for a, b in zip(true, predicted, strict=True)]
    # Both halves of the question docs/dataset.md asks: rows the emoticon stripper
    # left as a mangled fragment, and rows that reached [URL] masking intact.
    record["url_bug"] = {
        "_comment": (
            "The emoticon stripper eats the :// out of a URL before masking runs, so "
            "most URLs arrive as a fragment. Both subsets are reported because the "
            "one that would isolate the bug -- mangled against properly masked -- "
            "has too few masked rows to carry it."
        ),
        "mangled": summary.subset_error_rate(texts, hits, "http/"),
        "masked": summary.subset_error_rate(texts, hits, "[URL]"),
    }
    written = run.write_record(record, args.out)

    print(f"{record['total_samples']:,} held out, {record['total_errors']:,} wrong")
    print(f"  accuracy {record['accuracy']:.4f}")
    print()
    for name, block in sorted(record["classes"].items(), key=lambda kv: -kv[1]["error_rate"]):
        print(f"  {name:<9} {block['samples']:>6,} samples   {block['error_rate']:.2%} wrong")
    print()
    calibration = record["calibration"]
    print(f"  temperature {calibration['temperature']:.3f}, fitted on validation")
    for stage in ("before", "after"):
        part = calibration[stage]
        print(
            f"    {stage:<6} calibration error {part['expected_calibration_error']:.4f}   "
            f"confidence {part['correct']:.4f} right against {part['incorrect']:.4f} wrong"
        )
    print()
    print(f"wrote {written}")
    return 0


def cmd_training(args: argparse.Namespace) -> int:
    """What the fine-tune was, what it scored, and how that sits against the card."""
    from emotion_timeline.training import report as training

    record = training.TrainingReport.load(args.run, args.summary)
    problems = training.check_consistency(record)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    card = training.card_f1_from(args.card_metrics)
    for line in training.describe(record, card):
        print(line)
    return 0


def cmd_build_russian(args: argparse.Namespace) -> int:
    """Rebuild the Russian evaluation set and check the funnel against the record."""
    from emotion_timeline.data import ru
    from emotion_timeline.training import run

    try:
        frame, record, enthusiasm = ru.build(args.cache, args.drop_enthusiasm)
    except ImportError:
        print(
            "building needs the source corpus: install with --extra data (uv sync --extra data)",
            file=sys.stderr,
        )
        return 1

    _, alternative, _ = ru.build(args.cache, not args.drop_enthusiasm)
    moved = abs(record.rows - alternative.rows)

    for line in ru.iter_progress(record):
        print(line)
    print()
    print(f"{record.rows:,} rows")
    for name, count in sorted(record.counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<9} {count:>6,}  {count / record.rows:6.1%}")

    if args.write:
        written = run.write_record(
            ru.as_record(record, enthusiasm, moved, args.drop_enthusiasm), args.build_record
        )
        print()
        print(f"wrote {written}")
    else:
        expected = ru.RussianReport.load(args.build_record)
        differences = ru.compare(record, expected)
        for difference in differences:
            print(f"differs from the record: {difference}", file=sys.stderr)
        if differences:
            print("the build no longer reproduces the committed record", file=sys.stderr)
            return 1
        print()
        print(f"matches {expected.source.name} exactly")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


def cmd_russian(args: argparse.Namespace) -> int:
    """What the Russian evaluation set is, and what mapping it down cost."""
    from emotion_timeline.data import ru

    report = ru.RussianReport.load(args.build_record)
    problems = ru.check_consistency(report)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    print(f"{report.rows:,} rows from {report.raw['source']}")
    print()
    for step in report.steps:
        removed = step["rows_in"] - step["rows_out"]
        change = f"{-removed:>8,}" if removed else " " * 8
        print(f"  {step['name']:<16}{step['rows_out']:>8,} rows {change}  {step['note']}")
    print()
    for name, count in sorted(report.class_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<9} {count:>6,}  {count / report.rows:6.1%}")

    enthusiasm = report.enthusiasm
    print()
    if enthusiasm.get("mapped_to"):
        print(
            f"  enthusiasm is on {enthusiasm['rows']:,} rows and moves "
            f"{enthusiasm['rows_moved']:,} of them to {enthusiasm['mapped_to']}; "
            "the rest already carry a label the collapse prefers"
        )
    else:
        print(f"  enthusiasm dropped, taking {enthusiasm['rows_moved']:,} rows with it")
    print(f"  {', '.join(report.raw['dropped_columns'])} dropped: no seven-class equivalent")

    comparison = Path(args.comparison)
    if not comparison.exists():
        return 0

    from emotion_timeline.russian import compare

    scored = compare.Comparison.load(comparison)
    problems = compare.check_consistency(scored)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1
    print()
    for line in compare.describe(scored):
        print(line)

    priced = Path(args.translation_cost)
    if priced.exists():
        from emotion_timeline.russian import cost

        priced_report = cost.TranslationCost.load(priced)
        problems = cost.check_consistency(priced_report)
        for problem in problems:
            print(f"inconsistent record: {problem}", file=sys.stderr)
        if problems:
            return 1
        print()
        print("  and what translation costs, with everything else held constant:")
        for line in cost.describe(priced_report):
            print(line)
    return 0


def _russian_splits(dataset: str, manifest_path: str) -> dict[str, Any]:
    """The Russian set sliced into the committed splits, as plain lists."""
    from emotion_timeline.training import splits

    manifest = splits.SplitManifest.load(manifest_path)
    frame = _read_frame(dataset)
    texts = [str(v) for v in frame["text"]]
    labels = [str(v) for v in frame["label"]]
    keys = splits.row_keys(texts, labels, [str(v) for v in frame["source"]], seed=manifest.seed)
    assignment = splits.assign(keys, labels, manifest.fraction)
    out: dict[str, Any] = {}
    for name in (splits.VALIDATION, splits.TEST):
        chosen = [index for index, split in enumerate(assignment) if split == name]
        out[name] = {
            "texts": [texts[index] for index in chosen],
            "labels": [labels[index] for index in chosen],
        }
    return out


def cmd_compare_russian(args: argparse.Namespace) -> int:
    """Read the committed comparison, or rerun every approach behind --rescore.

    Rerunning loads four models and two translators and takes the better part of
    an hour. That is not what a command called `compare-russian` should do to
    somebody who typed it to see the comparison, so it is behind a flag -- the
    same shape as `valence --rescore`.
    """
    from emotion_timeline.russian import compare

    if args.rescore:  # pragma: no cover - runs four models
        return _rescore_russian(args)

    report = compare.Comparison.load(args.record)
    problems = compare.check_consistency(report)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1
    for line in compare.describe(report):
        print(line)
    print()
    print("  --rescore reruns every approach from the models (needs --extra model)")
    return 0


def _rescore_russian(args: argparse.Namespace) -> int:  # pragma: no cover - runs four models
    """Score every approach to Russian on the one held-out split."""
    import numpy as np

    from emotion_timeline.data import ru
    from emotion_timeline.data.labels import EMOTIONS
    from emotion_timeline.russian import baselines, compare, maps, translate
    from emotion_timeline.training import evaluate, run, splits

    parts = _russian_splits(args.dataset, args.manifest)
    validation, test = parts[splits.VALIDATION], parts[splits.TEST]
    true = test["labels"]
    print(f"{len(true):,} held-out Russian rows, {len(validation['labels']):,} for calibration")

    approaches: dict[str, Any] = {}

    print(f"A: translating with {translate.MODEL}")
    english = {
        name: translate.translate(part["texts"], progress=print)
        for name, part in (("validation", validation), ("test", test))
    }
    a_validation, _ = baselines.predict(args.weights, english["validation"], multi_label=False)
    a_test, _ = baselines.predict(args.weights, english["test"], multi_label=False)

    # The control for "a better translator would have won". `translation-cost`
    # already prices the two engines against each other on English rows; this is
    # the same question asked on the rows the ranking is actually decided on.
    print(f"A-NLLB: translating with {translate.NLLB}")
    nllb_english = translate.translate(test["texts"], model_id=translate.NLLB, progress=print)
    nllb_test, _ = baselines.predict(args.weights, nllb_english, multi_label=False)

    print(f"B: {args.rubert}")
    b_archive = np.load(args.rubert_predictions)
    b_validation = evaluate.softmax(b_archive["validation_logits"])
    b_test = evaluate.softmax(b_archive["test_logits"])

    # Each is scaled by its own temperature, fitted on its own validation rows.
    # Averaging raw probabilities would measure which model is more strident.
    index = {name: position for position, name in enumerate(EMOTIONS)}
    truth = np.array([index[label] for label in validation["labels"]], dtype=np.int64)
    temperatures = {
        "A": evaluate.fit_temperature(np.log(np.clip(a_validation, 1e-12, None)), truth),
        "B": evaluate.fit_temperature(np.log(np.clip(b_validation, 1e-12, None)), truth),
    }
    print(f"  temperatures: A {temperatures['A']:.3f}, B {temperatures['B']:.3f}")

    approaches["A translate, then ours"] = {
        "what": "",
        "model": f"{translate.MODEL} + {args.weights}",
        "temperature": round(temperatures["A"], 4),
        "probabilities": evaluate.softmax(np.log(np.clip(a_test, 1e-12, None)), temperatures["A"]),
    }
    approaches["B native ruBERT"] = {
        "what": "",
        "model": args.rubert,
        "temperature": round(temperatures["B"], 4),
        "probabilities": evaluate.softmax(np.log(np.clip(b_test, 1e-12, None)), temperatures["B"]),
    }

    approaches["A-NLLB translate, then ours"] = {
        "what": "",
        "model": f"{translate.NLLB} + {args.weights}",
        "probabilities": nllb_test,
    }

    for key, model_id, mapping, approximate in (
        ("C multilingual, off the shelf", args.multilingual, maps.MULTILINGUAL, maps.APPROXIMATE),
        ("D the one the pipeline shipped", args.incumbent, None, frozenset()),
    ):
        print(f"{key.split()[0]}: {model_id}")
        probabilities, labels = baselines.predict(model_id, test["texts"])
        # Djacon reports ru-izard's own columns, so ru.to_seven already maps them.
        if mapping is None:
            native = {column: ru.to_seven([column]) for column in labels}
            mapping = {column: target for column, target in native.items() if target is not None}
        folded = mapping
        scored = baselines.score(probabilities, labels, folded, approximate)
        approaches[key] = {
            "what": "",
            "model": model_id,
            "probabilities": scored["probabilities"],
            "approximate_share": scored["approximate_share"],
            **(
                {"caveat": maps.TRAINED_ON_THE_TEST_SET[model_id]}
                if model_id in maps.TRAINED_ON_THE_TEST_SET
                else {}
            ),
        }

    record = compare.build_record(
        true, approaches, pair=("A translate, then ours", "B native ruBERT")
    )
    written = run.write_record(record, args.record)
    print()
    for line in compare.describe(compare.Comparison.load(written)):
        print(line)
    print()
    print(f"wrote {written}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:  # pragma: no cover - needs network
    """A video or an audio file, turned into the three columns `timeline` reads."""
    from emotion_timeline.pipeline import transcribe

    audio = transcribe.fetch_audio(args.source, args.downloads)
    print(f"audio: {audio}")
    segments = transcribe.transcribe(
        audio, args.model, args.language, progress=print, vad=not args.no_vad
    )
    written = transcribe.write_segments(segments, args.out)
    minutes = segments[-1].end_s / 60 if segments else 0.0
    print(f"{len(segments):,} segments over {minutes:.1f} minutes")
    print(f"wrote {written}")
    print(f"next: emotion-timeline score-timeline --segments {written}")
    return 0


def cmd_score_timeline(args: argparse.Namespace) -> int:  # pragma: no cover - runs two models
    """Score every segment with both models and write the timeline record."""
    from emotion_timeline.pipeline import score as scoring
    from emotion_timeline.pipeline import timeline as pipeline
    from emotion_timeline.training import run

    record = scoring.score(
        pipeline.read_segments(args.segments),
        source=pipeline.relative(args.segments),
        gap=args.gap,
        chunk_chars=args.chunk_chars,
        weights=args.weights,
        rubert=args.rubert,
        comparison=args.comparison,
        progress=print,
        valence=args.valence,
    )
    written = run.write_record(record, args.out)
    print()
    for line in pipeline.describe(pipeline.Timeline.load(written)):
        print(line)
    print()
    print(f"wrote {written}")
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    """The per-scene emotion timeline, as a table and as a picture."""
    from emotion_timeline.pipeline import figures as pipeline_figures
    from emotion_timeline.pipeline import timeline as pipeline

    report = pipeline.Timeline.load(args.record)
    segments = pipeline.read_segments(args.segments)
    problems = pipeline.check_consistency(report, segments)
    for problem in problems:
        print(f"inconsistent timeline: {problem}", file=sys.stderr)
    if problems:
        return 1

    for line in pipeline.describe(report):
        print(line)

    if args.against:
        other = pipeline.Timeline.load(args.against)
        overlap = pipeline.runtime_agreement(report, other)
        print()
        print(
            f"  against {other.raw['source']}: the same emotion on "
            f"{float(overlap['share']):.1%} of the {overlap['covered']:,} seconds both cover"
        )
        for move, count in list(overlap["disagreements"].items())[:5]:
            print(f"    {move:<24} {count:>5}s")

    print()
    if not args.write:
        print("  --write regenerates the table and the figure from this record")
        return 0

    written = pipeline.write_csv(pipeline.table(report, segments), args.out)
    drawn = pipeline_figures.render_all(report, args.assets)
    print(f"wrote {written}")
    for path in drawn:
        print(f"wrote {path}")
    return 0


def cmd_translation_cost(args: argparse.Namespace) -> int:  # pragma: no cover - runs two engines
    """Round-trip the model's own English rows and price what translation costs."""
    import numpy as np
    import pandas as pd

    from emotion_timeline.data.labels import EMOTIONS
    from emotion_timeline.russian import baselines, cost, translate
    from emotion_timeline.training import evaluate, run, splits
    from emotion_timeline.training.summary import MARKERS

    manifest = splits.SplitManifest.load(args.manifest)
    frame = pd.read_csv(args.dataset)
    texts = [str(v) for v in frame["text"]]
    labels = [str(v) for v in frame["label"]]
    keys = splits.row_keys(texts, labels, [str(v) for v in frame["source"]], seed=manifest.seed)
    assignment = splits.assign(keys, labels, manifest.fraction)
    held = [index for index, split in enumerate(assignment) if split == splits.TEST]

    rng = np.random.default_rng(manifest.seed)
    size = min(args.rows, len(held))
    picked = sorted(rng.choice(len(held), size=size, replace=False).tolist())
    rows = [held[index] for index in picked]
    english = [texts[index] for index in rows]
    truth = [labels[index] for index in rows]
    print(f"{size:,} of {len(held):,} English held-out rows")

    def score(candidate: list[str]) -> dict[str, Any]:
        probabilities, _ = baselines.predict(args.weights, candidate, multi_label=False)
        predicted = [EMOTIONS[index] for index in probabilities.argmax(axis=1)]
        matrix = evaluate.confusion(truth, predicted, EMOTIONS)
        scores = evaluate.class_scores(matrix, EMOTIONS)
        return {
            "accuracy": round(evaluate.accuracy_of(matrix), 4),
            "macro_f1": round(evaluate.macro(scores, "f1"), 4),
            "markers": cost.marker_shares(candidate, list(MARKERS.values())),
        }

    baseline = score(english)
    print(f"  untranslated: {baseline['accuracy']:.4f}")

    engines: dict[str, Any] = {}
    for name, out_id, back_id in (
        ("opus-mt", translate.BACKWARDS, translate.MODEL),
        ("NLLB-600M", translate.NLLB, translate.NLLB),
    ):
        print(f"{name}: English -> Russian ...")
        russian = translate.translate(
            english, model_id=out_id, clean_output=False, source="en", target="ru"
        )
        print(f"{name}: Russian -> English ...")
        back = translate.translate(russian, model_id=back_id, clean_output=True)
        engines[name] = {"model": f"{out_id} + {back_id}" if out_id != back_id else out_id}
        engines[name].update(score(back))
        print(f"  {name}: {engines[name]['accuracy']:.4f}")

    record = cost.build_record(
        baseline, engines, rows=size, sampled_from=len(held), seed=manifest.seed
    )
    written = run.write_record(record, args.out)
    print()
    for line in cost.describe(cost.TranslationCost.load(written)):
        print(line)
    print()
    print(f"wrote {written}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    """Open the pipeline in a browser."""
    from emotion_timeline.web import app as web

    print(f"emotion-timeline on http://{args.host}:{args.port}")
    print("  a link or a file goes in; the committed example needs no GPU")
    web.serve(host=args.host, port=args.port, downloads=args.downloads)
    return 0


def _rescore_valence(args: argparse.Namespace) -> int:  # pragma: no cover - runs three models
    """Recompute the whole valence record: the model, the stacker, the rules."""
    import json

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from emotion_timeline.data.labels import EMOTIONS
    from emotion_timeline.russian import baselines, translate, va
    from emotion_timeline.training import evaluate, run, splits

    index_of = {name: position for position, name in enumerate(EMOTIONS)}
    manifest = splits.SplitManifest.load(args.manifest)
    frame = pd.read_csv(args.dataset)
    texts = [str(v) for v in frame["text"]]
    labels = [str(v) for v in frame["label"]]
    keys = splits.row_keys(texts, labels, [str(v) for v in frame["source"]], seed=manifest.seed)
    assignment = splits.assign(keys, labels, manifest.fraction)

    blocks = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    a_name, b_name = blocks["pair"]
    native = np.load(args.rubert_predictions)

    parts: dict[str, dict[str, Any]] = {}
    for split in (splits.VALIDATION, splits.TEST):
        chosen = [i for i, value in enumerate(assignment) if value == split]
        rows = [texts[i] for i in chosen]
        print(f"{split}: {len(rows):,} rows")
        print(f"  valence and arousal with {args.checkpoint}")
        valence, arousal = va.predict(rows, args.checkpoint, progress=print)
        print(f"  translating with {translate.MODEL}, then {args.weights}")
        english = translate.translate(rows, progress=print)
        a_raw, _ = baselines.predict(args.weights, english, multi_label=False)
        parts[split] = {
            "y": np.array([index_of[labels[i]] for i in chosen]),
            "labels": [labels[i] for i in chosen],
            "valence": valence,
            "arousal": arousal,
            "A": evaluate.softmax(
                np.log(np.clip(a_raw, 1e-12, None)),
                float(blocks["approaches"][a_name]["temperature"]),
            ),
            "B": evaluate.softmax(
                native[f"{split}_logits"], float(blocks["approaches"][b_name]["temperature"])
            ),
        }

    def stack(features: list[str]) -> np.ndarray:
        def build(part: dict[str, Any]) -> Any:
            return np.hstack(
                [part[k] if k in ("A", "B") else part[k].reshape(-1, 1) for k in features]
            )

        model = LogisticRegression(max_iter=3000, random_state=manifest.seed)
        model.fit(build(parts[splits.VALIDATION]), parts[splits.VALIDATION]["y"])
        chosen: np.ndarray = model.predict(build(parts[splits.TEST]))
        return chosen

    test = parts[splits.TEST]
    y = test["y"]
    argmax = test["B"].argmax(axis=1)
    without, with_va = stack(["A", "B"]), stack(["A", "B", "valence", "arousal"])

    contribution = {
        "_comment": (
            "A stacker over both classifiers, fitted on validation and scored on "
            "test, with and without the two dimensions as extra features."
        ),
        "without": round(float((without == y).mean()), 4),
        "with": round(float((with_va == y).mean()), 4),
        **va.mcnemar(y, without, with_va),
        "verdict": "",
    }
    contribution["verdict"] = (
        "no measurable gain, so this is display-only and off by default"
        if not contribution["significant"]
        else "a measurable gain, which would change how this is used"
    )

    # Everything that was tried and did not work. The chapter quotes these, so
    # they belong in the record rather than in a notebook nobody kept.
    neutral, joy = index_of["Neutral"], index_of["Joy"]
    validation = parts[splits.VALIDATION]

    def best_threshold(rule: str) -> float:
        a_v, b_v = validation["A"].argmax(axis=1), validation["B"].argmax(axis=1)
        disagree = a_v != b_v
        best = (0.0, 0.5)
        for threshold in np.arange(0.20, 0.85, 0.01):
            picked = _apply_rule(
                rule, validation, a_v, b_v, disagree, float(threshold), neutral, joy
            )
            score = float((picked == validation["y"]).mean())
            if score > best[0]:
                best = (score, float(threshold))
        return best[1]

    a_t, b_t = test["A"].argmax(axis=1), test["B"].argmax(axis=1)
    disagree_t = a_t != b_t
    alternatives: dict[str, Any] = {}
    for rule, label in (
        ("arousal-neutral", "low arousal means Neutral where they disagree"),
        ("valence-joy", "valence picks the positive candidate"),
    ):
        threshold = best_threshold(rule)
        picked = _apply_rule(rule, test, a_t, b_t, disagree_t, threshold, neutral, joy)
        measured = va.mcnemar(y, b_t, picked)
        alternatives[rule] = {
            "what": label,
            "threshold_fitted_on_validation": round(threshold, 2),
            "accuracy": round(float((picked == y).mean()), 4),
            "baseline": round(float((b_t == y).mean()), 4),
            **measured,
            "note": (
                f"{measured['gained']} right, {measured['lost']} wrong; "
                f"{float((picked == y).mean()):.4f} against {float((b_t == y).mean()):.4f}"
            ),
        }

    stacker = va.mcnemar(y, argmax, with_va)
    alternatives["learned-stacker"] = {
        "what": "a stacker over both classifiers, against the native model's argmax",
        "accuracy": round(float((with_va == y).mean()), 4),
        "baseline": round(float((argmax == y).mean()), 4),
        **stacker,
        "balanced_accuracy": va.balanced_accuracy(y, with_va),
        "baseline_balanced_accuracy": va.balanced_accuracy(y, argmax),
        "neutral_share": va.share_predicted(with_va, "Neutral"),
        "baseline_neutral_share": va.share_predicted(argmax, "Neutral"),
        "gold_neutral_share": round(float((y == neutral).mean()), 4),
        "note": (
            f"accuracy {float((with_va == y).mean()):.4f} against "
            f"{float((argmax == y).mean()):.4f}, p={stacker['p_value']:.4f} -- but balanced "
            f"accuracy {va.balanced_accuracy(y, with_va):.4f} against "
            f"{va.balanced_accuracy(y, argmax):.4f}, so it is the class prior, not skill"
        ),
    }

    record = va.build_record(
        test["valence"], test["arousal"], test["labels"], contribution, alternatives=alternatives
    )
    written = run.write_record(record, args.record)
    print()
    for line in va.describe(va.ValenceReport.load(written)):
        print(line)
    print()
    print(f"wrote {written}")
    return 0


def _apply_rule(  # pragma: no cover - only reached by --rescore
    rule: str,
    part: dict[str, Any],
    a_pick: Any,
    b_pick: Any,
    disagree: Any,
    threshold: float,
    neutral: int,
    joy: int,
) -> Any:
    """One tie-break rule, applied where the two classifiers disagree."""
    import numpy as np

    if rule == "arousal-neutral":
        touched = disagree & ((a_pick == neutral) | (b_pick == neutral))
        other = np.where(a_pick == neutral, b_pick, a_pick)
        return np.where(
            touched & (part["arousal"] < threshold), neutral, np.where(touched, other, b_pick)
        )
    touched = disagree & ((a_pick == joy) ^ (b_pick == joy))
    other = np.where(a_pick == joy, b_pick, a_pick)
    return np.where(touched, np.where(part["valence"] >= threshold, joy, other), b_pick)


def cmd_valence(args: argparse.Namespace) -> int:
    """What valence and arousal separate, and what they add to the label."""
    from emotion_timeline.russian import va

    if args.rescore:  # pragma: no cover - runs three models
        return _rescore_valence(args)

    report = va.ValenceReport.load(args.record)
    problems = va.check_consistency(report)
    for problem in problems:
        print(f"inconsistent record: {problem}", file=sys.stderr)
    if problems:
        return 1

    for line in va.describe(report):
        print(line)
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


#: The repository, so a default can be printed as the path somebody would type
#: rather than as wherever this checkout happens to live.
REPO = Path(__file__).resolve().parents[2]


def shown_default(value: Any) -> str | None:
    """How a default should read in `--help`, or None to leave it out.

    Two defaults are worth nothing to a reader. `None` is not a value anyone can
    pass, so a flag that has it explains itself in words or says nothing at all;
    and `False` on a switch is just what "off unless given" already means. What
    is left is written relative to the repository, because the absolute path is
    this machine's and no use to anybody reading it anywhere else.
    """
    if value is None or value is False or value is argparse.SUPPRESS:
        return None
    if isinstance(value, str) and value.startswith(str(REPO)):
        return Path(value).relative_to(REPO).as_posix()
    return str(value)


class Formatter(argparse.HelpFormatter):
    """Wrap prose, leave laid-out blocks alone, and show the useful defaults.

    argparse offers these as two formatters that cannot be usefully combined:
    `RawDescriptionHelpFormatter` leaves the epilog's examples alone but also
    stops wrapping the description, and `ArgumentDefaultsHelpFormatter` prints
    `(default: None)` on every optional path. A block that was laid out by hand
    has newlines in it and prose does not, which is the whole of the rule below.
    """

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        if "\n" in text:
            return "".join(f"{indent}{line}" for line in text.splitlines(keepends=True))
        return super()._fill_text(text, width, indent)

    def _get_help_string(self, action: argparse.Action) -> str | None:
        shown = shown_default(action.default)
        if not action.option_strings or shown is None or "%(default)" in (action.help or ""):
            return action.help
        return f"{action.help or ''} (default: {shown})".strip()


def global_flags() -> argparse.ArgumentParser:
    """Flags that mean the same thing before or after the command name.

    Attached to the top-level parser and to every subcommand, so both
    `emotion-timeline --debug timeline` and `emotion-timeline timeline --debug`
    work -- a user who has just been told to pass --debug should not also have to
    be told where to put it.
    """
    shared = argparse.ArgumentParser(add_help=False)
    # SUPPRESS, not False: a subparser parses into a fresh namespace and copies
    # every attribute back over the parent's, so a False default here would
    # silently undo a --debug given before the command name.
    shared.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help=f"print the traceback when something fails (or set {DEBUG_ENV}=1)",
    )
    return shared


def build_parser() -> argparse.ArgumentParser:
    shared = global_flags()
    parser = argparse.ArgumentParser(
        prog="emotion-timeline",
        description="Per-scene emotion analysis: dataset, model and pipeline.",
        epilog=command_listing(),
        formatter_class=Formatter,
        parents=[shared],
    )
    parser.add_argument(
        "--version", action="version", version=version_string(), help="print the version and exit"
    )
    # No `help=` reaches argparse: the one-line summaries live in CHAPTERS, which
    # is what the epilog above prints. Without them argparse lists no choices,
    # which is the point -- but `<command>` still appears in the usage line,
    # which suppressing the action outright would not have done.
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="<command>",
        help="one of the commands listed below",
    )

    wer_parser = sub.add_parser(
        "wer",
        formatter_class=Formatter,
        epilog=EXAMPLES["wer"],
        parents=[shared],
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
        help="time range to score in seconds, e.g. 0:00-18:09",
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
        formatter_class=Formatter,
        parents=[shared],
        description=(
            "Two records of the classifier survive and they describe different "
            "models: the group's model card claims DeBERTa-V2 evaluated "
            "single-label, and the committed training script is DistilBERT "
            "evaluated multi-label. The weights are gone from both university "
            "repositories, so nothing here is rerun. What is checked is each "
            "record's own arithmetic, and the identities that tell the two apart."
        ),
    )
    model_parser.add_argument(
        "--card-metrics", default=card_default, help="the model card's own reported metrics"
    )
    model_parser.set_defaults(func=cmd_model)

    dataset_parser = sub.add_parser(
        "dataset",
        formatter_class=Formatter,
        parents=[shared],
        description=(
            "Reads the committed build record. Every number it prints was produced "
            "by 'build-dataset' on this machine, not copied from the original "
            "notebook -- except the 9,151 synthetic Disgust rows, which are named "
            "as such because their source file no longer exists."
        ),
    )
    dataset_parser.add_argument(
        "--build-record",
        default=record_default,
        help="the committed build record this is checked against",
    )
    dataset_parser.set_defaults(func=cmd_dataset)

    build_dataset_parser = sub.add_parser(
        "build-dataset",
        formatter_class=Formatter,
        epilog=EXAMPLES["build-dataset"],
        parents=[shared],
        description=(
            "Downloads the source corpus and reruns the whole funnel, then checks "
            "the result against the committed record. Takes a few minutes and about "
            "a gigabyte of cache."
        ),
    )
    build_dataset_parser.add_argument(
        "--build-record",
        default=record_default,
        help="the committed build record this is checked against",
    )
    build_dataset_parser.add_argument(
        "--out", help="write the rebuilt dataset here as CSV; without it, nothing is written"
    )
    build_dataset_parser.add_argument("--cache", help="dataset download cache directory")
    build_dataset_parser.set_defaults(func=cmd_build_dataset)

    preflight_parser = sub.add_parser(
        "preflight",
        formatter_class=Formatter,
        epilog=EXAMPLES["preflight"],
        parents=[shared],
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
        help="free video memory the run needs, in MiB; %(default)s fits batch 64 at length 128",
    )
    preflight_parser.set_defaults(func=cmd_preflight)

    split_parser = sub.add_parser(
        "split",
        formatter_class=Formatter,
        epilog=EXAMPLES["split"],
        parents=[shared],
        description=(
            "Prints the split the fine-tune trains on. No rows are committed, so "
            "membership is pinned by a digest per split: --verify recomputes those "
            "from a rebuilt dataset, and --write regenerates the record from one."
        ),
    )
    split_parser.add_argument(
        "--manifest",
        default=str(BENCHMARKS / "training" / "split-manifest.json"),
        help="the committed split; membership is pinned by a digest per split",
    )
    split_parser.add_argument("--verify", help="recompute the split from this rebuilt CSV")
    split_parser.add_argument("--write", help="regenerate the manifest from this rebuilt CSV")
    split_parser.set_defaults(func=cmd_split)

    fine_tune_parser = sub.add_parser(
        "fine-tune",
        formatter_class=Formatter,
        epilog=EXAMPLES["fine-tune"],
        parents=[shared],
        description=(
            "Runs preflight first, then fine-tunes on the committed split. Writes "
            "the run record under benchmarks/, the weights and the per-sample "
            "logits outside it -- those ship as release assets, not in git."
        ),
    )
    fine_tune_parser.add_argument(
        "--dataset",
        default="data/dataset.csv",
        help="the rebuilt training CSV that `build-dataset --out` writes",
    )
    fine_tune_parser.add_argument(
        "--manifest",
        default=str(BENCHMARKS / "training" / "split-manifest.json"),
        help="the committed split; membership is pinned by a digest per split",
    )
    fine_tune_parser.add_argument(
        "--out",
        default=str(BENCHMARKS / "training" / "run-baseline.json"),
        help="where the run record is written",
    )
    fine_tune_parser.add_argument(
        "--weights",
        default="models/distilbert-v1",
        help="the fine-tuned English classifier (a release asset, not in git)",
    )
    fine_tune_parser.add_argument(
        "--predictions",
        default="models/predictions-v1.npz",
        help="where the per-sample logits are kept for `summarise`",
    )
    fine_tune_parser.add_argument(
        "--model-id",
        default=None,
        help="base checkpoint to fine-tune; unset uses the committed record's",
    )
    fine_tune_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epochs to train; unset uses the committed record's value",
    )
    fine_tune_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch size; unset uses the committed record's value",
    )
    fine_tune_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="learning rate; unset uses the committed record's value",
    )
    fine_tune_parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="token limit; unset uses the committed record's value",
    )
    fine_tune_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed; unset uses the committed record's value",
    )
    fine_tune_parser.add_argument(
        "--class-weights",
        action="store_true",
        help="weight the loss by inverse class frequency; off is what the card's model did",
    )
    fine_tune_parser.set_defaults(func=cmd_fine_tune)

    summarise_parser = sub.add_parser(
        "summarise",
        formatter_class=Formatter,
        parents=[shared],
        description=(
            "Reads the logits a fine-tune saved and writes the error-analysis "
            "record for them, in the same shape as the inherited one -- so the "
            "same command and the same figures read both."
        ),
    )
    summarise_parser.add_argument(
        "--predictions", default="models/predictions-v1.npz", help="the logits a fine-tune saved"
    )
    summarise_parser.add_argument(
        "--dataset",
        default="data/dataset.csv",
        help="the rebuilt training CSV that `build-dataset --out` writes",
    )
    summarise_parser.add_argument(
        "--manifest",
        default=str(BENCHMARKS / "training" / "split-manifest.json"),
        help="the committed split; membership is pinned by a digest per split",
    )
    summarise_parser.add_argument(
        "--out",
        default=str(BENCHMARKS / "training" / "held-out-summary.json"),
        help="where the held-out record is written",
    )
    summarise_parser.set_defaults(func=cmd_summarise)

    training_parser = sub.add_parser(
        "training",
        formatter_class=Formatter,
        parents=[shared],
        description=(
            "Reads only committed records, so it runs on a fresh clone with no "
            "dataset and no weights. Reproducing those records is what `fine-tune` "
            "and `summarise` do."
        ),
    )
    training_parser.add_argument(
        "--run",
        default=str(BENCHMARKS / "training" / "run-baseline.json"),
        help="the fine-tune's run record",
    )
    training_parser.add_argument(
        "--summary",
        default=str(BENCHMARKS / "training" / "held-out-summary.json"),
        help="the held-out record `summarise` writes",
    )
    training_parser.add_argument(
        "--card-metrics",
        default=str(BENCHMARKS / "model" / "card-metrics.json"),
        help="the model card's own reported metrics",
    )
    training_parser.set_defaults(func=cmd_training)

    build_russian_parser = sub.add_parser(
        "build-russian",
        formatter_class=Formatter,
        epilog=EXAMPLES["build-russian"],
        parents=[shared],
        description=(
            "Downloads Djacon/ru-izard-emotions and reruns the funnel, then checks "
            "the result against the committed record. Small and quick -- about "
            "25,000 rows."
        ),
    )
    build_russian_parser.add_argument(
        "--build-record",
        default=str(RU_RECORD),
        help="the committed build record this is checked against",
    )
    build_russian_parser.add_argument(
        "--out", help="write the rebuilt set here as CSV; without it, nothing is written"
    )
    build_russian_parser.add_argument("--cache", help="dataset download cache directory")
    build_russian_parser.add_argument(
        "--drop-enthusiasm",
        action="store_true",
        help="drop the enthusiasm rows instead of merging them into Joy",
    )
    build_russian_parser.add_argument(
        "--write", action="store_true", help="regenerate the record rather than check against it"
    )
    build_russian_parser.set_defaults(func=cmd_build_russian)

    russian_parser = sub.add_parser(
        "russian",
        formatter_class=Formatter,
        parents=[shared],
    )
    russian_parser.add_argument(
        "--build-record",
        default=str(RU_RECORD),
        help="the committed build record this is checked against",
    )
    russian_parser.add_argument(
        "--comparison",
        default=str(BENCHMARKS / "russian" / "comparison.json"),
        help="the Russian comparison, read for each model's fitted temperature",
    )
    russian_parser.add_argument(
        "--translation-cost",
        default=str(BENCHMARKS / "russian" / "translation-cost.json"),
        help="the priced record; shown too when it is there",
    )
    russian_parser.set_defaults(func=cmd_russian)

    compare_russian_parser = sub.add_parser(
        "compare-russian",
        formatter_class=Formatter,
        epilog=EXAMPLES["compare-russian"],
        parents=[shared],
        description=(
            "Runs the translator, our model, a native ruBERT and two off-the-shelf "
            "classifiers over the same rows, calibrates the two that answer in our "
            "seven classes, and writes the comparison."
        ),
    )
    compare_russian_parser.add_argument(
        "--dataset",
        default="data/russian.csv",
        help="the rebuilt Russian CSV that `build-russian --out` writes",
    )
    compare_russian_parser.add_argument(
        "--manifest",
        default=str(BENCHMARKS / "russian" / "split-manifest.json"),
        help="the committed split; membership is pinned by a digest per split",
    )
    compare_russian_parser.add_argument(
        "--weights",
        default="models/distilbert-v1",
        help="the fine-tuned English classifier (a release asset, not in git)",
    )
    compare_russian_parser.add_argument(
        "--rubert",
        default="models/rubert-v1",
        help="the native Russian classifier (a release asset, not in git)",
    )
    compare_russian_parser.add_argument(
        "--rubert-predictions",
        default="models/predictions-rubert.npz",
        help="the native model's saved logits, so it is not rerun here",
    )
    compare_russian_parser.add_argument(
        "--multilingual",
        default="tabularisai/multilingual-emotion-classification",
        help="the off-the-shelf multilingual model, approach C",
    )
    compare_russian_parser.add_argument(
        "--incumbent",
        default="Djacon/rubert-tiny2-russian-emotion-detection",
        help="the model the original pipeline shipped, approach D",
    )
    compare_russian_parser.add_argument(
        "--record",
        default=str(BENCHMARKS / "russian" / "comparison.json"),
        help="the committed comparison; read by default, rewritten by --rescore",
    )
    compare_russian_parser.add_argument(
        "--rescore",
        action="store_true",
        help="rerun every approach from the models instead of reading the record",
    )
    compare_russian_parser.set_defaults(func=cmd_compare_russian)

    transcribe_parser = sub.add_parser(
        "transcribe",
        formatter_class=Formatter,
        epilog=EXAMPLES["transcribe"],
        parents=[shared],
        description=(
            "Downloads audio if given a URL, converts it to 16 kHz mono, runs "
            "Whisper large-v3-turbo, and writes start_s, end_s and text. That "
            "CSV is the only interface the rest of the pipeline has to this."
        ),
    )
    transcribe_parser.add_argument("source", help="a video URL, or a local audio or video file")
    transcribe_parser.add_argument(
        "--out", default="data/segments.csv", help="where the segment CSV is written"
    )
    transcribe_parser.add_argument(
        "--downloads", default="downloads", help="where audio is cached; gitignored"
    )
    transcribe_parser.add_argument(
        "--model", default=TURBO, help="the Whisper checkpoint to transcribe with"
    )
    transcribe_parser.add_argument(
        "--language", default="ru", help="passed rather than detected; see the module docstring"
    )
    transcribe_parser.add_argument(
        "--no-vad",
        action="store_true",
        help="keep everything Whisper hears; the voice-activity filter drops speech over music",
    )
    transcribe_parser.set_defaults(func=cmd_transcribe)

    score_timeline_parser = sub.add_parser(
        "score-timeline",
        formatter_class=Formatter,
        epilog=EXAMPLES["score-timeline"],
        parents=[shared],
        description=(
            "Groups a segment CSV into scenes by silence, scores every segment "
            "with the native Russian model and with the translation path, "
            "applies each model's fitted temperature, and writes the record."
        ),
    )
    score_timeline_parser.add_argument(
        "--segments", default=str(SEGMENTS), help="the transcript to read: start_s, end_s and text"
    )
    score_timeline_parser.add_argument(
        "--gap",
        type=float,
        default=GAP_SECONDS,
        help="silence longer than this starts a new scene, in seconds",
    )
    score_timeline_parser.add_argument(
        "--chunk-chars",
        type=int,
        default=CHUNK_CHARS,
        help="the classification unit; keeps the timeline independent of the transcriber",
    )
    score_timeline_parser.add_argument(
        "--weights",
        default="models/distilbert-v1",
        help="the fine-tuned English classifier (a release asset, not in git)",
    )
    score_timeline_parser.add_argument(
        "--rubert",
        default="models/rubert-v1",
        help="the native Russian classifier (a release asset, not in git)",
    )
    score_timeline_parser.add_argument(
        "--comparison",
        default=str(BENCHMARKS / "russian" / "comparison.json"),
        help="the Russian comparison, read for each model's fitted temperature",
    )
    score_timeline_parser.add_argument(
        "--valence",
        nargs="?",
        const="models/va-v1",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "also read valence and arousal (display only; measured to add nothing "
            "to the label, see `emotion-timeline valence`). Off unless given."
        ),
    )
    score_timeline_parser.add_argument(
        "--out", default=str(TIMELINE), help="where the timeline record is written"
    )
    score_timeline_parser.set_defaults(func=cmd_score_timeline)

    timeline_parser = sub.add_parser(
        "timeline",
        formatter_class=Formatter,
        epilog=EXAMPLES["timeline"],
        parents=[shared],
        description=(
            "Reads the committed timeline record, joins it back to the "
            "transcript it was built from, and writes the table and the figure. "
            "No network, no model, no key."
        ),
    )
    timeline_parser.add_argument(
        "--record", default=str(TIMELINE), help="the timeline record to read"
    )
    timeline_parser.add_argument(
        "--segments", default=str(SEGMENTS), help="the transcript to read: start_s, end_s and text"
    )
    timeline_parser.add_argument(
        "--write",
        action="store_true",
        help="also regenerate the committed table and figure (off: this only reads)",
    )
    timeline_parser.add_argument(
        "--out",
        default=str(BENCHMARKS / "pipeline" / "timeline.csv"),
        help="where --write puts the per-scene table",
    )
    timeline_parser.add_argument("--assets", default="assets", help="where --write puts the figure")
    timeline_parser.add_argument(
        "--against",
        help="another timeline record; reports how much runtime the two put the same emotion on",
    )
    timeline_parser.set_defaults(func=cmd_timeline)

    translation_cost_parser = sub.add_parser(
        "translation-cost",
        formatter_class=Formatter,
        epilog=EXAMPLES["translation-cost"],
        parents=[shared],
        description=(
            "Round-trips the English held-out rows through two translation "
            "engines and re-scores them. Domain, labels and annotator are held "
            "constant, so the drop is the translation and nothing else."
        ),
    )
    translation_cost_parser.add_argument(
        "--dataset",
        default="data/dataset.csv",
        help="the rebuilt training CSV that `build-dataset --out` writes",
    )
    translation_cost_parser.add_argument(
        "--manifest",
        default=str(BENCHMARKS / "training" / "split-manifest.json"),
        help="the committed split; membership is pinned by a digest per split",
    )
    translation_cost_parser.add_argument(
        "--weights",
        default="models/distilbert-v1",
        help="the fine-tuned English classifier (a release asset, not in git)",
    )
    translation_cost_parser.add_argument(
        "--rows", type=int, default=3000, help="how many held-out English rows to round-trip"
    )
    translation_cost_parser.add_argument(
        "--out",
        default=str(BENCHMARKS / "russian" / "translation-cost.json"),
        help="where the priced record is written",
    )
    translation_cost_parser.set_defaults(func=cmd_translation_cost)

    serve_parser = sub.add_parser(
        "serve",
        formatter_class=Formatter,
        epilog=EXAMPLES["serve"],
        parents=[shared],
        description=(
            "A local page that takes a link or a file and draws the timeline. "
            "Binds to loopback: it hands URLs to yt-dlp and files to ffmpeg, so "
            "it is a tool you run for yourself rather than a service to expose."
        ),
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="loopback on purpose; see the module docstring before changing it",
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="port to listen on")
    serve_parser.add_argument(
        "--downloads",
        default="downloads",
        help="where fetched audio and uploads are cached; gitignored",
    )
    serve_parser.set_defaults(func=cmd_serve)

    valence_parser = sub.add_parser(
        "valence",
        formatter_class=Formatter,
        epilog=EXAMPLES["valence"],
        parents=[shared],
        description=(
            "The second model the original coursework ran, scored for the first "
            "time against this project's own labels. Valence separates Joy from "
            "the negative classes; arousal barely separates anything; neither "
            "improves the emotion label, which is why both are display-only."
        ),
    )
    valence_parser.add_argument(
        "--record",
        default=str(BENCHMARKS / "russian" / "valence-arousal.json"),
        help="the committed valence record; read by default",
    )
    valence_parser.add_argument(
        "--rescore",
        action="store_true",
        help="recompute the record from the models (needs --extra model and the checkpoint)",
    )
    valence_parser.add_argument(
        "--dataset",
        default="data/russian.csv",
        help="the rebuilt Russian CSV that `build-russian --out` writes",
    )
    valence_parser.add_argument(
        "--manifest",
        default=str(BENCHMARKS / "russian" / "split-manifest.json"),
        help="the committed split; membership is pinned by a digest per split",
    )
    valence_parser.add_argument(
        "--comparison",
        default=str(BENCHMARKS / "russian" / "comparison.json"),
        help="the Russian comparison, read for each model's fitted temperature",
    )
    valence_parser.add_argument(
        "--checkpoint", default="models/va-v1", help="the valence-arousal checkpoint"
    )
    valence_parser.add_argument(
        "--weights",
        default="models/distilbert-v1",
        help="the fine-tuned English classifier (a release asset, not in git)",
    )
    valence_parser.add_argument(
        "--rubert-predictions",
        default="models/predictions-rubert.npz",
        help="the native model's saved logits, so it is not rerun here",
    )
    valence_parser.set_defaults(func=cmd_valence)

    errors_parser = sub.add_parser(
        "errors",
        formatter_class=Formatter,
        parents=[shared],
    )
    errors_parser.add_argument(
        "--report", default=report_default, help="the committed error-analysis record"
    )
    errors_parser.set_defaults(func=cmd_errors)

    models_parser = sub.add_parser(
        "models",
        formatter_class=Formatter,
        parents=[shared],
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
        formatter_class=Formatter,
        epilog=EXAMPLES["figures"],
        parents=[shared],
    )
    figures_parser.add_argument(
        "--report", default=report_default, help="the committed error-analysis record"
    )
    figures_parser.add_argument(
        "--build-record",
        default=record_default,
        help="the committed build record this is checked against",
    )
    figures_parser.add_argument(
        "--card-metrics", default=card_default, help="the model card's own reported metrics"
    )
    figures_parser.add_argument(
        "--run",
        default=str(BENCHMARKS / "training" / "run-baseline.json"),
        help="the fine-tune's run record",
    )
    figures_parser.add_argument(
        "--summary",
        default=str(BENCHMARKS / "training" / "held-out-summary.json"),
        help="the held-out record `summarise` writes",
    )
    figures_parser.add_argument(
        "--comparison",
        default=str(BENCHMARKS / "russian" / "comparison.json"),
        help="the Russian comparison, read for each model's fitted temperature",
    )
    add_selection_arguments(figures_parser)
    figures_parser.add_argument(
        "--timeline", default=str(TIMELINE), help="the record the pipeline figure is drawn from"
    )
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

    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        # clig.dev: a program that needs an argument and is given none should
        # say what the arguments are, not just that one is missing. It still
        # exits 2 to stderr, because from a script this is a usage error.
        print(parser.format_help(), file=sys.stderr)
        return 2

    problem = suggest_command(tokens, COMMANDS)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2

    args = parser.parse_args(tokens)
    debug = bool(getattr(args, "debug", False) or os.environ.get(DEBUG_ENV))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        # clig.dev: say something at once and leave. Nothing here holds a lock
        # or a half-written record -- every writer writes to a temporary file and
        # renames -- so there is nothing to clean up on the way out.
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ImportError as error:
        if debug:
            raise
        print(missing_extra(args.command, error), file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        if debug:
            raise
        print(missing_file(args.command, error), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as error:
        # What a wrong path, a wrong flag or a hand-edited record produce.
        # Anything else is a bug, and gets the same one line plus --debug.
        if debug:
            raise
        print(unexpected(args.command, error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
