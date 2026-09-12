"""Command line entry point.

One subcommand per stage of the study. Subcommands appear here as the stage
they belong to lands, so `--help` is an honest statement of what works.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
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


def cmd_compare_russian(args: argparse.Namespace) -> int:  # pragma: no cover - runs four models
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
    written = run.write_record(record, args.out)
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
    segments = transcribe.transcribe(audio, args.model, args.language, progress=print)
    written = transcribe.write_segments(segments, args.out)
    minutes = segments[-1].end_s / 60 if segments else 0.0
    print(f"{len(segments):,} segments over {minutes:.1f} minutes")
    print(f"wrote {written}")
    print(f"next: emotion-timeline score-timeline --segments {written}")
    return 0


def cmd_score_timeline(args: argparse.Namespace) -> int:  # pragma: no cover - runs two models
    """Score every segment with both models and write the timeline record."""
    import json

    import numpy as np

    from emotion_timeline.pipeline import timeline as pipeline
    from emotion_timeline.russian import baselines, translate
    from emotion_timeline.training import evaluate, run

    segments = pipeline.read_segments(args.segments)
    scenes = pipeline.group(segments, args.gap)
    pieces = pipeline.chunks(scenes, args.chunk_chars)
    texts = [piece.text for piece in pieces]
    print(
        f"{len(segments):,} segments -> {len(scenes)} scenes at a {args.gap:g}s gap "
        f"-> {len(pieces)} chunks of at most {args.chunk_chars} characters"
    )

    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    blocks = comparison["approaches"]
    a_name, b_name = comparison["pair"]

    print(f"B: {args.rubert}")
    b_raw, _ = baselines.predict(args.rubert, texts, multi_label=False)
    print(f"A: {translate.MODEL} -> {args.weights}")
    english = translate.translate(texts, progress=print)
    a_raw, _ = baselines.predict(args.weights, english, multi_label=False)

    # The temperatures were fitted on each model's own validation rows in stage
    # 8. Refitting them here is impossible -- a documentary has no labels -- and
    # reusing them is the entire reason they were recorded.
    def calibrate(raw: np.ndarray, block: dict[str, Any]) -> np.ndarray:
        scaled: np.ndarray = evaluate.softmax(
            np.log(np.clip(raw, 1e-12, None)), float(block["temperature"])
        )
        return scaled

    record = pipeline.build_record(
        scenes,
        pieces,
        calibrate(b_raw, blocks[b_name]),
        calibrate(a_raw, blocks[a_name]),
        source=pipeline.relative(args.segments),
        gap=args.gap,
        primary_model={
            "name": b_name,
            "model": args.rubert,
            "temperature": blocks[b_name]["temperature"],
            "held_out_accuracy": blocks[b_name]["accuracy"],
        },
        second_model={
            "name": a_name,
            "model": f"{translate.MODEL} + {args.weights}",
            "temperature": blocks[a_name]["temperature"],
            "held_out_accuracy": blocks[a_name]["accuracy"],
        },
        agreement={
            "held_out_accuracy_where_they_agreed": comparison["combinations"]["agreement filter"][
                "accuracy"
            ],
            "measured_on": "the held-out ru-izard split, which is social-media register",
            "caveat": (
                "agreement here is a consistency signal, not an accuracy: this "
                "recording has no labels, and two models can agree and both be wrong"
            ),
        },
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

    written = pipeline.write_csv(pipeline.table(report, segments), args.out)
    drawn = pipeline_figures.render_all(report, args.assets)
    print()
    print(f"wrote {written}")
    for path in drawn:
        print(f"wrote {path}")
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

    fine_tune_parser = sub.add_parser(
        "fine-tune",
        help="train the classifier on the rebuilt dataset (needs --extra model and a GPU)",
        description=(
            "Runs preflight first, then fine-tunes on the committed split. Writes "
            "the run record under benchmarks/, the weights and the per-sample "
            "logits outside it -- those ship as release assets, not in git."
        ),
    )
    fine_tune_parser.add_argument("--dataset", default="data/dataset.csv")
    fine_tune_parser.add_argument(
        "--manifest", default=str(BENCHMARKS / "training" / "split-manifest.json")
    )
    fine_tune_parser.add_argument(
        "--out", default=str(BENCHMARKS / "training" / "run-baseline.json")
    )
    fine_tune_parser.add_argument("--weights", default="models/distilbert-v1")
    fine_tune_parser.add_argument("--predictions", default="models/predictions-v1.npz")
    fine_tune_parser.add_argument("--model-id", default=None)
    fine_tune_parser.add_argument("--epochs", type=int, default=None)
    fine_tune_parser.add_argument("--batch-size", type=int, default=None)
    fine_tune_parser.add_argument("--learning-rate", type=float, default=None)
    fine_tune_parser.add_argument("--max-length", type=int, default=None)
    fine_tune_parser.add_argument("--seed", type=int, default=None)
    fine_tune_parser.add_argument(
        "--class-weights",
        action="store_true",
        help="weight the loss by inverse class frequency (default: off, as the card's model was)",
    )
    fine_tune_parser.set_defaults(func=cmd_fine_tune)

    summarise_parser = sub.add_parser(
        "summarise",
        help="turn a run's kept predictions into the held-out record",
        description=(
            "Reads the logits a fine-tune saved and writes the error-analysis "
            "record for them, in the same shape as the inherited one -- so the "
            "same command and the same figures read both."
        ),
    )
    summarise_parser.add_argument("--predictions", default="models/predictions-v1.npz")
    summarise_parser.add_argument("--dataset", default="data/dataset.csv")
    summarise_parser.add_argument(
        "--manifest", default=str(BENCHMARKS / "training" / "split-manifest.json")
    )
    summarise_parser.add_argument(
        "--out", default=str(BENCHMARKS / "training" / "held-out-summary.json")
    )
    summarise_parser.set_defaults(func=cmd_summarise)

    training_parser = sub.add_parser(
        "training",
        help="the fine-tune: what it was, what it scored, and how it compares",
        description=(
            "Reads only committed records, so it runs on a fresh clone with no "
            "dataset and no weights. Reproducing those records is what `fine-tune` "
            "and `summarise` do."
        ),
    )
    training_parser.add_argument(
        "--run", default=str(BENCHMARKS / "training" / "run-baseline.json")
    )
    training_parser.add_argument(
        "--summary", default=str(BENCHMARKS / "training" / "held-out-summary.json")
    )
    training_parser.add_argument(
        "--card-metrics", default=str(BENCHMARKS / "model" / "card-metrics.json")
    )
    training_parser.set_defaults(func=cmd_training)

    build_russian_parser = sub.add_parser(
        "build-russian",
        help="rebuild the Russian evaluation set from the public corpus (needs --extra data)",
        description=(
            "Downloads Djacon/ru-izard-emotions and reruns the funnel, then checks "
            "the result against the committed record. Small and quick -- about "
            "25,000 rows."
        ),
    )
    build_russian_parser.add_argument("--build-record", default=str(RU_RECORD))
    build_russian_parser.add_argument("--out", help="write the rebuilt set here as CSV")
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
        help="the Russian evaluation set, and what mapping it to seven classes cost",
    )
    russian_parser.add_argument("--build-record", default=str(RU_RECORD))
    russian_parser.add_argument(
        "--comparison", default=str(BENCHMARKS / "russian" / "comparison.json")
    )
    russian_parser.set_defaults(func=cmd_russian)

    compare_russian_parser = sub.add_parser(
        "compare-russian",
        help="score every approach to Russian on the held-out split (needs --extra model)",
        description=(
            "Runs the translator, our model, a native ruBERT and two off-the-shelf "
            "classifiers over the same rows, calibrates the two that answer in our "
            "seven classes, and writes the comparison."
        ),
    )
    compare_russian_parser.add_argument("--dataset", default="data/russian.csv")
    compare_russian_parser.add_argument(
        "--manifest", default=str(BENCHMARKS / "russian" / "split-manifest.json")
    )
    compare_russian_parser.add_argument("--weights", default="models/distilbert-v1")
    compare_russian_parser.add_argument("--rubert", default="models/rubert-v1")
    compare_russian_parser.add_argument(
        "--rubert-predictions", default="models/predictions-rubert.npz"
    )
    compare_russian_parser.add_argument(
        "--multilingual", default="tabularisai/multilingual-emotion-classification"
    )
    compare_russian_parser.add_argument(
        "--incumbent", default="Djacon/rubert-tiny2-russian-emotion-detection"
    )
    compare_russian_parser.add_argument(
        "--out", default=str(BENCHMARKS / "russian" / "comparison.json")
    )
    compare_russian_parser.set_defaults(func=cmd_compare_russian)

    transcribe_parser = sub.add_parser(
        "transcribe",
        help="a video URL or an audio file into a segment CSV (needs --extra stt)",
        description=(
            "Downloads audio if given a URL, converts it to 16 kHz mono, runs "
            "Whisper large-v3-turbo, and writes start_s, end_s and text. That "
            "CSV is the only interface the rest of the pipeline has to this."
        ),
    )
    transcribe_parser.add_argument("source", help="a video URL, or a local audio or video file")
    transcribe_parser.add_argument("--out", default="data/segments.csv")
    transcribe_parser.add_argument(
        "--downloads", default="downloads", help="where audio is cached; gitignored"
    )
    transcribe_parser.add_argument("--model", default=TURBO)
    transcribe_parser.add_argument(
        "--language", default="ru", help="passed rather than detected; see the module docstring"
    )
    transcribe_parser.set_defaults(func=cmd_transcribe)

    score_timeline_parser = sub.add_parser(
        "score-timeline",
        help="run both models over a transcript and write the timeline record "
        "(needs --extra model)",
        description=(
            "Groups a segment CSV into scenes by silence, scores every segment "
            "with the native Russian model and with the translation path, "
            "applies each model's fitted temperature, and writes the record."
        ),
    )
    score_timeline_parser.add_argument("--segments", default=str(SEGMENTS))
    score_timeline_parser.add_argument(
        "--gap",
        type=float,
        default=GAP_SECONDS,
        help="silence longer than this starts a new scene (seconds)",
    )
    score_timeline_parser.add_argument(
        "--chunk-chars",
        type=int,
        default=CHUNK_CHARS,
        help="the classification unit; keeps the timeline independent of the transcriber",
    )
    score_timeline_parser.add_argument("--weights", default="models/distilbert-v1")
    score_timeline_parser.add_argument("--rubert", default="models/rubert-v1")
    score_timeline_parser.add_argument(
        "--comparison", default=str(BENCHMARKS / "russian" / "comparison.json")
    )
    score_timeline_parser.add_argument("--out", default=str(TIMELINE))
    score_timeline_parser.set_defaults(func=cmd_score_timeline)

    timeline_parser = sub.add_parser(
        "timeline",
        help="the per-scene emotion timeline over the committed transcript",
        description=(
            "Reads the committed timeline record, joins it back to the "
            "transcript it was built from, and writes the table and the figure. "
            "No network, no model, no key."
        ),
    )
    timeline_parser.add_argument("--record", default=str(TIMELINE))
    timeline_parser.add_argument("--segments", default=str(SEGMENTS))
    timeline_parser.add_argument("--out", default=str(BENCHMARKS / "pipeline" / "timeline.csv"))
    timeline_parser.add_argument("--assets", default="assets", help="where the figure goes")
    timeline_parser.add_argument(
        "--against",
        help="another timeline record; reports how much runtime the two put the same emotion on",
    )
    timeline_parser.set_defaults(func=cmd_timeline)

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
    figures_parser.add_argument("--run", default=str(BENCHMARKS / "training" / "run-baseline.json"))
    figures_parser.add_argument(
        "--summary", default=str(BENCHMARKS / "training" / "held-out-summary.json")
    )
    figures_parser.add_argument(
        "--comparison", default=str(BENCHMARKS / "russian" / "comparison.json")
    )
    add_selection_arguments(figures_parser)
    figures_parser.add_argument("--timeline", default=str(TIMELINE))
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
