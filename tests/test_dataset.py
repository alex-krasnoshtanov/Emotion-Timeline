"""The dataset build: the text rules, the label rules, and the funnel arithmetic.

Rebuilding the set needs a 550,000-row download, so CI cannot run it. What CI
can do is check that the recorded build holds together, that the rules that
produced it still behave, and -- the strongest evidence available -- that the
recorded dataset and the separately recorded evaluation describe the same thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from emotion_timeline.analysis import error_analysis as ea
from emotion_timeline.data import build as ds
from emotion_timeline.data import clean, labels
from emotion_timeline.data import figures as ds_figures

# --- the committed record ----------------------------------------------------


def test_the_recorded_build_is_internally_consistent() -> None:
    """Steps chain, class counts sum, and the synthetic rows account for the rest."""
    assert ds.check_consistency(ds.DatasetReport.load()) == []


def test_the_reproducible_part_plus_the_synthetic_rows_is_the_published_set() -> None:
    report = ds.DatasetReport.load()
    published = report.published
    assert report.rows == 419_180
    assert published["synthetic_disgust_rows"] == 9_151
    assert report.rows + 9_151 == published["rows"] == 428_331


def test_every_source_corpus_is_accounted_for() -> None:
    report = ds.DatasetReport.load()
    composition = report.composition
    assert sum(body["rows"] for body in composition.values()) == report.steps[0]["rows_out"]
    assert sum(body["kept"] for body in composition.values()) == report.steps[1]["rows_out"]
    # GoEmotions is the only corpus filtered, and almost all of it goes.
    filtered = {n for n, b in composition.items() if b["kept"] != b["rows"]}
    assert filtered == {"GoEmotions"}
    assert composition["GoEmotions"]["kept"] == 1_013


# --- the cross-check that matters --------------------------------------------


def test_the_evaluation_set_is_15_percent_of_this_dataset() -> None:
    """Two records written months apart, agreeing class by class.

    ``benchmarks/error-analysis/`` records an evaluation over 64,250 samples and
    says nothing about where they came from. This record says the training set
    holds 428,331 rows and says nothing about any evaluation. 15% of the second
    is the first -- exactly, in total, and within one row in all seven classes,
    the rounding a stratified split needs to hit an exact total.

    Neither number can be derived from the other, so their agreeing is real
    evidence that the error analysis describes a model trained on this data. It
    is the only such evidence available: the raw predictions are gone.
    """
    published = ds.DatasetReport.load().published
    report = ea.ErrorReport.load()

    assert round(published["rows"] * published["held_out_fraction"]) == report.total_samples

    for name, total in published["class_counts"].items():
        share = total * published["held_out_fraction"]
        assert abs(report.classes[name].samples - share) <= 1, name


def test_the_classes_the_model_fails_on_are_the_rare_ones() -> None:
    """The imbalance in the dataset and the error rates line up, in that order."""
    published = ds.DatasetReport.load().published["class_counts"]
    report = ea.ErrorReport.load()
    smallest = sorted(published, key=lambda name: published[name])[:3]
    hardest = [c.name for c in report.hardest(3)]
    assert set(hardest) <= set(smallest) | {"Fear"}
    assert "Neutral" in smallest and hardest[0] == "Neutral"


# --- text rules --------------------------------------------------------------


def test_shouting_is_marked_rather_than_erased() -> None:
    """The [CAPS] marker is why ALL-CAPS is measurable in the error analysis."""
    assert clean.mark_shouting("I am SO happy") == "i am [CAPS] so happy"
    assert clean.mark_shouting("[URL] IS DOWN") == "[URL] [CAPS] is [CAPS] down"
    # A single letter is not shouting; "I" and "A" would otherwise flood it.
    assert clean.mark_shouting("I A ok") == "i a ok"


def test_placeholders_are_never_treated_as_words() -> None:
    for step in (clean.mark_shouting, clean.squeeze_repeats):
        assert "[NAME]" in step("[NAME] said")
    assert clean.drop_repeated_placeholders("[TAG] [TAG] [TAG] hello") == "[TAG] hello"
    assert clean.drop_repeated_placeholders("[TAG] hi [TAG]") == "[TAG] hi [TAG]"


def test_slang_expands_before_numbers_are_masked() -> None:
    """`b4` has to become `before`, not `b[NUM]`."""
    assert clean.clean_text("b4 the show") == "before the show"
    assert clean.clean_text("I paid 20 dollars") == "i paid [NUM] dollars"


def test_emphasis_is_capped_rather_than_removed() -> None:
    assert clean.normalize_punctuation("amazing!!!!!!") == "amazing!!"
    assert clean.normalize_punctuation("what?????") == "what??"
    assert clean.squeeze_repeats("sooooo good") == "soo good"


def test_handles_and_numbers_become_markers() -> None:
    cleaned = clean.clean_text("@someone owes me 12 things")
    assert cleaned == "[TAG] owes me [NUM] things"


def test_the_emoticon_stripper_eats_the_slashes_out_of_urls() -> None:
    """A reproduced bug, kept because reproducing it is the point.

    ``:/`` is on the emoticon list and emoticons are removed before URLs are
    masked, so ``https://`` loses its ``://`` and the URL pattern no longer
    matches it. Of the 1,857 texts in the source corpus that contain a URL,
    only 199 still look like one by the time ``[URL]`` masking runs; the other
    1,658 reach the training data as a mangled fragment.

    The published model was trained on data with this in it, so the build
    reproduces it rather than quietly fixing it. ``docs/dataset.md`` says what
    fixing it would cost.
    """
    assert clean.strip_emoji("https://example.com") == "https/example.com"
    assert "[URL]" not in clean.clean_text("see https://example.com")
    # A URL that survives to the masking step is masked properly.
    assert clean.clean_text("see www.example.com") == "see [URL]"


def test_emoji_and_emoticons_go() -> None:
    assert clean.strip_emoji("happy 😂 today :)").strip() == "happy  today"


def test_the_order_of_the_pipeline_is_pinned() -> None:
    """Changing the order changes the dataset, so it is not changed by accident."""
    assert [step.__name__ for step in clean.PIPELINE] == [
        "normalize_quotes",
        "normalize_placeholders",
        "strip_emoji",
        "expand_slang",
        "mask_entities",
        "collapse_whitespace",
        "mark_shouting",
        "collapse_whitespace",
        "normalize_punctuation",
        "squeeze_repeats",
        "drop_repeated_placeholders",
        "collapse_whitespace",
    ]
    assert clean.PIPELINE == clean.BEFORE_DEDUPE + clean.AFTER_DEDUPE + clean.FINALLY


def test_deduplicating_late_would_lose_rows() -> None:
    """Why the pipeline is split in two rather than run end to end.

    The original build removed duplicates halfway through, before case was
    folded and punctuation capped. These two texts are distinct at that point
    and identical afterwards, so deduplicating at the end would discard one of
    them -- 671 rows across the corpus, and a build that no longer reproduces
    the published counts.
    """
    for a, b in (("Yes indeed", "yes indeed"), ("amazing!!!", "amazing!!!!!")):
        assert clean.clean_early(a) != clean.clean_early(b)
        assert clean.clean_text(a) == clean.clean_text(b)


# --- label rules -------------------------------------------------------------


def test_the_rarer_reading_wins_a_multi_label_row() -> None:
    assert labels.collapse(["Anger", "Disgust"]) == "Disgust"
    assert labels.collapse(["Joy", "Surprise"]) == "Surprise"
    assert labels.collapse(["Neutral", "Fear"]) == "Neutral"


def test_a_single_label_is_left_alone() -> None:
    assert labels.collapse(["Joy"]) == "Joy"
    assert labels.collapse([]) is None


def test_joy_wins_only_by_default() -> None:
    """Joy is a third of the data; it must not also win ties."""
    assert "Joy" not in labels.PRIORITY
    assert labels.collapse(["Joy", "Anger"]) == "Anger"
    assert labels.collapse(["Joy", "Love"]) == "Joy"  # nothing else applies


def test_the_source_annotation_overrules_the_collapse() -> None:
    assert labels.restore("Anger", ["disgust"]) == "Disgust"
    assert labels.restore("Joy", ["neutral"]) == "Neutral"
    assert labels.restore("Joy", ["excitement"]) == "Joy"


def test_a_row_annotated_both_disgust_and_neutral_ends_up_neutral() -> None:
    """33 rows, and the order of the two restore passes is what decides them."""
    assert labels.restore("Anger", ["disgust", "neutral"]) == "Neutral"
    assert labels.RESTORE_FROM_SOURCE == ("disgust", "neutral")


def test_love_is_dropped_because_it_has_no_home() -> None:
    assert labels.is_dropped(["Love"])
    assert not labels.is_dropped(["Joy"])
    assert "Love" not in labels.EMOTIONS


def test_the_seven_classes_are_the_ones_the_model_reports_on() -> None:
    assert set(labels.EMOTIONS) == set(ea.ErrorReport.load().classes)


# --- figures -----------------------------------------------------------------


def test_dataset_figures_are_stamped_with_the_record(tmp_path: Path) -> None:
    from emotion_timeline import figures as shared

    report = ds.DatasetReport.load()
    for path in ds_figures.render_all(report, tmp_path):
        assert shared.read_stamp(path) == report.digest
    assert ds_figures.check_figures_current(report, tmp_path) == []


def test_the_committed_dataset_figures_are_current() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    assert ds_figures.check_figures_current(ds.DatasetReport.load(), assets) == []


def test_a_broken_record_refuses_to_draw(tmp_path: Path) -> None:
    raw = json.loads(Path(ds.DEFAULT_RECORD).read_text(encoding="utf-8"))
    raw["class_counts"]["Joy"] += 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    problems = ds.check_consistency(ds.DatasetReport.load(path))
    assert any("class counts sum to" in p for p in problems)
    assert any("Joy" in p for p in problems)


def test_a_step_that_does_not_chain_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(ds.DEFAULT_RECORD).read_text(encoding="utf-8"))
    raw["steps"][2]["rows_in"] = 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = ds.check_consistency(ds.DatasetReport.load(path))
    assert any("takes 1 rows but" in p for p in problems)


# --- comparing a fresh build against the record ------------------------------


def _record_matching_the_committed_one() -> ds.BuildRecord:
    expected = ds.DatasetReport.load()
    record = ds.BuildRecord(source=expected.raw["source"])
    record.steps = [
        ds.Step(s["name"], s["rows_in"], s["rows_out"], s["note"]) for s in expected.steps
    ]
    record.counts = expected.class_counts
    return record


def test_a_matching_build_reports_no_differences() -> None:
    assert ds.compare(_record_matching_the_committed_one(), ds.DatasetReport.load()) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.counts.__setitem__("Joy", 1), "record says"),
        (lambda r: r.steps.pop(), "was not run"),
        (lambda r: r.counts.__setitem__("Ennui", 5), "not in the record"),
    ],
)
def test_a_build_that_drifts_is_caught(mutate: object, expected: str) -> None:
    record = _record_matching_the_committed_one()
    mutate(record)  # type: ignore[operator]
    problems = ds.compare(record, ds.DatasetReport.load())
    assert problems and any(expected in p for p in problems)


# --- the funnel, on a corpus small enough to read ----------------------------


def corpus() -> pd.DataFrame:
    """A miniature source corpus, one row per behaviour the funnel has to have."""
    rows = [
        # kept: ordinary MELD row
        ("A perfectly ordinary line.", ["Joy"], ["joy"], "MELD"),
        # dropped: GoEmotions without disgust
        ("Reddit comment about nothing.", ["Joy"], ["admiration"], "GoEmotions"),
        # kept: GoEmotions with disgust, and restored to Disgust from its source
        ("That is revolting to look at.", ["Anger"], ["disgust"], "GoEmotions"),
        # dropped: duplicate of the first row once cleaned
        ("A perfectly ordinary line.", ["Joy"], ["joy"], "SemEval"),
        # dropped: under three tokens
        ("Ok!", ["Joy"], ["joy"], "MELD"),
        # dropped: Love has no seven-class home
        ("I love this so much.", ["Love"], ["love"], "TwitterEmotion"),
        # kept: multi-label, collapsed by priority to the rarer reading
        ("Both at once somehow.", ["Anger", "Surprise"], ["anger", "surprise"], "ISEAR"),
        # kept: restored to Neutral from its source annotation
        ("Just stating a fact here.", ["Sadness"], ["neutral"], "ISEAR"),
    ]
    return pd.DataFrame(rows, columns=["text", "labels_str", "labels_source", "source"])


def test_each_filter_removes_exactly_what_it_is_for() -> None:
    frame = corpus()
    assert len(ds.filter_sources(frame)) == 7  # the GoEmotions row without disgust

    frame = ds.clean_early(ds.filter_sources(frame))
    assert len(ds.deduplicate(frame)) == 6  # the repeated line

    frame = ds.filter_by_length(ds.clean_late(ds.deduplicate(frame)))
    assert len(frame) == 5  # "Ok!"

    frame = ds.drop_love(ds.tidy(frame))
    assert len(frame) == 4


def test_labels_are_assigned_by_priority_then_source() -> None:
    frame = corpus()
    for step in (ds.filter_sources, ds.clean_early, ds.deduplicate, ds.clean_late):
        frame = step(frame)
    frame = ds.assign_labels(ds.drop_love(ds.tidy(ds.filter_by_length(frame))))
    assigned = sorted(frame["label"])
    assert assigned == ["Disgust", "Joy", "Neutral", "Surprise"]


def test_a_build_reports_every_step_and_its_own_class_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ds, "load_source", lambda cache_dir=None: corpus())
    frame, record = ds.build()

    assert record.rows == len(frame) == 4
    assert [step.name for step in record.steps] == [
        "load",
        "filter-sources",
        "normalise",
        "deduplicate",
        "fold-case",
        "filter-length",
        "tidy",
        "drop-love",
        "assign-labels",
    ]
    assert record.counts == {"Disgust": 1, "Joy": 1, "Neutral": 1, "Surprise": 1}
    assert sum(record.counts.values()) == record.rows
    # every step chains into the next, which is what check_consistency asserts
    # of the committed record
    for earlier, later in zip(record.steps, record.steps[1:], strict=False):
        assert earlier.rows_out == later.rows_in
    assert list(frame.columns) == ["text", "label", "source", "labels_source", "token_count"]


def test_progress_lines_name_every_step() -> None:
    record = _record_matching_the_committed_one()
    lines = list(ds.iter_progress(record))
    assert len(lines) == len(record.steps)
    assert "-34,940" in "\n".join(lines)  # the Love rows
