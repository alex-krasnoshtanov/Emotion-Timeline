"""The summary the retrain writes, and the proof it is the shape the chapter reads.

The test that matters most is the last one: a record built here is loaded by
`analysis/error_analysis.py` and passes its consistency checks. That is what makes
the existing four figures work on the new model without a line of new drawing
code, and it is the thing most likely to break quietly if either side drifts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_timeline.analysis import error_analysis as ea
from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.training import summary

CLASSES = ("Joy", "Sadness", "Anger")


# --- per class ---------------------------------------------------------------


def test_a_class_reports_its_own_errors_and_what_they_became() -> None:
    true = ["Joy", "Joy", "Joy", "Sadness"]
    predicted = ["Joy", "Sadness", "Anger", "Sadness"]
    breakdown = summary.class_breakdown(true, predicted, CLASSES)
    assert breakdown["Joy"]["samples"] == 3
    assert breakdown["Joy"]["errors"] == 2
    assert breakdown["Joy"]["error_rate"] == pytest.approx(2 / 3, abs=5e-5)
    assert breakdown["Joy"]["confused_with"] == {"Sadness": 1, "Anger": 1}
    assert breakdown["Sadness"]["errors"] == 0


def test_a_class_with_no_samples_is_reported_rather_than_skipped() -> None:
    breakdown = summary.class_breakdown(["Joy"], ["Joy"], CLASSES)
    assert breakdown["Anger"] == {
        "samples": 0,
        "errors": 0,
        "error_rate": 0.0,
        "confused_with": {},
    }


def test_only_the_worst_few_confusions_are_kept() -> None:
    true = ["Joy"] * 6
    predicted = ["Sadness", "Sadness", "Sadness", "Anger", "Anger", "Joy"]
    breakdown = summary.class_breakdown(true, predicted, CLASSES, top_confusions=1)
    assert breakdown["Joy"]["confused_with"] == {"Sadness": 3}
    assert breakdown["Joy"]["errors"] == 5


# --- surface markers ---------------------------------------------------------


def test_the_markers_are_the_ones_that_survive_cleaning() -> None:
    """Capitals are gone by the time the model sees the text; the marker is not."""
    assert summary.MARKERS["ALL-CAPS word"] == "[CAPS]"


def test_a_marker_is_measured_with_and_without() -> None:
    texts = ["[CAPS] angry", "[CAPS] loud", "calm", "quiet"]
    correct = [False, False, True, True]
    features = summary.textual_features(texts, correct)["ALL-CAPS word"]
    assert features["present_samples"] == 2
    assert features["error_rate_present"] == 1.0
    assert features["error_rate_absent"] == 0.0


def test_a_marker_nothing_carries_reports_zero_rather_than_dividing() -> None:
    features = summary.textual_features(["plain"], [True])
    assert features["Ellipsis"]["present_samples"] == 0
    assert features["Ellipsis"]["error_rate_present"] == 0.0


def test_every_marker_gets_a_row() -> None:
    features = summary.textual_features(["a!", "b?", "c...", "[CAPS] d"], [True] * 4)
    assert set(features) == {"_comment", *summary.MARKERS}


# --- length ------------------------------------------------------------------


def test_longer_answers_are_measured_against_shorter_ones() -> None:
    texts = ["a much longer sentence than the others here"] * 8 + ["short"] * 8
    correct = [True] * 8 + [False] * 8
    length = summary.length_statistics(texts, correct)
    assert length["correct"]["mean_characters"] > length["incorrect"]["mean_characters"]
    assert length["correct"]["mean_words"] == 8.0
    assert length["p_value"] < 0.01


def test_a_split_with_no_errors_still_produces_a_length_block() -> None:
    length = summary.length_statistics(["one", "two"], [True, True])
    assert length["incorrect"]["mean_characters"] == 0.0
    assert length["p_value"] == 1.0


# --- confidence --------------------------------------------------------------


def test_the_gap_and_the_errors_a_threshold_cannot_catch() -> None:
    block = summary.confidence_block([0.95, 0.9, 0.95, 0.4], [True, True, False, False])
    assert block["mean_when_correct"] == pytest.approx(0.925)
    assert block["mean_when_incorrect"] == pytest.approx(0.675)
    assert block["high_confidence_errors"] == 1
    assert block["high_confidence_error_share_of_errors"] == 0.5


def test_a_run_that_never_errs_reports_no_confident_errors() -> None:
    block = summary.confidence_block([0.9, 0.8], [True, True])
    assert block["high_confidence_errors"] == 0
    assert block["high_confidence_error_share_of_errors"] == 0.0


# --- vocabulary --------------------------------------------------------------


def test_a_word_needs_enough_occurrences_to_be_listed() -> None:
    texts = ["cursed"] * 3 + ["ordinary"] * 3
    correct = [False] * 3 + [True] * 3
    assert summary.vocabulary_bias(texts, correct, minimum=10)["error_biased"] == []


def test_the_word_that_rides_along_with_errors_comes_first() -> None:
    texts = ["cursed word"] * 50 + ["ordinary word"] * 50
    correct = [False] * 50 + [True] * 50
    bias = summary.vocabulary_bias(texts, correct, top=1, minimum=10)
    assert bias["error_biased"] == ["cursed"]
    assert bias["correct_biased"] == ["ordinary"]


# --- the whole record --------------------------------------------------------


def synthetic() -> dict[str, object]:
    true = (["Joy"] * 40) + (["Sadness"] * 30) + (["Anger"] * 30)
    predicted = (
        (["Joy"] * 36 + ["Sadness"] * 4)
        + (["Sadness"] * 27 + ["Anger"] * 3)
        + (["Anger"] * 25 + ["Joy"] * 5)
    )
    # Enough rows carrying a marker to clear the 100-sample floor the figure uses.
    texts = [f"sentence number {index}!" for index in range(100)]
    confidence = [0.9 if a == b else 0.4 for a, b in zip(true, predicted, strict=True)]
    return summary.held_out_summary(texts, true, predicted, confidence, CLASSES)


def test_the_headline_matches_the_per_class_rows() -> None:
    record = synthetic()
    assert record["total_samples"] == 100
    assert record["total_errors"] == 12
    assert record["accuracy"] == 0.88


def test_the_record_loads_and_passes_the_chapters_own_checks(tmp_path: Path) -> None:
    """The whole point of matching the schema: no new figure code for the new model."""
    path = tmp_path / "held-out.json"
    path.write_text(json.dumps(synthetic()), encoding="utf-8")
    report = ea.ErrorReport.load(path)
    assert ea.check_consistency(report) == []
    assert report.total_samples == 100


def test_the_new_record_draws_the_figures_the_old_one_draws(tmp_path: Path) -> None:
    path = tmp_path / "held-out.json"
    path.write_text(json.dumps(synthetic()), encoding="utf-8")
    report = ea.ErrorReport.load(path)
    written = ea.render_all(report, tmp_path / "assets")
    assert len(written) == len(ea.FIGURES)
    assert all(figure.exists() for figure in written)


def test_the_seven_real_classes_fit_the_record() -> None:
    true = list(EMOTIONS)
    record = summary.held_out_summary(
        ["a"] * 7, true, true, [0.9] * 7, EMOTIONS, split="held-out test"
    )
    assert set(record["classes"]) == set(EMOTIONS)
    assert record["error_rate"] == 0.0
    assert record["split"] == "held-out test"
