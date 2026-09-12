"""The fine-tune's committed records, and the comparison they are for.

These are the published numbers of the retrain, asserted the way every other
stage's are: if one of them changes, this file says so by name. The one that is
enforced rather than merely checked is Disgust, which must never be given a
delta -- see `training/report.py` for why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_timeline.training import evaluate
from emotion_timeline.training import report as training


def committed() -> training.TrainingReport:
    return training.TrainingReport.load()


def card() -> dict[str, float]:
    return training.card_f1_from("")


# --- the records hold together ----------------------------------------------


def test_the_committed_records_are_internally_consistent() -> None:
    assert training.check_consistency(committed()) == []


def test_the_run_is_the_one_the_documentation_describes() -> None:
    config = committed().config
    assert config["model_id"] == "distilbert-base-uncased"
    assert config["epochs"] == 3
    assert config["batch_size"] == 64
    assert config["weighted_loss"] is False


def test_the_published_held_out_figures() -> None:
    report = committed()
    scores = report.scores
    assert report.samples == 62_877
    assert report.accuracy == pytest.approx(0.9164, abs=5e-5)
    assert evaluate.macro(scores, "f1") == pytest.approx(0.8088, abs=5e-4)
    assert evaluate.weighted(scores, "f1") == pytest.approx(0.9164, abs=5e-4)


def test_weighted_recall_is_the_accuracy_here_too() -> None:
    """The identity docs/model-selection.md leaned on, holding on our own numbers."""
    report = committed()
    assert evaluate.weighted(report.scores, "recall") == pytest.approx(report.accuracy, abs=5e-4)


def test_every_class_f1_is_the_harmonic_mean_of_its_own_row() -> None:
    for score in committed().scores:
        assert score.f1 == pytest.approx(score.implied_f1, abs=5e-5)


def test_the_supports_are_the_split_the_manifest_committed() -> None:
    supports = {score.name: score.support for score in committed().scores}
    assert supports["Neutral"] == 2_010
    assert supports["Fear"] == 8_003
    assert supports["Anger"] == 8_695
    assert sum(supports.values()) == 62_877


# --- what it says against the card -------------------------------------------


def test_every_comparable_class_but_one_improves() -> None:
    """The result the stage exists for, and the shape of it matters.

    Five of the six comparable classes are ahead of the card's, Surprise is
    fractionally behind, and the biggest gain is Neutral -- the class the
    inherited chapter called the model's worst.
    """
    rows = {row.name: row for row in training.compare_to_card(committed(), card())}
    improved = [name for name, row in rows.items() if row.delta is not None and row.delta > 0]
    assert sorted(improved) == ["Anger", "Fear", "Joy", "Neutral", "Sadness"]
    assert rows["Surprise"].delta == pytest.approx(-0.0042, abs=5e-4)
    assert rows["Neutral"].delta == pytest.approx(0.0421, abs=5e-4)


def test_the_retrain_beats_the_cards_accuracy() -> None:
    assert committed().accuracy > 0.8995


def test_the_macro_average_is_lower_even_so() -> None:
    """And the whole of the difference is the class whose data is missing.

    Accuracy and weighted F1 are up, macro F1 is down, and that is not a
    contradiction: macro gives Disgust the same weight as Joy, and Disgust lost
    64% of its training rows with the synthetic file.
    """
    report = committed()
    assert evaluate.macro(report.scores, "f1") < 0.8127
    without_disgust = [score for score in report.scores if score.name != "Disgust"]
    card_scores = card()
    assert evaluate.macro(without_disgust, "f1") > sum(
        value for name, value in card_scores.items() if name != "Disgust"
    ) / len(without_disgust)


def test_disgust_is_never_given_a_number() -> None:
    """Enforced in code, because a caveat under a table gets read second."""
    rows = {row.name: row for row in training.compare_to_card(committed(), card())}
    assert rows["Disgust"].retrain_f1 is None
    assert rows["Disgust"].delta is None
    assert "synthetic" in (rows["Disgust"].reason or "")


def test_nothing_the_command_prints_puts_a_delta_beside_disgust() -> None:
    lines = list(training.describe(committed(), card()))
    disgust = [line for line in lines if "Disgust" in line and "card" in line]
    assert disgust and all("not comparable" in line for line in disgust)


def test_a_class_the_card_did_not_score_is_simply_absent() -> None:
    rows = training.compare_to_card(committed(), {"Joy": 0.9})
    assert [row.name for row in rows] == ["Joy"]


# --- calibration and the URL fragment ----------------------------------------


def test_the_model_is_overconfident_and_one_scalar_fixes_most_of_it() -> None:
    calibration = committed().calibration
    assert calibration["temperature"] == pytest.approx(1.499, abs=5e-3)
    assert calibration["after"]["expected_calibration_error"] < (
        calibration["before"]["expected_calibration_error"] / 2
    )


def test_a_mangled_url_costs_more_than_five_times_the_error_rate() -> None:
    """What docs/dataset.md deferred to a retrain, measured."""
    bug = committed().summary["url_bug"]
    assert bug["mangled"]["samples"] == 244
    assert bug["mangled"]["error_rate"] == pytest.approx(0.4713, abs=5e-4)
    assert bug["mangled"]["error_rate"] > 5 * bug["mangled"]["error_rate_elsewhere"]


def test_too_few_urls_survived_masking_to_isolate_the_bug() -> None:
    """Which is the limitation, and it is recorded rather than glossed."""
    assert committed().summary["url_bug"]["masked"]["samples"] < 20


def test_the_caps_marker_is_nothing_like_the_inherited_count() -> None:
    """750-ish, not 4,040: the inherited analysis cannot have used the cleaned text."""
    markers = committed().summary["textual_features"]
    assert markers["ALL-CAPS word"]["present_samples"] == 737


# --- records that do not hold together are refused ---------------------------


def broken(tmp_path: Path, **changes: object) -> training.TrainingReport:
    summary = json.loads(Path(training.DEFAULT_SUMMARY).read_text(encoding="utf-8"))
    summary.update(changes)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return training.TrainingReport.load(training.DEFAULT_RUN, path)


def test_a_record_with_no_matrix_cannot_be_scored(tmp_path: Path) -> None:
    summary = json.loads(Path(training.DEFAULT_SUMMARY).read_text(encoding="utf-8"))
    del summary["confusion"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    problems = training.check_consistency(training.TrainingReport.load(training.DEFAULT_RUN, path))
    assert problems == ["no confusion matrix, so no per-class score can be recomputed"]


def test_a_matrix_that_does_not_sum_to_the_header_is_caught(tmp_path: Path) -> None:
    problems = training.check_consistency(broken(tmp_path, total_samples=5))
    assert any("the header says 5" in problem for problem in problems)


def test_a_matrix_that_contradicts_the_error_count_is_caught(tmp_path: Path) -> None:
    problems = training.check_consistency(broken(tmp_path, total_errors=1))
    assert any("the header says 1" in problem for problem in problems)


def test_a_class_row_that_contradicts_its_own_sample_count_is_caught(tmp_path: Path) -> None:
    summary = json.loads(Path(training.DEFAULT_SUMMARY).read_text(encoding="utf-8"))
    summary["classes"]["Joy"]["samples"] = 3
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    problems = training.check_consistency(training.TrainingReport.load(training.DEFAULT_RUN, path))
    assert any("record says 3" in problem for problem in problems)


def test_a_negative_temperature_is_caught(tmp_path: Path) -> None:
    summary = json.loads(Path(training.DEFAULT_SUMMARY).read_text(encoding="utf-8"))
    summary["calibration"]["temperature"] = -1.0
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    problems = training.check_consistency(training.TrainingReport.load(training.DEFAULT_RUN, path))
    assert any("positive" in problem for problem in problems)


def test_a_matrix_over_different_classes_is_caught(tmp_path: Path) -> None:
    summary = json.loads(Path(training.DEFAULT_SUMMARY).read_text(encoding="utf-8"))
    summary["confusion"] = {"Only": {"Only": summary["total_samples"]}}
    summary["classes"] = {"Only": {"samples": summary["total_samples"], "errors": 0}}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    problems = training.check_consistency(training.TrainingReport.load(training.DEFAULT_RUN, path))
    assert any("same seven" in problem for problem in problems)


def test_the_two_records_together_have_one_digest() -> None:
    assert len(committed().digest) == 64
