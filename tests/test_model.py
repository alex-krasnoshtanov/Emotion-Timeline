"""The model card's arithmetic, and the identities that separate it from the script.

The weights are gone from both university repositories, so nothing here can be
rerun against a model. What can be checked is that each surviving record's
numbers hold together, and that the relations which distinguish a single-label
evaluation from a multi-label one say what the docs claim they say.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_timeline.data import build as ds
from emotion_timeline.model import card as mc
from emotion_timeline.model import figures as mc_figures

# --- the card holds together --------------------------------------------------


def test_both_records_are_internally_consistent() -> None:
    assert mc.check_consistency(mc.ModelReport.load()) == []


def test_every_class_f1_is_the_harmonic_mean_of_its_own_precision_and_recall() -> None:
    """Across all three tables. This is what says the metrics were computed."""
    report = mc.ModelReport.load()
    checked = 0
    for evaluation in report.evaluations:
        for score in evaluation.classes.values():
            assert score.f1 == pytest.approx(score.implied_f1, abs=5e-4), (
                f"{evaluation.name}/{score.name}"
            )
            checked += 1
    assert checked == 18  # 7 held-out + 5 CARER + 6 stress


def test_the_published_summary_metrics_follow_from_the_tables() -> None:
    report = mc.ModelReport.load()
    for name, accuracy, macro in (
        ("held_out", 0.8995, 0.8127),
        ("carer", 0.9255, 0.8865),
        ("stress", 0.3144, 0.2787),
    ):
        evaluation = report.evaluation(name)
        assert evaluation.accuracy == accuracy
        assert evaluation.macro_f1 == macro
        assert evaluation.implied_accuracy == pytest.approx(accuracy, abs=5e-4)
        assert evaluation.implied_macro_f1 == pytest.approx(macro, abs=5e-4)


def test_the_held_out_weighted_f1_and_macro_columns_reproduce() -> None:
    held = mc.ModelReport.load().evaluation("held_out")
    assert held.implied_weighted_f1 == pytest.approx(0.9028, abs=5e-4)
    assert held.samples == sum(c.support for c in held.classes.values()) == 64_250


# --- the two records are not the same model -----------------------------------


def test_the_cards_held_out_table_is_a_single_label_evaluation() -> None:
    """Micro F1 and accuracy are the same quantity when every prediction is one label.

    A false positive for one class is another class's false negative, so micro
    precision, micro recall and accuracy coincide. The card reports micro F1 and
    accuracy as the same number, which a thresholded multi-label head has no
    reason to produce.
    """
    report = mc.ModelReport.load()
    held = report.evaluation("held_out")
    assert held.raw["micro_f1"] == held.accuracy == 0.8995
    assert mc.single_label_identity_holds(held)


def test_the_scripts_own_metrics_are_multi_label() -> None:
    """Its micro F1 and subset accuracy differ, which single-label cannot do.

    And two of the metrics it reports -- hamming loss and subset accuracy -- have
    no single-label meaning at all.
    """
    metrics = mc.ModelReport.load().script_metrics
    assert metrics["f1_micro"] == 0.8724
    assert metrics["subset_accuracy"] == 0.8279
    assert metrics["f1_micro"] != metrics["subset_accuracy"]
    assert {"hamming_loss", "subset_accuracy"} <= set(metrics)


def test_the_scripts_micro_f1_follows_from_its_own_precision_and_recall() -> None:
    metrics = mc.ModelReport.load().script_metrics
    implied = mc.harmonic_mean(metrics["precision_micro"], metrics["recall_micro"])
    assert implied == pytest.approx(metrics["f1_micro"], abs=5e-4)


def test_the_hamming_loss_recovers_the_multi_label_rate() -> None:
    """The one number here that is derived, and the one that says which dataset.

    Solving the hamming loss for the average true labels per sample gives 1.025,
    so roughly one row in forty carried more than one label. The dataset the card
    names is single-label by construction, which would give exactly 1.
    """
    report = mc.ModelReport.load()
    assert report.script_labels_per_sample == pytest.approx(1.025, abs=5e-4)
    assert report.script_labels_per_sample > 1.0

    # A genuinely single-label run returns 1 from the same arithmetic.
    single = mc.labels_per_sample(recall_micro=0.9, precision_micro=0.9, hamming_loss=2 * 0.1 / 7)
    assert single == pytest.approx(1.0, abs=1e-9)


def test_the_surviving_tokenizer_is_not_the_one_the_card_claims() -> None:
    """30,522 WordPiece tokens is distilbert-base-uncased, exactly.

    The card claims a 128,100-token SentencePiece vocabulary, which is
    DeBERTa-V2's. The only tokenizer artefact that survived the training run is
    the other one.
    """
    report = mc.ModelReport.load()
    assert report.script["surviving_tokenizer_tokens"] == 30_522
    assert report.script["config_model_name"] == "distilbert-base-uncased"
    assert report.card["claimed_vocabulary_size"] == 128_100
    assert report.card["claimed_architecture"] == "DeBERTa-V2-Base"


def test_the_script_names_a_different_input_file_from_the_card() -> None:
    report = mc.ModelReport.load()
    assert report.script["config_data_path"] == "Super-Iter1.parquet"
    assert report.script["config_problem_type"] == "multi_label_classification"
    assert report.card["claimed_task"] == "single-label"


# --- the mislabelled dataset table --------------------------------------------


def test_the_cards_dataset_counts_are_the_ones_this_repository_builds() -> None:
    """The counts are right. They are the state the build passes through.

    Specifically the state after drop-love and before the priority collapse, plus
    the 9,151 synthetic Disgust rows. So the table is not made up -- it is
    correct data with the labels attached to the wrong rows.
    """
    report = mc.ModelReport.load()
    counts = {count for _, count in report.card_dataset_table}
    synthetic = report.raw["mislabelled_dataset_table"]["synthetic_disgust_rows"]
    assert synthetic == 9_151
    assert counts == set(report.corrected_labels) | {synthetic}


def test_four_of_the_seven_dataset_labels_are_on_the_wrong_row() -> None:
    mislabelled = dict((count, said) for count, said, _ in mc.ModelReport.load().mislabelled())
    assert mislabelled == {
        149_321: "Neutral",  # Joy
        127_866: "Happiness",  # Sadness
        54_041: "Sadness",  # Fear
        13_401: "Fear",  # Neutral
    }


def test_the_cards_own_support_column_proves_which_labels_are_right() -> None:
    """The contradiction is inside the card, not between the card and this repo.

    Its dataset table calls Joy's 149,321 rows Neutral. Its performance table,
    for the same model on the same data, gives Neutral a support of 2,010 -- 15%
    of 13,401, not of 149,321. The support column agrees with the corrected
    assignment in all seven classes and with the card's own dataset table in
    none of the four it got wrong.
    """
    report = mc.ModelReport.load()
    published = ds.DatasetReport.load().published["class_counts"]
    held = report.evaluation("held_out")

    # The support column is 15% of the corrected assignment, class by class.
    for name, total in published.items():
        assert abs(held.classes[name].support - total * mc.HELD_OUT_FRACTION) <= 1, name

    # And for every row the card got wrong, the support of the class it named is
    # nowhere near 15% of the count it attached to that name.
    for count, label, _truth in report.mislabelled():
        named = "Joy" if label == "Happiness" else label
        assert abs(held.classes[named].support - count * mc.HELD_OUT_FRACTION) > 1_000, label


def test_anger_and_surprise_are_right_only_by_coincidence() -> None:
    """They hold the same rank under both assignments, so the shuffle missed them."""
    report = mc.ModelReport.load()
    coincidence = report.raw["mislabelled_dataset_table"]["correct_by_coincidence"]
    assert coincidence == ["Anger", "Surprise"]
    wrong = {said for _, said, _ in report.mislabelled()}
    assert not ({"Anger", "Surprise"} & wrong)


# --- the stress test -----------------------------------------------------------


def test_the_stress_macro_average_divides_by_six_not_seven() -> None:
    """Surprise is absent from the table, so the published figure is over six."""
    stress = mc.ModelReport.load().evaluation("stress")
    assert stress.scored_classes == 6
    assert "Surprise" not in stress.classes
    assert stress.macro_f1_over(6) == pytest.approx(0.2787, abs=5e-4)
    assert stress.macro_f1_over(7) == pytest.approx(0.2389, abs=5e-4)


def test_the_control_group_is_outscored_by_inputs_built_to_break_the_model() -> None:
    """Which means the stress test is not measuring what it set out to measure.

    Emoji-heavy, typo-laden and deliberately subtle text all score higher than
    the untouched control. A control condition that is harder than the
    manipulation is not a baseline.
    """
    report = mc.ModelReport.load()
    beating = {row.name for row in report.outliers_beating_the_control()}
    assert beating == {"Emoji-heavy", "Typos/Slang", "Subtle emotion"}
    assert report.control.accuracy == 0.3147


def test_the_headline_stress_accuracy_is_the_controls_own_score() -> None:
    """The control is 70% of the samples, so it sets the overall figure."""
    report = mc.ModelReport.load()
    stress = report.evaluation("stress")
    control = report.control
    assert control.samples / stress.samples == pytest.approx(0.70, abs=0.01)
    assert control.accuracy == pytest.approx(stress.accuracy, abs=5e-4)


def test_the_outlier_breakdown_accounts_for_every_sample() -> None:
    report = mc.ModelReport.load()
    rows = report.outlier_types
    assert sum(row.samples for row in rows) == 5_000
    weighted = sum(row.accuracy * row.samples for row in rows) / 5_000
    assert weighted == pytest.approx(report.evaluation("stress").accuracy, abs=5e-4)


def test_three_categories_score_exactly_zero() -> None:
    """561 samples and not one right, which chance alone does not produce.

    Guessing uniformly over seven classes would land about one in seven. Three
    categories of 187 each scoring 0.0000 is a label-space problem, not a model
    that found them hard.
    """
    failures = mc.ModelReport.load().total_failures()
    assert {row.name for row in failures} == {"Sarcasm", "Negation", "Mixed emotions"}
    assert sum(row.samples for row in failures) == 561
    assert all(row.f1 == 0.0 for row in failures)


def test_the_reported_disgust_false_positives_do_not_follow_from_the_table() -> None:
    """2,753 sits outside the range the card's own Disgust precision allows."""
    report = mc.ModelReport.load()
    stress = report.evaluation("stress")
    low, high = mc.disgust_false_positive_bounds(stress)
    assert (low, high) == (2746, 2750)
    assert not low <= stress.raw["reported_disgust_false_positives"] <= high


def test_a_perfect_precision_bounds_the_false_positives_at_zero() -> None:
    """The bound is arithmetic, not a fudge factor: Fear's precision is 1.0."""
    stress = mc.ModelReport.load().evaluation("stress")
    assert stress.classes["Fear"].precision == 1.0
    assert stress.classes["Anger"].precision == 0.0
    assert stress.classes["Anger"].f1 == 0.0


# --- calibration ---------------------------------------------------------------


def test_the_confidence_gap_closes_where_the_model_stops_working() -> None:
    """In-domain the gap is 0.46; on the synthetic set it inverts."""
    report = mc.ModelReport.load()
    assert report.evaluation("held_out").confidence_gap == pytest.approx(0.4598, abs=5e-4)
    assert report.evaluation("carer").confidence_gap == pytest.approx(0.2094, abs=5e-4)
    assert report.evaluation("stress").confidence_gap < 0


# --- figures -------------------------------------------------------------------


def test_the_model_figures_are_stamped_with_the_record(tmp_path: Path) -> None:
    from emotion_timeline import figures as shared

    report = mc.ModelReport.load()
    for path in mc_figures.render_all(report, tmp_path):
        assert shared.read_stamp(path) == report.digest
    assert mc_figures.check_figures_current(report, tmp_path) == []


def test_the_committed_model_figures_are_current() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    assert mc_figures.check_figures_current(mc.ModelReport.load(), assets) == []


def test_a_broken_f1_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(mc.DEFAULT_CARD).read_text(encoding="utf-8"))
    raw["card"]["evaluations"]["held_out"]["classes"]["Joy"]["f1"] = 0.5
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    problems = mc.check_consistency(mc.ModelReport.load(path))
    assert any("held_out/Joy" in p for p in problems)
    assert any("macro F1" in p for p in problems)


def test_a_support_that_does_not_sum_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(mc.DEFAULT_CARD).read_text(encoding="utf-8"))
    raw["card"]["evaluations"]["carer"]["classes"]["Joy"]["support"] = 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    problems = mc.check_consistency(mc.ModelReport.load(path))
    assert any("supports sum to" in p for p in problems)
