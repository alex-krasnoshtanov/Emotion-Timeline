"""Mapping other vocabularies onto ours, and combining two models that answer in it.

None of this needs a model. Folding eleven classes into seven is array work, and
so are the three combination rules -- which is the point of keeping them separate
from the forward pass that produces the numbers they combine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.russian import baselines, compare, maps

THREE = ("Anger", "Disgust", "Fear")


# --- the label maps ----------------------------------------------------------


# --- the translator, and the bug that made approach A a measurement of itself ---


def test_a_row_is_split_into_its_sentences_before_translation() -> None:
    """Marian is a sentence model. Handed two, it translates one and says nothing."""
    from emotion_timeline.russian import translate

    assert translate.sentences("First one. Second one! Third?") == [
        "First one.",
        "Second one!",
        "Third?",
    ]


def test_a_row_with_no_terminator_is_still_one_sentence() -> None:
    from emotion_timeline.russian import translate

    assert translate.sentences("no terminator here") == ["no terminator here"]
    assert translate.sentences("   ") == []


def test_an_ellipsis_ends_a_sentence_and_a_decimal_does_not() -> None:
    from emotion_timeline.russian import translate

    assert len(translate.sentences("Well\u2026 quite.")) == 2
    assert len(translate.sentences("It cost 3.50 in total.")) == 1


def test_the_translation_is_cleaned_the_way_the_training_set_was() -> None:
    """The model was trained on cleaned text; raw translator output is a different surface."""
    from emotion_timeline.russian import translate

    cleaned = translate.to_model_english("Oh MY GOD, I missed it :( http://example.com")
    assert "[CAPS]" in cleaned
    assert ":(" not in cleaned
    # The URL bug the dataset build has is reproduced, not fixed -- the model was
    # trained on data carrying it. See CLAUDE.md.
    assert "http/example.com" in cleaned


def test_every_target_is_one_of_our_seven() -> None:
    assert maps.check_map(maps.MULTILINGUAL) == []
    assert set(maps.MULTILINGUAL.values()) <= set(EMOTIONS)


def test_the_eleven_class_model_is_fully_mapped() -> None:
    """A label left out would be silently dropped, which changes the score."""
    assert len(maps.MULTILINGUAL) == 11


def test_the_approximate_labels_are_the_ones_with_no_home() -> None:
    assert {"contempt", "frustration", "gratitude", "love"} == maps.APPROXIMATE
    assert all(label in maps.MULTILINGUAL for label in maps.APPROXIMATE)


def test_love_is_mapped_here_although_the_english_build_drops_it() -> None:
    """Dropping a training row and dropping a prediction are different things."""
    assert maps.MULTILINGUAL["love"] == "Joy"
    assert "love" in maps.APPROXIMATE


def test_the_incumbent_is_flagged_as_trained_on_the_test_set() -> None:
    caveat = maps.TRAINED_ON_THE_TEST_SET["Djacon/rubert-tiny2-russian-emotion-detection"]
    assert "trained on the corpus" in caveat


def test_a_map_onto_a_class_we_do_not_have_is_caught() -> None:
    problems = maps.check_map({**maps.MULTILINGUAL, "joy": "Elation"})
    assert any("not ours" in problem for problem in problems)


def test_a_map_that_forgets_an_approximate_label_is_caught() -> None:
    trimmed = {k: v for k, v in maps.MULTILINGUAL.items() if k != "love"}
    problems = maps.check_map(trimmed)
    assert any("never mapped" in problem for problem in problems)


# --- folding a model's classes into ours --------------------------------------


def test_two_source_classes_that_mean_the_same_thing_add_up() -> None:
    """contempt and disgust are one claim about Disgust made twice, not two rivals."""
    probabilities = np.array([[0.3, 0.3, 0.1]])
    folded = baselines.to_seven_matrix(
        probabilities, ["contempt", "disgust", "joy"], maps.MULTILINGUAL
    )
    assert folded[0][EMOTIONS.index("Disgust")] == pytest.approx(0.6 / 0.7)
    assert folded[0][EMOTIONS.index("Joy")] == pytest.approx(0.1 / 0.7)


def test_every_row_comes_out_a_distribution() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.random((20, 11))
    folded = baselines.to_seven_matrix(probabilities, list(maps.MULTILINGUAL), maps.MULTILINGUAL)
    assert folded.shape == (20, 7)
    assert folded.sum(axis=1) == pytest.approx(np.ones(20))


def test_a_class_nobody_could_map_stops_contributing() -> None:
    folded = baselines.to_seven_matrix(np.array([[0.9, 0.1]]), ["nonsense", "joy"], {"joy": "Joy"})
    assert folded[0][EMOTIONS.index("Joy")] == pytest.approx(1.0)


def test_a_row_with_no_opinion_stays_uniform_rather_than_dividing_by_zero() -> None:
    folded = baselines.to_seven_matrix(np.array([[0.0]]), ["joy"], {"joy": "Joy"})
    assert folded[0] == pytest.approx(np.full(7, 1 / 7))


def test_columns_that_do_not_match_the_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="columns against"):
        baselines.to_seven_matrix(np.zeros((2, 3)), ["joy"], {"joy": "Joy"})


def test_the_approximate_share_says_how_much_is_our_map_rather_than_the_model() -> None:
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    share = baselines.approximate_share(probabilities, ["love", "joy"], frozenset({"love"}))
    assert share == pytest.approx(0.5)


def test_a_model_with_no_approximate_labels_reports_none() -> None:
    assert baselines.approximate_share(np.eye(2), ["joy", "anger"], frozenset()) == 0.0
    assert (
        baselines.approximate_share(np.zeros((0, 2)), ["joy", "anger"], frozenset({"joy"})) == 0.0
    )


def test_sigmoid_and_softmax_are_the_two_heads_they_belong_to() -> None:
    assert baselines.sigmoid(np.array([[0.0]]))[0][0] == pytest.approx(0.5)
    assert baselines.softmax(np.array([[1.0, 1.0]]))[0] == pytest.approx([0.5, 0.5])


def test_score_reports_a_prediction_and_a_confidence_per_row() -> None:
    probabilities = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    result = baselines.score(
        probabilities, ["joy", "anger", "fear"], maps.MULTILINGUAL, maps.APPROXIMATE
    )
    assert result["predicted"] == ["Joy", "Anger"]
    assert len(result["confidence"]) == 2
    assert result["probabilities"].shape == (2, 7)


# --- combining two models -----------------------------------------------------


def two_models() -> tuple[np.ndarray, np.ndarray]:
    """Four rows: agree twice, disagree twice, with different confidences."""
    first = np.array(
        [
            [0.8, 0.1, 0.1],  # Anger, confident
            [0.1, 0.8, 0.1],  # Disgust
            [0.6, 0.2, 0.2],  # Anger, less sure
            [0.2, 0.2, 0.6],  # Fear
        ]
    )
    second = np.array(
        [
            [0.7, 0.2, 0.1],  # Anger -- agrees
            [0.2, 0.7, 0.1],  # Disgust -- agrees
            [0.1, 0.9, 0.0],  # Disgust -- disagrees, and is surer
            [0.9, 0.05, 0.05],  # Anger -- disagrees, and is surer
        ]
    )
    return first, second


def test_the_soft_vote_averages_rather_than_letting_one_model_win() -> None:
    first, second = two_models()
    chosen = compare.soft_vote(first, second)
    # Row 2: 0.6/0.2 against 0.1/0.9 averages to 0.35/0.55, so Disgust.
    assert list(chosen) == [0, 1, 1, 0]


def test_the_confidence_pick_takes_whichever_was_surer() -> None:
    first, second = two_models()
    chosen = compare.confidence_pick(first, second)
    assert list(chosen) == [0, 1, 1, 0]


def test_the_agreement_filter_covers_only_where_they_match() -> None:
    first, second = two_models()
    mask, shared = compare.agreement_filter(first, second)
    assert list(mask) == [True, True, False, False]
    assert list(shared[:2]) == [0, 1]


def test_coverage_is_reported_with_the_accuracy_it_belongs_to() -> None:
    """The whole point: an accuracy over two rows is not an accuracy over four."""
    true = ["Anger", "Disgust", "Anger", "Fear"]
    predicted = ["Anger", "Disgust", "Disgust", "Anger"]
    result = compare.measure_filtered(true, predicted, [True, True, False, False])
    assert result["coverage"] == 0.5
    assert result["samples"] == 2
    assert result["accuracy"] == 1.0


def test_a_filter_that_covers_nothing_reports_nothing_rather_than_dividing() -> None:
    result = compare.measure_filtered(["Anger"], ["Anger"], [False])
    assert result == {"coverage": 0.0, "samples": 0, "accuracy": 0.0, "macro_f1": 0.0}


def test_measure_is_the_same_arithmetic_the_other_stages_use() -> None:
    true = ["Anger", "Anger", "Disgust", "Fear"]
    predicted = ["Anger", "Disgust", "Disgust", "Fear"]
    result = compare.measure(true, predicted, THREE)
    assert result["samples"] == 4
    assert result["accuracy"] == 0.75
    assert sum(entry["support"] for entry in result["classes"].values()) == 4


# --- the record ---------------------------------------------------------------


def four_rows() -> tuple[list[str], dict[str, dict[str, object]]]:
    """Two approaches over four rows, agreeing on half of them."""
    true = ["Anger", "Disgust", "Anger", "Fear"]
    first, second = two_models()
    wide = np.zeros((4, 7))
    wide[:, [EMOTIONS.index(name) for name in THREE]] = first
    other = np.zeros((4, 7))
    other[:, [EMOTIONS.index(name) for name in THREE]] = second
    return true, {
        "A": {"what": "translate", "probabilities": wide, "temperature": 1.2},
        "B": {"what": "native", "probabilities": other, "caveat": "only a test"},
    }


def test_the_record_scores_every_approach_on_the_same_rows() -> None:
    true, approaches = four_rows()
    record = compare.build_record(true, approaches)
    assert record["held_out_rows"] == 4
    assert set(record["approaches"]) == {"A", "B"}
    assert all(block["samples"] == 4 for block in record["approaches"].values())


def test_the_reasons_to_discount_a_number_travel_with_it() -> None:
    """A caveat in prose gets read second; one in the block cannot be separated."""
    true, approaches = four_rows()
    record = compare.build_record(true, approaches)
    assert record["approaches"]["B"]["caveat"] == "only a test"
    assert record["approaches"]["A"]["temperature"] == 1.2


def test_the_probabilities_themselves_are_not_committed() -> None:
    true, approaches = four_rows()
    record = compare.build_record(true, approaches)
    assert "probabilities" not in record["approaches"]["A"]


def test_the_three_combinations_land_only_when_a_pair_is_named() -> None:
    true, approaches = four_rows()
    assert "combinations" not in compare.build_record(true, approaches)
    record = compare.build_record(true, approaches, pair=("A", "B"))
    assert set(record["combinations"]) == {"soft vote", "confidence pick", "agreement filter"}
    assert record["pair"] == ["A", "B"]


def test_only_the_agreement_filter_reports_coverage() -> None:
    true, approaches = four_rows()
    combinations = compare.build_record(true, approaches, pair=("A", "B"))["combinations"]
    assert combinations["agreement filter"]["coverage"] == 0.5
    assert "coverage" not in combinations["soft vote"]
    assert combinations["soft vote"]["samples"] == 4


def written(tmp_path: Path, record: dict[str, object]) -> compare.Comparison:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return compare.Comparison.load(path)


def test_a_record_built_here_passes_its_own_checks(tmp_path: Path) -> None:
    true, approaches = four_rows()
    report = written(tmp_path, compare.build_record(true, approaches, pair=("A", "B")))
    assert compare.check_consistency(report) == []
    assert report.samples == 4


def test_the_best_row_is_the_one_with_the_highest_accuracy(tmp_path: Path) -> None:
    true, approaches = four_rows()
    report = written(tmp_path, compare.build_record(true, approaches, pair=("A", "B")))
    name, accuracy = report.best()
    everything = {**report.approaches, **report.combinations}
    assert accuracy == max(float(block["accuracy"]) for block in everything.values())
    assert name in everything


def test_describe_names_every_approach_and_its_caveat(tmp_path: Path) -> None:
    true, approaches = four_rows()
    report = written(tmp_path, compare.build_record(true, approaches, pair=("A", "B")))
    lines = list(compare.describe(report))
    assert any("4 held-out Russian rows" in line for line in lines)
    assert any("agreement filter" in line for line in lines)
    assert any("only a test" in line for line in lines)


def test_an_accuracy_that_is_not_a_share_is_caught(tmp_path: Path) -> None:
    true, approaches = four_rows()
    record = compare.build_record(true, approaches)
    record["approaches"]["A"]["accuracy"] = 1.5
    problems = compare.check_consistency(written(tmp_path, record))
    assert any("not a share" in problem for problem in problems)


def test_supports_that_do_not_sum_to_the_header_are_caught(tmp_path: Path) -> None:
    true, approaches = four_rows()
    record = compare.build_record(true, approaches)
    record["approaches"]["A"]["classes"]["Anger"]["support"] = 99
    problems = compare.check_consistency(written(tmp_path, record))
    assert any("supports sum to" in problem for problem in problems)


def test_an_f1_that_is_not_the_harmonic_mean_is_caught(tmp_path: Path) -> None:
    true, approaches = four_rows()
    record = compare.build_record(true, approaches)
    record["approaches"]["A"]["classes"]["Anger"]["f1"] = 0.999
    problems = compare.check_consistency(written(tmp_path, record))
    assert any("harmonic mean" in problem for problem in problems)


def test_an_approach_scored_on_the_wrong_rows_is_caught(tmp_path: Path) -> None:
    true, approaches = four_rows()
    record = compare.build_record(true, approaches)
    record["held_out_rows"] = 9
    problems = compare.check_consistency(written(tmp_path, record))
    assert any("the set holds 9" in problem for problem in problems)


def test_a_full_coverage_rule_that_answers_fewer_rows_is_caught(tmp_path: Path) -> None:
    """A rule with no coverage figure has to have answered everything."""
    true, approaches = four_rows()
    record = compare.build_record(true, approaches, pair=("A", "B"))
    record["combinations"]["soft vote"]["samples"] = 2
    problems = compare.check_consistency(written(tmp_path, record))
    assert any("reports no coverage" in problem for problem in problems)


def test_a_coverage_that_is_not_a_share_is_caught(tmp_path: Path) -> None:
    true, approaches = four_rows()
    record = compare.build_record(true, approaches, pair=("A", "B"))
    record["combinations"]["agreement filter"]["coverage"] = 4.0
    problems = compare.check_consistency(written(tmp_path, record))
    assert any("coverage 4.0 is not a share" in problem for problem in problems)


# --- the committed comparison -------------------------------------------------


def committed() -> compare.Comparison:
    return compare.Comparison.load()


def test_the_committed_comparison_is_consistent() -> None:
    assert compare.check_consistency(committed()) == []


def test_every_approach_was_scored_on_the_same_rows() -> None:
    report = committed()
    assert report.samples == 3_715
    assert all(block["samples"] == 3_715 for block in report.approaches.values())


def test_on_this_corpus_classifying_russian_directly_beats_translating_it() -> None:
    """Measured -- and the name says "on this corpus" because that is the limit.

    ru-izard is DeepL-translated English, so A translates twice to be scored here
    and B trains on its own test distribution. The margin is real and it is not a
    recommendation; see the chapter.
    """
    approaches = committed().approaches
    native = float(approaches["B native ruBERT"]["accuracy"])
    translated = float(approaches["A translate, then ours"]["accuracy"])
    assert native == pytest.approx(0.4816, abs=5e-4)
    assert translated == pytest.approx(0.3728, abs=5e-4)
    assert native > translated + 0.1


def test_a_far_better_translator_does_not_close_the_gap() -> None:
    """The control for "a better translator would have won". It would not have."""
    approaches = committed().approaches
    opus = float(approaches["A translate, then ours"]["accuracy"])
    nllb = float(approaches["A-NLLB translate, then ours"]["accuracy"])
    assert nllb == pytest.approx(0.3612, abs=5e-4)
    # Six times the parameters, and on this set it does not even match opus-mt.
    assert nllb < opus
    assert nllb < float(approaches["B native ruBERT"]["accuracy"]) - 0.1


def test_our_native_model_beats_the_one_the_pipeline_shipped() -> None:
    """And beats it despite the incumbent having trained on this very corpus."""
    approaches = committed().approaches
    assert float(approaches["B native ruBERT"]["accuracy"]) > float(
        approaches["D the one the pipeline shipped"]["accuracy"]
    )


def test_combining_them_does_not_raise_accuracy_at_full_coverage() -> None:
    """The negative result, pinned so nobody quietly claims the opposite.

    A soft vote and a confidence pick both land *below* the stronger model alone:
    averaging a 0.36 model into a 0.48 one drags it down. The ensemble intuition
    does not survive two models this unequal.
    """
    report = committed()
    best_alone = float(report.approaches["B native ruBERT"]["accuracy"])
    for rule in ("soft vote", "confidence pick"):
        assert float(report.combinations[rule]["accuracy"]) < best_alone


def test_the_agreement_filter_buys_accuracy_and_pays_in_coverage() -> None:
    """Where they agree the answer is much better -- on 44% of the rows."""
    agreement = committed().combinations["agreement filter"]
    assert float(agreement["accuracy"]) == pytest.approx(0.5718, abs=5e-4)
    assert float(agreement["coverage"]) == pytest.approx(0.446, abs=5e-3)
    assert float(agreement["accuracy"]) > 0.48


def test_no_accuracy_is_published_without_the_coverage_it_was_measured_over() -> None:
    """The 240-row word error rate, prevented rather than repeated."""
    for name, block in committed().combinations.items():
        assert "coverage" in block or block["samples"] == 3_715, name


def test_the_incumbent_carries_its_caveat_into_the_record() -> None:
    caveat = committed().approaches["D the one the pipeline shipped"]["caveat"]
    assert "trained on the corpus" in caveat


def test_the_russian_numbers_are_far_below_the_english_one() -> None:
    """0.48 against 0.9164, which nobody should read as a translation problem alone."""
    assert float(committed().approaches["B native ruBERT"]["accuracy"]) < 0.6


def chapter() -> str:
    return (Path(__file__).resolve().parents[1] / "docs" / "russian.md").read_text(encoding="utf-8")


def test_the_chapter_quotes_every_score_in_the_record() -> None:
    text = chapter()
    report = committed()
    for name, block in {**report.approaches, **report.combinations}.items():
        assert f"{float(block['accuracy']):.4f}" in text, name
        assert f"{float(block['macro_f1']):.4f}" in text, name


def test_the_chapter_says_the_ensemble_did_not_help() -> None:
    text = chapter()
    assert "does not raise accuracy" in text
    assert "44.6%" in text


def test_the_chapter_ends_by_saying_what_it_does_not_establish() -> None:
    assert "## What this does not establish" in chapter()


def test_the_chapter_says_what_the_corpus_is_and_what_that_costs_the_comparison() -> None:
    """The audit's finding. Without this sentence the table reads as a recommendation."""
    text = chapter()
    assert "DeepL-translated GoEmotions" in text
    assert "translates twice" in text
    assert "is **not supported**" in text


def test_the_chapter_owns_the_bug_rather_than_quietly_fixing_it() -> None:
    """0.3631 was partly a broken harness. The correction is published, not buried."""
    text = chapter()
    assert "0.3631" in text and "0.3728" in text
    assert "88.1%" in text
