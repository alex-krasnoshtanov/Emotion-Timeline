"""Mapping other vocabularies onto ours, and combining two models that answer in it.

None of this needs a model. Folding eleven classes into seven is array work, and
so are the three combination rules -- which is the point of keeping them separate
from the forward pass that produces the numbers they combine.
"""

from __future__ import annotations

import numpy as np
import pytest

from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.russian import baselines, compare, maps

THREE = ("Anger", "Disgust", "Fear")


# --- the label maps ----------------------------------------------------------


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
