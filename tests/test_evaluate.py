"""Scoring: the arithmetic, and the identity that ties it to the other chapters.

No torch and no GPU. A confusion matrix is just a table, so every published
number of the fine-tune can be checked here on cases small enough to work out by
hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.training import evaluate

CLASSES = ("a", "b", "c")


def test_the_matrix_counts_what_each_sample_was_called() -> None:
    true = ["a", "a", "b", "c"]
    predicted = ["a", "b", "b", "a"]
    assert evaluate.confusion(true, predicted, CLASSES) == [
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]


def test_precision_and_recall_come_off_the_column_and_the_row() -> None:
    matrix = [[8, 2, 0], [1, 5, 0], [1, 0, 3]]
    scores = {score.name: score for score in evaluate.class_scores(matrix, CLASSES)}
    # 'a' was called 10 times and 8 were right; 'a' occurred 10 times and 8 were found.
    assert scores["a"].precision == pytest.approx(0.8)
    assert scores["a"].recall == pytest.approx(0.8)
    assert scores["a"].f1 == pytest.approx(0.8)
    assert scores["a"].support == 10
    # 'b' was called 7 times, 5 right, out of 6 that occurred.
    assert scores["b"].precision == pytest.approx(5 / 7)
    assert scores["b"].recall == pytest.approx(5 / 6)


def test_every_f1_is_the_harmonic_mean_of_its_own_row() -> None:
    """The same assertion docs/model.md makes about the card, on the new numbers."""
    matrix = [[8, 2, 0], [1, 5, 0], [1, 0, 3]]
    for score in evaluate.class_scores(matrix, CLASSES):
        assert score.f1 == pytest.approx(score.implied_f1)


def test_a_class_nobody_predicted_scores_zero_rather_than_dividing_by_it() -> None:
    matrix = [[0, 2], [0, 3]]
    scores = evaluate.class_scores(matrix, ("a", "b"))
    assert scores[0].precision == 0.0
    assert scores[0].f1 == 0.0
    assert scores[0].support == 2


def test_accuracy_is_the_trace_over_the_total() -> None:
    matrix = [[8, 2, 0], [1, 5, 0], [1, 0, 3]]
    assert evaluate.accuracy_of(matrix) == pytest.approx(16 / 20)


def test_an_empty_matrix_scores_zero_rather_than_raising() -> None:
    assert evaluate.accuracy_of([[0, 0], [0, 0]]) == 0.0
    assert evaluate.macro([], "f1") == 0.0
    assert evaluate.weighted([], "f1") == 0.0


def test_support_weighted_recall_is_the_accuracy() -> None:
    """The identity docs/model-selection.md used to work out which average was reported.

    It holds for weighted averaging and fails for macro, which is exactly what
    made it usable as evidence there. Asserting it here keeps the two chapters
    arithmetically consistent.
    """
    matrix = [[8, 2, 0], [1, 5, 0], [1, 0, 3]]
    scores = evaluate.class_scores(matrix, CLASSES)
    assert evaluate.weighted(scores, "recall") == pytest.approx(evaluate.accuracy_of(matrix))
    assert evaluate.macro(scores, "recall") != pytest.approx(evaluate.accuracy_of(matrix))


def test_macro_is_the_plain_mean_of_the_column() -> None:
    matrix = [[8, 2, 0], [1, 5, 0], [1, 0, 3]]
    scores = evaluate.class_scores(matrix, CLASSES)
    assert evaluate.macro(scores, "f1") == pytest.approx(
        float(np.mean([score.f1 for score in scores]))
    )


def test_the_seven_classes_score_as_seven() -> None:
    matrix = [[1 if row == column else 0 for column in range(7)] for row in range(7)]
    scores = evaluate.class_scores(matrix, EMOTIONS)
    assert [score.name for score in scores] == list(EMOTIONS)
    assert evaluate.macro(scores, "f1") == pytest.approx(1.0)


# --- calibration -------------------------------------------------------------


def test_softmax_gives_a_distribution() -> None:
    probabilities = evaluate.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probabilities[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_a_higher_temperature_flattens_the_distribution() -> None:
    logits = np.array([[4.0, 1.0, 0.0]])
    assert evaluate.softmax(logits, 5.0).max() < evaluate.softmax(logits, 1.0).max()


def test_temperature_scaling_recovers_a_known_overconfidence() -> None:
    """Labels drawn from the honest distribution, logits handed over three times too sharp."""
    generator = np.random.default_rng(0)
    honest = generator.normal(size=(4000, 7))
    probabilities = evaluate.softmax(honest)
    drawn = np.array(
        [generator.choice(7, p=row) for row in probabilities],
        dtype=np.int64,
    )
    fitted = evaluate.fit_temperature(honest * 3.0, drawn)
    assert fitted == pytest.approx(3.0, abs=0.25)


def test_a_calibrated_set_needs_no_temperature() -> None:
    generator = np.random.default_rng(1)
    logits = generator.normal(size=(4000, 7))
    drawn = np.array(
        [generator.choice(7, p=row) for row in evaluate.softmax(logits)],
        dtype=np.int64,
    )
    assert evaluate.fit_temperature(logits, drawn) == pytest.approx(1.0, abs=0.2)


def test_the_bins_account_for_every_sample() -> None:
    confidence = [0.05, 0.2, 0.55, 0.9, 1.0]
    correct = [False, False, True, True, True]
    bins = evaluate.calibration(confidence, correct, bins=4)
    assert sum(int(section["samples"]) for section in bins) == 5
    assert [section["lower"] for section in bins] == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_a_perfectly_calibrated_set_has_no_calibration_error() -> None:
    confidence = [1.0] * 6
    correct = [True] * 6
    assert evaluate.expected_calibration_error(evaluate.calibration(confidence, correct)) == 0.0


def test_confident_and_wrong_is_what_the_error_measures() -> None:
    bins = evaluate.calibration([1.0, 1.0], [False, False], bins=2)
    assert evaluate.expected_calibration_error(bins) == pytest.approx(1.0)


def test_no_samples_means_no_error_rather_than_a_division() -> None:
    assert evaluate.expected_calibration_error([]) == 0.0
    assert evaluate.expected_calibration_error(evaluate.calibration([], [])) == 0.0


def test_the_confidence_gap_is_the_pair_the_card_reports() -> None:
    gap = evaluate.confidence_gap([0.9, 0.8, 0.4, 0.3], [True, True, False, False])
    assert gap["correct"] == pytest.approx(0.85)
    assert gap["incorrect"] == pytest.approx(0.35)
    assert gap["gap"] == pytest.approx(0.5)


def test_a_gap_needs_both_kinds_of_answer() -> None:
    assert evaluate.confidence_gap([0.9], [True])["incorrect"] == 0.0
    assert evaluate.confidence_gap([0.2], [False])["correct"] == 0.0
