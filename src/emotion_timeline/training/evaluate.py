"""Scoring predictions with the same arithmetic that audited the model card.

``harmonic_mean`` and :class:`~emotion_timeline.model.card.ClassScore` are
imported rather than rewritten, so the fine-tune's table and the card's table
cannot come to mean different things by F1. That matters more than the few lines
it saves: the whole point of the new chapter is a comparison against the old one,
and a comparison between two definitions is not one.

Everything here is numpy over a confusion matrix or an array of probabilities.
Nothing imports torch, so the scoring runs -- and is tested -- on a machine with
no GPU, which is also what lets CI check the published numbers.

Calibration is here because `docs/error-analysis.md` asks for it: the inherited
model's confidence separated right from wrong by 0.460 in domain and by -0.019 on
the stress set, and its advice to threshold at 0.7 only ever held for the first
of those. Temperature scaling is the cheapest honest answer, it is fitted on
validation and applied to test, and it costs one scalar.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from emotion_timeline.model.card import ClassScore, harmonic_mean


def confusion(
    true: Sequence[str],
    predicted: Sequence[str],
    classes: Sequence[str],
) -> list[list[int]]:
    """Rows are what a sample is, columns are what it was called."""
    index = {name: position for position, name in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for actual, guess in zip(true, predicted, strict=True):
        matrix[index[actual]][index[guess]] += 1
    return matrix


def class_scores(matrix: Sequence[Sequence[int]], classes: Sequence[str]) -> list[ClassScore]:
    """Precision, recall and F1 per class, straight off the matrix."""
    counts = np.asarray(matrix, dtype=np.int64)
    scores: list[ClassScore] = []
    for position, name in enumerate(classes):
        hits = int(counts[position, position])
        called = int(counts[:, position].sum())
        actual = int(counts[position, :].sum())
        precision = hits / called if called else 0.0
        recall = hits / actual if actual else 0.0
        scores.append(
            ClassScore(
                name=name,
                precision=precision,
                recall=recall,
                f1=harmonic_mean(precision, recall),
                support=actual,
            )
        )
    return scores


def accuracy_of(matrix: Sequence[Sequence[int]]) -> float:
    """The trace over the total: the share of samples put in the right row."""
    counts = np.asarray(matrix, dtype=np.int64)
    total = int(counts.sum())
    return float(np.trace(counts)) / total if total else 0.0


def macro(scores: Sequence[ClassScore], field: str) -> float:
    """The plain mean of a column, every class weighted the same."""
    return float(np.mean([getattr(score, field) for score in scores])) if scores else 0.0


def weighted(scores: Sequence[ClassScore], field: str) -> float:
    """The support-weighted mean, which for recall is the accuracy."""
    total = sum(score.support for score in scores)
    if not total:
        return 0.0
    total_score = sum(float(getattr(score, field)) * score.support for score in scores)
    return total_score / total


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Probabilities from logits, cooled or warmed by ``temperature``."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def negative_log_likelihood(
    logits: np.ndarray, true_index: np.ndarray, temperature: float
) -> float:
    """Mean NLL of the true classes, which is what temperature scaling minimises."""
    probabilities = softmax(logits, temperature)
    chosen = probabilities[np.arange(len(true_index)), true_index]
    return float(-np.log(np.clip(chosen, 1e-12, None)).mean())


def fit_temperature(
    logits: np.ndarray,
    true_index: np.ndarray,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """The single scalar that best calibrates these logits.

    Fitted on validation and applied to test, never fitted on the set it is
    reported over. One dimension, so a bounded scalar search is enough and scipy
    is already a core dependency.
    """
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda t: negative_log_likelihood(logits, true_index, t),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def calibration(
    confidence: Sequence[float],
    correct: Sequence[bool],
    bins: int = 20,
) -> list[dict[str, float]]:
    """Confidence against accuracy, in equal-width bins.

    A perfectly calibrated model answers correctly 70% of the time when it says
    0.7. Each bin reports both, so the gap is visible rather than summarised away.
    """
    scores = np.asarray(confidence, dtype=np.float64)
    hits = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict[str, float]] = []
    for lower, upper in pairwise(edges):
        inside = (scores > lower) & (scores <= upper) if lower > 0 else (scores <= upper)
        count = int(inside.sum())
        out.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "samples": count,
                "confidence": float(scores[inside].mean()) if count else 0.0,
                "accuracy": float(hits[inside].mean()) if count else 0.0,
            }
        )
    return out


def expected_calibration_error(bins: Sequence[dict[str, float]]) -> float:
    """How far stated confidence sits from measured accuracy, weighted by bin size."""
    total = sum(int(section["samples"]) for section in bins)
    if not total:
        return 0.0
    weighted_error = sum(
        int(section["samples"]) * abs(section["confidence"] - section["accuracy"])
        for section in bins
    )
    return weighted_error / total


def confidence_gap(confidence: Sequence[float], correct: Sequence[bool]) -> dict[str, float]:
    """Mean confidence when right against when wrong, the card's own measure.

    The card reports 0.8873 against 0.4275 in domain. Reporting the same pair for
    the new model is what makes the two comparable at all.
    """
    scores = np.asarray(confidence, dtype=np.float64)
    hits = np.asarray(correct, dtype=bool)
    right = float(scores[hits].mean()) if hits.any() else 0.0
    wrong = float(scores[~hits].mean()) if (~hits).any() else 0.0
    return {"correct": right, "incorrect": wrong, "gap": right - wrong}
