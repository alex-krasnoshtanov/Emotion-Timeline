"""Turning kept predictions into the record the error-analysis chapter already reads.

The output deliberately matches ``benchmarks/error-analysis/held-out-64250.json``
field for field, because that means ``analysis/error_analysis.py`` loads it, checks
it and draws all four of its figures with no new code -- the loader already takes
a path and the command already takes ``--report``.

The difference is where the numbers come from. That file is a summary somebody
wrote down, and `docs/error-analysis.md` says so under Provenance: the per-sample
predictions behind it were not kept, so nothing in it can be re-derived. These are
computed from predictions that do still exist, which is the one thing a retrain
can fix about that chapter and the reason the logits are saved at all.

Surface markers are counted on the **cleaned** text, which is what the model was
trained and scored on. That is worth stating because it does not reproduce the
inherited figure: `[CAPS]` survives cleaning in 1.19% of the corpus, so a 15%
held-out slice holds about 750 of them, against the 4,040 of 64,250 -- 6.29% --
the inherited analysis reports. A factor of five apart is not a rounding
difference, and the likeliest explanation is that the original counted capitals on
the raw text before `mark_shouting` folded them away.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

#: Markers as they survive `data/clean.py`, which is the text the model sees.
#: `mark_shouting` replaces a shouted word with a `[CAPS]` marker and lowercases
#: it, so searching the cleaned text for capitals finds nothing at all.
MARKERS = {
    "ALL-CAPS word": "[CAPS]",
    "Exclamation mark": "!",
    "Question mark": "?",
    "Ellipsis": "...",
}

#: The card advises trusting a prediction at or above this. An error made at this
#: confidence is one a threshold will never catch, which is why they are counted.
CONFIDENT = 0.7

WORD = re.compile(r"[a-z']{3,}")


def class_breakdown(
    true: Sequence[str],
    predicted: Sequence[str],
    classes: Sequence[str],
    top_confusions: int = 3,
) -> dict[str, dict[str, Any]]:
    """Per class: how many, how many wrong, and what they were mistaken for."""
    out: dict[str, dict[str, Any]] = {}
    for name in classes:
        rows = [guess for actual, guess in zip(true, predicted, strict=True) if actual == name]
        errors = Counter(guess for guess in rows if guess != name)
        out[name] = {
            "samples": len(rows),
            "errors": sum(errors.values()),
            "error_rate": round(sum(errors.values()) / len(rows), 4) if rows else 0.0,
            "confused_with": {label: count for label, count in errors.most_common(top_confusions)},
        }
    return out


def length_statistics(texts: Sequence[str], correct: Sequence[bool]) -> dict[str, Any]:
    """Whether right answers are longer than wrong ones, and whether it is real.

    Mann-Whitney rather than a t-test: text lengths are not normal, and the
    inherited chapter reported the same statistic, so the two stay comparable.
    """
    from scipy.stats import mannwhitneyu

    hits = np.asarray(correct, dtype=bool)
    characters = np.array([len(text) for text in texts], dtype=np.float64)
    words = np.array([len(text.split()) for text in texts], dtype=np.float64)

    right, wrong = characters[hits], characters[~hits]
    if not len(right) or not len(wrong):
        statistic, probability = 0.0, 1.0
    else:
        result = mannwhitneyu(right, wrong, alternative="two-sided")
        statistic, probability = float(result.statistic), float(result.pvalue)

    return {
        "correct": {
            "mean_characters": round(float(right.mean()), 2) if len(right) else 0.0,
            "mean_words": round(float(words[hits].mean()), 2) if len(right) else 0.0,
        },
        "incorrect": {
            "mean_characters": round(float(wrong.mean()), 2) if len(wrong) else 0.0,
            "mean_words": round(float(words[~hits].mean()), 2) if len(wrong) else 0.0,
        },
        "mann_whitney_u": statistic,
        "p_value": probability,
    }


def textual_features(texts: Sequence[str], correct: Sequence[bool]) -> dict[str, Any]:
    """The error rate with each surface marker present, and without it."""
    hits = np.asarray(correct, dtype=bool)
    out: dict[str, Any] = {
        "_comment": (
            "Surface markers, and the error rate with and without each. Counted on "
            "the cleaned text the model was trained on, where a shouted word is a "
            "[CAPS] marker rather than capitals."
        )
    }
    for name, marker in MARKERS.items():
        present = np.array([marker in text for text in texts], dtype=bool)
        absent = ~present
        out[name] = {
            "present_samples": int(present.sum()),
            "error_rate_present": (
                round(float((~hits[present]).mean()), 4) if present.any() else 0.0
            ),
            "error_rate_absent": round(float((~hits[absent]).mean()), 4) if absent.any() else 0.0,
        }
    return out


def confidence_block(confidence: Sequence[float], correct: Sequence[bool]) -> dict[str, Any]:
    """The gap the card measures, and the errors a threshold cannot catch."""
    scores = np.asarray(confidence, dtype=np.float64)
    hits = np.asarray(correct, dtype=bool)
    errors = int((~hits).sum())
    confident_errors = int(((~hits) & (scores >= CONFIDENT)).sum())
    return {
        "mean_when_correct": round(float(scores[hits].mean()), 4) if hits.any() else 0.0,
        "mean_when_incorrect": round(float(scores[~hits].mean()), 4) if errors else 0.0,
        "high_confidence_errors": confident_errors,
        "high_confidence_error_share_of_errors": (
            round(confident_errors / errors, 4) if errors else 0.0
        ),
        "high_confidence_threshold": CONFIDENT,
    }


def vocabulary_bias(
    texts: Sequence[str],
    correct: Sequence[bool],
    top: int = 5,
    minimum: int = 40,
) -> dict[str, Any]:
    """Words that show up disproportionately in wrong answers, and in right ones.

    A floor on how often a word has to appear, because without one the list is
    whatever rare token happened to land on the wrong side once.
    """
    right: Counter[str] = Counter()
    wrong: Counter[str] = Counter()
    for text, hit in zip(texts, correct, strict=True):
        (right if hit else wrong).update(set(WORD.findall(text.lower())))

    rates: dict[str, tuple[float, int]] = {}
    for word in set(right) | set(wrong):
        total = right[word] + wrong[word]
        if total >= minimum:
            rates[word] = (wrong[word] / total, total)

    # Ties are broken by how often the word appears, not alphabetically. Plenty of
    # words sit at exactly zero, and sorting those by name lists whatever happens
    # to come last in the alphabet rather than whatever carries the most evidence.
    worst = sorted(rates, key=lambda word: (-rates[word][0], -rates[word][1], word))
    best = sorted(rates, key=lambda word: (rates[word][0], -rates[word][1], word))
    return {
        "_comment": (
            f"Words appearing at least {minimum} times, ranked by the share of their "
            "occurrences that sit in a wrong prediction, ties broken by how often "
            "the word appears."
        ),
        "error_biased": worst[:top],
        "correct_biased": best[:top],
    }


def confusion_matrix(
    true: Sequence[str],
    predicted: Sequence[str],
    classes: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Rows are what a sample is, columns are what it was called, as plain names."""
    from emotion_timeline.training.evaluate import confusion

    counts = confusion(true, predicted, classes)
    return {
        name: dict(zip(classes, (int(value) for value in row), strict=True))
        for name, row in zip(classes, counts, strict=True)
    }


def held_out_summary(
    texts: Sequence[str],
    true: Sequence[str],
    predicted: Sequence[str],
    confidence: Sequence[float],
    classes: Sequence[str],
    split: str = "held-out test",
    comment: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The whole record, in the shape `analysis/error_analysis.py` already reads."""
    correct = [actual == guess for actual, guess in zip(true, predicted, strict=True)]
    errors = sum(1 for hit in correct if not hit)
    total = len(correct)
    return {
        "_comment": list(comment)
        if comment
        else [
            "Error analysis of the fine-tune, recomputed from the predictions it",
            "kept rather than from a summary somebody wrote down. Every figure",
            "here derives from the per-sample logits shipped with the weights,",
            "so it can be re-derived rather than only checked for consistency.",
        ],
        "split": split,
        "total_samples": total,
        "total_errors": errors,
        "accuracy": round((total - errors) / total, 4) if total else 0.0,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "classes": class_breakdown(true, predicted, classes),
        # The whole 7x7, not just the worst three confusions per class: 49 integers
        # is nothing to commit, and without the full matrix per-class precision --
        # and so F1, and so any comparison against the model card's table -- cannot
        # be recomputed from this file at all.
        "confusion": confusion_matrix(true, predicted, classes),
        "length": length_statistics(texts, correct),
        "textual_features": textual_features(texts, correct),
        "vocabulary": vocabulary_bias(texts, correct),
        "confidence": confidence_block(confidence, correct),
    }


def decode(
    logits: np.ndarray,
    classes: Sequence[str],
    temperature: float = 1.0,
) -> tuple[list[str], list[float]]:
    """The class each row was given, and how sure the model was about it."""
    from emotion_timeline.training.evaluate import softmax

    probabilities = softmax(np.asarray(logits), temperature)
    chosen = probabilities.argmax(axis=1)
    return (
        [classes[position] for position in chosen],
        [float(row[position]) for row, position in zip(probabilities, chosen, strict=True)],
    )


def calibration_report(
    validation_logits: np.ndarray,
    validation_true: np.ndarray,
    test_logits: np.ndarray,
    test_true: np.ndarray,
    bins: int = 20,
) -> dict[str, Any]:
    """What one scalar fitted on validation does to the held-out set.

    `docs/error-analysis.md` asks for calibration rather than a threshold, and the
    card's own advice to trust anything above 0.7 rests on a gap it measured in
    domain and nowhere else. Fitting on validation and reporting on test is the
    only honest order: a temperature fitted on the set it is scored over would
    flatter itself.
    """
    from emotion_timeline.training import evaluate

    temperature = evaluate.fit_temperature(validation_logits, validation_true)

    measured: dict[str, Any] = {}
    for name, value in (("before", 1.0), ("after", temperature)):
        probabilities = evaluate.softmax(test_logits, value)
        chosen = probabilities.argmax(axis=1)
        correct = [bool(a == b) for a, b in zip(chosen, test_true, strict=True)]
        confidence = [
            float(row[position]) for row, position in zip(probabilities, chosen, strict=True)
        ]
        bucket = evaluate.calibration(confidence, correct, bins)
        measured[name] = {
            "expected_calibration_error": round(evaluate.expected_calibration_error(bucket), 4),
            **{
                key: round(value_, 4)
                for key, value_ in evaluate.confidence_gap(confidence, correct).items()
            },
        }

    return {
        "_comment": (
            "One scalar fitted on the validation split by minimising its negative "
            "log likelihood, then applied to the held-out set. Accuracy cannot "
            "change -- temperature does not reorder the classes -- so what moves "
            "is only how much the stated confidence can be believed."
        ),
        "temperature": round(temperature, 4),
        "fitted_on": "validation",
        "reported_over": "held-out test",
        **measured,
    }


def subset_error_rate(
    texts: Sequence[str],
    correct: Sequence[bool],
    marker: str,
) -> dict[str, Any]:
    """The error rate on rows containing ``marker``, against the rest.

    Written for the URL bug `docs/dataset.md` defers to a retrain. Stripping `:/`
    as an emoticon before URLs are masked leaves 1,658 of 1,857 URLs in the corpus
    as a mangled fragment, and this is what that costs on the held-out set --
    measured on the few hundred rows it actually touches rather than inferred from
    a macro F1 that the difference could not move.
    """
    hits = np.asarray(correct, dtype=bool)
    inside = np.array([marker in text for text in texts], dtype=bool)
    outside = ~inside
    return {
        "marker": marker,
        "samples": int(inside.sum()),
        "error_rate": round(float((~hits[inside]).mean()), 4) if inside.any() else 0.0,
        "error_rate_elsewhere": round(float((~hits[outside]).mean()), 4) if outside.any() else 0.0,
    }
