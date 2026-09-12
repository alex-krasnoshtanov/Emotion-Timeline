"""Scoring an off-the-shelf classifier on the Russian set, in our seven classes.

The models reported here answer in their own vocabulary and with their own head:
both are multi-label, so their outputs are independent sigmoids rather than a
distribution. Getting to a seven-class probability vector takes two steps and
both are choices worth stating.

**Sigmoid, then sum into the target class.** A model that reports both
``contempt`` and ``disgust`` is making one claim about Disgust twice, so the two
probabilities add rather than compete. Summing is the reduction that respects
what a multi-label head means; taking the max would throw away the second piece
of evidence.

**Then normalise.** Summed sigmoids do not total one, and everything downstream
-- the calibration, the soft vote, the confidence threshold -- assumes a
distribution. Normalising is what makes two different models comparable at all,
and it is also why an uncalibrated comparison between them would be meaningless.

Only the forward pass needs a GPU, and it is small: a few thousand rows through a
base-sized encoder. The mapping and the scoring are ordinary array work and are
tested without one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from emotion_timeline.data.labels import EMOTIONS

#: Enough for a base encoder over short social-media text; the corpus's longest
#: row is 357 characters.
MAX_LENGTH = 128


def to_seven_matrix(
    probabilities: np.ndarray,
    source_labels: Sequence[str],
    mapping: Mapping[str, str],
) -> np.ndarray:
    """Fold a model's own classes into our seven, summing then normalising.

    Columns whose label has no entry in ``mapping`` are dropped, which is how a
    class nobody can place stops contributing rather than being guessed at.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape[1] != len(source_labels):
        raise ValueError(f"{probabilities.shape[1]} columns against {len(source_labels)} labels")
    folded = np.zeros((len(probabilities), len(EMOTIONS)), dtype=np.float64)
    for column, label in enumerate(source_labels):
        target = mapping.get(label)
        if target is None:
            continue
        folded[:, EMOTIONS.index(target)] += probabilities[:, column]

    totals = folded.sum(axis=1, keepdims=True)
    # A row where every mapped class came out at zero has no opinion; leave it
    # uniform rather than dividing by zero and calling the result confident.
    safe = np.where(totals > 0, totals, 1.0)
    folded = np.where(totals > 0, folded / safe, 1.0 / len(EMOTIONS))
    return folded


def approximate_share(
    probabilities: np.ndarray,
    source_labels: Sequence[str],
    approximate: frozenset[str],
) -> float:
    """How often the model's top answer was a class we could only approximate.

    The number that says how much of a score is measuring our label map rather
    than the model.
    """
    probabilities = np.asarray(probabilities)
    if not len(probabilities):
        return 0.0
    chosen = probabilities.argmax(axis=1)
    flagged = {index for index, label in enumerate(source_labels) if label in approximate}
    return float(np.isin(chosen, list(flagged)).mean()) if flagged else 0.0


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Independent probabilities, which is what a multi-label head reports."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def softmax(logits: np.ndarray) -> np.ndarray:
    """A distribution, which is what a single-label head reports."""
    from emotion_timeline.training.evaluate import softmax as shared

    return shared(logits)


def predict(
    model_id: str,
    texts: Sequence[str],
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
    multi_label: bool = True,
) -> tuple[np.ndarray, list[str]]:  # pragma: no cover - needs the model and a GPU
    """Per-class probabilities and the model's own label order.

    ``multi_label`` decides sigmoid against softmax, and it has to match the head
    the checkpoint was trained with. The two off-the-shelf baselines are
    multi-label; our own fine-tune is not, and running softmax over a sigmoid head
    or the reverse quietly changes every number downstream.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            out.append(model(**encoded).logits.float().cpu().numpy())
    raw = np.concatenate(out)
    return (sigmoid(raw) if multi_label else softmax(raw)), labels


def score(
    probabilities: np.ndarray,
    source_labels: Sequence[str],
    mapping: Mapping[str, str],
    approximate: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Everything the comparison needs from one model's raw output."""
    folded = to_seven_matrix(probabilities, source_labels, mapping)
    chosen = folded.argmax(axis=1)
    return {
        "predicted": [EMOTIONS[index] for index in chosen],
        "confidence": [float(row[index]) for row, index in zip(folded, chosen, strict=True)],
        "probabilities": folded,
        "approximate_share": round(approximate_share(probabilities, source_labels, approximate), 4),
    }
