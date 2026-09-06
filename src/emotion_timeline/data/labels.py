"""Collapsing many emotion vocabularies down to seven single labels.

The source corpora between them use 35 emotion names. The client wanted seven.
Getting from one to the other is three decisions, and each of them loses
something:

1. **Priority, not frequency.** A row labelled both Anger and Disgust becomes
   Disgust, because the rarer, more specific reading is the informative one.
   Taking the commonest label instead would collapse the small classes into
   Sadness and Joy and leave the model with four usable classes.
2. **Source labels overrule the collapse.** The bundled seven-class mapping in
   the upstream dataset routes some disgust and neutral rows elsewhere, so a
   row whose *original* annotation said disgust is put back to Disgust. This
   moved 5,165 rows into Disgust that the collapse alone would have lost.
3. **Love has no seven-class home, so it is dropped.** 34,940 rows, 7.7% of the
   data at that point. See :func:`drop_love` -- the original build meant to
   relabel them and could not.

The order of 2 is load-bearing: disgust is restored first and neutral second,
so a row annotated as both ends up Neutral. That is 33 rows, and it is written
down because it is the sort of detail that silently changes a class count.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# The client's seven classes.
EMOTIONS = ("Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise")

# Ties break leftwards. Joy is deliberately absent: it is the largest class by
# a wide margin, so it wins only when a row carries nothing else, which is what
# falling through to the first remaining label achieves.
PRIORITY = ("Disgust", "Neutral", "Fear", "Sadness", "Surprise", "Anger")

# Applied in this order, so a row annotated both disgust and neutral is Neutral.
RESTORE_FROM_SOURCE = ("disgust", "neutral")

DROPPED = "Love"


def collapse(labels: Sequence[str]) -> str | None:
    """One label from many, by priority. ``None`` if there were none at all."""
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    for candidate in PRIORITY:
        if candidate in labels:
            return candidate
    return labels[0]


def restore(label: str | None, source_labels: Iterable[str]) -> str | None:
    """Put a row back to the class its original annotation named.

    The upstream seven-class mapping is not always the one this project wants:
    a row annotated ``disgust`` can arrive mapped to Anger, because disgust is
    not one of the upstream target classes. This undoes that, for the two
    classes where it matters.
    """
    lowered = {str(name).lower() for name in source_labels}
    # Later entries win, matching the original build, which applied one whole
    # pass per class in this order. A row annotated both ends up Neutral.
    for candidate in RESTORE_FROM_SOURCE:
        if candidate in lowered:
            label = candidate.capitalize()
    return label


def is_dropped(labels: Sequence[str]) -> bool:
    """Whether this row carries the label with no seven-class equivalent."""
    return DROPPED in labels
