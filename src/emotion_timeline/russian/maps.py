"""Mapping other people's emotion vocabularies onto our seven.

Every model scored against the Russian set reports its own classes, and none of
them is our seven. Mapping is therefore not a formality: it changes the score,
so the map is committed here where it can be read and argued with rather than
buried in a script.

Two of them are needed.

`tabularisai/multilingual-emotion-classification` reports **eleven**, and four
have no exact home. `contempt` goes to Disgust and `frustration` to Anger, which
are the standard neighbours. `gratitude` and `love` both go to Joy -- and `love`
is the uncomfortable one, because the English build **drops** Love outright for
having no seven-class equivalent (`dataset.md`, 34,940 rows). Dropping a training
row and dropping a prediction are different things: a prediction we refuse to map
is a row the model is scored wrong on regardless, which measures our map rather
than the model. So it is mapped, and the share of predictions that land on a
mapped-not-native label is reported beside the score.

`Djacon/rubert-tiny2-russian-emotion-detection` reports **ten**, and they are
exactly `ru-izard-emotions`'s own columns -- which is the tell that it was trained
on the corpus we are testing on. `data/ru.to_seven` already maps that vocabulary,
so it is reused rather than restated. What cannot be reused is the assumption
that its score means what the others' mean; see :data:`TRAINED_ON_THE_TEST_SET`.
"""

from __future__ import annotations

from emotion_timeline.data.labels import EMOTIONS

#: `tabularisai/multilingual-emotion-classification`, eleven classes.
MULTILINGUAL = {
    "anger": "Anger",
    "contempt": "Disgust",
    "disgust": "Disgust",
    "fear": "Fear",
    "frustration": "Anger",
    "gratitude": "Joy",
    "joy": "Joy",
    "love": "Joy",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "surprise": "Surprise",
}

#: Labels with no seven-class equivalent, mapped to a neighbour rather than
#: refused. Reported as a share of predictions so the reader can discount it.
APPROXIMATE = frozenset({"contempt", "frustration", "gratitude", "love"})

#: Models whose training data includes the corpus they are being scored on, so
#: their number is an upper bound rather than a measurement. Djacon's classifier
#: reports ru-izard's own ten columns, and ru-izard is the test set.
TRAINED_ON_THE_TEST_SET = {
    "Djacon/rubert-tiny2-russian-emotion-detection": (
        "reports ru-izard-emotions' own ten columns, so it was trained on the "
        "corpus this scores it against; its split is unknown and may overlap "
        "these rows"
    )
}


def check_map(mapping: dict[str, str]) -> list[str]:
    """Every target has to be one of ours, and nothing may be left unmapped."""
    problems: list[str] = []
    unknown = sorted(set(mapping.values()) - set(EMOTIONS))
    if unknown:
        problems.append(f"maps onto classes that are not ours: {unknown}")
    missing = sorted(APPROXIMATE - set(mapping))
    if missing:
        problems.append(f"labels called approximate but never mapped: {missing}")
    return problems
