"""Valence and arousal, measured before they are shown.

The original coursework ran a second model alongside the classifier and used its
arousal output as an "intensity" score, cut into five levels at 0.2/0.4/0.6/0.8.
The model is real and published -- Mendes & Martins, ECIR 2023, XLM-RoBERTa over
34 psycho-linguistic datasets, and it reads Russian without translation. What was
missing is the only thing this project cares about: nobody ever scored it.

Scoring it takes no new data. Russell's circumplex makes two predictions about
any working valence-arousal model, and the seven-class labels this project
already has can check both:

* **valence** should put Joy above Anger, Disgust, Fear and Sadness;
* **arousal** should put Anger, Fear and Surprise above Sadness and Neutral.

Held out, that is an AUC each, and the answer is lopsided. Valence separates its
classes at **0.8223**. Arousal manages **0.5734**, where 0.5 is nothing at all --
so the dimension the original chose as its intensity measure is the weaker of the
two by a wide margin.

**And neither adds accuracy.** Stacked on top of the two classifiers and fitted
on validation, valence and arousal together move the test score by ten rows in
3,715 -- 54 right, 44 wrong, p = 0.3634. That is why this is display-only and off
by default: it tells a reader something the emotion label does not, and it does
not make the label better. Both halves of that are measured, and
:func:`check_consistency` refuses a record that reports one without the other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emotion_timeline.data.labels import EMOTIONS

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "russian"
DEFAULT_RECORD = BENCHMARK / "valence-arousal.json"

#: The published checkpoint, mirrored as a release asset. See `weights.py`.
MODEL = "va-v1"
CITATION = (
    "Mendes & Martins, Quantifying Valence and Arousal in Text with "
    "Multilingual Pre-trained Transformers, ECIR 2023 (arXiv:2302.14021)"
)

#: Russell's circumplex, as psychology has it. These are *predictions* the model
#: either satisfies or does not; nothing here is fitted to them.
HIGH_AROUSAL = ("Anger", "Fear", "Surprise")
LOW_AROUSAL = ("Sadness", "Neutral")
POSITIVE = ("Joy",)
NEGATIVE = ("Anger", "Disgust", "Fear", "Sadness")

#: The original's five intensity levels. Kept so the record can show what they do
#: to a real distribution rather than describing it.
ORIGINAL_CUTS = (0.2, 0.4, 0.6, 0.8)


def auc(scores: Sequence[float] | np.ndarray, positive: Sequence[bool] | np.ndarray) -> float:
    """Rank-based separation, so it does not assume the scale is calibrated.

    0.5 is no separation at all. Returns 0.5 when either group is empty, because
    a separation between a group and nothing is not a measurement.
    """
    from scipy.stats import rankdata

    values = np.asarray(scores, dtype=np.float64)
    flags = np.asarray(positive, dtype=bool)
    if flags.all() or not flags.any():
        return 0.5
    # Average ranks for ties. `argsort` would break them arbitrarily, which
    # silently turns "these two scored the same" into "one beat the other" --
    # and on a dimension that is mostly flat, that is most of the pairs.
    ranks = rankdata(values, method="average")
    wins, losses = float(flags.sum()), float((~flags).sum())
    return float((ranks[flags].sum() - wins * (wins + 1) / 2) / (wins * losses))


def separation(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[str],
    above: Sequence[str],
    below: Sequence[str],
) -> dict[str, Any]:
    """How well one dimension ranks one group of classes over another."""
    values = np.asarray(scores, dtype=np.float64)
    names = np.asarray(labels)
    high, low = np.isin(names, above), np.isin(names, below)
    kept = high | low
    return {
        "above": list(above),
        "below": list(below),
        "samples": int(kept.sum()),
        "mean_above": round(float(values[high].mean()), 4) if high.any() else 0.0,
        "mean_below": round(float(values[low].mean()), 4) if low.any() else 0.0,
        "gap": round(
            float(values[high].mean() - values[low].mean()) if high.any() and low.any() else 0.0, 4
        ),
        "auc": round(auc(values[kept], high[kept]), 4),
    }


def by_class(
    scores: Sequence[float] | np.ndarray, labels: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Mean score per gold class, which is what makes the table in the chapter."""
    values = np.asarray(scores, dtype=np.float64)
    names = np.asarray(labels)
    out: dict[str, dict[str, float]] = {}
    for name in EMOTIONS:
        mask = names == name
        if mask.any():
            out[name] = {
                "samples": int(mask.sum()),
                "mean": round(float(values[mask].mean()), 4),
            }
    return out


def level_counts(
    scores: Sequence[float] | np.ndarray, cuts: Sequence[float] = ORIGINAL_CUTS
) -> list[int]:
    """How many rows land in each of the original's five intensity levels."""
    return [int(n) for n in np.histogram(np.asarray(scores), bins=[0.0, *cuts, 1.0])[0]]


def mcnemar(truth: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Paired comparison of two sets of predictions over the same rows.

    Paired, because an unpaired standard error treats two models scored on
    identical rows as independent samples and overstates the uncertainty. What
    matters is only the rows where they differ.
    """
    from scipy.stats import binomtest

    gained = int(((second == truth) & (first != truth)).sum())
    lost = int(((first == truth) & (second != truth)).sum())
    p = float(binomtest(gained, gained + lost, 0.5).pvalue) if gained + lost else 1.0
    return {
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "delta": round(float((second == truth).mean() - (first == truth).mean()), 4),
        "p_value": round(p, 4),
        "significant": bool(p < 0.05),
    }


def balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Mean per-class recall, which ignores how common each class is.

    The number that catches a model winning by guessing the majority class more
    often. Plain accuracy cannot tell that apart from getting better.
    """
    recalls = []
    for index in range(len(EMOTIONS)):
        mask = truth == index
        if mask.any():
            recalls.append(float((predicted[mask] == index).mean()))
    return round(float(np.mean(recalls)), 4) if recalls else 0.0


def share_predicted(predicted: np.ndarray, label: str) -> float:
    """How often a set of predictions reaches for one class."""
    return round(float((predicted == EMOTIONS.index(label)).mean()), 4)


def build_record(
    valence: Sequence[float] | np.ndarray,
    arousal: Sequence[float] | np.ndarray,
    labels: Sequence[str],
    contribution: dict[str, Any],
    model: str = MODEL,
    alternatives: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The committed record: what each dimension separates, and what it buys.

    ``alternatives`` carries the things that were tried and did not work -- the
    tie-break rules, and the stacker whose gain turned out to be class prior
    rather than skill. They are in the record because the chapter quotes them,
    and this project does not quote a number nothing recomputes.
    """
    return {
        "_comment": [
            "Valence and arousal from a published multilingual regressor, scored",
            "against this project's own seven-class labels. Russell's circumplex",
            "predicts valence ranks Joy above the negative classes and arousal",
            "ranks Anger/Fear/Surprise above Sadness/Neutral; both are checked",
            "here. `contribution` is the other half: what the two dimensions add",
            "to the classifiers, measured by a stacker fitted on validation and",
            "scored on test; `alternatives` holds what else was tried and failed.",
            "Written by `emotion-timeline valence --rescore` (needs --extra model).",
        ],
        "model": model,
        "citation": CITATION,
        "samples": len(labels),
        "valence": {
            "by_class": by_class(valence, labels),
            "separation": separation(valence, labels, POSITIVE, NEGATIVE),
        },
        "arousal": {
            "by_class": by_class(arousal, labels),
            "separation": separation(arousal, labels, HIGH_AROUSAL, LOW_AROUSAL),
            "original_levels": {
                "cuts": list(ORIGINAL_CUTS),
                "counts": level_counts(arousal),
            },
        },
        "contribution": contribution,
        **({"alternatives": alternatives} if alternatives else {}),
    }


def predict(  # pragma: no cover - needs the checkpoint and a GPU
    texts: Sequence[str],
    model_dir: str | Path,
    batch_size: int = 32,
    max_length: int = 128,
    progress: object = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Valence and arousal for each row, in order, both on 0..1.

    The checkpoint declares `XLMRobertaForSequenceClassificationSig` -- the stock
    model with a sigmoid in `forward`. Loaded as the stock class it returns raw
    logits, so the sigmoid is applied here. Getting that wrong does not raise; it
    silently returns numbers on the wrong scale, which is why it is stated rather
    than left to the reader.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out.append(torch.sigmoid(model(**encoded).logits).float().cpu().numpy())
            if callable(progress) and start % (batch_size * 20) == 0:
                progress(
                    f"  valence/arousal {min(start + batch_size, len(texts)):,}/{len(texts):,}"
                )
    scored = np.concatenate(out)
    return scored[:, 0], scored[:, 1]


@dataclass(frozen=True, slots=True)
class ValenceReport:
    """The committed measurement of the two dimensions."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RECORD) -> ValenceReport:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def samples(self) -> int:
        return int(self.raw["samples"])

    def auc_of(self, dimension: str) -> float:
        return float(self.raw[dimension]["separation"]["auc"])

    @property
    def contribution(self) -> dict[str, Any]:
        return dict(self.raw["contribution"])

    def helps(self) -> bool:
        """Whether the dimensions measurably improve the classifier. They do not."""
        return bool(self.contribution["significant"])


def check_consistency(report: ValenceReport) -> list[str]:
    """Every relation the record has to satisfy before anything is drawn from it."""
    problems: list[str] = []

    for dimension in ("valence", "arousal"):
        block = report.raw[dimension]
        found = sum(entry["samples"] for entry in block["by_class"].values())
        if found != report.samples:
            problems.append(
                f"{dimension}: classes cover {found} rows, the header says {report.samples}"
            )
        auc_value = float(block["separation"]["auc"])
        if not 0.0 <= auc_value <= 1.0:
            problems.append(f"{dimension}: AUC {auc_value} is not in range")
        for key in ("mean_above", "mean_below"):
            value = float(block["separation"][key])
            if not 0.0 <= value <= 1.0:
                problems.append(f"{dimension}: {key} {value} is not a sigmoid output")
        gap = float(block["separation"]["gap"])
        implied = float(block["separation"]["mean_above"]) - float(
            block["separation"]["mean_below"]
        )
        if abs(gap - implied) > 5e-4:
            problems.append(f"{dimension}: the gap is not the difference of its own two means")

    counts = report.raw["arousal"]["original_levels"]["counts"]
    if sum(counts) != report.samples:
        problems.append(f"the five intensity levels hold {sum(counts)} rows, not {report.samples}")

    contribution = report.contribution
    # The record exists to say both things. One without the other is the mistake
    # this whole project is about.
    if "significant" not in contribution or "p_value" not in contribution:
        problems.append(
            "the contribution block has to say whether it is significant, and at what p"
        )
    elif bool(contribution["significant"]) != (float(contribution["p_value"]) < 0.05):
        problems.append("the significance flag disagrees with its own p value")
    if {"with", "without", "delta"} <= set(contribution):
        implied = float(contribution["with"]) - float(contribution["without"])
        if abs(implied - float(contribution["delta"])) > 1e-3:
            problems.append("the contribution's delta is not the difference of its own two scores")
    if {"gained", "lost", "net"} <= set(contribution) and int(contribution["net"]) != int(
        contribution["gained"]
    ) - int(contribution["lost"]):
        problems.append("the contribution's net is not gained minus lost")

    return problems


def describe(report: ValenceReport) -> Iterator[str]:
    """The measurement, in the terms the chapter uses."""
    yield f"{report.samples:,} held-out Russian rows, scored by {report.raw['model']}"
    yield f"  {report.raw['citation']}"
    yield ""
    yield f"  {'class':<9} {'n':>6} {'valence':>9} {'arousal':>9}"
    for name in EMOTIONS:
        valence = report.raw["valence"]["by_class"].get(name)
        arousal = report.raw["arousal"]["by_class"].get(name)
        if valence and arousal:
            yield (
                f"  {name:<9} {valence['samples']:>6} "
                f"{valence['mean']:>9.4f} {arousal['mean']:>9.4f}"
            )
    yield ""
    for dimension in ("valence", "arousal"):
        block = report.raw[dimension]["separation"]
        verdict = "separates them" if float(block["auc"]) >= 0.7 else "barely separates them"
        yield (
            f"  {dimension:<8} {'/'.join(block['above']):<22} over "
            f"{'/'.join(block['below']):<24} AUC {float(block['auc']):.4f}  {verdict}"
        )

    counts = report.raw["arousal"]["original_levels"]["counts"]
    edges = [0.0, *report.raw["arousal"]["original_levels"]["cuts"], 1.0]
    yield ""
    yield "  the original's five intensity levels, on this data:"
    for position, count in enumerate(counts):
        share = count / report.samples if report.samples else 0.0
        yield (
            f"    level {position + 1}  arousal {edges[position]:.1f}-{edges[position + 1]:.1f}  "
            f"{count:>6,} {share:>6.1%}  {'#' * round(share * 40)}"
        )

    contribution = report.contribution
    yield ""
    yield (
        f"  what it adds to the classifiers: {contribution['delta']:+.4f} "
        f"({contribution['gained']} rows right, {contribution['lost']} wrong, "
        f"p = {float(contribution['p_value']):.4f})"
    )
    yield f"  {contribution['verdict']}"

    alternatives = report.raw.get("alternatives")
    if alternatives:
        yield ""
        yield "  things tried on the way, and what they did:"
        for name, block in alternatives.items():
            yield f"    {name:<34} {block['note']}"
