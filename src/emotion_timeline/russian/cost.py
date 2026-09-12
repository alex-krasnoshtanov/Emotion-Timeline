"""What does translation cost, with domain and labels held constant?

Approach A scores 0.3755 on ru-izard where the same model scores 0.9164 on its
own English test set. Those are different corpora, so the gap between them is
translation *plus* domain *plus* one annotator's conventions against another's,
and nothing in that comparison separates the three. Published side by side they
invite exactly the wrong reading -- that translating Russian costs 55 points.

This separates them, and needs no Russian ground truth at all. Take the model's
**own English held-out rows**, push them through English → Russian → English with
the same machinery approach A uses, and score them again. Domain, labels,
annotator and model are all held fixed, so whatever moves is the translation.

Two things make it worth committing rather than running once:

**A round trip is an upper bound, deliberately.** It is two translation passes
where approach A makes one, so the true one-way cost is smaller. That is the safe
direction for the error to run: it cannot make translation look better than it is.

**A second engine is the control.** "A better translator would fix it" is the
obvious objection, and the only way to answer it is to run a better translator.
`facebook/nllb-200-distilled-600M` is roughly six times the size of the opus-mt
pair and trained on 200 languages. If the loss is the translator's quality, it
closes most of the gap. If it recovers a couple of points, the loss is the
*paraphrase* -- and no translator fixes that, which is a much more useful thing to
know than a ranking.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "russian"
DEFAULT_COST = BENCHMARK / "translation-cost.json"


#: The surface markers `error-analysis.md` found the model leans on. Translation
#: rewrites punctuation and capitalisation freely, so whether they survive is the
#: first hypothesis anyone would reach for -- which is why it is measured here
#: rather than asserted in the prose.
def marker_shares(texts: Sequence[str], markers: Sequence[str]) -> dict[str, float]:
    """The share of rows carrying each marker, for one version of the text."""
    if not texts:
        return {marker: 0.0 for marker in markers}
    return {
        marker: round(sum(1 for text in texts if marker in text) / len(texts), 4)
        for marker in markers
    }


def build_record(
    baseline: dict[str, Any],
    engines: dict[str, dict[str, Any]],
    *,
    rows: int,
    sampled_from: int,
    seed: int,
) -> dict[str, Any]:
    """The committed record: the rows as they were, and after each round trip."""
    return {
        "_comment": [
            "What translation costs this classifier, measured on its own English",
            "held-out rows so that domain, labels and annotator are held constant.",
            "Each engine round-trips the rows English -> Russian -> English; the",
            "drop from the baseline is the translation and nothing else. A round",
            "trip is two passes where the pipeline makes one, so every cost here",
            "is an upper bound. Written by `emotion-timeline translation-cost`.",
        ],
        "rows": rows,
        "sampled_from": sampled_from,
        "seed": seed,
        "baseline": baseline,
        "engines": engines,
    }


@dataclass(frozen=True, slots=True)
class TranslationCost:
    """The committed round-trip record."""

    raw: dict[str, Any]
    source: Path
    digest: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_COST) -> TranslationCost:
        source = Path(path)
        payload = source.read_bytes()
        return cls(
            raw=json.loads(payload.decode("utf-8")),
            source=source,
            digest=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def rows(self) -> int:
        return int(self.raw["rows"])

    @property
    def baseline(self) -> dict[str, Any]:
        return dict(self.raw["baseline"])

    @property
    def engines(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw["engines"])

    def cost_of(self, engine: str) -> float:
        """How much accuracy this engine's round trip destroyed."""
        return round(float(self.baseline["accuracy"]) - float(self.engines[engine]["accuracy"]), 4)

    def best_engine(self) -> str:
        """The engine that lost least, which is the one the argument turns on."""
        return max(self.engines, key=lambda name: float(self.engines[name]["accuracy"]))

    def recovered(self) -> float:
        """What the better engine buys back, against what the worse one lost.

        The number the chapter rests on: if a far larger translator recovers a
        small fraction of the loss, the loss is not the translator's quality.
        """
        costs = [self.cost_of(name) for name in self.engines]
        return round(max(costs) - min(costs), 4)


def check_consistency(report: TranslationCost) -> list[str]:
    """Every relation the record has to satisfy before it is quoted."""
    problems: list[str] = []
    blocks = {"baseline": report.baseline, **report.engines}

    for name, block in blocks.items():
        for key in ("accuracy", "macro_f1"):
            value = float(block[key])
            if not 0.0 <= value <= 1.0:
                problems.append(f"{name}: {key} {value} is not a share")
        for marker, share in dict(block.get("markers", {})).items():
            if not 0.0 <= float(share) <= 1.0:
                problems.append(f"{name}/{marker}: share {share} is not a share")

    if report.rows > int(report.raw["sampled_from"]):
        problems.append(
            f"{report.rows} rows sampled from {report.raw['sampled_from']}, which is fewer"
        )
    if not report.engines:
        problems.append("the record carries no engine to compare the baseline against")

    for name in report.engines:
        if report.cost_of(name) < 0:
            problems.append(f"{name} scores above the untranslated rows, which needs explaining")

    return problems


def describe(report: TranslationCost) -> Iterator[str]:
    """The round trip, in the terms the chapter uses."""
    baseline = report.baseline
    yield (
        f"{report.rows:,} of the model's own English held-out rows, sampled from "
        f"{int(report.raw['sampled_from']):,}"
    )
    yield ""
    yield f"  {'':<34} {'accuracy':>9} {'macro F1':>9} {'cost':>9}"
    yield (
        f"  {'the English rows themselves':<34} {float(baseline['accuracy']):>9.4f} "
        f"{float(baseline['macro_f1']):>9.4f} {'':>9}"
    )
    for name, block in report.engines.items():
        yield (
            f"  {name:<34} {float(block['accuracy']):>9.4f} "
            f"{float(block['macro_f1']):>9.4f} {report.cost_of(name):>9.4f}"
        )
    yield ""
    best = report.best_engine()
    yield (
        f"  the strongest engine here, {best}, recovers {report.recovered():.4f} of the "
        f"{max(report.cost_of(n) for n in report.engines):.4f} the weakest loses"
    )
    yield "  so the loss is the paraphrase, not the translator"
