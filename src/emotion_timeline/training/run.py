"""The fine-tune: the parts CI can check, and the loop it cannot.

The split between them is deliberate and is the rule
``.claude/skills/add-stage/SKILL.md`` sets for GPU work. Building the
configuration, choosing class weights, slicing the dataset into its three splits
and writing the record are all ordinary functions over ordinary data, so they are
tested against fakes and counted by the coverage floor like everything else. Only
the loop itself and the forward pass over the held-out set are marked
``# pragma: no cover``, because a runner has no GPU and the floor would otherwise
be met by widening ``omit`` until it stopped meaning anything.

The loop is written out rather than handed to ``transformers.Trainer``. It is
about forty lines either way, and the forty here do not change between library
versions -- which matters for a repository whose claim is that its numbers can be
reproduced years from now.

Tokenising happens per batch rather than once up front. The fast tokeniser does
the whole training split in a few seconds an epoch, and dynamic padding to the
longest sequence in each batch is faster than padding everything to 128 as well
as smaller in memory.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.training import splits

if TYPE_CHECKING:
    import pandas as pd

#: What the surviving tokenizer of the original run actually was: 30,522 tokens,
#: exactly `distilbert-base-uncased`. The card claims DeBERTa-V2 and docs/model.md
#: declines to say which record is wrong; training the one the evidence points at
#: is what makes the two tables comparable.
DEFAULT_MODEL = "distilbert-base-uncased"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything that decides what the run produces, written into the record."""

    model_id: str = DEFAULT_MODEL
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 5e-5
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06
    seed: int = splits.SEED
    #: Inverse-frequency weights in the loss. Off by default: the point of the
    #: first run is a comparison against the card, and the card's model was not
    #: weighted either.
    weighted_loss: bool = False
    classes: tuple[str, ...] = field(default_factory=lambda: tuple(EMOTIONS))

    @property
    def label_index(self) -> dict[str, int]:
        return {name: position for position, name in enumerate(self.classes)}


def build_config(**overrides: Any) -> TrainConfig:
    """A config from command-line arguments, refusing the ones that cannot work."""
    given = {key: value for key, value in overrides.items() if value is not None}
    unknown = set(given) - set(TrainConfig.__slots__)
    if unknown:
        raise ValueError(f"unknown setting {sorted(unknown)}")
    config = TrainConfig(**given)
    if config.epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {config.epochs}")
    if config.batch_size < 1:
        raise ValueError(f"batch size must be at least 1, got {config.batch_size}")
    if not 0 < config.learning_rate < 1:
        raise ValueError(f"learning rate {config.learning_rate} is not plausible")
    if not 0 <= config.warmup_fraction < 1:
        raise ValueError(f"warmup fraction {config.warmup_fraction} is not a fraction")
    return config


def class_weights(counts: Mapping[str, int], classes: Sequence[str]) -> list[float]:
    """Inverse frequency, normalised so the mean weight is one.

    Normalising matters: without it the loss scale moves with the class balance,
    so the learning rate would have to change whenever the dataset does. With it,
    the only thing that changes is how much each class is worth relative to the
    others.
    """
    missing = [name for name in classes if counts.get(name, 0) <= 0]
    if missing:
        raise ValueError(f"no rows for {missing}, so no weight can be given")
    raw = np.array([1.0 / counts[name] for name in classes], dtype=np.float64)
    return [float(value) for value in raw / raw.mean()]


def batches(count: int, size: int, shuffle: bool, seed: int) -> Iterator[np.ndarray]:
    """Index batches over ``count`` rows, shuffled reproducibly or left in order."""
    order = np.arange(count)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, count, size):
        yield order[start : start + size]


def load_split_frames(
    dataset: str | Path,
    manifest: splits.SplitManifest,
) -> dict[str, pd.DataFrame]:
    """Slice a rebuilt dataset into the three committed splits.

    The split is recomputed here rather than stored, which is the whole point of
    the manifest: the digests say whether what came out is what was recorded, and
    ``emotion-timeline split --verify`` is how a reader checks that separately.
    """
    import pandas as pd

    frame = pd.read_csv(dataset)
    texts = [str(value) for value in frame["text"]]
    labels = [str(value) for value in frame["label"]]
    sources = [str(value) for value in frame["source"]]
    keys = splits.row_keys(texts, labels, sources, seed=manifest.seed)
    assignment = splits.assign(keys, labels, manifest.fraction)

    frame = frame.assign(row_key=keys, split=assignment)
    out: dict[str, pd.DataFrame] = {}
    for name in splits.SPLITS:
        part = frame[frame["split"] == name].reset_index(drop=True)
        recorded = int(manifest.part(name)["rows"])
        if len(part) != recorded:
            raise ValueError(f"{name} came out {len(part)} rows, the record says {recorded}")
        out[name] = part
    return out


def run_record(
    config: TrainConfig,
    manifest: splits.SplitManifest,
    history: Sequence[Mapping[str, float]],
    seconds: float,
    peak_mib: int,
    device_name: str,
) -> dict[str, Any]:
    """What the run was, in the shape every other record in this repository takes."""
    return {
        "_comment": [
            "One fine-tune of the classifier on the rebuilt dataset.",
            "Written by `emotion-timeline fine-tune`. The weights themselves ship",
            "as a release asset with the digest recorded below; this file is the",
            "part small enough to read in a diff. Scores live beside it in",
            "held-out-summary.json, which `emotion-timeline training` checks.",
        ],
        # classes as a list, not a tuple: JSON has no tuples, and a record that
        # does not read back as what was written is not a record.
        "config": {**asdict(config), "classes": list(config.classes)},
        "split_manifest_sha256": manifest.digest,
        "device": device_name,
        "python": platform.python_version(),
        "seconds": round(seconds, 1),
        "peak_mib": peak_mib,
        "epochs": [dict(entry) for entry in history],
    }


def write_record(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write a run record where the committed one lives."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out
