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


def warmup_then_decay(step: int, warmup: int, total: int) -> float:
    """Learning-rate multiplier: up over ``warmup`` steps, then down to zero.

    Written out rather than taken from ``transformers``, which is five lines
    either way and one fewer library API to track across versions. It is also
    then testable, which the transformers version is not.
    """
    if warmup > 0 and step < warmup:
        return step / warmup
    remaining = total - warmup
    return max(0.0, (total - step) / remaining) if remaining > 0 else 0.0


def encode(tokenizer: Any, texts: Sequence[str], max_length: int) -> Any:
    """Tokenise one batch, padded to its own longest sequence rather than to 128."""
    return tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


def forward_all(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    config: TrainConfig,
    device: str = "cuda",
) -> np.ndarray:  # pragma: no cover - needs a GPU
    """Logits for every row of a split, in the frame's own order."""
    import torch

    model.eval()
    out: list[np.ndarray] = []
    texts = [str(value) for value in frame["text"]]
    with torch.no_grad():
        for index in batches(len(texts), config.batch_size, shuffle=False, seed=0):
            encoded = encode(tokenizer, [texts[position] for position in index], config.max_length)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**encoded).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def train(
    config: TrainConfig,
    frames: Mapping[str, pd.DataFrame],
    weights_dir: str | Path,
    progress: Any = print,
) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:  # pragma: no cover - needs a GPU
    """Fine-tune, reporting each epoch, and return the history and the logits.

    Validation is scored every epoch so the history shows whether the run was
    still improving when it stopped -- which is the question a reader asks of
    three epochs, and which the inherited record cannot answer about its own.
    """
    import time

    import torch
    from torch.nn import CrossEntropyLoss
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(config.seed)
    index = config.label_index
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=len(config.classes),
        id2label=dict(enumerate(config.classes)),
        label2id=index,
    ).to("cuda")

    train_frame = frames[splits.TRAIN]
    texts = [str(value) for value in train_frame["text"]]
    targets = np.array([index[str(value)] for value in train_frame["label"]], dtype=np.int64)

    weight = None
    if config.weighted_loss:
        counts = splits.counts_of(str(value) for value in train_frame["label"])
        weight = torch.tensor(
            class_weights(counts, config.classes), dtype=torch.float32, device="cuda"
        )
    loss_function = CrossEntropyLoss(weight=weight)

    steps_per_epoch = (len(texts) + config.batch_size - 1) // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    warmup_steps = int(total_steps * config.warmup_fraction)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: warmup_then_decay(step, warmup_steps, total_steps)
    )

    torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, float]] = []
    started = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for step, batch in enumerate(
            batches(len(texts), config.batch_size, shuffle=True, seed=config.seed + epoch), start=1
        ):
            encoded = encode(tokenizer, [texts[position] for position in batch], config.max_length)
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            labels = torch.from_numpy(targets[batch]).to("cuda")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**encoded).logits
            loss = loss_function(logits.float(), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad(set_to_none=True)

            running += float(loss.item()) * len(batch)
            seen += len(batch)
            if step % 200 == 0 or step == steps_per_epoch:
                progress(
                    f"  epoch {epoch}  step {step:>5,}/{steps_per_epoch:,}  "
                    f"loss {running / seen:.4f}  {time.monotonic() - started:.0f}s"
                )

        validation = frames[splits.VALIDATION]
        logits = forward_all(model, tokenizer, validation, config)
        truth = np.array([index[str(value)] for value in validation["label"]], dtype=np.int64)
        accuracy = float((logits.argmax(axis=1) == truth).mean())
        history.append(
            {"epoch": float(epoch), "train_loss": running / seen, "val_accuracy": accuracy}
        )
        progress(f"  epoch {epoch}  validation accuracy {accuracy:.4f}")

    out = Path(weights_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    progress(f"  weights written to {out}")

    logits = {
        name: forward_all(model, tokenizer, frames[name], config)
        for name in (splits.VALIDATION, splits.TEST)
    }
    history.append(
        {
            "seconds": time.monotonic() - started,
            "peak_mib": float(torch.cuda.max_memory_allocated() // (1024 * 1024)),
        }
    )
    return history, logits


def predictions_table(
    frame: pd.DataFrame,
    logits: np.ndarray,
    config: TrainConfig,
) -> dict[str, np.ndarray]:
    """Row keys, true class and logits, aligned, ready to be saved.

    The per-sample predictions of the inherited model were not kept, which is why
    `docs/error-analysis.md` renders from a summary rather than recomputing.
    Keeping these is the one thing this run can do that the original cannot be
    made to do retrospectively.
    """
    if len(frame) != len(logits):
        raise ValueError(f"{len(frame)} rows against {len(logits)} rows of logits")
    index = config.label_index
    return {
        "row_key": np.array([str(value) for value in frame["row_key"]]),
        "true": np.array([index[str(value)] for value in frame["label"]], dtype=np.int16),
        "logits": np.asarray(logits, dtype=np.float32),
    }


def save_predictions(path: str | Path, tables: Mapping[str, Mapping[str, np.ndarray]]) -> Path:
    """One compressed archive per run, holding every split it scored."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flattened = {
        f"{split}_{field}": array
        for split, table in tables.items()
        for field, array in table.items()
    }
    # numpy's stub types every keyword of savez_compressed as its own
    # allow_pickle flag, so the arrays have to go through an untyped call.
    save: Any = np.savez_compressed
    save(out, **flattened)
    return out
