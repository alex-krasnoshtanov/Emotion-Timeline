"""The fine-tune, minus the loop.

Everything the training run decides before it touches the GPU is an ordinary
function over ordinary data, so it is checked here: the configuration it refuses,
the class weights it computes, the way it slices the dataset, and the record it
leaves behind. What is left unchecked is the loop, and that is marked in the
source with the reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.training import run, splits


def tiny_dataset(path: Path, per_class: int = 20) -> Path:
    rows = ["text,label,source"]
    for label in EMOTIONS:
        rows += [f"{label.lower()} sentence {index},{label},MELD" for index in range(per_class)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def tiny_manifest(tmp_path: Path, dataset: Path) -> splits.SplitManifest:
    import pandas as pd

    frame = pd.read_csv(dataset)
    record = splits.build_manifest(
        [str(value) for value in frame["text"]],
        [str(value) for value in frame["label"]],
        [str(value) for value in frame["source"]],
    )
    return splits.SplitManifest.load(splits.write_manifest(record, tmp_path / "manifest.json"))


# --- configuration -----------------------------------------------------------


def test_the_default_run_is_the_one_the_evidence_points_at() -> None:
    """distilbert-base-uncased, because the surviving vocabulary is its 30,522 tokens."""
    config = run.build_config()
    assert config.model_id == "distilbert-base-uncased"
    assert config.epochs == 3
    assert config.seed == splits.SEED
    assert config.classes == tuple(EMOTIONS)
    assert config.weighted_loss is False


def test_settings_that_were_not_given_keep_their_defaults() -> None:
    config = run.build_config(batch_size=None, epochs=5)
    assert config.epochs == 5
    assert config.batch_size == 64


def test_a_setting_nobody_has_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        run.build_config(learnign_rate=1e-5)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("epochs", 0, "at least 1"),
        ("batch_size", 0, "at least 1"),
        ("learning_rate", 0.0, "not plausible"),
        ("learning_rate", 2.0, "not plausible"),
        ("warmup_fraction", 1.0, "not a fraction"),
        ("warmup_fraction", -0.1, "not a fraction"),
    ],
)
def test_a_run_that_cannot_work_is_refused_before_the_gpu_is_touched(
    setting: str, value: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run.build_config(**{setting: value})


def test_the_label_index_is_the_order_the_classes_are_declared_in() -> None:
    assert run.build_config().label_index == dict(zip(EMOTIONS, range(7), strict=True))


# --- class weights -----------------------------------------------------------


def test_weights_are_inverse_frequency_with_a_mean_of_one() -> None:
    weights = run.class_weights({"a": 100, "b": 100, "c": 100}, ["a", "b", "c"])
    assert weights == pytest.approx([1.0, 1.0, 1.0])


def test_a_rare_class_is_worth_more_than_a_common_one() -> None:
    weights = run.class_weights({"common": 900, "rare": 100}, ["common", "rare"])
    assert weights[1] == pytest.approx(9 * weights[0])
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_the_mean_stays_one_whatever_the_balance() -> None:
    """So the loss scale does not move with the dataset, and the learning rate can stay."""
    counts = {"Joy": 147_869, "Neutral": 13_401, "Disgust": 5_165}
    weights = run.class_weights(counts, list(counts))
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_a_class_with_no_rows_cannot_be_weighted() -> None:
    with pytest.raises(ValueError, match=r"no rows for \['missing'\]"):
        run.class_weights({"a": 10}, ["a", "missing"])


# --- batching ----------------------------------------------------------------


def test_every_row_appears_in_exactly_one_batch() -> None:
    seen = [index for batch in run.batches(55, 8, shuffle=False, seed=0) for index in batch]
    assert sorted(seen) == list(range(55))


def test_the_last_batch_is_short_rather_than_dropped() -> None:
    sizes = [len(batch) for batch in run.batches(55, 8, shuffle=False, seed=0)]
    assert sizes == [8, 8, 8, 8, 8, 8, 7]


def test_shuffling_is_reproducible_from_the_seed() -> None:
    first = [i for b in run.batches(40, 8, shuffle=True, seed=3) for i in b]
    same = [i for b in run.batches(40, 8, shuffle=True, seed=3) for i in b]
    other = [i for b in run.batches(40, 8, shuffle=True, seed=4) for i in b]
    assert first == same
    assert first != other
    assert sorted(first) == list(range(40))


def test_not_shuffling_leaves_the_order_alone() -> None:
    assert [i for b in run.batches(6, 3, shuffle=False, seed=9) for i in b] == list(range(6))


# --- slicing the dataset -----------------------------------------------------


def test_the_three_splits_come_out_the_size_the_record_says(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path / "tiny.csv")
    manifest = tiny_manifest(tmp_path, dataset)
    frames = run.load_split_frames(dataset, manifest)
    assert set(frames) == set(splits.SPLITS)
    for name in splits.SPLITS:
        assert len(frames[name]) == manifest.part(name)["rows"]


def test_the_splits_do_not_share_a_row(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path / "tiny.csv")
    frames = run.load_split_frames(dataset, tiny_manifest(tmp_path, dataset))
    keys = {name: set(frame["row_key"]) for name, frame in frames.items()}
    assert keys["train"] & keys["test"] == set()
    assert keys["train"] & keys["validation"] == set()
    assert keys["test"] & keys["validation"] == set()


def test_a_dataset_that_no_longer_matches_the_record_is_refused(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path / "tiny.csv")
    manifest = tiny_manifest(tmp_path, dataset)
    shorter = tmp_path / "shorter.csv"
    lines = dataset.read_text(encoding="utf-8").splitlines()
    shorter.write_text("\n".join(lines[:-14]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="the record says"):
        run.load_split_frames(shorter, manifest)


# --- the record --------------------------------------------------------------


def test_the_record_carries_what_the_run_was(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path / "tiny.csv")
    manifest = tiny_manifest(tmp_path, dataset)
    config = run.build_config(epochs=1)
    record = run.run_record(
        config,
        manifest,
        history=[{"epoch": 1.0, "train_loss": 0.5, "val_accuracy": 0.8}],
        seconds=61.25,
        peak_mib=4500,
        device_name="NVIDIA GeForce RTX 5070",
    )
    assert record["config"]["model_id"] == "distilbert-base-uncased"
    assert record["split_manifest_sha256"] == manifest.digest
    assert record["seconds"] == 61.2
    assert record["epochs"][0]["val_accuracy"] == 0.8


def test_the_record_round_trips_through_the_file(tmp_path: Path) -> None:
    dataset = tiny_dataset(tmp_path / "tiny.csv")
    manifest = tiny_manifest(tmp_path, dataset)
    record = run.run_record(
        run.build_config(), manifest, history=[], seconds=1.0, peak_mib=1, device_name="cpu"
    )
    path = run.write_record(record, tmp_path / "nested" / "run.json")
    assert json.loads(path.read_text(encoding="utf-8")) == record
