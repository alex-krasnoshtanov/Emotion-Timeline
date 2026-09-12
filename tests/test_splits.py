"""The split: the rounding that loses a row, and the bridge to the inherited records.

Nothing here needs the dataset. The quota is arithmetic over the recorded class
counts, and the membership behaviour is checked on a handful of synthetic rows,
so the whole file runs on a machine that has never downloaded the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_timeline.data import build as ds
from emotion_timeline.training import splits

RECORD = json.loads(
    (
        Path(__file__).resolve().parents[1] / "benchmarks" / "dataset" / "build-record.json"
    ).read_text(encoding="utf-8")
)
CLASS_COUNTS: dict[str, int] = RECORD["class_counts"]


def rows(counts: dict[str, int]) -> tuple[list[str], list[str]]:
    """Synthetic rows with distinct text, one label each."""
    texts, labels = [], []
    for label, count in counts.items():
        for index in range(count):
            texts.append(f"{label}-{index}")
            labels.append(label)
    return splits.row_keys(texts, labels, ["test"] * len(texts)), labels


# --- the quota ---------------------------------------------------------------


def test_plain_rounding_loses_a_row_that_largest_remainder_keeps() -> None:
    """The reason this is not seven calls to round().

    15% of 419,180 is exactly 62,877. Rounding each class on its own gives
    62,876, and the missing row is not a rounding detail -- it is the difference
    between a split that sums to the set and one that does not.
    """
    rounded = sum(round(count * 0.15) for count in CLASS_COUNTS.values())
    exact = round(sum(CLASS_COUNTS.values()) * 0.15)
    assert rounded == 62_876
    assert exact == 62_877
    assert sum(splits.quota(CLASS_COUNTS, 0.15).values()) == exact


def test_the_spare_rows_go_to_the_largest_remainders() -> None:
    counts = {"a": 10, "b": 23, "c": 7}
    # A quarter of 40 is 10. The floors are 2, 5 and 1, so two rows are left to
    # place, and b (.75) and c (.75) are ahead of a (.5).
    taken = splits.quota(counts, 0.25)
    assert sum(taken.values()) == 10
    assert taken == {"a": 2, "b": 6, "c": 2}


def test_a_tie_is_broken_by_class_name() -> None:
    """Somewhere has to decide, and dictionary order is not a decision."""
    counts = {"zeta": 10, "alpha": 10}
    assert splits.quota(counts, 0.15) == {"alpha": 2, "zeta": 1}


def test_a_whole_fraction_needs_no_tie_break() -> None:
    assert splits.quota({"a": 10, "b": 30}, 0.5) == {"a": 5, "b": 15}


# --- what the split is -------------------------------------------------------


def test_the_held_out_set_is_exactly_fifteen_percent_of_the_build() -> None:
    taken = splits.quota(CLASS_COUNTS, 0.15)
    assert sum(taken.values()) == 62_877
    assert sum(CLASS_COUNTS.values()) - 2 * sum(taken.values()) == 293_426


def test_five_of_the_seven_classes_land_on_the_model_cards_own_supports() -> None:
    """The evidence that this split is comparable to the inherited evaluation.

    The card's per-class held-out supports were recorded months earlier, over a
    set that included 9,151 synthetic Disgust rows this build cannot reproduce.
    Five classes still agree to the row. Joy is one short because the original
    resolved its rounding tie differently, and Disgust cannot agree at all --
    which is exactly why no Disgust comparison is published.
    """
    report = ds.DatasetReport.load()
    card_supports = {
        "Anger": 8_695,
        "Fear": 8_003,
        "Neutral": 2_010,
        "Sadness": 18_842,
        "Surprise": 2_372,
    }
    taken = splits.quota(report.class_counts, 0.15)
    assert {name: taken[name] for name in card_supports} == card_supports
    assert taken["Joy"] == 22_180  # the card says 22,181
    assert taken["Disgust"] == 775  # the card says 2,147, of which 1,373 are synthetic


def test_the_held_out_set_plus_the_synthetic_share_is_the_cards_64250() -> None:
    """Two documents written months apart, meeting on a number neither derives.

    This is the same cross-check docs/dataset.md makes for the whole set, carried
    down to the split: the rows this build cannot reproduce account for the
    difference exactly, with nothing left over.
    """
    report = ds.DatasetReport.load()
    synthetic = int(report.published["synthetic_disgust_rows"])
    held_out = sum(splits.quota(report.class_counts, 0.15).values())
    assert held_out + round(0.15 * synthetic) == report.published["held_out_rows"]


def test_validation_and_test_hold_the_same_rows_of_each_class() -> None:
    keys, labels = rows({"a": 40, "b": 20})
    assignment = splits.assign(keys, labels, 0.25)
    counts = splits.counts_of(assignment)
    assert counts[splits.TEST] == counts[splits.VALIDATION] == 15
    assert counts[splits.TRAIN] == 30


def test_every_row_lands_in_exactly_one_split() -> None:
    keys, labels = rows({"a": 37, "b": 11, "c": 3})
    assignment = splits.assign(keys, labels)
    assert len(assignment) == len(keys)
    assert set(assignment) <= set(splits.SPLITS)
    assert sum(splits.counts_of(assignment).values()) == 51


def test_a_class_too_small_to_split_stays_in_train() -> None:
    keys, labels = rows({"a": 100, "rare": 2})
    assignment = splits.assign(keys, labels, 0.15)
    rare = [split for split, label in zip(assignment, labels, strict=True) if label == "rare"]
    assert rare == [splits.TRAIN, splits.TRAIN]


# --- reproducibility ---------------------------------------------------------


def test_a_rows_split_does_not_depend_on_the_order_it_arrives_in() -> None:
    """Why rows are keyed by their text rather than by position."""
    keys, labels = rows({"a": 30, "b": 18})
    forward = dict(zip(keys, splits.assign(keys, labels), strict=True))
    backward = dict(zip(keys[::-1], splits.assign(keys[::-1], labels[::-1]), strict=True))
    assert forward == backward


def test_the_digest_pins_the_set_and_not_the_order() -> None:
    keys, _ = rows({"a": 12})
    assert splits.membership_digest(keys) == splits.membership_digest(reversed(keys))
    assert splits.membership_digest(keys) != splits.membership_digest(keys[1:])


def test_the_same_seed_gives_the_same_membership_and_another_seed_does_not() -> None:
    labels = ["a"] * 40 + ["b"] * 40
    texts = [f"row-{index}" for index in range(80)]
    sources = ["test"] * 80
    first = splits.manifest(splits.row_keys(texts, labels, sources), labels)
    same = splits.manifest(splits.row_keys(texts, labels, sources), labels)
    other = splits.manifest(splits.row_keys(texts, labels, sources, seed=7), labels, seed=7)
    assert first == same
    assert first["splits"] != other["splits"]


def test_a_digest_is_sixty_four_hex_characters() -> None:
    digest = splits.membership_digest(["one", "two"])
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_keys_and_labels_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="2 keys against 1 labels"):
        splits.assign(["a", "b"], ["x"])


# --- the manifest ------------------------------------------------------------


def test_the_manifest_accounts_for_every_row_it_was_given() -> None:
    keys, labels = rows({"a": 60, "b": 24, "c": 9})
    record = splits.manifest(keys, labels)
    assert record["source_rows"] == 93
    per_split = record["splits"]
    assert isinstance(per_split, dict)
    assert sum(int(part["rows"]) for part in per_split.values()) == 93
    for part in per_split.values():
        assert sum(part["class_counts"].values()) == part["rows"]


def test_the_manifest_class_counts_sum_to_the_counts_it_started_from() -> None:
    keys, labels = rows({"a": 60, "b": 24, "c": 9})
    record = splits.manifest(keys, labels)
    per_split = record["splits"]
    assert isinstance(per_split, dict)
    totalled: dict[str, int] = {}
    for part in per_split.values():
        for name, count in part["class_counts"].items():
            totalled[name] = totalled.get(name, 0) + count
    assert totalled == record["class_counts"]


# --- keys, where text alone is not enough ------------------------------------


def test_the_same_text_under_two_labels_gets_two_keys() -> None:
    """The case that made text alone unusable as a key.

    444 texts reach the training set more than once because deduplication runs
    before case is folded, and 178 of them carry more than one label.
    """
    keys = splits.row_keys(["a shout", "a shout"], ["Joy", "Anger"], ["MELD", "MELD"])
    assert len(set(keys)) == 2


def test_rows_identical_in_every_field_are_numbered() -> None:
    keys = splits.row_keys(["same"] * 3, ["Joy"] * 3, ["MELD"] * 3)
    assert len(set(keys)) == 3


def test_numbering_identical_rows_still_gives_the_same_set_either_way() -> None:
    """Why occurrence numbering does not smuggle order back in."""
    texts = ["same", "other", "same", "same"]
    labels = ["Joy", "Joy", "Joy", "Joy"]
    sources = ["MELD"] * 4
    forward = splits.row_keys(texts, labels, sources)
    backward = splits.row_keys(texts[::-1], labels[::-1], sources[::-1])
    assert sorted(forward) == sorted(backward)


# --- what splitting by row rather than by text costs -------------------------


def test_overlap_counts_held_out_rows_whose_text_is_also_in_train() -> None:
    texts = ["shared", "shared", "alone"]
    labels = ["Joy", "Joy", "Anger"]
    assignment = [splits.TRAIN, splits.TEST, splits.TEST]
    assert splits.text_overlap(texts, labels, assignment) == {
        "held_out_rows_sharing_a_training_text": 1,
        "of_those_labelled_differently": 0,
    }


def test_overlap_separates_the_pairs_that_cannot_both_be_right() -> None:
    """A held-out row whose text is in train under another label is a certain error."""
    texts = ["shared", "shared"]
    labels = ["Joy", "Anger"]
    assignment = [splits.TRAIN, splits.TEST]
    assert splits.text_overlap(texts, labels, assignment) == {
        "held_out_rows_sharing_a_training_text": 1,
        "of_those_labelled_differently": 1,
    }


def test_a_split_with_nothing_shared_reports_nothing() -> None:
    texts = ["one", "two", "three"]
    labels = ["Joy", "Joy", "Anger"]
    assignment = [splits.TRAIN, splits.TEST, splits.VALIDATION]
    assert splits.text_overlap(texts, labels, assignment) == {
        "held_out_rows_sharing_a_training_text": 0,
        "of_those_labelled_differently": 0,
    }


# --- the committed manifest --------------------------------------------------


def committed() -> splits.SplitManifest:
    return splits.SplitManifest.load()


def test_the_committed_manifest_is_internally_consistent() -> None:
    assert splits.check_consistency(committed()) == []


def test_the_committed_split_is_the_one_the_documentation_describes() -> None:
    record = committed()
    assert record.source_rows == 419_180
    assert record.part(splits.TRAIN)["rows"] == 293_426
    assert record.part(splits.TEST)["rows"] == 62_877
    assert record.part(splits.VALIDATION)["rows"] == 62_877
    assert record.class_counts == CLASS_COUNTS


def test_the_committed_split_holds_the_supports_the_error_analysis_scored() -> None:
    held = committed().part(splits.TEST)["class_counts"]
    assert held["Neutral"] == 2_010
    assert held["Fear"] == 8_003
    assert held["Anger"] == 8_695


def test_the_overlap_the_split_admits_to_is_small_and_recorded() -> None:
    """Splitting by row lets 185 texts reach both halves; 75 cannot be got right."""
    overlap = committed().overlap
    assert overlap["held_out_rows_sharing_a_training_text"] == 185
    assert overlap["of_those_labelled_differently"] == 75


def test_describe_says_what_the_command_prints() -> None:
    lines = list(splits.describe(committed()))
    assert "419,180 rows, split 70% / 15% / 15%, seed 20251115" in lines[0]
    assert any("185 held-out rows share their text" in line for line in lines)


# --- a record that does not hold together is refused -------------------------


def broken(tmp_path: Path, **changes: object) -> splits.SplitManifest:
    raw = json.loads(Path(splits.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    for key, value in changes.items():
        raw[key] = value
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return splits.SplitManifest.load(path)


def test_a_missing_split_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(splits.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    del raw["splits"][splits.VALIDATION]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = splits.check_consistency(splits.SplitManifest.load(path))
    assert any("expected" in problem for problem in problems)


def test_class_counts_that_do_not_sum_to_the_header_are_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(splits.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    raw["splits"][splits.TEST]["rows"] = 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = splits.check_consistency(splits.SplitManifest.load(path))
    assert any("header says 1" in problem for problem in problems)


def test_a_digest_that_is_not_a_sha256_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(splits.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    raw["splits"][splits.TRAIN]["membership_sha256"] = "not a digest"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = splits.check_consistency(splits.SplitManifest.load(path))
    assert any("not a sha256" in problem for problem in problems)


def test_a_held_out_set_that_is_not_the_quota_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(splits.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    raw["splits"][splits.TEST]["class_counts"]["Joy"] -= 1
    raw["splits"][splits.TEST]["rows"] -= 1
    raw["splits"][splits.TRAIN]["class_counts"]["Joy"] += 1
    raw["splits"][splits.TRAIN]["rows"] += 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = splits.check_consistency(splits.SplitManifest.load(path))
    assert any("quota" in problem for problem in problems)


def test_splits_that_do_not_add_back_to_their_class_counts_are_caught(tmp_path: Path) -> None:
    record = broken(tmp_path, class_counts={**CLASS_COUNTS, "Joy": 1})
    problems = splits.check_consistency(record)
    assert any("do not add back up" in problem for problem in problems)


def test_more_contradictions_than_shared_rows_is_caught(tmp_path: Path) -> None:
    record = broken(
        tmp_path,
        text_overlap={
            "held_out_rows_sharing_a_training_text": 1,
            "of_those_labelled_differently": 2,
        },
    )
    problems = splits.check_consistency(record)
    assert any("contradict" in problem for problem in problems)


def test_a_row_total_that_contradicts_the_header_is_caught(tmp_path: Path) -> None:
    record = broken(tmp_path, source_rows=5)
    problems = splits.check_consistency(record)
    assert any("the record says 5" in problem for problem in problems)


# --- verifying against a rebuilt dataset -------------------------------------


def tiny() -> tuple[list[str], list[str], list[str]]:
    texts = [f"row {index}" for index in range(60)]
    labels = ["a"] * 40 + ["b"] * 20
    return texts, labels, ["test"] * 60


def test_verify_agrees_with_the_rows_the_record_was_built_from(tmp_path: Path) -> None:
    texts, labels, sources = tiny()
    path = splits.write_manifest(splits.build_manifest(texts, labels, sources), tmp_path / "m.json")
    record = splits.SplitManifest.load(path)
    assert splits.check_consistency(record) == []
    assert splits.verify(record, texts, labels, sources) == []


def test_verify_catches_a_row_whose_text_changed(tmp_path: Path) -> None:
    texts, labels, sources = tiny()
    path = splits.write_manifest(splits.build_manifest(texts, labels, sources), tmp_path / "m.json")
    record = splits.SplitManifest.load(path)
    changed = [*texts[:-1], "something else entirely"]
    assert splits.verify(record, changed, labels, sources) != []


def test_verify_refuses_a_dataset_of_the_wrong_size(tmp_path: Path) -> None:
    texts, labels, sources = tiny()
    path = splits.write_manifest(splits.build_manifest(texts, labels, sources), tmp_path / "m.json")
    record = splits.SplitManifest.load(path)
    problems = splits.verify(record, texts[:10], labels[:10], sources[:10])
    assert problems == ["10 rows given, the record was built from 60"]


def test_verify_catches_an_overlap_that_no_longer_matches(tmp_path: Path) -> None:
    texts, labels, sources = tiny()
    record_path = splits.write_manifest(
        splits.build_manifest(texts, labels, sources), tmp_path / "m.json"
    )
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    raw["text_overlap"]["held_out_rows_sharing_a_training_text"] = 99
    record_path.write_text(json.dumps(raw), encoding="utf-8")
    record = splits.SplitManifest.load(record_path)
    problems = splits.verify(record, texts, labels, sources)
    assert any("text overlap" in problem for problem in problems)


def test_rows_that_cannot_be_told_apart_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate rows are fine; two rows sharing a key are not.

    Occurrence numbering is what keeps keys distinct when every field matches, so
    this guards a regression in that rather than a SHA-256 collision. Forcing the
    keys is the only way to reach it.
    """
    monkeypatch.setattr(splits, "row_keys", lambda *a, **k: ["same", "same"])
    with pytest.raises(ValueError, match="could not be told apart"):
        splits.build_manifest(["x", "y"], ["a", "a"], ["s", "s"], fraction=0.5)


def test_the_manifest_round_trips_through_the_file(tmp_path: Path) -> None:
    texts, labels, sources = tiny()
    record = splits.build_manifest(texts, labels, sources)
    path = splits.write_manifest(record, tmp_path / "nested" / "m.json")
    assert json.loads(path.read_text(encoding="utf-8")) == record
