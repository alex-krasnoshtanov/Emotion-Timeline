"""The Russian evaluation set: the funnel, the one mapping decision, and its cost.

Rebuilding needs a download, so what runs here is the committed record and the
pure mapping functions behind it -- the same division `tests/test_dataset.py` uses
for the English build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from emotion_timeline.data import build as ds
from emotion_timeline.data import ru
from emotion_timeline.data.build import BuildRecord
from emotion_timeline.data.labels import EMOTIONS


def committed() -> ru.RussianReport:
    return ru.RussianReport.load()


# --- the committed record ----------------------------------------------------


def test_the_record_is_internally_consistent() -> None:
    assert ru.check_consistency(committed()) == []


def test_the_funnel_is_the_one_the_chapter_describes() -> None:
    steps = {step["name"]: step for step in committed().steps}
    assert steps["load"]["rows_in"] == 24_891
    assert steps["deduplicate"]["rows_out"] == 24_853
    assert steps["drop-unlabelled"]["rows_out"] == 24_766
    assert committed().rows == 24_766


def test_every_class_is_one_of_ours() -> None:
    assert set(committed().class_counts) == set(EMOTIONS)


def test_the_russian_set_is_neutral_heavy_where_the_english_one_is_starved() -> None:
    """The confound that will dominate any cross-language comparison.

    Our English model saw Neutral in 3.2% of its training rows and is worst at it.
    This set is 31.3% Neutral -- ten times the share, on the class the model is
    least able to get right. A score measured here is measuring that as much as it
    is measuring the language.
    """
    russian = committed()
    english = ds.DatasetReport.load()
    russian_share = russian.class_counts["Neutral"] / russian.rows
    english_share = english.class_counts["Neutral"] / english.rows
    assert russian_share == pytest.approx(0.313, abs=5e-3)
    assert english_share == pytest.approx(0.032, abs=5e-3)
    assert russian_share > 9 * english_share


def test_the_enthusiasm_decision_is_smaller_than_the_column_suggests() -> None:
    """5,185 rows carry it; only 1,115 change class because of it."""
    enthusiasm = committed().enthusiasm
    assert enthusiasm["rows"] == 5_185
    assert enthusiasm["rows_moved"] == 1_115
    assert enthusiasm["mapped_to"] == "Joy"
    assert enthusiasm["rows_moved"] < enthusiasm["rows"] / 4


def test_the_classes_with_no_home_are_named() -> None:
    assert committed().raw["dropped_columns"] == ["guilt", "shame"]


# --- the mapping -------------------------------------------------------------


def test_enthusiasm_becomes_joy() -> None:
    assert ru.to_seven(["enthusiasm"]) == "Joy"


def test_enthusiasm_and_joy_together_are_one_joy() -> None:
    """Merged before the collapse, so it cannot vote for Joy twice."""
    assert ru.to_seven(["joy", "enthusiasm"]) == "Joy"


def test_dropping_enthusiasm_leaves_a_row_with_nothing_else_unlabelled() -> None:
    assert ru.to_seven(["enthusiasm"], drop_enthusiasm=True) is None


def test_dropping_enthusiasm_keeps_a_row_that_carries_something_else() -> None:
    assert ru.to_seven(["enthusiasm", "anger"], drop_enthusiasm=True) == "Anger"


def test_guilt_and_shame_never_reach_a_class() -> None:
    assert ru.to_seven(["guilt", "shame"]) is None
    assert ru.to_seven(["guilt", "fear"]) == "Fear"


def test_multi_label_collapses_the_way_the_english_build_collapses() -> None:
    """Same PRIORITY, so the same pair resolves the same way in either language."""
    assert ru.to_seven(["anger", "disgust"]) == "Disgust"
    assert ru.to_seven(["joy", "neutral"]) == "Neutral"


def test_a_row_with_nothing_set_has_no_class() -> None:
    assert ru.to_seven([]) is None


def test_the_kept_columns_change_with_the_decision() -> None:
    assert "enthusiasm" in ru.kept_columns()
    assert "enthusiasm" not in ru.kept_columns(drop_enthusiasm=True)
    assert not set(ru.DROPPED_COLUMNS) & set(ru.kept_columns())


# --- a record that does not hold together is refused -------------------------


def broken(tmp_path: Path, **changes: object) -> ru.RussianReport:
    raw = json.loads(Path(ru.DEFAULT_RECORD).read_text(encoding="utf-8"))
    raw.update(changes)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return ru.RussianReport.load(path)


def test_class_counts_that_do_not_sum_to_the_build_are_caught(tmp_path: Path) -> None:
    problems = ru.check_consistency(broken(tmp_path, rows=5))
    assert any("the build produced 5" in problem for problem in problems)


def test_a_step_that_does_not_chain_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(ru.DEFAULT_RECORD).read_text(encoding="utf-8"))
    raw["steps"][1]["rows_in"] = 3
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = ru.check_consistency(ru.RussianReport.load(path))
    assert any("starts from 3" in problem for problem in problems)


def test_a_class_that_is_not_ours_is_caught(tmp_path: Path) -> None:
    counts = dict(committed().class_counts)
    counts["Enthusiasm"] = 0
    problems = ru.check_consistency(broken(tmp_path, class_counts=counts))
    assert any("not ours" in problem for problem in problems)


def test_moving_more_rows_than_carried_the_label_is_caught(tmp_path: Path) -> None:
    problems = ru.check_consistency(
        broken(tmp_path, enthusiasm={"rows": 10, "rows_moved": 99, "mapped_to": "Joy"})
    )
    assert any("more rows changed class" in problem for problem in problems)


def test_mapping_onto_a_class_we_do_not_have_is_caught(tmp_path: Path) -> None:
    problems = ru.check_consistency(
        broken(tmp_path, enthusiasm={"rows": 10, "rows_moved": 1, "mapped_to": "Enthusiasm"})
    )
    assert any("not one of the seven" in problem for problem in problems)


def test_a_funnel_that_ends_somewhere_else_is_caught(tmp_path: Path) -> None:
    raw = json.loads(Path(ru.DEFAULT_RECORD).read_text(encoding="utf-8"))
    raw["steps"][-1]["rows_out"] = 7
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = ru.check_consistency(ru.RussianReport.load(path))
    assert any("the funnel ends at" in problem for problem in problems)


# --- the funnel, on rows small enough to count by hand ------------------------


def fake_source() -> pd.DataFrame:
    """Eight rows covering every case the mapping has to handle."""
    rows = [
        # text,            neutral joy sad ang enth sur dis fear guilt shame
        ("plain joy", 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        ("enthusiasm only", 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ("enthusiasm and joy", 0, 1, 0, 0, 1, 0, 0, 0, 0, 0),
        ("anger and disgust", 0, 0, 0, 1, 0, 0, 1, 0, 0, 0),
        ("guilt only", 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        ("shame and fear", 0, 0, 0, 0, 0, 0, 0, 1, 0, 1),
        ("plain joy", 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # duplicate text
        ("nothing at all", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    return pd.DataFrame(rows, columns=["text", *ru.SOURCE_COLUMNS])


def built(
    monkeypatch: pytest.MonkeyPatch, drop_enthusiasm: bool = False
) -> tuple[pd.DataFrame, BuildRecord, int]:
    monkeypatch.setattr(ru, "load_source", lambda *a, **k: fake_source())
    return ru.build(drop_enthusiasm=drop_enthusiasm)


def test_the_funnel_removes_the_duplicate_and_the_homeless_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, record, enthusiasm = built(monkeypatch)
    steps = {step.name: step for step in record.steps}
    assert steps["load"].rows_out == 8
    assert steps["deduplicate"].rows_out == 7  # the repeated text
    assert steps["drop-unlabelled"].rows_out == 5  # guilt-only and the empty row
    assert len(frame) == 5
    assert enthusiasm == 2


def test_the_labels_are_the_ones_the_mapping_promises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, record, _ = built(monkeypatch)
    labels = dict(zip(frame["text"], frame["label"], strict=True))
    assert labels["plain joy"] == "Joy"
    assert labels["enthusiasm only"] == "Joy"
    assert labels["enthusiasm and joy"] == "Joy"
    assert labels["anger and disgust"] == "Disgust"
    assert labels["shame and fear"] == "Fear"
    assert record.counts["Joy"] == 3


def test_dropping_enthusiasm_loses_only_the_row_that_had_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, record, _ = built(monkeypatch, drop_enthusiasm=True)
    assert len(frame) == 4
    assert record.counts["Joy"] == 2


def test_a_rebuilt_set_that_matches_the_record_reports_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, record, enthusiasm = built(monkeypatch)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(ru.as_record(record, enthusiasm, 1)), encoding="utf-8")
    assert ru.compare(record, ru.RussianReport.load(path)) == []


def test_a_rebuilt_set_that_drifts_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, record, enthusiasm = built(monkeypatch)
    raw = ru.as_record(record, enthusiasm, 1)
    raw["class_counts"] = {**raw["class_counts"], "Joy": 99}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = ru.compare(record, ru.RussianReport.load(path))
    assert any("record says 99" in problem for problem in problems)


def test_a_class_the_build_no_longer_produces_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, record, enthusiasm = built(monkeypatch)
    raw = ru.as_record(record, enthusiasm, 1)
    raw["class_counts"] = {**raw["class_counts"], "Surprise": 5}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = ru.compare(record, ru.RussianReport.load(path))
    assert any("in the record but not in the build" in problem for problem in problems)


def test_progress_names_every_step(monkeypatch: pytest.MonkeyPatch) -> None:
    _, record, _ = built(monkeypatch)
    lines = list(ru.iter_progress(record))
    assert len(lines) == len(record.steps)
    assert any("deduplicate" in line for line in lines)


def test_a_rebuild_with_the_wrong_row_count_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, record, enthusiasm = built(monkeypatch)
    raw = ru.as_record(record, enthusiasm, 1)
    raw["rows"] = 99
    path = tmp_path / "record.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = ru.compare(record, ru.RussianReport.load(path))
    assert any("the record says 99" in problem for problem in problems)
