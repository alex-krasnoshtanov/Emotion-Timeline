"""The recorded statistics have to hold together, because they cannot be re-derived."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from emotion_timeline.analysis import error_analysis as ea


def test_report_is_internally_consistent() -> None:
    """Supports sum, errors sum, rates match their own numerator and denominator."""
    assert ea.check_consistency(ea.ErrorReport.load()) == []


def test_class_error_rates_match_the_model_card_recalls() -> None:
    """Cross-check against a document produced separately for the same split.

    The model card records per-class recall on the held-out validation set; this
    report records per-class error rate. They are the same measurement from two
    directions, so recall must equal 1 - error rate. Two independently written
    documents agreeing to four decimal places is the strongest evidence available
    that the split described here is the split the model card describes.
    """
    model_card_recall = {
        "Anger": 0.9381,
        "Disgust": 0.9157,
        "Fear": 0.7513,
        "Joy": 0.9374,
        "Neutral": 0.6323,
        "Sadness": 0.9427,
        "Surprise": 0.7732,
    }
    report = ea.ErrorReport.load()
    for name, recall in model_card_recall.items():
        assert abs(report.classes[name].recall - recall) < 5e-4, name


def test_the_hardest_classes_are_the_rare_ones() -> None:
    report = ea.ErrorReport.load()
    hardest = [c.name for c in report.hardest(3)]
    assert hardest == ["Neutral", "Fear", "Surprise"]
    # Neutral carries 2,010 of 64,250 samples and the worst error rate of any class.
    assert report.classes["Neutral"].samples < report.classes["Joy"].samples / 10


def test_surface_markers_multiply_the_error_rate() -> None:
    worst = ea.ErrorReport.load().worst_features()
    assert worst[0][0] == "ALL-CAPS word"
    for _, ratio in worst[:3]:
        assert ratio > 5.0  # every one of the three at least sextuples it


def test_figures_render(tmp_path: Path) -> None:
    written = ea.render_all(ea.ErrorReport.load(), tmp_path)
    assert len(written) == len(ea.FIGURES)
    for path in written:
        assert path.exists() and path.stat().st_size > 10_000


# --- staleness ---------------------------------------------------------------


def test_figures_record_the_report_they_were_drawn_from(tmp_path: Path) -> None:
    report = ea.ErrorReport.load()
    for path in ea.render_all(report, tmp_path):
        assert ea.read_stamp(path) == report.digest
    assert ea.check_figures_current(report, tmp_path) == []


def test_a_changed_report_makes_the_committed_figures_stale(tmp_path: Path) -> None:
    """The check has to actually fail when the numbers move underneath it."""
    report = ea.ErrorReport.load()
    ea.render_all(report, tmp_path)

    edited = json.loads(Path(ea.DEFAULT_REPORT).read_text(encoding="utf-8"))
    edited["confidence"]["mean_when_correct"] = 0.5
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(edited), encoding="utf-8")

    stale = ea.check_figures_current(ea.ErrorReport.load(altered), tmp_path)
    assert len(stale) == len(ea.FIGURES)
    assert all("drawn from" in problem for problem in stale)


def test_missing_figures_are_reported(tmp_path: Path) -> None:
    problems = ea.check_figures_current(ea.ErrorReport.load(), tmp_path)
    assert len(problems) == len(ea.FIGURES)
    assert all(p.endswith("missing") for p in problems)


def test_the_committed_figures_are_current() -> None:
    """The assets in the tree match the report in the tree. CI runs this too."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    assert ea.check_figures_current(ea.ErrorReport.load(), assets) == []


# --- the consistency checks themselves ---------------------------------------


Report = dict[str, Any]


def altered(tmp_path: Path, mutate: Callable[[Report], None]) -> ea.ErrorReport:
    """The committed report with one number broken, to check a guard fires."""
    raw: Report = json.loads(Path(ea.DEFAULT_REPORT).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return ea.ErrorReport.load(path)


def test_a_class_that_does_not_sum_to_the_header_is_caught(tmp_path: Path) -> None:
    def mutate(raw: Report) -> None:
        raw["classes"]["Joy"]["samples"] += 5_000

    problems = ea.check_consistency(altered(tmp_path, mutate))
    assert any("class samples sum to" in p for p in problems)
    assert any("Joy: error rate" in p for p in problems)


def test_an_error_count_that_contradicts_the_accuracy_is_caught(tmp_path: Path) -> None:
    def mutate(raw: Report) -> None:
        raw["total_errors"] = 5000

    problems = ea.check_consistency(altered(tmp_path, mutate))
    assert any("class errors sum to" in p for p in problems)
    assert any("errors/samples imply" in p for p in problems)


def test_confusions_cannot_outnumber_the_errors_they_explain(tmp_path: Path) -> None:
    def mutate(raw: Report) -> None:
        raw["classes"]["Disgust"]["confused_with"] = {"Anger": 99_999}

    problems = ea.check_consistency(altered(tmp_path, mutate))
    assert any("listed confusions" in p for p in problems)


# --- the stamp ---------------------------------------------------------------


def test_an_unstamped_image_is_reported_as_unstamped(tmp_path: Path) -> None:
    """A figure produced by anything other than the renderer carries no digest."""
    report = ea.ErrorReport.load()
    ea.render_all(report, tmp_path)
    victim = tmp_path / next(iter(ea.FIGURES))
    stripped = victim.read_bytes().replace(ea.STAMP_KEY.encode(), b"Other-Header")
    victim.write_bytes(stripped)

    problems = ea.check_figures_current(report, tmp_path)
    assert problems == [f"{victim.name}: carries no {ea.STAMP_KEY} stamp"]


def test_a_file_that_is_not_a_png_has_no_stamp(tmp_path: Path) -> None:
    path = tmp_path / "not-an-image.png"
    path.write_bytes(b"this is not a PNG")
    assert ea.read_stamp(path) is None


def test_the_marker_headline_counts_rather_than_asserts() -> None:
    """It used to be a string, and would have been drawn over any other report."""
    assert ea.marker_headline([56.8, 56.6, 55.8, 14.3]) == (
        "Three surface markers each take the error rate past 55%"
    )
    assert ea.marker_headline([60.0]) == "One surface marker takes the error rate past 55%"
    assert (
        ea.marker_headline([10.0, 20.0]) == "No surface markers each take the error rate past 55%"
    )
    assert ea.marker_headline([]) == "No surface markers each take the error rate past 55%"


def test_the_committed_report_still_says_three() -> None:
    report = ea.ErrorReport.load()
    rates = [
        block["error_rate_present"] * 100
        for name, block in report.textual_features.items()
        if name != "_comment" and block["present_samples"] >= 100
    ]
    assert ea.marker_headline(rates) == "Three surface markers each take the error rate past 55%"


def test_a_report_with_no_errors_is_checked_rather_than_crashing(tmp_path: Path) -> None:
    """A small enough evaluation set can be got entirely right; the check divided by it."""
    raw = json.loads(Path(ea.DEFAULT_REPORT).read_text(encoding="utf-8"))
    raw["total_errors"] = 0
    raw["error_rate"] = 0.0
    raw["accuracy"] = 1.0
    for block in raw["classes"].values():
        block["errors"] = 0
        block["error_rate"] = 0.0
        block["confused_with"] = {}
    raw["confidence"]["high_confidence_errors"] = 0
    raw["confidence"]["high_confidence_error_share_of_errors"] = 0.0
    path = tmp_path / "perfect.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert ea.check_consistency(ea.ErrorReport.load(path)) == []

    raw["confidence"]["high_confidence_errors"] = 3
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = ea.check_consistency(ea.ErrorReport.load(path))
    assert any("no errors recorded" in problem for problem in problems)
