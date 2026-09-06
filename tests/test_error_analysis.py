"""The recorded statistics have to hold together, because they cannot be re-derived."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_figures_render(tmp_path) -> None:
    written = ea.render_all(ea.ErrorReport.load(), tmp_path)
    assert len(written) == len(ea.FIGURES)
    for path in written:
        assert path.exists() and path.stat().st_size > 10_000


# --- staleness ---------------------------------------------------------------


def test_figures_record_the_report_they_were_drawn_from(tmp_path) -> None:
    report = ea.ErrorReport.load()
    for path in ea.render_all(report, tmp_path):
        assert ea.read_stamp(path) == report.digest
    assert ea.check_figures_current(report, tmp_path) == []


def test_a_changed_report_makes_the_committed_figures_stale(tmp_path) -> None:
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


def test_missing_figures_are_reported(tmp_path) -> None:
    problems = ea.check_figures_current(ea.ErrorReport.load(), tmp_path)
    assert len(problems) == len(ea.FIGURES)
    assert all(p.endswith("missing") for p in problems)


def test_the_committed_figures_are_current() -> None:
    """The assets in the tree match the report in the tree. CI runs this too."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    assert ea.check_figures_current(ea.ErrorReport.load(), assets) == []
