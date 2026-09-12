"""What translation costs, measured with everything else held constant.

The claim this file exists to protect is the one the Russian chapter turns on:
that the gap between a translated approach and a native one is **not** the
translator's quality. That only holds because a much larger engine was run as a
control and recovered almost none of the loss, so the test that matters most here
is the one asserting the two engines land close together and far below the
untranslated rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emotion_timeline.russian import cost

ROOT = Path(__file__).resolve().parents[1]


def committed() -> cost.TranslationCost:
    return cost.TranslationCost.load()


def made(**changes: Any) -> cost.TranslationCost:
    """A small record, with one part overridden."""
    raw: dict[str, Any] = {
        "rows": 100,
        "sampled_from": 1000,
        "seed": 1,
        "baseline": {"accuracy": 0.9, "macro_f1": 0.8, "markers": {"!": 0.1}},
        "engines": {
            "weak": {"accuracy": 0.5, "macro_f1": 0.4, "markers": {"!": 0.1}},
            "strong": {"accuracy": 0.55, "macro_f1": 0.45, "markers": {"!": 0.1}},
        },
    }
    raw.update(changes)
    return cost.TranslationCost(raw=raw, source=Path("x.json"), digest="0" * 64)


# --- the arithmetic -----------------------------------------------------------


def test_the_cost_is_the_drop_from_the_untranslated_rows() -> None:
    report = made()
    assert report.cost_of("weak") == 0.4
    assert report.cost_of("strong") == pytest.approx(0.35)


def test_the_better_engine_is_the_one_the_argument_turns_on() -> None:
    report = made()
    assert report.best_engine() == "strong"
    # 0.40 lost against 0.35 lost: five points of the forty bought back.
    assert report.recovered() == pytest.approx(0.05)


def test_markers_are_counted_as_a_share_of_rows() -> None:
    texts = ["a! b", "c? d", "e! f!", "plain"]
    assert cost.marker_shares(texts, ["!", "?"]) == {"!": 0.5, "?": 0.25}


def test_counting_markers_in_nothing_is_zero_rather_than_a_crash() -> None:
    assert cost.marker_shares([], ["!"]) == {"!": 0.0}


# --- what the consistency check catches ---------------------------------------


def test_an_accuracy_that_is_not_a_share_is_caught() -> None:
    broken = made(baseline={"accuracy": 1.4, "macro_f1": 0.8})
    assert any("not a share" in problem for problem in cost.check_consistency(broken))


def test_a_marker_share_above_one_is_caught() -> None:
    broken = made(baseline={"accuracy": 0.9, "macro_f1": 0.8, "markers": {"!": 1.2}})
    assert any("share 1.2" in problem for problem in cost.check_consistency(broken))


def test_sampling_more_rows_than_exist_is_caught() -> None:
    broken = made(rows=5000)
    assert any("which is fewer" in problem for problem in cost.check_consistency(broken))


def test_a_record_with_nothing_to_compare_against_is_caught() -> None:
    assert any("no engine" in problem for problem in cost.check_consistency(made(engines={})))


def test_an_engine_that_beats_the_untranslated_rows_has_to_be_explained() -> None:
    """Round-tripping cannot add signal. If it looks like it did, something is wrong."""
    broken = made(engines={"impossible": {"accuracy": 0.95, "macro_f1": 0.9}})
    assert any("needs explaining" in problem for problem in cost.check_consistency(broken))


# --- the committed record -----------------------------------------------------


def test_the_committed_record_holds_together() -> None:
    assert cost.check_consistency(committed()) == []


def test_the_model_scores_on_its_own_rows_what_the_fine_tune_reported() -> None:
    """The sample has to be representative, or nothing below it means anything."""
    baseline = committed().baseline
    run = json.loads(
        (ROOT / "benchmarks" / "training" / "run-baseline.json").read_text(encoding="utf-8")
    )
    published = run["epochs"][-1]["val_accuracy"]
    assert abs(float(baseline["accuracy"]) - published) < 0.01


def test_translating_costs_this_classifier_a_third_of_its_accuracy() -> None:
    report = committed()
    assert report.rows == 3000
    for name in report.engines:
        assert report.cost_of(name) > 0.30


def test_a_far_larger_translator_recovers_almost_none_of_the_loss() -> None:
    """The control, and the reason the chapter blames the paraphrase not the engine."""
    report = committed()
    assert set(report.engines) == {"opus-mt", "NLLB-600M"}
    assert report.best_engine() == "NLLB-600M"
    assert report.recovered() < 0.05
    worst = max(report.cost_of(name) for name in report.engines)
    assert report.recovered() < worst / 10


def test_the_surface_markers_mostly_survive_translation() -> None:
    """The obvious explanation, measured and ruled out rather than asserted."""
    report = committed()
    before = report.baseline["markers"]
    for name, block in report.engines.items():
        for marker, share in dict(block["markers"]).items():
            if float(before[marker]) < 0.005:
                continue
            ratio = float(share) / float(before[marker])
            assert 0.5 < ratio < 2.0, f"{name}/{marker} moved by more than the accuracy did"


def test_describe_says_what_the_control_bought() -> None:
    out = "\n".join(cost.describe(committed()))
    assert "English held-out rows" in out
    assert "NLLB-600M" in out and "opus-mt" in out
    assert "the paraphrase, not the translator" in out


def test_the_chapter_quotes_the_record() -> None:
    chapter = (ROOT / "docs" / "russian.md").read_text(encoding="utf-8")
    report = committed()
    assert f"{float(report.baseline['accuracy']):.4f}" in chapter
    for name in report.engines:
        assert f"{float(report.engines[name]['accuracy']):.4f}" in chapter
