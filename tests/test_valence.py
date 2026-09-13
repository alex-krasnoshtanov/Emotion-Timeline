"""Valence and arousal: what they separate, and what they are not allowed to claim.

The record this covers exists to say two things at once — that valence separates
the classes well, and that neither dimension improves the label — and the tests
that matter most are the ones stopping either half from being published alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emotion_timeline.russian import va

ROOT = Path(__file__).resolve().parents[1]


def committed() -> va.ValenceReport:
    return va.ValenceReport.load()


def altered(**changes: Any) -> va.ValenceReport:
    raw = json.loads(json.dumps(committed().raw))
    raw.update(changes)
    return va.ValenceReport(raw=raw, source=Path("x.json"), digest="0" * 64)


# --- the separation statistic -------------------------------------------------


def test_a_dimension_that_ranks_one_group_above_the_other_scores_one() -> None:
    assert va.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0


def test_a_dimension_that_knows_nothing_scores_a_half() -> None:
    """Interleaved, so neither group sits above the other."""
    assert va.auc([0.1, 0.9, 0.9, 0.1], [True, True, False, False]) == 0.5
    assert va.auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_a_dimension_that_ranks_them_backwards_scores_zero() -> None:
    """Not the same as knowing nothing, and the record would show it."""
    assert va.auc([0.1, 0.1, 0.9, 0.9], [True, True, False, False]) == 0.0


def test_separating_a_group_from_nothing_is_not_a_measurement() -> None:
    """An empty side would divide by zero; it returns 'no information' instead."""
    assert va.auc([0.1, 0.2], [True, True]) == 0.5
    assert va.auc([0.1, 0.2], [False, False]) == 0.5


def test_separation_reports_its_own_gap() -> None:
    found = va.separation([0.2, 0.8], ["Joy", "Anger"], ["Joy"], ["Anger"])
    assert found["mean_above"] == 0.2
    assert found["mean_below"] == 0.8
    assert found["gap"] == pytest.approx(-0.6)
    assert found["samples"] == 2


def test_classes_outside_the_two_groups_are_left_out() -> None:
    found = va.separation([0.9, 0.1, 0.5], ["Joy", "Anger", "Surprise"], ["Joy"], ["Anger"])
    assert found["samples"] == 2


def test_the_original_five_levels_are_counted_not_described() -> None:
    counts = va.level_counts([0.1, 0.3, 0.5, 0.7, 0.9, 0.55])
    assert counts == [1, 1, 2, 1, 1]
    assert sum(counts) == 6


# --- the committed record -----------------------------------------------------


def test_the_committed_record_holds_together() -> None:
    assert va.check_consistency(committed()) == []


def test_valence_separates_the_classes_and_arousal_barely_does() -> None:
    """The finding: the original used arousal, which is the weaker dimension."""
    report = committed()
    assert report.auc_of("valence") == pytest.approx(0.8223, abs=5e-4)
    assert report.auc_of("arousal") == pytest.approx(0.5734, abs=5e-4)
    assert report.auc_of("valence") > report.auc_of("arousal") + 0.2


def test_joy_is_the_most_positive_class_and_disgust_the_least() -> None:
    means = {name: block["mean"] for name, block in committed().raw["valence"]["by_class"].items()}
    assert max(means, key=lambda name: means[name]) == "Joy"
    assert min(means, key=lambda name: means[name]) == "Disgust"


def test_neither_dimension_improves_the_emotion_label() -> None:
    """The other half of the record, and the reason this is off by default."""
    report = committed()
    contribution = report.contribution
    assert contribution["p_value"] > 0.05
    assert contribution["significant"] is False
    assert report.helps() is False
    assert abs(float(contribution["delta"])) < 0.01


def test_the_five_intensity_levels_the_original_used_are_nearly_empty_at_the_ends() -> None:
    counts = committed().raw["arousal"]["original_levels"]["counts"]
    total = sum(counts)
    assert (counts[0] + counts[-1]) / total < 0.10
    assert counts[2] / total > 0.4


def test_the_things_that_did_not_work_are_in_the_record_too() -> None:
    """The chapter quotes them, so a command has to recompute them."""
    alternatives = committed().raw["alternatives"]
    assert set(alternatives) == {"arousal-neutral", "valence-joy", "learned-stacker"}
    for name in ("arousal-neutral", "valence-joy"):
        block = alternatives[name]
        assert block["accuracy"] < block["baseline"], name
        assert block["lost"] > block["gained"], name


def test_the_stacker_gain_is_the_class_prior_and_the_record_says_so() -> None:
    """Accuracy up, balanced accuracy down: the trap this nearly fell into."""
    block = committed().raw["alternatives"]["learned-stacker"]
    assert block["accuracy"] > block["baseline"]
    assert block["p_value"] < 0.05
    assert block["balanced_accuracy"] <= block["baseline_balanced_accuracy"]
    assert block["neutral_share"] > block["baseline_neutral_share"] > block["gold_neutral_share"]


def test_the_contribution_arithmetic_is_its_own() -> None:
    contribution = committed().contribution
    assert contribution["net"] == contribution["gained"] - contribution["lost"]
    assert contribution["delta"] == pytest.approx(
        contribution["with"] - contribution["without"], abs=1e-3
    )


def test_every_number_the_chapter_quotes_is_in_the_record() -> None:
    """The rule this project runs on, applied to its own newest chapter."""
    chapter = (ROOT / "docs" / "valence.md").read_text(encoding="utf-8")
    report = committed()
    alternatives = report.raw["alternatives"]
    for value in (
        f"{report.contribution['without']:.4f}",
        f"{report.contribution['with']:.4f}",
        f"{report.contribution['p_value']:.4f}",
        f"{alternatives['arousal-neutral']['accuracy']:.4f}",
        f"{alternatives['valence-joy']['accuracy']:.4f}",
        f"{alternatives['learned-stacker']['balanced_accuracy']:.4f}",
    ):
        assert value in chapter, value


# --- what the consistency check refuses ---------------------------------------


def test_a_record_claiming_significance_it_does_not_have_is_caught() -> None:
    """The failure that would matter: a gain asserted without the p value agreeing."""
    broken = altered(contribution={**committed().contribution, "significant": True})
    assert any("disagrees with its own p" in problem for problem in va.check_consistency(broken))


def test_a_record_that_omits_the_contribution_is_caught() -> None:
    broken = altered(contribution={"delta": 0.2})
    assert any("whether it is significant" in problem for problem in va.check_consistency(broken))


def test_class_counts_that_do_not_cover_the_rows_are_caught() -> None:
    assert any("classes cover" in problem for problem in va.check_consistency(altered(samples=99)))


def test_a_gap_that_is_not_the_difference_of_its_own_means_is_caught() -> None:
    raw = json.loads(json.dumps(committed().raw))
    raw["valence"]["separation"]["gap"] = 0.9
    broken = va.ValenceReport(raw=raw, source=Path("x"), digest="0" * 64)
    assert any("not the difference" in problem for problem in va.check_consistency(broken))


def test_a_score_outside_the_sigmoid_range_is_caught() -> None:
    raw = json.loads(json.dumps(committed().raw))
    raw["arousal"]["separation"]["mean_above"] = 1.4
    broken = va.ValenceReport(raw=raw, source=Path("x"), digest="0" * 64)
    assert any("not a sigmoid output" in problem for problem in va.check_consistency(broken))


# --- what it prints -----------------------------------------------------------


def test_describe_says_both_halves() -> None:
    out = "\n".join(va.describe(committed()))
    assert "separates them" in out
    assert "barely separates them" in out
    assert "no measurable gain" in out
    assert "arXiv:2302.14021" in out


def test_the_chapter_quotes_the_record() -> None:
    chapter = (ROOT / "docs" / "valence.md").read_text(encoding="utf-8")
    report = committed()
    assert f"{report.auc_of('valence'):.4f}" in chapter
    assert f"{report.auc_of('arousal'):.4f}" in chapter
    assert "off by default" in chapter


def test_the_checkpoint_is_credited_wherever_it_is_named() -> None:
    """It is somebody else's model, mirrored. MIT asks for attribution."""
    for name in ("docs/valence.md", "README.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "Mendes" in text, name
