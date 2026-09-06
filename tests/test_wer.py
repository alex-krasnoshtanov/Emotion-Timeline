"""The arithmetic, and the mistake the harness exists to prevent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from emotion_timeline.stt import wer

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks" / "stt"


def frame(rows: list[tuple[float, float, str, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(wer.REQUIRED_COLUMNS))


def test_reference_length_adds_back_deletions_and_removes_insertions() -> None:
    # Ten hypothesis words, two of which the system invented, and three
    # reference words it dropped: the reference held 10 - 2 + 3 = 11.
    result = wer.score(frame([(0.0, 1.0, " ".join("abcdefghij"), 0, 2, 3)]), "t")
    assert result.hypothesis_tokens == 10
    assert result.reference_tokens == 11
    assert result.errors == 5
    assert result.wer == pytest.approx(100 * 5 / 11)


def test_perfect_transcript_scores_zero() -> None:
    assert wer.score(frame([(0.0, 1.0, "one two three", 0, 0, 0)]), "t").wer == 0.0


def test_empty_window_is_zero_not_a_crash() -> None:
    result = wer.score(frame([]), "t")
    assert result.segments == 0
    assert result.wer == 0.0


def test_window_filters_on_start_time() -> None:
    f = frame([(0.0, 5.0, "a b", 1, 0, 0), (60.0, 65.0, "c d", 0, 0, 0)])
    assert len(wer.in_window(f, 0.0, 30.0)) == 1
    assert len(wer.in_window(f, 0.0, None)) == 2


def test_annotated_extent_reports_the_last_marked_error() -> None:
    f = frame([(0.0, 5.0, "a", 1, 0, 0), (600.0, 605.0, "b", 0, 0, 0)])
    assert wer.annotated_extent(f) == 0.0


# --- the committed benchmark ------------------------------------------------


@pytest.fixture(scope="module")
def systems() -> dict[str, pd.DataFrame]:
    return {
        "whisper-large-v3": wer.load(BENCHMARKS / "whisper-large-v3.csv"),
        "assemblyai-best": wer.load(BENCHMARKS / "assemblyai-best.csv"),
    }


def test_both_systems_transcribe_the_same_recording(systems: dict[str, pd.DataFrame]) -> None:
    ends = {name: f["end_s"].max() for name, f in systems.items()}
    assert max(ends.values()) - min(ends.values()) < 2.0, ends


def test_published_numbers_reproduce(systems: dict[str, pd.DataFrame]) -> None:
    # 18.1 minutes: as far as the Whisper annotation runs. Both systems are
    # scored over it, which is what makes the two numbers comparable.
    results = {r.system: r for r in wer.compare(systems, end_s=1089.0)}
    assert results["whisper-large-v3"].wer == pytest.approx(3.33, abs=0.01)
    assert results["assemblyai-best"].wer == pytest.approx(0.81, abs=0.01)
    # Near-identical reference lengths are what make the two rates comparable
    # at all: 2,105 tokens against 2,101 over the same 18.1 minutes.
    assert (
        abs(
            results["whisper-large-v3"].reference_tokens
            - results["assemblyai-best"].reference_tokens
        )
        < 25
    )


def test_the_window_is_what_makes_it_a_comparison(systems: dict[str, pd.DataFrame]) -> None:
    """Row counts are not a window, and the difference is not small.

    Taking 240 rows from each compares 18 minutes of Whisper against 41
    minutes of AssemblyAI, because AssemblyAI emits longer segments. That is
    how the original analysis reached 0.61% for AssemblyAI instead of 0.82%.
    """
    by_rows = wer.score(systems["assemblyai-best"].head(240), "rows")
    by_time = wer.score(wer.in_window(systems["assemblyai-best"], end_s=1089.0), "time")
    assert by_rows.window_end_s > 2400  # past 40 minutes
    assert by_time.window_end_s < 1200  # inside 20
    assert by_rows.wer < by_time.wer  # the longer slice flatters it
