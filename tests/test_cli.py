"""The command line is the published interface, so it is tested like one.

Every number in the README is reachable by a command a reader can run. If one of
these commands stops printing what the documentation says it prints, the
documentation is wrong and nobody notices. These tests are what makes that fail
loudly instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_timeline import cli

ROOT = Path(__file__).resolve().parents[1]


# --- argument parsing --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("42", 42.0),
        ("1:30", 90.0),
        ("18:09", 1089.0),
        ("1:00:00", 3600.0),
        ("0:01.5", 1.5),
    ],
)
def test_timestamps_read_as_seconds(text: str, seconds: float) -> None:
    assert cli.parse_timestamp(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "1:xx"])
def test_a_malformed_timestamp_is_rejected(text: str) -> None:
    with pytest.raises(Exception, match="not a timestamp"):
        cli.parse_timestamp(text)


def test_either_side_of_a_window_may_be_left_open() -> None:
    assert cli.parse_window("0:00-18:09") == (0.0, 1089.0)
    assert cli.parse_window("-18:09") == (0.0, 1089.0)
    assert cli.parse_window("5:00-") == (300.0, None)


def test_a_window_that_ends_before_it_starts_is_rejected() -> None:
    with pytest.raises(Exception, match="ends before it starts"):
        cli.parse_window("10:00-5:00")
    with pytest.raises(Exception, match="0:00-18:09"):
        cli.parse_window("18:09")


# --- wer ---------------------------------------------------------------------


def test_wer_prints_the_two_figures_the_readme_quotes(capsys: pytest.CaptureFixture[str]) -> None:
    """The comparison the README leads with, over the window it names."""
    assert cli.main(["wer", "--window", "0:00-18:09"]) == 0
    out = capsys.readouterr().out
    assert "window 0.0-18.1 min" in out
    scores = {line.split()[0]: float(line.split()[1].rstrip("%")) for line in out.splitlines()[1:]}
    assert scores["assemblyai-best"] == pytest.approx(0.81, abs=0.005)
    assert scores["whisper-large-v3"] == pytest.approx(3.33, abs=0.005)


def test_scoring_past_the_annotated_window_warns(capsys: pytest.CaptureFixture[str]) -> None:
    """Silently comparing an annotated stretch against an unannotated one is the
    original bug. Going wider is allowed, going wider quietly is not."""
    assert cli.main(["wer", "--window", "0:00-40:00"]) == 0
    assert "the comparison is uneven" in capsys.readouterr().err


def test_wer_reports_an_empty_benchmark_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["wer", "--benchmarks", str(tmp_path)]) == 1
    assert "no annotated transcripts" in capsys.readouterr().err


# --- errors ------------------------------------------------------------------


def test_errors_prints_the_headline_and_the_hardest_classes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["errors"]) == 0
    out = capsys.readouterr().out
    assert "64,250 samples" in out
    assert "6,454 errors" in out
    assert "accuracy 0.8995" in out
    # Neutral is the hardest class and ALL-CAPS the worst marker; both are
    # claims the README makes in prose.
    assert out.index("Neutral") < out.index("Fear")
    assert "ALL-CAPS word" in out


# --- figures -----------------------------------------------------------------


def test_figures_writes_every_figure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from emotion_timeline.analysis import error_analysis as ea

    assert cli.main(["figures", "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("wrote ") == len(ea.FIGURES)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == sorted(ea.FIGURES)


def test_check_passes_against_the_committed_assets(capsys: pytest.CaptureFixture[str]) -> None:
    """The same command CI runs. If this fails, `assets/` needs regenerating."""
    assert cli.main(["figures", "--out", str(ROOT / "assets"), "--check"]) == 0
    assert "figures current" in capsys.readouterr().out


def test_check_fails_when_a_figure_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["figures", "--out", str(tmp_path), "--check"]) == 1
    err = capsys.readouterr().err
    assert "stale figure" in err
    assert "commit the result" in err


def test_figures_refuses_to_draw_an_inconsistent_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Publishing a contradiction as a picture is worse than publishing nothing."""
    from emotion_timeline.analysis import error_analysis as ea

    broken = json.loads(Path(ea.DEFAULT_REPORT).read_text(encoding="utf-8"))
    broken["total_errors"] = 1
    report = tmp_path / "broken.json"
    report.write_text(json.dumps(broken), encoding="utf-8")

    assert cli.main(["figures", "--report", str(report), "--out", str(tmp_path)]) == 1
    assert "inconsistent report" in capsys.readouterr().err
    assert list(tmp_path.glob("*.png")) == []


# --- the parser itself -------------------------------------------------------


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


@pytest.mark.parametrize("command", ["wer", "errors", "figures"])
def test_every_subcommand_documents_itself(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--help` is meant to be an honest statement of what works."""
    with pytest.raises(SystemExit) as exit_code:
        cli.main([command, "--help"])
    assert exit_code.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: emotion-timeline {command}")
