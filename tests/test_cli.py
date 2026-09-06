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


def all_figures() -> set[str]:
    """Every figure the repository publishes, across every stage."""
    from emotion_timeline.analysis import error_analysis as ea
    from emotion_timeline.data import figures as ds_figures

    return set(ea.FIGURES) | set(ds_figures.FIGURES)


def test_figures_writes_every_figure_from_every_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["figures", "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("wrote ") == len(all_figures())
    assert {p.name for p in tmp_path.glob("*.png")} == all_figures()


def test_check_passes_against_the_committed_assets(capsys: pytest.CaptureFixture[str]) -> None:
    """The same command CI runs. If this fails, `assets/` needs regenerating."""
    assert cli.main(["figures", "--out", str(ROOT / "assets"), "--check"]) == 0
    out = capsys.readouterr().out
    assert f"{len(all_figures())} figures current" in out


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


@pytest.mark.parametrize("command", ["wer", "errors", "figures", "dataset", "build-dataset"])
def test_every_subcommand_documents_itself(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--help` is meant to be an honest statement of what works."""
    with pytest.raises(SystemExit) as exit_code:
        cli.main([command, "--help"])
    assert exit_code.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: emotion-timeline {command}")


# --- dataset -----------------------------------------------------------------


def test_dataset_prints_the_composition_and_the_published_split(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["dataset"]) == 0
    out = capsys.readouterr().out
    assert "552,821 rows across six corpora" in out
    assert "419,180 reproducible + 9,151 synthetic" in out
    assert "428,331 rows" in out
    # Joy first, Neutral last: the imbalance is the point of the section.
    assert out.index("Joy") < out.index("Neutral")
    assert "GoEmotions" in out and "51,521 dropped" in out


def test_dataset_refuses_an_inconsistent_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.data import build as ds

    raw = json.loads(Path(ds.DEFAULT_RECORD).read_text(encoding="utf-8"))
    raw["rows"] = 1
    record = tmp_path / "broken.json"
    record.write_text(json.dumps(raw), encoding="utf-8")

    assert cli.main(["dataset", "--build-record", str(record)]) == 1
    assert "inconsistent record" in capsys.readouterr().err


def test_build_dataset_checks_itself_against_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A build that does not reproduce the committed funnel is a failure, not a note."""
    from emotion_timeline.data import build as ds
    from tests.test_dataset import corpus

    monkeypatch.setattr(ds, "load_source", lambda cache_dir=None: corpus())
    out = tmp_path / "nested" / "dataset.csv"
    assert cli.main(["build-dataset", "--out", str(out)]) == 1

    printed = capsys.readouterr()
    assert "differs from the record" in printed.err
    assert "no longer reproduces" in printed.err
    assert "load" in printed.out
    # The CSV is only written once the build agrees with the record.
    assert not out.exists()


def test_build_dataset_writes_the_csv_when_the_build_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.data import build as ds
    from tests.test_dataset import corpus

    monkeypatch.setattr(ds, "load_source", lambda cache_dir=None: corpus())
    monkeypatch.setattr(ds, "compare", lambda record, expected: [])
    out = tmp_path / "dataset.csv"
    assert cli.main(["build-dataset", "--out", str(out)]) == 0
    assert "matches build-record.json exactly" in capsys.readouterr().out
    assert out.exists() and out.read_text(encoding="utf-8").startswith("text,label,")


def test_build_dataset_explains_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`datasets` is an optional extra, so the failure has to say how to fix it."""
    from emotion_timeline.data import build as ds

    def no_datasets(cache_dir: object = None) -> None:
        raise ImportError("No module named 'datasets'")

    monkeypatch.setattr(ds, "load_source", no_datasets)
    assert cli.main(["build-dataset"]) == 1
    assert "--extra data" in capsys.readouterr().err
