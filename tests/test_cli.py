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
    from emotion_timeline.model import figures as model_figures
    from emotion_timeline.selection import figures as sel_figures

    return (
        set(ea.FIGURES)
        | set(ds_figures.FIGURES)
        | set(sel_figures.FIGURES)
        | set(model_figures.FIGURES)
    )


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


@pytest.mark.parametrize(
    "command", ["wer", "errors", "figures", "dataset", "build-dataset", "model", "models"]
)
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


# --- models ------------------------------------------------------------------


def test_models_prints_the_audit_the_readme_summarises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "8 submitted rows, 101 logged runs" in out
    assert "79 of 101 accuracies" in out
    assert "27 distinct divisors" in out
    assert "6,044,800 samples" in out
    # Both halves of the finding: the ranking moves under the other average,
    # and the two headline runs were scored on different data.
    assert "gru #6" in out and "3 places up" in out
    assert "pytorch_mlp_gpu" in out and "divisible by 3,200" in out
    assert "distilbert" in out and "divisible by 1,889" in out
    assert "Neutral capped at 10,000" in out and "Neutral capped at  6,000" in out
    assert "17 of the 101 runs are provably a model that answered with one class" in out


def test_models_refuses_an_inconsistent_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.selection import runs as sel

    raw = json.loads(Path(sel.DEFAULT_SUBMITTED_LOG).read_text(encoding="utf-8"))
    raw["rows"][0]["recall_weighted"] = 0.5
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(raw), encoding="utf-8", newline="\n")

    assert cli.main(["models", "--submitted-log", str(broken)]) == 1
    assert "inconsistent record" in capsys.readouterr().err


def test_figures_refuses_to_draw_an_inconsistent_selection_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rendering gate is per stage, and this stage sits behind it too."""
    from emotion_timeline.selection import runs as sel

    raw = json.loads(Path(sel.DEFAULT_SUBMITTED_LOG).read_text(encoding="utf-8"))
    raw["rows"][3]["neutral_cap"] = 4000
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(raw), encoding="utf-8", newline="\n")

    assert cli.main(["figures", "--submitted-log", str(broken), "--out", str(tmp_path)]) == 1
    assert "inconsistent report" in capsys.readouterr().err
    assert list(tmp_path.glob("*.png")) == []


# --- model -------------------------------------------------------------------


def test_model_prints_what_separates_the_two_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["model"]) == 0
    out = capsys.readouterr().out
    # the three evaluations and the six-class average
    assert "64,250 samples" in out and "0.8995" in out
    assert "(6 of 7 classes)" in out
    assert "over all seven classes would be 0.2389" in out
    # the identities that tell the records apart
    assert "single-label" in out
    assert "1.025 true labels per sample" in out
    assert "30,522 tokens" in out and "128,100" in out
    # the mislabelled table and the stress-test control
    assert "called Neutral    are Joy" in out
    assert "beats the control" in out
    assert "score exactly zero over 561 samples" in out
    assert "2,746-2,750" in out


def test_model_refuses_an_inconsistent_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.model import card as mc

    raw = json.loads(Path(mc.DEFAULT_CARD).read_text(encoding="utf-8"))
    raw["card"]["evaluations"]["held_out"]["accuracy"] = 0.5
    record = tmp_path / "broken.json"
    record.write_text(json.dumps(raw), encoding="utf-8")

    assert cli.main(["model", "--card-metrics", str(record)]) == 1
    assert "inconsistent record" in capsys.readouterr().err


# --- split -------------------------------------------------------------------


def test_split_prints_the_committed_split(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["split"]) == 0
    out = capsys.readouterr().out
    assert "419,180 rows, split 70% / 15% / 15%" in out
    assert any(line.strip().startswith("test") and "62,877" in line for line in out.splitlines())
    assert "185 held-out rows share their text with a training row" in out


def test_split_refuses_an_inconsistent_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.training import splits

    raw = json.loads(Path(splits.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    raw["source_rows"] = 3
    manifest = tmp_path / "broken.json"
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    assert cli.main(["split", "--manifest", str(manifest)]) == 1
    assert "inconsistent record" in capsys.readouterr().err


def test_split_writes_and_then_verifies_a_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole round trip, on sixty rows rather than 419,180."""
    dataset = tmp_path / "tiny.csv"
    rows = ["text,label,source"]
    rows += [f"row {index},a,test" for index in range(40)]
    rows += [f"row {index},b,test" for index in range(40, 60)]
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"

    assert cli.main(["split", "--manifest", str(manifest), "--write", str(dataset)]) == 0
    assert cli.main(["split", "--manifest", str(manifest), "--verify", str(dataset)]) == 0
    assert "reproduces every split exactly" in capsys.readouterr().out


def test_split_reports_a_dataset_that_no_longer_gives_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "tiny.csv"
    header = "text,label,source"
    rows = [f"row {index},a,test" for index in range(20)]
    dataset.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    assert cli.main(["split", "--manifest", str(manifest), "--write", str(dataset)]) == 0

    changed = tmp_path / "changed.csv"
    rows[0] = "something else,a,test"
    changed.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    assert cli.main(["split", "--manifest", str(manifest), "--verify", str(changed)]) == 1
    assert "does not give the committed split" in capsys.readouterr().err


def test_split_rejects_a_csv_missing_a_column(tmp_path: Path) -> None:
    dataset = tmp_path / "wrong.csv"
    dataset.write_text("text,label\nhello,a\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="source"):
        cli.main(["split", "--verify", str(dataset)])


# --- preflight ---------------------------------------------------------------


def fake_device() -> object:
    from emotion_timeline.training import preflight

    return preflight.Device(
        name="NVIDIA GeForce RTX 5070",
        capability=(12, 0),
        total_mib=12_227,
        free_mib=11_000,
        torch_version="2.11.0+cu128",
        cuda_version="12.8",
        arch_list=("sm_90", "sm_120"),
    )


def test_preflight_reports_a_card_it_can_train_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.training import preflight

    monkeypatch.setattr(preflight, "probe", fake_device)
    monkeypatch.setattr(preflight, "smoke", lambda: None)
    assert cli.main(["preflight"]) == 0
    out = capsys.readouterr().out
    assert "RTX 5070" in out
    assert "sm_120" in out
    assert "ready to train" in out


def test_preflight_fails_with_the_fix_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.training import preflight

    monkeypatch.setattr(preflight, "probe", lambda: None)
    monkeypatch.setattr(preflight, "smoke", lambda: None)
    assert cli.main(["preflight"]) == 1
    err = capsys.readouterr().err
    assert "cannot train here" in err
    assert "--extra model" in err


def test_preflight_fails_when_the_multiply_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.training import preflight

    monkeypatch.setattr(preflight, "probe", fake_device)
    monkeypatch.setattr(preflight, "smoke", lambda: "no kernel image is available")
    assert cli.main(["preflight"]) == 1
    assert "matrix multiply on the device failed" in capsys.readouterr().err


def test_preflight_takes_the_memory_a_smaller_run_needs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from emotion_timeline.training import preflight

    monkeypatch.setattr(preflight, "probe", fake_device)
    monkeypatch.setattr(preflight, "smoke", lambda: None)
    assert cli.main(["preflight", "--need-mib", "99999"]) == 1
    assert "11,000 MiB free" in capsys.readouterr().err
