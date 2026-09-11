"""Reading keys from the environment, and the precedence rule that matters.

The rule worth pinning is that an exported variable beats a file on disk. Get it
backwards and a stale `.env` on a developer's laptop silently shadows the key a
CI runner injected, which is the kind of bug that looks like a service outage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_timeline import credentials as creds


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    """The common case: nobody has a .env, and no command needs one."""
    assert creds.load_env_file(tmp_path / "absent") == {}


def test_values_are_read_into_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text("EXAMPLE_KEY=abc123\n", encoding="utf-8")
    monkeypatch.delenv("EXAMPLE_KEY", raising=False)

    assert creds.load_env_file(path) == {"EXAMPLE_KEY": "abc123"}
    assert creds.get("EXAMPLE_KEY") == "abc123"


def test_an_exported_variable_beats_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precedence that keeps a stale file from shadowing a CI secret."""
    path = tmp_path / ".env"
    path.write_text("EXAMPLE_KEY=from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("EXAMPLE_KEY", "from-the-environment")

    assert creds.load_env_file(path) == {}
    assert creds.get("EXAMPLE_KEY") == "from-the-environment"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("EXAMPLE_KEY=plain", "plain"),
        ('EXAMPLE_KEY="double quoted"', "double quoted"),
        ("EXAMPLE_KEY='single quoted'", "single quoted"),
        ("export EXAMPLE_KEY=exported", "exported"),
        ("  EXAMPLE_KEY = spaced  ", "spaced"),
        ("EXAMPLE_KEY=has=equals=inside", "has=equals=inside"),
    ],
)
def test_the_shapes_a_line_can_take(
    line: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text(line + "\n", encoding="utf-8")
    monkeypatch.delenv("EXAMPLE_KEY", raising=False)
    assert creds.load_env_file(path) == {"EXAMPLE_KEY": expected}


def test_comments_blanks_and_junk_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed line must not stop a command that did not need the file."""
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n\n   \nnot-a-pair\nEXAMPLE_KEY=kept\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("EXAMPLE_KEY", raising=False)
    assert creds.load_env_file(path) == {"EXAMPLE_KEY": "kept"}


def test_an_empty_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env.example` ships every name with a blank value; copying it sets nothing."""
    monkeypatch.setenv("EXAMPLE_KEY", "   ")
    assert creds.get("EXAMPLE_KEY") is None


def test_a_missing_key_explains_how_to_get_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(creds.MissingCredential) as raised:
        creds.require("HF_TOKEN")

    message = str(raised.value)
    assert "huggingface.co/settings/tokens" in message
    assert ".env.example" in message
    # And it says the truth about this repository: nothing committed needs it.
    assert "not needed for anything committed here" in message.lower()


def test_an_unknown_variable_still_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOT_A_KNOWN_KEY", raising=False)
    with pytest.raises(creds.MissingCredential, match="NOT_A_KNOWN_KEY is not set"):
        creds.require("NOT_A_KNOWN_KEY")


def test_require_returns_the_key_when_it_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-value")
    assert creds.require("HF_TOKEN") == "hf-value"


# --- the repository's own claim ------------------------------------------------


def test_the_example_file_lists_every_known_variable_and_no_values() -> None:
    """`.env.example` is committed, so it must never carry a value."""
    root = Path(__file__).resolve().parents[1]
    lines = (root / ".env.example").read_text(encoding="utf-8").splitlines()
    pairs = [line for line in lines if "=" in line and not line.startswith("#")]

    assert {line.partition("=")[0] for line in pairs} == set(creds.KNOWN)
    for line in pairs:
        assert line.endswith("="), f"{line} carries a value"


def test_no_committed_command_requires_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim the README makes: the whole study reproduces with no account.

    Every published number comes from files under benchmarks/, so each read-only
    command has to run with the environment stripped of every key this project
    knows about.
    """
    from emotion_timeline import cli

    for variable in creds.KNOWN:
        monkeypatch.delenv(variable, raising=False)

    for command in (["dataset"], ["errors"], ["models"], ["model"], ["wer"]):
        assert cli.main(command) == 0, command
