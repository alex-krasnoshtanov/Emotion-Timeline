"""Reading API keys from the environment, and failing usefully when they are absent.

No stage committed so far needs a credential: the source corpus is public and
ungated, and every published number derives from files in ``benchmarks/``. That
is deliberate and worth keeping -- a reader can reproduce the whole study with no
account anywhere.

The stages still to land are not all like that. The prompted-LLM baseline calls a
hosted model and the transcriber adapters call AssemblyAI, so this module exists
to give those one way in, before anyone is tempted to paste a key into a file
"just to test it".

**Why this is here at all.** The upstream coursework notebook opened with

    os.environ["HF_TOKEN"] = "hf_..."

a live read-scoped token on a personal account, committed to a repository every
member of the university organisation could read. A read token is enough to pull
any private or gated repository as its owner. The token was also unnecessary:
the dataset it was fetching is public. The `no-credentials` pre-commit hook now
refuses that shape of string, and this module is the thing to reach for instead.

Precedence is environment first, ``.env`` second. A key set in the shell or by a
CI runner always wins over a file on disk, because the file is the convenience
and the environment is the real configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENV_FILE = Path(".env")


@dataclass(frozen=True, slots=True)
class Credential:
    """One API key: what reads it, what it is for, and where to get one."""

    variable: str
    purpose: str
    obtain_from: str
    needed_by: str

    @property
    def message(self) -> str:
        """What a reader sees when it is missing. Says what to do, not what broke."""
        return (
            f"{self.variable} is not set.\n"
            f"  {self.purpose}\n"
            f"  Get one from {self.obtain_from}\n"
            f"  Then either export {self.variable}=... or put it in a .env file "
            f"(see .env.example).\n"
            f"  Needed by: {self.needed_by}"
        )


KNOWN: dict[str, Credential] = {
    "HF_TOKEN": Credential(
        variable="HF_TOKEN",
        purpose=(
            "Hugging Face access token. Not needed for anything committed here -- "
            "the source corpus is public and ungated -- only for private or gated "
            "repositories."
        ),
        obtain_from="https://huggingface.co/settings/tokens",
        needed_by="nothing yet; read natively by huggingface_hub when set",
    ),
    "ASSEMBLYAI_API_KEY": Credential(
        variable="ASSEMBLYAI_API_KEY",
        purpose="AssemblyAI key, for transcribing new audio with the winning system.",
        obtain_from="https://www.assemblyai.com/app/account",
        needed_by="transcribing new audio; every committed command needs none",
    ),
}


class MissingCredential(RuntimeError):
    """Raised instead of a bare KeyError, so the message says how to fix it."""


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """Read ``KEY=value`` lines into the environment, without overriding it.

    Deliberately not python-dotenv. The core install is four packages and a
    twenty-line parser is not worth a fifth, especially one that would only ever
    run on a developer's machine.

    Returns the names it set, which is what the tests assert on. Blank lines,
    ``#`` comments and a leading ``export`` are ignored; surrounding single or
    double quotes are stripped; a line with no ``=`` is skipped rather than
    raising, because a malformed .env should not stop the command working when
    the variable is already exported.
    """
    file = Path(path)
    if not file.exists():
        return {}

    applied: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.removeprefix("export ").partition("=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # The environment wins: a CI secret must not be shadowed by a stale file.
        if name and name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def get(variable: str) -> str | None:
    """The key if it is set and non-empty, otherwise ``None``."""
    value = os.environ.get(variable, "").strip()
    return value or None


def require(variable: str) -> str:
    """The key, or a :class:`MissingCredential` saying how to obtain one."""
    value = get(variable)
    if value is not None:
        return value
    known = KNOWN.get(variable)
    raise MissingCredential(known.message if known else f"{variable} is not set.")
