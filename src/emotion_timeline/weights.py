"""Downloading a trained checkpoint, and refusing one that is not the right file.

The weights are 268 MB and the repository is meant to stay clonable in seconds, so
they ship as a GitHub release asset rather than in git. That only works if a
reader can tell they got the file this study measured, which is what the digest
is for: a download whose SHA-256 differs is deleted rather than cached, because a
substituted checkpoint does not fail, it quietly predicts something else.

Ported from ``cv_pipeline/weights.py`` in CV-Pipeline-Deployment-Platform, as
``CLAUDE.md`` says to, with one change. That version uses ``requests``; the core
install here is numpy, pandas, matplotlib and scipy, and this is not the stage to
add a fifth. ``urllib.request`` streams, follows redirects and exposes the content
type, which is all the download path used it for -- and it takes ``file://``
URLs, so the whole thing is testable without a network.

Verification runs after a download and not on a cache hit. Re-hashing 268 MB on
every process start costs more than it protects, and writing to a temporary name
and renaming on success already rules out a half-written cache entry.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

RELEASE = "https://github.com/alex-krasnoshtanov/Emotion-Timeline/releases/download"


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """Where a checkpoint lives and what it has to hash to."""

    url: str
    #: ``None`` disables the check, which is only right for a local or throwaway
    #: URL whose digest nobody has recorded.
    sha256: str | None = None


#: Version string -> where to get it. A bare string stands in for a spec with no
#: digest. Every entry here is an asset on a public release, so no credential is
#: involved and `test_no_committed_command_requires_a_credential` stays true.
REGISTRY: dict[str, WeightSpec | str] = {
    "distilbert-v1": WeightSpec(
        url=f"{RELEASE}/weights-v1/emotion-timeline-distilbert-v1.safetensors",
        sha256="a4095f789c024d87ffc1f1d362d698263c5a4804e05e6c7972f7cd5a1a5f3520",
    ),
}

_DEFAULT_CACHE = Path.home() / ".cache" / "emotion-timeline" / "models"


def get_cache_dir() -> Path:
    """Where downloaded checkpoints live. ``EMOTION_TIMELINE_CACHE_DIR`` overrides it."""
    raw = os.getenv("EMOTION_TIMELINE_CACHE_DIR")
    cache = Path(raw).expanduser() if raw else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def list_versions() -> list[str]:
    """Every checkpoint this package knows how to fetch."""
    return sorted(REGISTRY)


def spec_for(version: str) -> WeightSpec:
    """``REGISTRY[version]`` as a spec, wrapping a bare URL."""
    entry = REGISTRY[version]
    return entry if isinstance(entry, WeightSpec) else WeightSpec(url=entry)


def get_weights(version: str, cache_dir: str | Path | None = None) -> Path:
    """The local path to a checkpoint, downloading and verifying it if it is not cached."""
    if version not in REGISTRY:
        raise KeyError(
            f"unknown version {version!r}. Known: {list_versions()}. "
            "Add an entry to REGISTRY in emotion_timeline/weights.py."
        )
    cache = Path(cache_dir) if cache_dir else get_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{version}.safetensors"
    if target.exists():
        return target

    spec = spec_for(version)
    download(spec.url, target)
    if spec.sha256:
        verify(target, spec.sha256)
    return target


def digest_of(path: str | Path) -> str:
    """The SHA-256 of a file, read a megabyte at a time."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: str | Path, expected: str) -> None:
    """Check a downloaded file, and delete it if it is the wrong one.

    Deleting is the part that matters: leaving it would make the next call a cache
    hit, so one bad download would poison every run after it.
    """
    target = Path(path)
    actual = digest_of(target)
    if actual != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {target.name}: expected {expected}, got {actual}. "
            "The download was corrupted or the release asset was replaced. The bad "
            "file has been deleted; retry, and if it persists check the digest "
            "pinned in emotion_timeline/weights.py."
        )


def reject_html(url: str, content_type: str) -> None:
    """Refuse a landing page served where a binary was asked for.

    A wrong tag, a missing asset or a private release all answer with HTML and a
    200, so without this the cache fills with a web page named like a checkpoint.
    """
    if "html" in content_type.lower():
        raise RuntimeError(
            f"expected a binary response from {url} but got content-type "
            f"{content_type!r}. The release asset is probably missing or not public."
        )


def download(url: str, target: str | Path) -> Path:
    """Stream a URL to a file, writing to a temporary name and renaming on success."""
    import urllib.request

    out = Path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response:
        reject_html(url, response.headers.get("Content-Type", ""))
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, 1024 * 1024)
    temporary.replace(out)
    return out
