"""Fetching a checkpoint, checked with no network and no 268 MB file.

``urllib`` takes ``file://`` URLs, so every path here -- including the download
and the rename-on-success -- runs against ``tmp_path``. The one that matters is
the mismatch: a checkpoint that is not the one this study measured has to become
an error, not a cache entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_timeline import weights


def source(tmp_path: Path, content: bytes = b"weights") -> str:
    path = tmp_path / "source.safetensors"
    path.write_bytes(content)
    return path.as_uri()


@pytest.mark.parametrize("version", ["distilbert-v1", "rubert-v1"])
def test_every_shipped_checkpoint_has_a_digest_recorded(version: str) -> None:
    """Without one, a substituted asset predicts something else and says nothing."""
    spec = weights.spec_for(version)
    assert spec.sha256 is not None
    assert len(spec.sha256) == 64
    assert spec.url.endswith(".safetensors")


def test_the_registry_lists_what_it_knows() -> None:
    assert weights.list_versions() == ["distilbert-v1", "rubert-v1"]


def test_a_version_nobody_has_says_what_is_available() -> None:
    with pytest.raises(KeyError, match="distilbert-v1"):
        weights.get_weights("no-such-model")


def test_a_bare_url_becomes_a_spec_with_no_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(weights.REGISTRY, "bare", "file:///tmp/x")
    assert weights.spec_for("bare") == weights.WeightSpec(url="file:///tmp/x", sha256=None)


def test_a_download_lands_and_is_not_fetched_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        weights.REGISTRY,
        "local",
        weights.WeightSpec(
            url=source(tmp_path), sha256=weights.digest_of(tmp_path / "source.safetensors")
        ),
    )
    cache = tmp_path / "cache"
    first = weights.get_weights("local", cache)
    assert first.read_bytes() == b"weights"

    # A second call must not go back to the URL, so removing the source proves it.
    (tmp_path / "source.safetensors").unlink()
    assert weights.get_weights("local", cache) == first


def test_a_checkpoint_that_is_not_the_right_one_is_deleted_rather_than_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaving it would make the next call a cache hit on a bad file."""
    monkeypatch.setitem(
        weights.REGISTRY, "local", weights.WeightSpec(url=source(tmp_path), sha256="0" * 64)
    )
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        weights.get_weights("local", cache)
    assert not (cache / "local.safetensors").exists()


def test_a_spec_with_no_digest_is_taken_on_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(weights.REGISTRY, "local", source(tmp_path))
    assert weights.get_weights("local", tmp_path / "cache").read_bytes() == b"weights"


def test_a_landing_page_served_instead_of_a_file_is_refused() -> None:
    with pytest.raises(RuntimeError, match="probably missing or not public"):
        weights.reject_html("http://example/x", "text/html; charset=utf-8")


def test_a_binary_content_type_passes() -> None:
    """It raises or it does not; these are the content types that must not."""
    weights.reject_html("http://example/x", "application/octet-stream")
    weights.reject_html("http://example/x", "")


def test_a_partial_download_never_becomes_the_cached_file(tmp_path: Path) -> None:
    out = tmp_path / "cache" / "x.safetensors"
    weights.download(source(tmp_path), out)
    assert out.read_bytes() == b"weights"
    assert not out.with_suffix(".safetensors.part").exists()


def test_the_cache_directory_can_be_moved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMOTION_TIMELINE_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert weights.get_cache_dir() == tmp_path / "elsewhere"
    assert weights.get_cache_dir().is_dir()


def test_without_the_variable_it_falls_back_to_the_home_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMOTION_TIMELINE_CACHE_DIR", raising=False)
    assert weights.get_cache_dir().parts[-2:] == ("emotion-timeline", "models")


def test_the_digest_is_the_files_own(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "x"
    path.write_bytes(b"abc" * 1000)
    assert weights.digest_of(path) == hashlib.sha256(b"abc" * 1000).hexdigest()
