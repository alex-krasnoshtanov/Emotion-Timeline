"""The browser front end: its validation, its job state machine, its routes.

Two things here are worth more than the rest. The first is that **the page works
with no GPU and no network** — `/api/demo` serves the committed timeline, and a
test asserts it, because an interface nobody can open on an ordinary laptop is
not an interface. The second is the validation: a URL goes to a downloader and a
file goes to ffmpeg, so those two inputs are the trust boundary and each of the
ways they can be abused gets a named test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from emotion_timeline.web import jobs as job_module
from emotion_timeline.web.jobs import Job, JobStore, State

pytest.importorskip("fastapi", reason="the browser front end is the `web` extra")

import threading

from fastapi.testclient import TestClient

from emotion_timeline.web import app as web

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store() -> JobStore:
    return JobStore()


@pytest.fixture
def client(store: JobStore, tmp_path: Path) -> TestClient:
    return TestClient(web.build_app(downloads=tmp_path, store=store))


# --- the trust boundary -------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "file:///etc/passwd",
        "ftp://example.com/x.mp4",
        "javascript:alert(1)",
        "data:video/mp4;base64,AAAA",
        "",
        "   ",
        "not a url",
    ],
)
def test_a_url_we_will_not_hand_to_a_downloader_is_refused(raw: str) -> None:
    with pytest.raises(ValueError, match=r"link|http"):
        web.validate_url(raw)


def test_an_ordinary_link_is_accepted_and_trimmed() -> None:
    assert web.validate_url("  https://example.com/watch?v=x  ") == "https://example.com/watch?v=x"


@pytest.mark.parametrize(
    "name",
    [
        "../../../../etc/passwd",
        "video.mp4.exe",
        "notes.txt",
        "archive.zip",
        "no-extension",
        "x.sh",
        "",
    ],
)
def test_an_upload_we_will_not_hand_to_ffmpeg_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="unsupported file type"):
        web.safe_suffix(name)


def test_only_the_extension_of_an_upload_is_ever_used() -> None:
    """The name is attacker-controlled; the suffix is all that survives."""
    assert web.safe_suffix("../../etc/evil.mp4") == ".mp4"
    assert web.safe_suffix("HOLIDAY.MOV") == ".mov"


@pytest.mark.parametrize("gap", [-1.0, 61.0, 1e9])
def test_a_nonsensical_scene_gap_is_refused(gap: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 60"):
        web.validate_gap(gap)


def test_the_server_binds_to_loopback_by_default() -> None:
    """It hands URLs to yt-dlp. That is a tool you run for yourself."""
    import inspect

    assert inspect.signature(web.serve).parameters["host"].default == "127.0.0.1"


# --- the job state machine ----------------------------------------------------


def test_a_job_runs_from_queued_to_done() -> None:
    job = Job(id="a", source="x", kind="link")
    assert job.state is State.QUEUED and not job.state.finished
    job.start()
    # Through summary(), because that is what the polling endpoint actually
    # returns -- and because mypy narrows the attribute to its dataclass default.
    assert job.summary()["state"] == "running"
    assert not job.summary()["finished"]
    job.finish({"timeline": [1, 2, 3]})
    assert job.state.finished
    assert job.summary()["scenes"] == 3


def test_a_failed_job_reports_the_message_and_not_the_stack() -> None:
    job = Job(id="a", source="x", kind="link")
    job.fail("Traceback (most recent call last):\n  File x\nRuntimeError: no speech here")
    assert job.error == "RuntimeError: no speech here"
    assert job.state is State.FAILED


def test_a_failure_with_nothing_to_say_still_says_something() -> None:
    job = Job(id="a", source="x", kind="link")
    job.fail("   ")
    assert job.error == "failed"


def test_the_progress_log_is_capped_and_keeps_the_tail() -> None:
    """A long transcription logs per batch, so this is unbounded otherwise."""
    job = Job(id="a", source="x", kind="link")
    for index in range(job_module.MAX_LOG_LINES + 50):
        job.say(f"line {index}")
    assert len(job.log) == job_module.MAX_LOG_LINES
    assert job.log[-1] == f"line {job_module.MAX_LOG_LINES + 49}"


def test_the_store_evicts_finished_jobs_once_it_is_full(store: JobStore) -> None:
    small = JobStore(limit=3)
    made = [small.create(f"source {index}", "link") for index in range(3)]
    for job in made:
        job.finish({"timeline": []})
    newest = small.create("source 3", "link")
    assert len(small) == 3
    assert small.get(made[0].id) is None
    assert small.get(newest.id) is not None


def test_a_running_job_is_never_evicted() -> None:
    """A worker thread holds it; forgetting it would leave that thread writing to nothing."""
    small = JobStore(limit=1)
    running = small.create("busy", "link")
    running.start()
    second = small.create("also busy", "link")
    second.start()
    assert len(small) == 2
    assert small.get(running.id) is not None


# --- the routes ---------------------------------------------------------------


def test_the_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Emotion Timeline" in response.text
    assert "<svg" in response.text


def test_the_page_takes_its_palette_from_the_one_that_draws_the_figures(
    client: TestClient,
) -> None:
    """One source of truth, so the browser and the PNG cannot disagree."""
    from emotion_timeline import figures
    from emotion_timeline.data.labels import EMOTIONS

    meta = client.get("/api/meta").json()
    assert meta["colours"] == dict(figures.EMOTION_COLOURS)
    assert set(meta["emotions"]) == set(EMOTIONS)
    assert meta["default_gap"] == 1.0


def test_the_committed_example_loads_with_no_gpu_and_no_network(client: TestClient) -> None:
    """The claim that makes this an interface rather than a demo reel."""
    payload = client.get("/api/demo").json()
    assert len(payload["timeline"]) == 47
    assert payload["header"]["segments"] == 316
    first = payload["timeline"][0]
    assert {"start_s", "emotion", "confidence", "second_opinion", "agreed", "text"} <= set(first)
    assert first["text"]


def test_the_caveat_travels_with_the_agreement_rate(client: TestClient) -> None:
    payload = client.get("/api/demo").json()
    assert "not an accuracy" in payload["header"]["agreement"]["caveat"]
    assert "not an accuracy" in client.get("/api/meta").json()["caveat"]


def test_asking_for_both_a_link_and_a_file_is_refused(client: TestClient, tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not really a video")
    with media.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            data={"url": "https://example.com/x"},
            files={"file": ("clip.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 400
    assert "not both" in response.json()["detail"]


def test_asking_for_neither_is_refused(client: TestClient) -> None:
    response = client.post("/api/jobs", data={"url": "   "})
    assert response.status_code == 400


def test_a_bad_link_is_refused_before_any_download_starts(
    client: TestClient, store: JobStore
) -> None:
    response = client.post("/api/jobs", data={"url": "file:///etc/passwd"})
    assert response.status_code == 400
    assert "http" in response.json()["detail"]
    assert len(store) == 0


def test_a_bad_upload_is_refused_before_anything_is_written(
    client: TestClient, store: JobStore, tmp_path: Path
) -> None:
    response = client.post(
        "/api/jobs", files={"file": ("payload.sh", b"#!/bin/sh", "application/x-sh")}
    )
    assert response.status_code == 400
    assert len(store) == 0
    assert not list(tmp_path.glob("*.sh"))


def test_submitting_a_link_starts_a_job(
    client: TestClient, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Any] = []
    monkeypatch.setattr(
        threading,
        "Thread",
        lambda **kwargs: type("Fake", (), {"start": lambda self: started.append(kwargs)})(),
    )
    response = client.post("/api/jobs", data={"url": "https://example.com/watch?v=x", "gap": "2"})
    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "link"
    assert body["state"] == "queued"
    assert len(store) == 1
    assert started and started[0]["args"][2] == 2.0


def test_an_upload_is_written_under_a_generated_name(
    client: TestClient, store: JobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        threading, "Thread", lambda **kwargs: type("Fake", (), {"start": lambda self: None})()
    )
    response = client.post(
        "/api/jobs", files={"file": ("../../holiday.MP4", b"bytes", "video/mp4")}
    )
    assert response.status_code == 202
    assert response.json()["kind"] == "file"
    written = list(tmp_path.glob("*.mp4"))
    assert len(written) == 1
    assert "holiday" not in written[0].name


def test_an_unknown_job_says_so_rather_than_crashing(client: TestClient) -> None:
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/timeline").status_code == 404


def test_asking_for_a_result_that_is_not_ready_says_which_state_it_is_in(
    client: TestClient, store: JobStore
) -> None:
    job = store.create("https://example.com/x", "link")
    job.start()
    response = client.get(f"/api/jobs/{job.id}/timeline")
    assert response.status_code == 409
    assert "running" in response.json()["detail"]


def test_a_finished_job_hands_back_its_timeline(client: TestClient, store: JobStore) -> None:
    job = store.create("https://example.com/x", "link")
    job.finish({"header": {"scenes": 1}, "timeline": [{"emotion": "Joy"}]})
    assert client.get(f"/api/jobs/{job.id}").json()["state"] == "done"
    assert client.get(f"/api/jobs/{job.id}/timeline").json()["timeline"][0]["emotion"] == "Joy"


def test_the_status_endpoint_never_ships_the_whole_timeline(
    client: TestClient, store: JobStore
) -> None:
    """The page polls this every 1.5s; the rows go over once, at the end."""
    job = store.create("https://example.com/x", "link")
    job.finish({"header": {}, "timeline": [{"emotion": "Joy"}] * 400})
    body = client.get(f"/api/jobs/{job.id}").json()
    assert body["scenes"] == 400
    assert "timeline" not in body


# --- joining a record back to its transcript ----------------------------------


def test_rows_carry_the_transcript_they_were_scored_from() -> None:
    from emotion_timeline.pipeline import timeline as pipeline

    report = pipeline.Timeline.load()
    segments = pipeline.read_segments(pipeline.DEFAULT_SEGMENTS)
    rows = web.rows_with_text(report.raw, segments)
    assert len(rows) == len(report.scenes)
    assert rows[0]["text"].startswith("Козацкая")
