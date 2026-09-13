"""A local FastAPI front end: paste a link or drop a file, get a timeline.

**This is a localhost tool and it binds to 127.0.0.1 on purpose.** It hands a URL
to yt-dlp and a file to ffmpeg, so anyone who can reach it can make this machine
fetch a URL of their choosing. That is fine for a tool you run for yourself and
not fine on an open port, which is why the default host is loopback and why this
docstring says so rather than a comment nobody opens.

Three things are validated before any of that happens, because they are the
boundary: the URL has to be `http`/`https` (no `file://` reads, no arbitrary
scheme handed to a downloader), the upload has to be a media extension and under
:data:`MAX_UPLOAD_BYTES`, and the uploaded filename is never used as a path — the
extension is taken and the name is generated.

The page itself also runs with **no GPU and no network**: `/api/demo` serves the
committed timeline, so the interface can be opened, read and tested on a machine
that could not run a model.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from emotion_timeline import figures
from emotion_timeline.pipeline import score as scoring
from emotion_timeline.pipeline import timeline as pipeline
from emotion_timeline.web.jobs import Job, JobStore

# FastAPI resolves route annotations at import time, so `UploadFile | None` has
# to be a real name here rather than a forward reference to something imported
# inside a function. That makes this module need the `web` extra -- which is why
# `cli.py` imports it inside `cmd_serve` and nowhere else.

# FastAPI's calling convention is defaults that are marker objects, so B008 does
# not apply to a route signature.
URL_FIELD = Form(default="")
GAP_FIELD = Form(default=pipeline.GAP_SECONDS)
VAD_FIELD = Form(default=True)
FILE_FIELD = File(default=None)

STATIC = Path(__file__).resolve().parent / "static"

#: A 52-minute episode is about 90 MB of audio. This is generous for that and
#: still refuses a file somebody leans on the server with.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

#: What ffmpeg is willing to read here. The check is a allowlist rather than a
#: denylist because the point is to keep surprises out, not to catch known-bad.
MEDIA_SUFFIXES = frozenset(
    {".mp3", ".mp4", ".m4a", ".mkv", ".mov", ".wav", ".webm", ".ogg", ".opus", ".flac", ".avi"}
)

SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def validate_url(raw: str) -> str:
    """A URL we are willing to hand to a downloader, or a reason we are not."""
    candidate = raw.strip()
    if not candidate:
        raise ValueError("give a link or choose a file")
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http and https links are accepted")
    if not parsed.netloc:
        raise ValueError(f"that does not look like a link: {candidate[:80]}")
    return candidate


def safe_suffix(filename: str) -> str:
    """The extension of an upload, checked. The name itself is never reused.

    An uploaded filename is attacker-controlled: it can contain separators, `..`,
    a null byte, or 4,000 characters. Only the suffix is taken, and only if it is
    one we expect.
    """
    suffix = Path(filename.strip()).suffix.lower()
    if not SAFE_SUFFIX.match(suffix) or suffix not in MEDIA_SUFFIXES:
        raise ValueError(f"unsupported file type {suffix or '(none)'}; expected audio or video")
    return suffix


def validate_gap(raw: float) -> float:
    """The silence threshold, kept inside the range where it means anything."""
    if not 0.0 <= raw <= 60.0:
        raise ValueError("the scene gap has to be between 0 and 60 seconds")
    return float(raw)


def rows_with_text(
    record: dict[str, Any], segments: list[pipeline.Segment]
) -> list[dict[str, Any]]:
    """The record's scenes, each carrying the transcript it was scored from."""
    scenes, _ = scoring.plan(segments, float(record["gap_seconds"]))
    return [
        {**row, "text": scene.text} for row, scene in zip(record["timeline"], scenes, strict=True)
    ]


def demo_payload() -> dict[str, Any]:
    """The committed timeline, so the page works with no GPU and no network."""
    report = pipeline.Timeline.load()
    segments = pipeline.read_segments(pipeline.DEFAULT_SEGMENTS)
    return {
        "header": {key: value for key, value in report.raw.items() if key != "timeline"},
        "timeline": pipeline.table(report, segments),
    }


def run_job(  # pragma: no cover - downloads, transcribes and loads three models
    job: Job,
    upload: Path | None,
    gap: float,
    vad: bool,
    downloads: Path,
) -> None:
    """The whole pipeline for one job, on a worker thread."""
    from emotion_timeline.pipeline import transcribe

    try:
        job.start()
        job.say("fetching audio")
        audio = transcribe.fetch_audio(str(upload) if upload else job.source, downloads)
        job.say(f"transcribing with whisper {transcribe.MODEL}")
        segments = transcribe.transcribe(audio, progress=job.say, vad=vad)
        if not segments:
            raise RuntimeError("the transcript came back empty; is there speech in this?")
        job.say(f"{len(segments):,} segments transcribed")
        record = scoring.score(segments, source=job.source, gap=gap, progress=job.say)
        job.finish(
            {
                "header": {k: v for k, v in record.items() if k != "timeline"},
                "timeline": rows_with_text(record, segments),
            }
        )
        job.say("done")
    except Exception as exc:
        job.fail(f"{type(exc).__name__}: {exc}")
    finally:
        if upload is not None:
            upload.unlink(missing_ok=True)


def build_app(downloads: str | Path = "downloads", store: JobStore | None = None) -> FastAPI:
    """The application. Takes its store so a test can inspect one."""
    jobs = store if store is not None else JobStore()
    downloads_dir = Path(downloads)
    app = FastAPI(title="Emotion Timeline", docs_url=None, redoc_url=None)
    app.state.jobs = jobs

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        """Everything the page needs to render before anything has been run."""
        return JSONResponse(
            {
                "emotions": list(figures.EMOTION_COLOURS),
                "colours": dict(figures.EMOTION_COLOURS),
                "default_gap": pipeline.GAP_SECONDS,
                "chunk_chars": pipeline.CHUNK_CHARS,
                "caveat": scoring.AGREEMENT_CAVEAT,
                "demo_available": pipeline.DEFAULT_TIMELINE.exists(),
            }
        )

    @app.get("/api/demo")
    def demo() -> JSONResponse:
        if not pipeline.DEFAULT_TIMELINE.exists():
            raise HTTPException(404, "no committed timeline in this checkout")
        return JSONResponse(demo_payload())

    @app.post("/api/jobs")
    async def create(
        url: str = URL_FIELD,
        gap: float = GAP_FIELD,
        vad: bool = VAD_FIELD,
        file: UploadFile | None = FILE_FIELD,
    ) -> JSONResponse:
        has_file = file is not None and bool(file.filename)
        if has_file == bool(url.strip()):
            raise HTTPException(400, "give either a link or a file, not both and not neither")

        try:
            checked_gap = validate_gap(gap)
            if has_file:
                assert file is not None
                suffix = safe_suffix(file.filename or "")
            else:
                source = validate_url(url)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        upload: Path | None = None
        if has_file:
            assert file is not None
            downloads_dir.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(suffix=suffix, dir=downloads_dir)
            upload = Path(name)
            with open(descriptor, "wb") as handle:
                written = 0
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        upload.unlink(missing_ok=True)
                        raise HTTPException(413, "that file is larger than this accepts")
                    handle.write(chunk)
            source = file.filename or upload.name

        job = jobs.create(source=source, kind="file" if has_file else "link")
        threading.Thread(
            target=run_job,
            args=(job, upload, checked_gap, vad, downloads_dir),
            daemon=True,
        ).start()
        return JSONResponse(job.summary(), status_code=202)

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job; the server may have restarted")
        return JSONResponse(job.summary())

    @app.get("/api/jobs/{job_id}/timeline")
    def result(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job; the server may have restarted")
        if job.result is None:
            raise HTTPException(409, f"that run is {job.state.value}, not finished")
        return JSONResponse(job.result)

    return app


def serve(  # pragma: no cover - starts a server
    host: str = "127.0.0.1",
    port: int = 8000,
    downloads: str | Path = "downloads",
) -> None:
    """Run it. Loopback by default -- see the module docstring for why."""
    import uvicorn

    if shutil.which("ffmpeg") is None:
        print("warning: ffmpeg is not on PATH, so uploads and downloads will fail")
    uvicorn.run(build_app(downloads), host=host, port=port, log_level="info")
