"""The optional front end: a video URL or an audio file in, a segment CSV out.

Everything here is above the seam, which is why it is the only module in this
package that is allowed to be untested. It needs a network, a GPU, ffmpeg and a
few hundred megabytes of audio that must never be committed, and none of those
can be in CI. What it produces is three columns, and those *are* tested, in
`timeline.py`.

**Whisper large-v3-turbo, not large-v3.** The decoder is pruned from 32 layers to
4, the model from 1.55B parameters to 809M, and it runs at roughly 216 times real
time against large-v3's 10-ish, for a word error rate of 7.75% against 7.4%. On a
51-minute recording that is the difference between minutes and tens of minutes,
for about a third of a point. It did not exist when the original coursework ran.

**The word error rate here is not the one in `stt-benchmark.md`.** That number,
3.33%, was measured on `large-v3` over a hand-annotated eighteen-minute window of
one specific recording. These are the model card's figures on a general
benchmark. Do not compare them.

`language="ru"` is passed rather than detected. Auto-detection reads the first
thirty seconds, and a recording that opens with a title card or music can come
back as the wrong language and silently transcribe into it.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from emotion_timeline.pipeline.timeline import Segment

#: Roughly 216x real time, within a third of a point of large-v3's word error
#: rate. See the module docstring for why that trade is the right way round.
MODEL = "large-v3-turbo"

SEGMENT_COLUMNS = ("start_s", "end_s", "text")

#: 16 kHz mono is what every Whisper family model resamples to internally, so
#: doing it once with ffmpeg costs nothing and makes the intermediate file small.
SAMPLE_RATE = 16_000


def write_segments(segments: Sequence[Segment], path: str | Path) -> Path:
    """The three columns that are the whole interface to the rest of the pipeline.

    The same shape as `benchmarks/stt/assemblyai-best.csv`, minus the hand-marked
    error columns that only the word error rate harness needs. Anything that
    writes these three columns can feed `timeline`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SEGMENT_COLUMNS)
        for segment in segments:
            writer.writerow([f"{segment.start_s:.3f}", f"{segment.end_s:.3f}", segment.text])
    return target


def require(tool: str) -> str:  # pragma: no cover - depends on what is installed
    """The path to an external binary, or a message saying how to get it."""
    found = shutil.which(tool)
    if found is None:
        raise RuntimeError(
            f"{tool} is not on PATH. `transcribe` needs it; nothing else in this "
            "repository does, which is why it is not a dependency."
        )
    return found


def fetch_audio(source: str, out_dir: str | Path) -> Path:  # pragma: no cover - needs network
    """Audio for a URL or a local file, as 16 kHz mono WAV.

    A local path skips the download and goes straight to conversion, so the same
    command works on a file somebody already has.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    local = Path(source)

    if local.exists():
        downloaded = local
    else:
        import yt_dlp

        template = str(directory / "%(id)s.%(ext)s")
        with yt_dlp.YoutubeDL({"format": "bestaudio/best", "outtmpl": template}) as handle:
            info = handle.extract_info(source, download=True)
            downloaded = Path(handle.prepare_filename(info))

    wav = directory / f"{downloaded.stem}.{SAMPLE_RATE // 1000}k.wav"
    subprocess.run(
        [
            require("ffmpeg"),
            "-y",
            "-i",
            str(downloaded),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    return wav


def transcribe(  # pragma: no cover - needs the model, a GPU and audio
    audio: str | Path,
    model_id: str = MODEL,
    language: str = "ru",
    progress: object = None,
) -> list[Segment]:
    """Timestamped segments, in order, as Whisper produced them.

    No merging and no splitting. Whatever Whisper thought was one utterance stays
    one row, and grouping them into scenes is `timeline.group`'s job, with a
    threshold a reader can change.
    """
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        model_id, device=device, compute_type="float16" if device == "cuda" else "int8"
    )
    segments, info = model.transcribe(str(audio), language=language, vad_filter=True)

    out: list[Segment] = []
    for segment in segments:
        out.append(Segment(float(segment.start), float(segment.end), segment.text.strip()))
        if callable(progress) and len(out) % 25 == 0:
            progress(
                f"  {len(out):,} segments, {out[-1].end_s / 60:.1f} min of {info.duration / 60:.1f}"
            )
    return out
