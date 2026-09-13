# syntax=docker/dockerfile:1.9
#
# Two images out of one file, because the extras in `pyproject.toml` already
# split this project the same way:
#
#   --target study   core + web.  Every command that reads a committed record,
#                    and the browser page serving the committed example. ~400 MB.
#   --target app     everything above plus ffmpeg, Whisper and the classifiers,
#                    built against CUDA, so it runs the whole pipeline on a
#                    video of your own. Large, because CUDA kernels are.
#
#                      docker run --gpus all -p 127.0.0.1:8000:8000 ...
#
#                    Without `--gpus all`, or on a machine with no NVIDIA card,
#                    it still runs: torch reports no CUDA and everything falls
#                    back to the CPU. A 52-minute recording takes minutes that
#                    way rather than about one.
#
# The project is installed into /app rather than into site-packages on purpose.
# Every read-only command finds its records relative to its own file, so a wheel
# on its own looks for `<venv>/Lib/benchmarks/...` and finds nothing. The image
# ships the repository layout, which is what those commands need.
#
#   docker build --target study -t emotion-timeline:study .
#   docker run --rm emotion-timeline:study timeline
#   docker run --rm -p 127.0.0.1:8000:8000 emotion-timeline:study

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- base ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app

# /data is the only writable path, so the image runs read-only if you ask it to.
# `models/` is a symlink into it because that is where the CLI's defaults point.
RUN useradd --create-home --uid 10001 timeline \
 && install -d -o timeline -g timeline \
      /data /data/downloads /data/models /data/cache /data/cache/matplotlib \
 && ln -s /data/models /app/models

# matplotlib builds a font cache on first import and warns if it has nowhere to
# put it. Every command that draws anything imports it, `timeline` included.
ENV MPLCONFIGDIR=/data/cache/matplotlib

# --- dependencies, cached apart from the source -------------------------------
FROM base AS deps
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra web

# --- study: everything that reads a committed record --------------------------
FROM deps AS study
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY assets/ ./assets/
COPY docs/ ./docs/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra web

LABEL org.opencontainers.image.title="Emotion Timeline" \
      org.opencontainers.image.description="Per-scene emotion analysis: the study, and the page that reads it" \
      org.opencontainers.image.source="https://github.com/alex-krasnoshtanov/Emotion-Timeline" \
      org.opencontainers.image.licenses="MIT"

USER timeline
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/meta')"

# 0.0.0.0 binds inside the container's own network namespace and nothing more.
# Publish it to loopback -- `-p 127.0.0.1:8000:8000` -- because the page hands a
# URL to yt-dlp and a file to ffmpeg, so it is a tool you run for yourself.
ENTRYPOINT ["emotion-timeline"]
CMD ["serve", "--host", "0.0.0.0", "--downloads", "/data/downloads"]

# --- app: the whole pipeline --------------------------------------------------
FROM study AS app
USER root

# ffmpeg is what turns a download into 16 kHz mono, and `serve` warns on start
# when it is missing.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# torch comes from the cu128 index, the same one the Windows machine uses, so
# the sm_120 kernels this project's card needs are actually in here. That is
# most of the image's size and all of its speed: `transcribe.py` asks torch
# whether CUDA is there and hands the answer to faster-whisper, so a CPU-only
# torch would put Whisper on the CPU as well, and Whisper is the wall clock.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra web --extra stt --extra model

# faster-whisper is CTranslate2 rather than torch, and it dynamically loads
# cuBLAS and cuDNN by name. torch brings both as pip packages but only puts them
# on its own loader path, so without this the classifiers would find the GPU and
# the transcriber would not.
RUN printf '%s\n' /opt/venv/lib/python3*/site-packages/nvidia/*/lib \
      > /etc/ld.so.conf.d/nvidia-from-pip.conf \
 && ldconfig

# Both caches go to the volume, so the first run's download survives the second.
ENV HF_HOME=/data/models \
    XDG_CACHE_HOME=/data/cache

LABEL org.opencontainers.image.description="Per-scene emotion analysis: the full pipeline, from a link or a file"

USER timeline
VOLUME ["/data"]
