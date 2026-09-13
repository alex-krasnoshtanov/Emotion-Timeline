# syntax=docker/dockerfile:1.9
#
# Two images out of one file, because the extras in `pyproject.toml` already
# split this project the same way:
#
#   --target study   core + web.  Every command that reads a committed record,
#                    and the browser page serving the committed example. ~400 MB.
#   --target app     everything above plus ffmpeg, Whisper and the classifiers,
#                    so it runs the whole pipeline on a video of your own.
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

# torch here is the CPU build, pinned that way in `pyproject.toml` for Linux:
# a container has no GPU without the nvidia runtime, and the CUDA wheel brings
# forty-three nvidia packages and several gigabytes with it. The GPU stage in
# this project is the fine-tune, and that runs on the host.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra web --extra stt --extra model

# Both caches go to the volume, so the first run's download survives the second.
ENV HF_HOME=/data/models \
    XDG_CACHE_HOME=/data/cache

LABEL org.opencontainers.image.description="Per-scene emotion analysis: the full pipeline, from a link or a file"

USER timeline
VOLUME ["/data"]
