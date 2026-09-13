# syntax=docker/dockerfile:1.9
#
# Two images out of one file, because the extras in `pyproject.toml` already
# split this project the same way:
#
#   --target study   core + web.  Every command that reads a committed record,
#                    and the browser page serving the committed example.
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

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- base ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_FROZEN=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# /data is the only path the container writes to, so `models/` is a symlink into
# it: that is where the CLI's own defaults point.
RUN useradd --create-home --uid 10001 timeline \
 && install -d -o timeline -g timeline \
      /data /data/downloads /data/models /data/cache /data/cache/matplotlib \
 && ln -s /data/models /app/models

# matplotlib builds a font cache on first import and warns if it has nowhere to
# put it. Every command that draws anything imports it, `timeline` included.
ENV MPLCONFIGDIR=/data/cache/matplotlib

# uv is mounted for the length of a RUN and never copied, so it does no work in
# the shipped image and costs none of its 52 MB. pyproject and the lock are
# mounted for the same reason and one better: a README edit no longer
# invalidates the dependency layer, which is most of both images.
#
# Two `uv sync` calls per target, always. The first resolves dependencies from
# the lock alone, so it is cached until the lock changes; the second installs
# the project, which is a few hundred kilobytes and changes constantly.

# --- study --------------------------------------------------------------------
FROM base AS study-deps
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --no-install-project --extra web

FROM study-deps AS study
COPY --link README.md LICENSE ./
COPY --link src/ ./src/
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --extra web

# After the install, so editing a record rebuilds nothing but its own layer.
COPY --link benchmarks/ ./benchmarks/
COPY --link assets/ ./assets/
COPY --link docs/ ./docs/

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

# --- app ----------------------------------------------------------------------
# From `base` rather than from `study`: inheriting it would leave study's own
# 412 MB virtual environment in the image underneath the one that replaces it,
# and a layer that is overwritten is still a layer that is pulled.
FROM base AS app-deps

# ffmpeg is what turns a download into 16 kHz mono, and `serve` warns on start
# when it is missing.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg

# torch comes from the cu128 index, the same one the Windows machine uses, so
# the sm_120 kernels this project's card needs are actually in here. That is
# most of the image's size and all of its speed: `transcribe.py` asks torch
# whether CUDA is there and hands the answer to faster-whisper, so a CPU-only
# torch would put Whisper on the CPU as well, and Whisper is the wall clock.
#
# `triton` is the torch.compile JIT and nothing here calls torch.compile, so it
# is 641 MB of LLVM that never runs. `tests/test_container.py` asserts that stays
# true, because that is what the saving rests on.
#
# Its two obvious neighbours stay. `nccl` is multi-GPU collectives and `nvshmem`
# is multi-node shared memory, neither of which a single card needs -- but torch
# 2.11 links both into `_C`, so dropping either costs `import torch` itself:
#
#   ImportError: libnvshmem_host.so.3: cannot open shared object file
#   ImportError: libnccl.so.2: cannot open shared object file
#
# That was measured on the card rather than reasoned about, which is the only
# reason it is not in here.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --no-install-project --extra web --extra stt --extra model \
      --no-install-package triton

# faster-whisper is CTranslate2 rather than torch, and it dynamically loads
# cuBLAS and cuDNN by name. torch brings both as pip packages but only puts them
# on its own loader path, so without this the classifiers would find the GPU and
# the transcriber would not.
RUN printf '%s\n' /opt/venv/lib/python3*/site-packages/nvidia/*/lib \
      > /etc/ld.so.conf.d/nvidia-from-pip.conf \
 && ldconfig

FROM app-deps AS app
COPY --link README.md LICENSE ./
COPY --link src/ ./src/
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --extra web --extra stt --extra model \
      --no-install-package triton

COPY --link benchmarks/ ./benchmarks/
COPY --link assets/ ./assets/
COPY --link docs/ ./docs/

# Both caches go to the volume, so the first run's download survives the second.
ENV HF_HOME=/data/models \
    XDG_CACHE_HOME=/data/cache

LABEL org.opencontainers.image.title="Emotion Timeline" \
      org.opencontainers.image.description="Per-scene emotion analysis: the full pipeline, from a link or a file" \
      org.opencontainers.image.source="https://github.com/alex-krasnoshtanov/Emotion-Timeline" \
      org.opencontainers.image.licenses="MIT"

USER timeline
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/meta')"

ENTRYPOINT ["emotion-timeline"]
CMD ["serve", "--host", "0.0.0.0", "--downloads", "/data/downloads"]
