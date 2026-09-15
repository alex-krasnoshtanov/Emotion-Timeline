# The container: what ships, and where it runs

Two images come out of one Dockerfile, split the way the extras in
`pyproject.toml` already split this project. `:study` reads every committed
record and serves [the page](pipeline.md#the-browser-front-end). `:latest` adds
ffmpeg, Whisper and the classifiers, and runs the whole pipeline on a video of
your own.

```bash
docker run --rm ghcr.io/alex-krasnoshtanov/emotion-timeline:study timeline
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/alex-krasnoshtanov/emotion-timeline:study

docker compose up        # the full pipeline, on the GPU, models cached in a volume
docker compose up study  # the small one, on http://127.0.0.1:8001
```

Both publish to loopback rather than to every interface, because the page hands
a URL to yt-dlp and a file to ffmpeg.

## Why there is no package to install

The project is installed into `/app` in both images and not into site-packages,
and that is also the reason this repository publishes no wheel. Every read-only
command finds its records relative to its own file, so an installed wheel looks
for `<venv>/Lib/benchmarks/pipeline/timeline.json` and finds nothing:

```
$ emotion-timeline timeline
emotion-timeline timeline: cannot find .../site-packages/../../benchmarks/...
```

Twenty of the twenty-one commands are in that position. The image ships the
repository layout, which is the thing a wheel cannot, and `git clone` covers
everyone else.

## What the two images cost

**Two images, for the two things people want.** `ghcr.io/...:study` is core plus
the page, 592 MB, and runs every command that reads a committed record.
`ghcr.io/...:latest` adds ffmpeg, Whisper and the classifiers, built against
CUDA and run with `--gpus all`. It is 12.1 GB, and worth it: the transcriber is
most of the wall clock, and it takes its device from `torch.cuda.is_available()`,
so a CPU-only torch would slow down the one stage that matters. faster-whisper is
CTranslate2 rather than torch and loads cuBLAS and cuDNN by name, so the image
puts torch's copies of both on the loader path; without that the classifiers
would find the card and the transcriber would not.

**Both are smaller than they were**, by measurement rather than by guess: the
study image went from 801 MB to 592 MB and the app one from 13.5 GB to 12.1 GB.

| | |
| --- | --- |
| uv is bind-mounted for one `RUN` and never copied | −52 MB each |
| bytecode compilation off | −157 MB on the study image, for 100 ms on a command that takes a second |
| the app image built from the same base as the study one rather than *from* it | −412 MB, a virtual environment left underneath the one that replaced it |
| `triton` left out, being a JIT nothing here calls | −641 MB |

`nccl` and `nvshmem` look equally droppable on a single card and are not: torch
2.11 links both into `_C`, and removing either breaks `import torch` outright.
That was found by deleting them inside the running container on an actual GPU,
which is the only reason this file does not confidently leave them out too.

Both install the
project into `/app` rather than into site-packages, because the commands find
their records relative to their own file: the image ships the repository layout,
which is the thing a wheel cannot. The published images are built by
[`release.yml`](../.github/workflows/release.yml), which runs `timeline` inside
each one and checks the numbers before pushing it.

## Which cards the app image runs on

`torch 2.11.0+cu128` carries kernels for **sm_75, sm_80, sm_86, sm_90, sm_100
and sm_120**, read out of the built image rather than recalled. A cubin is
binary-compatible forward across the minor revisions of one major architecture,
so that list covers more cards than it names:

| | | |
| --- | --- | --- |
| Turing | sm_75 | T4, RTX 20-series, GTX 16-series |
| Ampere | sm_80, sm_86 | A100, A30, RTX 30-series, A10, A40 |
| Ada | sm_89, **via sm_86** | RTX 40-series, L4, L40S |
| Hopper | sm_90 | H100, H200 |
| Blackwell | sm_100, sm_120 | B200, GB200, RTX 50-series |

Ada is the row worth reading twice. `sm_89` appears in no arch list torch ships
and an RTX 4090 is the most likely card anyone points this at; it works on the
sm_86 kernels, and a check that asked `architecture in arch_list` would have sent
every one of them to the CPU. This one did, for about an hour.

**What it will not run on.** Maxwell and Pascal were dropped from the cu128
wheels, and torch 2.11 dropped Volta with them, so a GTX 10-series card or a
V100 has nothing to run. Neither does anything newer than sm_120: the build
ships no `compute_*` entry, so there is no PTX for the driver to compile forward.
The driver floor is 525.60.13 for the CUDA 12.x family and 570 for Blackwell.
The app image is `linux/amd64` only, because neither torch nor CTranslate2
publishes generic aarch64 CUDA wheels; `:study` is built for arm64 as well.

**None of that used to be checked.** Every model here picked its device with
`torch.cuda.is_available()`, which is perfectly true of a card the wheel has no
kernels for, and the run then died part of the way through a transcription with
`no kernel image is available for execution on the device`.
`preflight.usable_device` now asks the harder question and falls back to the CPU
with a line saying why, which is slow and right rather than fast and broken.
`emotion-timeline preflight` prints the whole picture, and runs inside the
container:

```bash
docker run --rm --gpus all ghcr.io/alex-krasnoshtanov/emotion-timeline preflight
```

faster-whisper is the other half and it answers separately, being CTranslate2
rather than torch. 4.8.2 reaches sm_120 through the driver's PTX JIT, and the
pipeline asks it for `float16` on CUDA, which sidesteps the INT8 path that
crashes on Blackwell with `CUBLAS_STATUS_NOT_SUPPORTED`. Verified by transcribing
on the card rather than by reading: `get_cuda_device_count()` counts devices
through the driver API and says nothing about whether a kernel can launch.

## What this does not establish

- **That the images are reproducible byte for byte.** `uv.lock` pins every
  Python dependency and the base image is pinned by tag rather than by digest,
  so `python:3.12-slim` moving underneath is a difference nothing here would
  catch. The published provenance attestation records what was built.
- **That 12.1 GB is a floor.** It is the floor for these dependencies. cuDNN and
  cuBLAS are most of it and both are genuinely loaded; a smaller image means a
  smaller job than this one.
- **That the CPU fallback is fast enough to be useful.** It is correct and it is
  several times slower. Nothing here measures by how much on a machine that is
  not this one.
- **Anything about arm64 with a GPU.** The app image is `linux/amd64`, so Jetson
  and Grace Hopper are untested and unbuilt rather than known to fail.
