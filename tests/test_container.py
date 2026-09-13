"""What the images promise, checked here rather than in a twenty-minute build.

A Dockerfile is not unit-testable, but the claims it rests on are. The app image
drops three of torch's dependencies, which is only safe while nothing here wants
a second GPU. It excludes them by name, and a name with a typo in it excludes
nothing and says nothing. And `.dockerignore` is an allowlist, so a path left out
of it is a file the image quietly does without -- which for `benchmarks/` would
mean twenty commands with nothing to read.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8")


def test_nothing_here_wants_a_second_gpu() -> None:
    """`triton`, `nccl` and `nvshmem` are excluded from the image on this basis.

    torch.compile needs triton; torch.distributed needs the other two. Neither is
    used, this is single-GPU inference plus a single-GPU fine-tune, and about a
    gigabyte of wheels rides on that staying true.
    """
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    for banned in ("torch.compile", "torch.distributed", "DataParallel", "init_process_group"):
        assert banned not in source, f"{banned} is used, so the image cannot drop its wheels"


def test_every_excluded_package_is_a_real_one() -> None:
    """`--no-install-package` for a name nothing provides is a silent no-op."""
    excluded = set(re.findall(r"--no-install-package ([A-Za-z0-9_.-]+)", DOCKERFILE))
    assert excluded, "the Dockerfile stopped excluding anything; drop this test with it"
    locked = set(re.findall(r'^name = "(.+)"$', (ROOT / "uv.lock").read_text(), re.M))
    assert excluded <= locked, sorted(excluded - locked)


def test_the_image_is_allowed_to_carry_what_the_commands_read() -> None:
    """`.dockerignore` excludes everything and puts back a list. This is the list."""
    allowed = {line[1:] for line in DOCKERIGNORE.splitlines() if line.startswith("!")}
    for needed in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "src", "benchmarks"):
        assert needed in allowed, f"{needed} would not reach the image"


def test_both_targets_are_still_there() -> None:
    targets = set(re.findall(r"^FROM .+ AS ([a-z-]+)$", DOCKERFILE, re.M))
    assert {"study", "app"} <= targets, sorted(targets)


def test_the_extras_the_images_install_are_extras_that_exist() -> None:
    """A `--extra` that no longer exists fails the build, twenty minutes in."""
    declared = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
    )
    used = set(re.findall(r"--extra ([a-z]+)", DOCKERFILE))
    assert used, "the Dockerfile installs no extras at all"
    assert used <= declared, sorted(used - declared)
