"""Checking the GPU is usable before an hour of training finds out it is not.

Both ways this fails are quiet. A torch wheel built for an older architecture
raises ``no kernel image is available for execution on the device`` only once a
kernel actually launches, which can be several minutes into a run; and the
default Windows wheel is CPU-only, so the fine-tune does not fail at all -- it
just takes forty times longer while every log line looks healthy.

So the gate runs a real matrix multiply rather than trusting
``torch.cuda.is_available()``, and it checks the compute capability against the
architectures the build was compiled for. ``get_arch_list()`` can name an
architecture the runtime still cannot launch on, which is why both checks are
here and the multiply is the one that decides.

:func:`assess` is a pure function over what was measured, in the same shape as
every other stage's ``check_consistency``: it returns the problems, one readable
line each, and says what to do about them. The three functions that touch the
hardware are the only ones CI cannot run.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

#: Enough for DistilBERT at batch 64 and sequence length 128, with room to spare.
NEED_MIB = 6000

CU128 = "https://download.pytorch.org/whl/cu128"


@dataclass(frozen=True, slots=True)
class Device:
    """What was measured about the machine, so :func:`assess` can stay pure."""

    name: str
    capability: tuple[int, int]
    total_mib: int
    free_mib: int
    torch_version: str
    #: ``None`` when torch was built without CUDA at all, which is the default
    #: PyPI wheel on Windows and the failure that looks most like success.
    cuda_version: str | None
    #: The architectures this build carries kernels for, e.g. ``("sm_90", "sm_120")``.
    arch_list: tuple[str, ...]

    @property
    def architecture(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"


def assess(device: Device | None, smoke_error: str | None, need_mib: int = NEED_MIB) -> list[str]:
    """Everything wrong with this machine for training, and what to do about each."""
    if device is None:
        return [
            "torch is not installed. `uv sync --extra model` installs it, and "
            f"pyproject.toml points it at {CU128} so the wheel carries the right kernels."
        ]

    if device.cuda_version is None:
        return [
            f"torch {device.torch_version} was built without CUDA, so training would run "
            "on the CPU and take roughly forty times longer without reporting an error. "
            f"Reinstall from {CU128}: `uv sync --reinstall-package torch --extra model`."
        ]

    problems: list[str] = []
    if device.architecture not in device.arch_list:
        problems.append(
            f"{device.name} is compute capability {device.capability[0]}.{device.capability[1]} "
            f"({device.architecture}), and this torch build carries kernels for "
            f"{', '.join(device.arch_list) or 'nothing'}. A kernel launch will fail with "
            f"'no kernel image is available for execution on the device'. Install torch>=2.7 "
            f"from {CU128}."
        )
    if device.free_mib < need_mib:
        problems.append(
            f"{device.free_mib:,} MiB free of {device.total_mib:,}, and this run wants about "
            f"{need_mib:,}. Close whatever else is on the card, or pass a smaller --batch-size."
        )
    if smoke_error is not None:
        problems.append(
            f"a 256x256 matrix multiply on the device failed: {smoke_error}. The architecture "
            "check above passed, so this is a driver or wheel mismatch rather than a missing "
            "kernel -- see the WSL2 note in docs/fine-tune.md."
        )
    return problems


def describe(device: Device) -> Iterator[str]:
    """The device, as the command prints it."""
    major, minor = device.capability
    cuda = device.cuda_version or "not built in"
    yield f"{device.name}"
    yield f"  compute capability   {major}.{minor} ({device.architecture})"
    yield f"  memory               {device.free_mib:,} MiB free of {device.total_mib:,}"
    yield f"  torch                {device.torch_version}, CUDA {cuda}"
    yield f"  kernels compiled for {', '.join(device.arch_list) or 'nothing'}"


def probe() -> Device | None:  # pragma: no cover - needs a GPU
    """Measure the machine. ``None`` when torch is not installed at all."""
    try:
        import torch
    except ImportError:
        return None

    cuda = torch.version.cuda
    if cuda is None or not torch.cuda.is_available():
        return Device(
            name="no CUDA device",
            capability=(0, 0),
            total_mib=0,
            free_mib=0,
            torch_version=torch.__version__,
            cuda_version=cuda,
            arch_list=tuple(torch.cuda.get_arch_list()),
        )

    free, total = torch.cuda.mem_get_info(0)
    return Device(
        name=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        total_mib=total // (1024 * 1024),
        free_mib=free // (1024 * 1024),
        torch_version=torch.__version__,
        cuda_version=cuda,
        arch_list=tuple(torch.cuda.get_arch_list()),
    )


def smoke() -> str | None:  # pragma: no cover - needs a GPU
    """Launch one real kernel. The only check that proves anything."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        left = torch.randn(256, 256, device="cuda")
        right = torch.randn(256, 256, device="cuda")
        float((left @ right).sum().item())
    except Exception as error:  # any failure here is the answer, not a crash
        return str(error)
    return None


def require(need_mib: int = NEED_MIB) -> Device:  # pragma: no cover - needs a GPU
    """The gate every training entry point calls first."""
    device = probe()
    problems = assess(device, smoke(), need_mib)
    if problems or device is None:
        raise RuntimeError("\n".join(problems))
    return device
