"""The GPU gate, checked without a GPU.

Everything that touches the hardware lives in three pragma'd functions; what is
tested here is the judgement they feed, which is where the useful part is. Each
case is a way the machine is wrong, and the assertion is that the message says
what to do about it rather than only that something failed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from emotion_timeline.training import preflight

BLACKWELL = preflight.Device(
    name="NVIDIA GeForce RTX 5070",
    capability=(12, 0),
    total_mib=12_227,
    free_mib=11_000,
    torch_version="2.11.0+cu128",
    cuda_version="12.8",
    arch_list=("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"),
)


def test_a_card_the_build_has_kernels_for_passes() -> None:
    assert preflight.assess(BLACKWELL, None) == []


def test_no_torch_at_all_says_how_to_install_it() -> None:
    problems = preflight.assess(None, None)
    assert len(problems) == 1
    assert "--extra model" in problems[0]
    assert preflight.CU128 in problems[0]


def test_a_cpu_only_build_is_the_failure_that_looks_like_success() -> None:
    """The default Windows wheel. It does not error, it just runs forty times slower."""
    cpu = preflight.Device(
        name="no CUDA device",
        capability=(0, 0),
        total_mib=0,
        free_mib=0,
        torch_version="2.11.0",
        cuda_version=None,
        arch_list=(),
    )
    problems = preflight.assess(cpu, None)
    assert len(problems) == 1
    assert "without CUDA" in problems[0]
    assert preflight.CU128 in problems[0]


def test_a_build_without_this_cards_kernels_is_caught_before_it_launches_one() -> None:
    older = replace(BLACKWELL, arch_list=("sm_80", "sm_90"))
    problems = preflight.assess(older, None)
    assert any("no kernel image is available" in problem for problem in problems)
    assert any("sm_120" in problem for problem in problems)


def test_a_card_with_no_room_left_is_caught() -> None:
    busy = replace(BLACKWELL, free_mib=900)
    problems = preflight.assess(busy, None)
    assert any("900 MiB free" in problem for problem in problems)
    assert any("--batch-size" in problem for problem in problems)


def test_a_failed_multiply_outranks_a_passing_architecture_list() -> None:
    """get_arch_list() can name an architecture the runtime still cannot use."""
    problems = preflight.assess(BLACKWELL, "CUDA error: no kernel image is available")
    assert len(problems) == 1
    assert "matrix multiply on the device failed" in problems[0]
    assert "driver or wheel mismatch" in problems[0]


def test_every_problem_with_the_machine_is_reported_at_once() -> None:
    broken = replace(BLACKWELL, arch_list=("sm_90",), free_mib=100)
    assert len(preflight.assess(broken, "boom")) == 3


def test_a_smaller_run_is_allowed_a_smaller_card() -> None:
    small = replace(BLACKWELL, free_mib=2_000)
    assert preflight.assess(small, None, need_mib=1_000) == []
    assert preflight.assess(small, None, need_mib=4_000) != []


def test_the_architecture_is_named_the_way_torch_names_it() -> None:
    assert BLACKWELL.architecture == "sm_120"


def test_describe_prints_what_was_measured() -> None:
    lines = list(preflight.describe(BLACKWELL))
    assert lines[0] == "NVIDIA GeForce RTX 5070"
    assert any("12.0 (sm_120)" in line for line in lines)
    assert any("11,000 MiB free of 12,227" in line for line in lines)
    assert any("2.11.0+cu128, CUDA 12.8" in line for line in lines)


def test_describe_says_so_when_cuda_was_never_built_in() -> None:
    cpu = replace(BLACKWELL, cuda_version=None, arch_list=())
    lines = list(preflight.describe(cpu))
    assert any("not built in" in line for line in lines)
    assert any("compiled for nothing" in line for line in lines)


@pytest.mark.parametrize("need", [0, preflight.NEED_MIB])
def test_the_default_requirement_fits_the_card_it_was_chosen_for(need: int) -> None:
    assert preflight.assess(BLACKWELL, None, need_mib=need) == []


# --- can this build launch here at all ---------------------------------------


#: What `torch 2.11.0+cu128` actually reports, read out of the built image.
SHIPPED = ("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")


def test_a_card_the_build_has_kernels_for_is_used() -> None:
    assert preflight.unusable_reason((12, 0), ("sm_90", "sm_120"), "2.11.0+cu128") is None


@pytest.mark.parametrize(
    ("capability", "card"),
    [
        ((7, 5), "T4, RTX 20-series"),
        ((8, 0), "A100"),
        ((8, 6), "RTX 30-series, A10"),
        ((8, 9), "RTX 40-series, L4 -- sm_89, and not in the list"),
        ((9, 0), "H100"),
        ((10, 0), "B200"),
        ((12, 0), "RTX 50-series"),
    ],
)
def test_the_cards_this_image_is_meant_for(capability: tuple[int, int], card: str) -> None:
    """sm_89 is the one worth a test of its own.

    Ada Lovelace is not in any arch list torch ships, and an RTX 4090 is the most
    common card anyone would point this at. It works because a cubin is
    binary-compatible forward across minor revisions, and checking `in arch_list`
    would have sent every one of them to the CPU.
    """
    assert preflight.unusable_reason(capability, SHIPPED, "2.11.0+cu128") is None, card


@pytest.mark.parametrize(
    ("capability", "card"),
    [
        ((5, 2), "GTX 900-series, dropped from the cu128 wheels"),
        ((6, 1), "GTX 10-series"),
        ((7, 0), "V100 -- Volta, dropped by torch 2.11, and sm_75 is a later minor"),
    ],
)
def test_the_cards_it_will_not_run_on(capability: tuple[int, int], card: str) -> None:
    assert preflight.unusable_reason(capability, SHIPPED, "2.11.0+cu128") is not None, card


def test_this_build_ships_no_ptx_so_a_future_card_is_not_covered() -> None:
    """No `compute_*` entry means no JIT fallback for an architecture after these."""
    assert not [entry for entry in SHIPPED if entry.startswith("compute_")]
    assert preflight.unusable_reason((13, 0), SHIPPED, "2.11.0+cu128") is not None


def test_a_card_the_build_has_no_kernels_for_falls_back() -> None:
    """The cu128 wheels dropped Maxwell and Pascal, and 2.11 dropped Volta.

    `torch.cuda.is_available()` is true for all three, which is why the pipeline
    used to die part-way through a transcription instead of at the start.
    """
    reason = preflight.unusable_reason((6, 1), ("sm_75", "sm_90", "sm_120"), "2.11.0+cu128")
    assert reason is not None
    assert "sm_61" in reason and "sm_75" in reason
    assert "CPU" in reason


def test_ptx_for_an_older_architecture_is_forward_compatible() -> None:
    """The driver compiles PTX for a newer card at load time, so it runs."""
    assert preflight.unusable_reason((12, 0), ("sm_90", "compute_90"), "2.11.0+cu128") is None


def test_ptx_for_a_newer_architecture_does_not_help_an_older_card() -> None:
    """Compared as numbers: "90" sorts after "120" as a string, and did once."""
    assert preflight.unusable_reason((7, 5), ("sm_120", "compute_120"), "2.11.0") is not None


def test_a_later_minor_revision_does_not_run_on_an_earlier_card() -> None:
    """Compatibility runs one way. sm_86 does not run on an sm_80 A100."""
    assert preflight.unusable_reason((8, 0), ("sm_86",), "2.11.0") is not None


def test_a_build_with_no_cuda_kernels_at_all_says_so() -> None:
    reason = preflight.unusable_reason((8, 6), (), "2.11.0")
    assert reason is not None and "nothing" in reason
