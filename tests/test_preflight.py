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
