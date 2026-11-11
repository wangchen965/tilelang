"""Numerics for B16 indexed accesses under surrounding serial loops."""

import pytest
import torch

from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_serial import (
    serial_b16_gather_with_view_origin,
    serial_b16_scatter_kernel,
    symbolic_serial_b16_scatter_kernel,
)

npu = _npu


@pytest.mark.pto
@pytest.mark.parametrize("case", ["serial", "interleaved", "nested", "nonzero_min", "sequential"])
def test_serial_b16_scatter_numerics(case, npu):
    source = (torch.arange(128, dtype=torch.float32) - 64).to(torch.float16)
    output = torch.full((1536,), 211, dtype=torch.float16)
    expected = output.clone()
    lane = torch.arange(128)
    if case == "interleaved":
        offsets = [2 * lane + t for t in range(2)]
    elif case == "nested":
        offsets = [t * 512 + u * 256 + 2 * lane for t in range(2) for u in range(2)]
    elif case == "nonzero_min":
        offsets = [t * 256 + 2 * lane for t in range(2, 4)]
    elif case == "sequential":
        offsets = [t * 256 + 2 * lane for t in (0, 1, 3, 4)]
    else:
        offsets = [t * 256 + 2 * lane for t in range(2)]
    for index in offsets:
        expected[index] = source
    _run(f"serial_b16_scatter_{case}", serial_b16_scatter_kernel(case), [source, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("outside_vf", [False, True])
def test_symbolic_serial_b16_scatter_numerics(outside_vf, npu):
    source = (torch.arange(128, dtype=torch.float32) - 64).to(torch.float16)
    output = torch.full((1536,), 211, dtype=torch.float16)
    expected = output.clone()
    for start in (0, 256, 768, 1024):
        expected[start : start + 256 : 2] = source
    _run(
        "symbolic_serial_b16_scatter",
        symbolic_serial_b16_scatter_kernel(outside_vf),
        [source, output],
        expected,
        npu,
    )


@pytest.mark.pto
def test_serial_b16_gather_with_view_origin_numerics(npu):
    source = (torch.arange(65568) % 113).to(torch.float16)
    index = (torch.arange(128) * 37) % 128
    output = torch.empty((2, 128), dtype=torch.float16)
    expected = torch.stack((source[32 + index], source[33 + index]))
    _run("b16_gather_symbolic_view", serial_b16_gather_with_view_origin, [source, index.to(torch.uint16), output], expected, npu)
