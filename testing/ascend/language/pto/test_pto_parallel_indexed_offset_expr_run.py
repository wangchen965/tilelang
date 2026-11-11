"""NPU verification of scalar coordinates inside full indexed offsets."""

import pytest
import torch

from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_offset_expr import scalar_coordinate_cast_kernel

npu = _npu


@pytest.mark.pto
@pytest.mark.parametrize("coordinate", ["chunk", "select"])
@pytest.mark.parametrize("extent", [256, 150])
def test_scalar_coordinate_cast_keeps_indexed_address_semantics(coordinate, extent, npu):
    source = torch.arange(512, dtype=torch.float32) - 123
    index = (torch.arange(256, dtype=torch.int64) * 37 + 11) % 128
    index[0] = 0
    output = torch.full((256,), -777.0)
    expected = output.clone()
    offset = index[:extent] + (torch.arange(extent) // 128 * 128 if coordinate == "chunk" else 1)
    expected[:extent] = source[offset]
    index[extent:] = -1
    _run(
        f"scalar_coordinate_cast_{coordinate}_{extent}",
        scalar_coordinate_cast_kernel(coordinate, extent),
        [source, index.to(torch.int32), output],
        expected,
        npu,
    )
