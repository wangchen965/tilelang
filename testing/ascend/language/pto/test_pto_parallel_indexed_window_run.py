"""NPU verification of block-dependent windows in a large B16 owner."""

import pytest
import torch

from test_pto_parallel_indexed_window import alternate_axis_window_kernel, outer_range_window_kernel
from test_pto_parallel_gather_scatter_run import _run, npu as npu


@pytest.mark.pto
def test_b16_window_with_block_dependent_base_numerics(npu):
    source = (torch.arange(70000, dtype=torch.int64) % 113 - 56).to(torch.float16)
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 256
    output = torch.full((2, 128), 211, dtype=torch.float16)
    expected = torch.stack([source[index], source[35000 + index]])
    _run("outer_range_window", outer_range_window_kernel, [source, index.to(torch.uint16), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_alternate_axis_with_scalar_row_base_numerics(kind, npu):
    if kind == "gather":
        source = (torch.arange(128 * 600, dtype=torch.int64) % 113 - 56).reshape(128, 600).to(torch.float16)
        output = torch.full((2, 128), 211, dtype=torch.float16)
        expected = source[:, :4:2].T.contiguous()
    else:
        source = (torch.arange(256, dtype=torch.int64) % 113 - 56).reshape(2, 128).to(torch.float16)
        output = torch.full((128, 600), 211, dtype=torch.float16)
        expected = output.clone()
        expected[:, :4:2] = source.T
    _run(f"alternate_axis_{kind}", alternate_axis_window_kernel(kind), [source, output], expected, npu)
