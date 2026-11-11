"""NPU verification of chunk-uniform windows in a large B16 owner."""

import pytest
import torch

from test_pto_parallel_indexed_window import chunk_window_kernel
from test_pto_parallel_gather_scatter_run import _run


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_chunk_uniform_b16_window_numerics(kind):
    import os

    import torch_npu

    if not torch.npu.is_available():
        pytest.skip("NPU runtime unavailable")
    device = torch.device(os.environ.get("TILELANG_GS_NPU", "npu:0"))
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    source_size = 70000 if kind == "gather" else 256
    output_size = 256 if kind == "gather" else 70000
    x = (torch.arange(source_size, dtype=torch.int64) % 113 - 56).to(torch.float16)
    index = (torch.arange(256, dtype=torch.int64) * 37 + 11) % 256
    position = 66000 + torch.arange(150) // 128 * 256 + index[:150]
    output = torch.full((output_size,), 211, dtype=torch.float16)
    expected = output.clone()
    if kind == "gather":
        expected[:150] = x[position]
    else:
        expected[position] = x[:150]
    index[150:] = 65535
    _run(f"chunk_window_{kind}_150", chunk_window_kernel(kind), [x, index.to(torch.uint16), output], expected, device)
