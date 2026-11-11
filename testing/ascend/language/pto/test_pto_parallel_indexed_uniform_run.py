"""NPU acceptance for chunk-uniform scalar index coordinates."""

import importlib.util
from pathlib import Path

import pytest

from test_pto_parallel_indexed_uniform import chunk_uniform_kernel


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("extent", [256, 150])
def test_chunk_uniform_coordinate_numerics(kind, extent):
    import os

    import torch
    import torch_npu

    if not torch.npu.is_available():
        pytest.skip("NPU runtime unavailable")
    device = torch.device(os.environ.get("TILELANG_GS_NPU", "npu:0"))
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    spec = importlib.util.spec_from_file_location(
        "indexed_uniform_runtime", Path(__file__).with_name("test_pto_parallel_gather_scatter_run.py")
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    x = torch.arange(512, dtype=torch.float32).reshape(2, 256)
    index = (torch.arange(256, dtype=torch.int64) * 37 + 11) % 256
    output = torch.full((256,) if kind == "gather" else (2, 256), -777.0)
    expected = output.clone()
    logical = torch.arange(extent)
    if kind == "gather":
        expected[:extent] = x[logical // 128, index[:extent]]
    else:
        expected[logical // 128, index[:extent]] = x[0, :extent]
    index[extent:] = 0x3FFF0000
    runtime._run(f"chunk_uniform_{kind}_{extent}", chunk_uniform_kernel(kind, extent), [x, index.to(torch.int32), output], expected, device)
