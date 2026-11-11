"""NPU numerics for legal alternate vector axes."""

import pytest
import torch

import tilelang

from test_pto_parallel_candidate_legality import (
    auto_version_axis,
    b16_row_gather,
    b16_view_gather,
    misaligned_continuous_store_axis,
    padded_version_axis,
    runtime_float_computation_axis,
)
from test_pto_parallel_gather_scatter_run import _run, npu as _npu

npu = _npu


@pytest.mark.pto
def test_b16_row_gather_with_legal_continuous_axis_numerics(npu):
    source = (torch.arange(128 * 128, dtype=torch.int64) % 113 - 56).reshape(128, 128).to(torch.float16)
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 128
    output = torch.full((128, 128), 211, dtype=torch.float16)
    expected = output.clone()
    expected[:, :2] = source[index, :2]
    _run("candidate_b16_row", b16_row_gather, [source, index.to(torch.int32), output], expected, npu)


@pytest.mark.pto
def test_b16_view_gather_with_legal_scatter_axis_numerics(npu):
    source = (torch.arange(512, dtype=torch.int64) % 113 - 56).to(torch.float16)
    index = (torch.arange(256, dtype=torch.int64) * 37 + 11) % 256
    output = torch.full((4, 256), 211, dtype=torch.float16)
    expected = source[32 + index].expand(4, 256).contiguous()
    _run("candidate_b16_view", b16_view_gather, [source, index.to(torch.uint16), output], expected, npu)


@pytest.mark.pto
def test_runtime_float_computation_axis_numerics(npu):
    byte_source = torch.arange(128, dtype=torch.uint8)
    float_source = torch.arange(128, dtype=torch.float32) - 16
    expected_bytes = torch.full((32, 128), 211, dtype=torch.uint8)
    expected_floats = torch.full((32, 128), -211, dtype=torch.float32)
    byte_output = expected_bytes.to(npu)
    float_output = expected_floats.to(npu)
    kernel = tilelang.compile(runtime_float_computation_axis, target="pto", out_idx=None)
    kernel(byte_source.to(npu), float_source.to(npu), byte_output, float_output)
    torch.npu.synchronize()
    expected_bytes[:, :4] = byte_source[:4]
    expected_floats[:, :4] = float_source[:4] + 1
    torch.testing.assert_close(byte_output.cpu(), expected_bytes, rtol=0, atol=0)
    torch.testing.assert_close(float_output.cpu(), expected_floats, rtol=0, atol=0)


@pytest.mark.pto
def test_misaligned_continuous_store_uses_scatter_numerics(npu):
    source = (torch.arange(256) - 128).to(torch.int8)
    index = (torch.arange(256) * 37 + 11) % 256
    output = torch.full((4, 264), -211, dtype=torch.int32)
    expected = output.clone()
    expected[:, 1:257] = source[index].to(torch.int32)
    _run(
        "candidate_misaligned_continuous_store",
        misaligned_continuous_store_axis,
        [source, index.to(torch.uint16), output],
        expected,
        npu,
    )


@pytest.mark.pto
def test_padded_version_axis_numerics(npu):
    source = torch.arange(2 * 264, dtype=torch.float32).reshape(2, 264)
    index = torch.zeros(128, dtype=torch.int32)
    index[:2] = torch.tensor([37, 256], dtype=torch.int32)
    output = torch.full((2, 128), -19, dtype=torch.float32)
    expected = output.clone()
    expected[:, :2] = source[torch.arange(2), index[:2].to(torch.int64)]
    _run("padded_version_axis", padded_version_axis, [source, index, output], expected, npu)


@pytest.mark.pto
def test_auto_version_axis_numerics(npu):
    source = (torch.arange(4 * 256).reshape(4, 256) % 197 - 98).to(torch.int16)
    index = (torch.arange(256) * 37 + torch.arange(4)[:, None] * 11) % 256
    output = torch.full((4, 4, 256), -211, dtype=torch.int32)
    expected = source.to(torch.int32).gather(1, index)[:, None, :].expand_as(output)
    _run("auto_version_axis", auto_version_axis, [source, index.to(torch.uint16), output], expected, npu)
