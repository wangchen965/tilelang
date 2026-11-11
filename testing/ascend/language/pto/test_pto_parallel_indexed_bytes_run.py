"""NPU checks for byte gather/scatter semantics and byte offset signedness."""

import pytest
import torch

from test_pto_parallel_gather_scatter import indexed_kernel
from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_bytes import (
    byte_gather_kernel,
    byte_gather_scatter_chain_kernel,
    byte_scatter_kernel,
    cast_bind_kernel,
    explicitly_widened_byte_offset_kernel,
)

npu = _npu

pytestmark = pytest.mark.pto


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_cast_bind_numerics(dtype, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    output = torch.zeros(256, dtype=torch.int32)
    _run(f"cast_bind_{dtype}", cast_bind_kernel(dtype), [source, output], source.to(torch.int32) + 1, npu)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("widen", [False, True])
def test_byte_gather_source_sign_and_tail(dtype, widen, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    # Includes -128, -1, 0, 1, 127 in the gathered source for signed payloads.
    offsets = ((torch.arange(256, dtype=torch.int32) * 37 + 11) % 256).to(torch.uint16)
    output = torch.full((256,), 73, dtype=torch.int32 if widen else source.dtype)
    expected = output.clone()
    values = source[offsets[:150].to(torch.int64)]
    expected[:150] = values.to(torch.int32) + 1 if widen else values
    _run(f"byte_gather_{dtype}_{widen}", byte_gather_kernel(dtype, widen=widen), [source, offsets, output], expected, npu)


@pytest.mark.parametrize("index_dtype", ["int8", "uint8"])
def test_explicit_byte_offset_widening_numerics(index_dtype, npu):
    owner_size = 65536 if index_dtype == "int8" else 256
    source = (torch.arange(owner_size, dtype=torch.int64) % 113).to(torch.float16)
    offsets = ((torch.arange(128, dtype=torch.int32) * 37 + 131) % 256).to(getattr(torch, index_dtype))
    output = torch.empty(128, dtype=torch.float16)
    # An explicit uint16 cast sign-extends signed bytes before reinterpreting
    # them. Negative inputs address the end of the 65536-element owner.
    positions = offsets.to(torch.int64) % 65536
    _run(
        f"widen_byte_offset_{index_dtype}",
        explicitly_widened_byte_offset_kernel(index_dtype),
        [source, offsets, output],
        source[positions],
        npu,
    )


@pytest.mark.parametrize("payload", ["float16", "float32"])
@pytest.mark.parametrize("index_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_byte_offsets_reinterpret_before_widening(payload, index_dtype, kind, npu):
    source = torch.arange(256, dtype=torch.float32).to(getattr(torch, payload))
    # The signed representation is negative for positions 128..255. Offsets
    # are same-width unsigned positions before extension, as in PTODSL.
    positions = (torch.arange(256, dtype=torch.int32) * 37 + 131) % 256
    offsets = positions.to(getattr(torch, index_dtype))
    output = torch.full((256,), -19, dtype=source.dtype)
    expected = output.clone()
    if kind == "gather":
        expected[:150] = source[positions[:150].to(torch.int64)]
    else:
        expected[positions[:150].to(torch.int64)] = source[:150]
    _run(
        f"byte_offsets_{payload}_{index_dtype}_{kind}",
        indexed_kernel(kind, payload, 128, 150, index_dtype=index_dtype),
        [source, offsets, output],
        expected,
        npu,
    )


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_scatter_source_sign_and_tail(dtype, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    # A permutation keeps every scattered target unique, so the result does
    # not depend on duplicate-offset write behavior.
    targets = ((torch.arange(256, dtype=torch.int32) * 37 + 11) % 256).to(torch.uint16)
    output = torch.full((256,), 73, dtype=source.dtype)
    expected = output.clone()
    expected[targets[:150].to(torch.int64)] = source[:150]
    _run(f"byte_scatter_{dtype}", byte_scatter_kernel(dtype), [source, targets, output], expected, npu)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_gather_scatter_chain_numerics(dtype, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    gather_offsets = ((torch.arange(256, dtype=torch.int32) * 37 + 11) % 256).to(torch.uint16)
    scatter_targets = ((torch.arange(256, dtype=torch.int32) * 53 + 7) % 256).to(torch.uint16)
    output = torch.full((256,), 73, dtype=source.dtype)
    expected = output.clone()
    expected[scatter_targets[:150].to(torch.int64)] = source[gather_offsets[:150].to(torch.int64)]
    _run(
        f"byte_chain_{dtype}",
        byte_gather_scatter_chain_kernel(dtype),
        [source, gather_offsets, scatter_targets, output],
        expected,
        npu,
    )
