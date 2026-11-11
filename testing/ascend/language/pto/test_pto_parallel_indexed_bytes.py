"""Byte data and offset carriers through automatic Parallel lowering."""

import pytest

import tilelang.ascend.language as T
from test_pto_parallel_gather_scatter import calls, lower_case

pytestmark = pytest.mark.pto


def cast_bind_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), C: T.Tensor((256,), "int32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            y = T.alloc_shared((256,), "int32")
            T.copy(A, x)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    value = T.cast(x[i], "int32")
                    y[i] = value + 1
            T.copy(y, C)

    return main


def byte_gather_kernel(dtype, index_dtype="uint16", widen=False):
    result_dtype = "int32" if widen else dtype

    @T.prim_func
    def main(A: T.Tensor((256,), dtype), I: T.Tensor((256,), index_dtype), C: T.Tensor((256,), result_dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            idx = T.alloc_shared((256,), index_dtype)
            y = T.alloc_shared((256,), result_dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(150):
                    value = x[idx[i]]
                    if widen:
                        promoted = T.cast(value, "int32")
                        y[i] = promoted + 1
                    else:
                        y[i] = value
            T.copy(y, C)

    return main


def byte_scatter_kernel(dtype, index_dtype="uint16"):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), I: T.Tensor((256,), index_dtype), C: T.Tensor((256,), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            idx = T.alloc_shared((256,), index_dtype)
            y = T.alloc_shared((256,), dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(150):
                    y[idx[i]] = x[i]
            T.copy(y, C)

    return main


def byte_gather_scatter_chain_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), I: T.Tensor((256,), "uint16"), J: T.Tensor((256,), "uint16"), C: T.Tensor((256,), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            idx = T.alloc_shared((256,), "uint16")
            dst = T.alloc_shared((256,), "uint16")
            y = T.alloc_shared((256,), dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(J, dst)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(150):
                    y[dst[i]] = x[idx[i]]
            T.copy(y, C)

    return main


def explicitly_widened_byte_offset_kernel(index_dtype):
    owner_size = 65536 if index_dtype == "int8" else 256

    @T.prim_func
    def main(A: T.Tensor((owner_size,), "float16"), I: T.Tensor((128,), index_dtype), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((owner_size,), "float16")
            idx = T.alloc_shared((128,), index_dtype)
            y = T.alloc_shared((128,), "float16")
            T.copy(A, x)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    y[i] = x[T.cast(idx[i], "uint16")]
            T.copy(y, C)

    return main


@pytest.mark.parametrize("index_dtype", ["int8", "uint8"])
def test_byte_offset_keeps_explicit_widening(index_dtype):
    _, captured = lower_case("widen_byte_offset", explicitly_widened_byte_offset_kernel(index_dtype))
    (gather,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    assert str(gather.args[2].dtype) == "uint16x128"


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_ordinary_byte_cast_bind_uses_value_classification(dtype):
    _, captured = lower_case(f"cast_bind_{dtype}", cast_bind_kernel(dtype))
    mod = captured["VectorizeParallelToPTO_after"]
    assert len(calls(mod, "tl.vmi.vcvt")) == 1
    assert len(calls(mod, "tl.vmi.vadd")) == 1
    assert not calls(mod, "tl.vmi.vgather")


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("widen", [False, True])
def test_byte_gather_restores_source_before_consuming_bind(dtype, widen):
    _, captured = lower_case(f"byte_gather_{dtype}_{widen}", byte_gather_kernel(dtype, widen=widen))
    mod = captured["VectorizeParallelToPTO_after"]
    gather = calls(mod, "tl.vmi.vgather")
    assert len(gather) == 1
    assert str(gather[0].dtype) == ("int16x128" if dtype == "int8" else "uint16x128")
    conversions = calls(mod, "tl.vmi.vcvt")
    assert str(conversions[0].dtype) == f"{dtype}x128"
    assert str(conversions[0].annotations["saturate"].value) == "NOSAT"
    if widen:
        assert str(conversions[1].dtype) == ("int32x128" if dtype == "int8" else "uint32x128")
        if dtype == "uint8":
            assert str(calls(mod, "tl.vmi.vinterpret_cast")[0].dtype) == "int32x128"
    assert len(calls(mod, "tl.vmi.vsel")) == 1


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_scatter_keeps_source_dtype(dtype):
    _, captured = lower_case(f"byte_scatter_{dtype}", byte_scatter_kernel(dtype))
    mod = captured["VectorizeParallelToPTO_after"]
    (scatter,) = calls(mod, "tl.vmi.vscatter")
    # The scattered value keeps the source element dtype; a continuous byte
    # load feeds the scatter without any restore or widening step.
    assert str(scatter.args[0].dtype) == f"{dtype}x128"
    assert str(scatter.args[3].dtype) == "uint16x128"
    assert not calls(mod, "tl.vmi.vgather")
    assert not calls(mod, "tl.vmi.vcvt")


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_scatter_consumes_the_restored_gather(dtype):
    _, captured = lower_case(f"byte_chain_{dtype}", byte_gather_scatter_chain_kernel(dtype))
    mod = captured["VectorizeParallelToPTO_after"]
    assert len(calls(mod, "tl.vmi.vgather")) == 1
    (scatter,) = calls(mod, "tl.vmi.vscatter")
    # The byte gather is restored to the source dtype before its scatter
    # consumer; the widened gather result never reaches the store value.
    conversions = calls(mod, "tl.vmi.vcvt")
    assert str(conversions[0].dtype) == f"{dtype}x128"
    assert str(conversions[0].annotations["saturate"].value) == "NOSAT"
    assert str(scatter.args[0].dtype) == f"{dtype}x128"
