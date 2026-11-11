"""Choose a lowerable vector axis before comparing costs."""

import pytest

from test_pto_parallel_gather_scatter import T, ascend_transform, assert_vector_axis, calls, lower_case, normalize_pto_target, tvm


@T.prim_func
def b16_row_gather(A: T.Tensor((128, 128), "float16"), I: T.Tensor((128,), "int32"), C: T.Tensor((128, 128), "float16")):
    with T.Kernel(1):
        x = T.alloc_shared((128, 128), "float16")
        idx = T.alloc_shared((128,), "int32")
        y = T.alloc_shared((128, 128), "float16")
        T.copy(A, x)
        T.copy(I, idx)
        T.copy(C, y)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(128, 2):
                y[i, j] = x[idx[i], j]
        T.copy(y, C)


@T.prim_func
def b16_view_gather(A: T.Tensor((512,), "float16"), I: T.Tensor((256,), "uint16"), C: T.Tensor((4, 256), "float16")):
    with T.Kernel(1):
        x = T.alloc_shared((512,), "float16")
        view = tvm.tirx.decl_buffer((256,), "float16", data=x.data, elem_offset=32, scope="shared.dyn")
        idx = T.alloc_shared((256,), "uint16")
        y = T.alloc_shared((4, 256), "float16")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                y[i, j] = view[idx[j]]
        T.copy(y, C)


@T.prim_func
def padded_version_axis(A: T.Tensor((2, 264), "float32"), I: T.Tensor((128,), "int32"), C: T.Tensor((2, 128), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((2, 257), "float32")
        T.annotate_manual_multi_buffer(x)
        idx = T.alloc_shared((128,), "int32")
        y = T.alloc_shared((2, 128), "float32")
        T.copy(A[0, :257], x[0, :])
        T.copy(A[1, :257], x[1, :])
        T.copy(I, idx)
        T.copy(C, y)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(2, 2):
                y[i, j] = x[j, idx[j]]
        T.copy(y, C)


@pytest.mark.pto
@pytest.mark.parametrize("auto_schedule", [False, True])
def test_candidate_keeps_padded_version_selector_uniform(auto_schedule):
    _, captured = lower_case("padded_version_axis", padded_version_axis, auto_schedule=auto_schedule)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 0)
    converted = captured["VectorizeParallelToPTO_after"]
    assert not calls(converted, "tl.vmi.vgather")
    assert len(calls(converted, "tl.vmi.vscatter")) == 1


@T.prim_func
def byte_offset_mul(A: T.Tensor((256,), "int32"), I: T.Tensor((256,), "uint8"), C: T.Tensor((4, 256), "int32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "int32")
        idx = T.alloc_shared((256,), "uint8")
        y = T.alloc_shared((4, 256), "int32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                y[i, j] = x[idx[j] * T.uint8(3)]
        T.copy(y, C)


@T.prim_func
def fp8_uniform_store(A: T.Tensor((128,), "float8_e4m3fn"), C: T.Tensor((128,), "float8_e4m3fn")):
    with T.Kernel(1):
        x = T.alloc_shared((128,), "float8_e4m3fn")
        y = T.alloc_shared((128,), "float8_e4m3fn")
        T.copy(A, x)
        T.copy(C, y)
        with T.SimdVF(lanes=128):
            for i, _j in T.Parallel(4, 128):
                y[i] = x[i]
        T.copy(y, C)


@T.prim_func
def reinterpreted_gather(A: T.Tensor((256,), "int32"), I: T.Tensor((256,), "int32"), C: T.Tensor((4, 256), "float32")):
    with T.Kernel(1):
        raw = T.alloc_shared((256,), "int32")
        view = T.view(raw, shape=(256,), dtype="float32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((4, 256), "float32")
        T.copy(A, raw)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                y[i, j] = view[idx[j]]
        T.copy(y, C)


@T.prim_func
def nested_gather_in_view_origin(
    A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), J: T.Tensor((256,), "uint16"), C: T.Tensor((4, 256), "float32")
):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        idx32 = T.alloc_shared((256,), "int32")
        idx16 = T.alloc_shared((256,), "uint16")
        y = T.alloc_shared((4, 256), "float32")
        T.copy(A, x)
        T.copy(I, idx32)
        T.copy(J, idx16)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                view = tvm.tirx.decl_buffer((1,), "float32", data=x.data, elem_offset=idx16[idx32[j]], scope="shared.dyn")
                y[i, j] = view[0]
        T.copy(y, C)


@pytest.mark.pto
@pytest.mark.parametrize(
    "kernel,axis,scatters",
    [
        (b16_row_gather, 1, 0),
        (b16_view_gather, 0, 1),
        (byte_offset_mul, 0, 1),
        (fp8_uniform_store, 0, 0),
        (nested_gather_in_view_origin, 0, 1),
        (reinterpreted_gather, 0, 1),
    ],
)
def test_candidate_rejects_unsupported_indexed_operation_before_ranking(kernel, axis, scatters):
    _, captured = lower_case(kernel.attrs["global_symbol"], kernel)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], axis)
    converted = captured["VectorizeParallelToPTO_after"]
    assert not calls(converted, "tl.vmi.vgather")
    assert len(calls(converted, "tl.vmi.vscatter")) == scatters
    assert not calls(converted, "tl.vmi.vcvt"), "choosing another axis must not truncate loaded offsets"


@T.prim_func
def runtime_float_computation_axis(
    A: T.Tensor((128,), "uint8"),
    B: T.Tensor((128,), "float32"),
    C: T.Tensor((32, 128), "uint8"),
    D: T.Tensor((32, 128), "float32"),
):
    with T.Kernel(1):
        x = T.alloc_shared((128,), "uint8")
        b = T.alloc_shared((128,), "float32")
        y = T.alloc_shared((32, 128), "uint8")
        z = T.alloc_shared((32, 128), "float32")
        T.copy(A, x)
        T.copy(B, b)
        T.copy(C, y)
        T.copy(D, z)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(32, 4):
                y[i, j] = x[j]
                z[i, j] = b[j] + T.float32(1)
        T.copy(y, C)
        T.copy(z, D)


@pytest.mark.pto
def test_candidate_keeps_runtime_float_computation_vectorized():
    _, captured = lower_case("runtime_float_computation_axis", runtime_float_computation_axis)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 1)
    converted = captured["VectorizeParallelToPTO_after"]
    assert len(calls(converted, "tl.vmi.vadd")) == 1
    assert not calls(converted, "tl.vmi.vscatter")


@T.prim_func
def misaligned_continuous_store_axis(A: T.Tensor((256,), "int8"), I: T.Tensor((256,), "uint16"), C: T.Tensor((4, 264), "int32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "int8")
        idx = T.alloc_shared((256,), "uint16")
        y = T.alloc_shared((4, 264), "int32")
        T.copy(A, x)
        T.copy(I, idx)
        T.copy(C, y)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                y[i, j + 1] = T.cast(x[idx[j]], "int32")
        T.copy(y, C)


@pytest.mark.pto
def test_candidate_rejects_misaligned_continuous_store():
    _, captured = lower_case("misaligned_continuous_store_axis", misaligned_continuous_store_axis)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 0)
    converted = captured["VectorizeParallelToPTO_after"]
    assert not calls(converted, "tl.vmi.vgather")
    assert len(calls(converted, "tl.vmi.vscatter")) == 1


@T.prim_func
def no_b16_candidate(A: T.Tensor((256,), "float16"), I: T.Tensor((128, 128), "int32"), C: T.Tensor((128, 128), "float16")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float16")
        idx = T.alloc_shared((128, 128), "int32")
        y = T.alloc_shared((128, 128), "float16")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(128, 128):
                y[i, j] = x[idx[i, j]]
        T.copy(y, C)


@pytest.mark.pto
def test_rejected_candidates_report_each_axis_failure():
    with pytest.raises(Exception, match=r"inner \(j\).*implicit narrowing.*outer \(i\).*implicit narrowing"):
        lower_case("no_b16_candidate", no_b16_candidate)


@T.prim_func
def auto_version_axis(A: T.Tensor((4, 256), "int16"), I: T.Tensor((4, 256), "uint16"), C: T.Tensor((4, 4, 256), "int32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "int16")
        idx = T.alloc_shared((256,), "uint16")
        y = T.alloc_shared((4, 256), "int32")
        T.annotate_buffer_versions({x: 2, idx: 2, y: 2})
        for t in T.Pipelined(4, num_stages=2):
            T.copy(A[t, :], x)
            T.copy(I[t, :], idx)
            with T.SimdVF(lanes=128):
                for i, j in T.Parallel(4, 256):
                    y[i, j] = T.cast(x[idx[j]], "int32")
            T.copy(y, C[t, :, :])


@pytest.mark.pto
def test_auto_versioning_preserves_inferred_axis_and_native_offset():
    _, captured = lower_case("auto_version_axis", auto_version_axis)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 1)
    assert_vector_axis(captured["VerifyParallelToPTO_after"], 1)
    converted = captured["VectorizeParallelToPTO_after"]
    (gather,) = calls(converted, "tl.vmi.vgather")
    assert str(gather.args[2].dtype) == "uint16x128"
    origin, extent, _ = gather.args[0].args
    assert [int(dim) for dim in origin.buffer.shape] == [2, 256]
    assert not isinstance(origin.indices[0], tvm.tirx.IntImm)
    assert int(origin.indices[1]) == 0
    assert int(extent) == 256


@pytest.mark.pto
def test_empty_axis_reports_loop_extent_before_padding():
    @T.prim_func
    def main():
        with T.Kernel(1):
            y = T.alloc_shared((4, 128), "int32")
            with T.SimdVF(lanes=128):
                for i, j in T.Parallel(4, 0):
                    y[i, j] = 1

    target = normalize_pto_target("pto")
    with target, pytest.raises(Exception, match="positive compile-time constant loop extents, got 0 for loop j"):
        ascend_transform.AscendLayoutInference()(tvm.IRModule.from_expr(main.with_attr("target", target)))
