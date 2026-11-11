"""Chunk-uniform B16 windows within allocations larger than ui16 offsets."""

import pytest

from test_pto_parallel_gather_scatter import T, calls, lower_case, tvm


def chunk_window_kernel(kind="gather", extent=150):
    source_size = 70000 if kind == "gather" else 256
    output_size = 256 if kind == "gather" else 70000

    @T.prim_func
    def main(A: T.Tensor((source_size,), "float16"), I: T.Tensor((256,), "uint16"), C: T.Tensor((output_size,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((source_size,), "float16")
            idx = T.alloc_shared((256,), "uint16")
            y = T.alloc_shared((output_size,), "float16")
            x_view = tvm.tirx.decl_buffer((2, 256), "float16", data=x.data, strides=(256, 1), elem_offset=66000, scope="shared.dyn")
            y_view = tvm.tirx.decl_buffer((2, 256), "float16", data=y.data, strides=(256, 1), elem_offset=66000, scope="shared.dyn")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    if kind == "gather":
                        y[i] = x_view[i // 128, idx[i]]
                    else:
                        y_view[i // 128, idx[i]] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("extent", [150, 256])
def test_chunk_uniform_b16_base_uses_the_emitted_chunk(kind, extent):
    _, captured = lower_case(f"chunk_window_{kind}_{extent}", chunk_window_kernel(kind, extent))
    mod = captured["VectorizeParallelToPTO_after"]
    operation = calls(mod, "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter")[0]
    pointer_index = 0 if kind == "gather" else 1
    pointer = operation.args[pointer_index]
    assert int(pointer.args[1]) == 70000
    assert int(pointer.args[2]) == (1 if kind == "gather" else 2)
    base = operation.args[pointer_index + 1]
    assert "q" in str(base) and "i" not in str(base)
    assert "66000" in str(base) and "256" in str(base)
    assert str(operation.args[pointer_index + 2].dtype) == "uint16x128"


def alternate_axis_window_kernel(kind="gather"):
    source_shape = (128, 600) if kind == "gather" else (2, 128)
    output_shape = (2, 128) if kind == "gather" else (128, 600)

    @T.prim_func
    def main(A: T.Tensor(source_shape, "float16"), C: T.Tensor(output_shape, "float16")):
        with T.Kernel(1):
            x = T.alloc_shared(source_shape, "float16")
            y = T.alloc_shared(output_shape, "float16")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i, j in T.Parallel(2, 128):
                    if kind == "gather":
                        y[i, j] = x[j, 2 * i]
                    else:
                        y[j, 2 * i] = x[i, j]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_affine_offsets_reject_unrepresentable_window_or_implicit_narrowing(kind):
    # Neither axis is lowerable: the wide direction exceeds the ui16 window,
    # while the other still requires forbidden implicit int32-to-ui16 narrowing.
    with pytest.raises(Exception, match="implicit narrowing|ui16 relative window"):
        lower_case(f"rejected_alternate_axis_{kind}", alternate_axis_window_kernel(kind))
