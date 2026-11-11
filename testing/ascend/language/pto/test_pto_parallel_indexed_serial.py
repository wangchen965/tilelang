"""B16 indexed accesses under surrounding serial loops."""

import pytest

import tilelang.ascend.language as T
from tilelang import tvm
from test_pto_parallel_gather_scatter import calls, lower_case


def serial_b16_scatter_kernel(case="serial"):
    @T.prim_func
    def main(A: T.Tensor((128,), "float16"), C: T.Tensor((1536,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((128,), "float16")
            y = T.alloc_shared((1536,), "float16")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                if case == "serial":
                    for t in T.serial(2):
                        for i in T.Parallel(128):
                            y[t * 256 + 2 * i] = x[i]
                elif case == "interleaved":
                    for t in T.serial(2):
                        for i in T.Parallel(128):
                            y[2 * i + t] = x[i]
                elif case == "nested":
                    for t in T.serial(2):
                        for u in T.serial(2):
                            for i in T.Parallel(128):
                                y[t * 512 + u * 256 + 2 * i] = x[i]
                elif case == "nonzero_min":
                    for t in T.serial(2, 4):
                        for i in T.Parallel(128):
                            y[t * 256 + 2 * i] = x[i]
                else:
                    for t in T.serial(2):
                        for i in T.Parallel(128):
                            y[t * 256 + 2 * i] = x[i]
                    for u in T.serial(3, 5):
                        for j in T.Parallel(128):
                            y[u * 256 + 2 * j] = x[j]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("case", ["serial", "interleaved", "nested", "nonzero_min", "sequential"])
def test_b16_scatter_preserves_surrounding_serial_ranges(case):
    artifact, captured = lower_case(f"b16_scatter_{case}", serial_b16_scatter_kernel(case))
    converted = captured["VectorizeParallelToPTO_after"]
    scatters = calls(converted, "tl.vmi.vscatter")
    assert len(scatters) == (2 if case == "sequential" else 1)
    assert all(str(scatter.args[3].dtype) == "uint16x128" for scatter in scatters)
    assert not calls(converted, "tl.vmi.vcvt")
    assert "pto.vmi.vscatter(" in artifact.kernel_source


def symbolic_serial_b16_scatter_kernel(outside_vf):
    @T.prim_func
    def main(A: T.Tensor((128,), "float16"), C: T.Tensor((1536,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((128,), "float16")
            y = T.alloc_shared((1536,), "float16")
            T.copy(A, x)
            T.copy(C, y)
            if outside_vf:
                for t in T.serial(2):
                    for u in T.serial(t, t + 2):
                        with T.SimdVF(lanes=128):
                            for i in T.Parallel(128):
                                y[(2 * t + u) * 256 + 2 * i] = x[i]
            else:
                with T.SimdVF(lanes=128):
                    for t in T.serial(2):
                        for u in T.serial(t, t + 2):
                            for i in T.Parallel(128):
                                y[(2 * t + u) * 256 + 2 * i] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("outside_vf", [False, True])
@pytest.mark.parametrize("auto_schedule", [False, True])
def test_b16_scatter_preserves_symbolic_serial_range(outside_vf, auto_schedule):
    _, captured = lower_case("b16_scatter_symbolic_serial", symbolic_serial_b16_scatter_kernel(outside_vf), auto_schedule=auto_schedule)
    (scatter,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vscatter")
    assert str(scatter.args[3].dtype) == "uint16x128"


@T.prim_func
def serial_b16_gather_with_view_origin(
    A: T.Tensor((65568,), "float16"),
    I: T.Tensor((128,), "uint16"),
    C: T.Tensor((2, 128), "float16"),
):
    with T.Kernel(1):
        x = T.alloc_shared((65568,), "float16")
        idx = T.alloc_shared((128,), "uint16")
        y = T.alloc_shared((2, 128), "float16")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for t in T.serial(2):
                view = tvm.tirx.decl_buffer((128,), "float16", data=x.data, elem_offset=T.min(t, 1) + 32, scope="shared.dyn")
                for i in T.Parallel(128):
                    y[t, i] = view[idx[i]]
        T.copy(y, C)


@pytest.mark.pto
def test_b16_gather_rebases_symbolic_view_origin():
    _, captured = lower_case("b16_gather_symbolic_view", serial_b16_gather_with_view_origin)
    converted = captured["VectorizeParallelToPTO_after"]
    (gather,) = calls(converted, "tl.vmi.vgather")
    assert str(gather.args[2].dtype) == "uint16x128"
    assert not calls(converted, "tl.vmi.vcvt")
