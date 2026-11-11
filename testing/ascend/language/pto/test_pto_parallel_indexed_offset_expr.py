"""Verify complete indexed offsets using the expression lowering contract."""

import pytest

from test_pto_parallel_gather_scatter import ascend_transform, calls, indexed_kernel, lower_case, tvm
from test_pto_parallel_indexed_control_flow import replace_unit_body
from tilelang.ascend.target import normalize_pto_target
import tilelang.ascend.language as T


def scalar_coordinate_cast_kernel(coordinate="chunk", extent=256):
    @T.prim_func
    def main(A: T.Tensor((512,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((512,), "float32")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    if coordinate == "chunk":
                        y[i] = x[idx[i] + T.Cast("int32", T.Cast("int16", i // 128)) * 128]
                    else:
                        y[i] = x[T.Cast("uint32", idx[i] + T.Select(idx[0] > 0, 0, 1))]
            T.copy(y, C)

    return main


@pytest.fixture(scope="module")
def source_module():
    _, captured = lower_case("indexed_offset_expression_base", indexed_kernel())
    return captured["VerifyParallelToPTO_before"]


@pytest.mark.pto
@pytest.mark.parametrize("coordinate", ["chunk", "select"])
def test_cast_input_may_include_a_scalar_index_coordinate(source_module, coordinate):
    def make_body(i, payload, index, output):
        if coordinate == "chunk":
            scalar = tvm.tirx.Cast("int16", i // 128)
            offset = index[i] + tvm.tirx.Cast("int32", scalar) * 128
        else:
            scalar = tvm.tirx.Select(index[0] > 0, 0, 1)
            offset = tvm.tirx.Cast("uint32", index[i] + scalar)
        return tvm.tirx.BufferStore(output, payload[offset], [i])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"):
        verified = ascend_transform.VerifyParallelToPTO()(mod)
        assert tvm.ir.structural_equal(mod, verified), "verification must not rewrite scalar coordinates or source casts"
        converted = ascend_transform.VectorizeParallelToPTO()(verified)
    assert len(calls(converted, "tl.vmi.vgather")) == 1
    assert len(calls(converted, "tl.vmi.vload")) >= 1
    if coordinate == "select":
        assert len(calls(converted, "tl.vmi.vinterpret_cast")) == 1, "the explicit same-width source Cast must survive"


@pytest.mark.pto
def test_complete_offset_checks_references_introduced_by_view_metadata(source_module):
    def make_body(i, payload, index, output):
        ghost = tvm.tirx.Var("undefined_view_origin", "int32")
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=ghost, scope=payload.scope())
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="undefined_view_origin.*never defined"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
def test_uniform_store_checks_operations_introduced_by_view_metadata(source_module):
    def make_body(i, payload, index, output):
        origin = tvm.tirx.call_extern("int32", "opaque_origin")
        view = tvm.tirx.decl_buffer(output.shape, output.dtype, data=output.data, elem_offset=origin, scope=output.scope())
        return tvm.tirx.BufferStore(view, payload[i], [0])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="complete indexed offset.*unsupported call"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["predicate", "conditional_gather"])
def test_metadata_index_reads_keep_their_own_admission_rules(source_module, kind):
    def make_body(i, payload, index, output):
        if kind == "predicate":
            origin = tvm.tirx.BufferLoad(index, [i], predicate=i < 150)
        else:
            origin = tvm.tirx.Select(index[0] > 0, index[index[i]], index[i])
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=origin, scope=payload.scope())
        if kind == "conditional_gather":
            return tvm.tirx.Evaluate(view[index[i]])
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    diagnostic = (
        "non-trivial access predicate"
        if kind == "predicate"
        else "requires gather.*Select true value|unsupported indexed offset node.*Select"
    )
    with normalize_pto_target("pto"), pytest.raises(Exception, match=diagnostic):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
def test_bound_external_view_origin_survives_output_reference_check(source_module):
    base = tvm.tirx.Var("external_view_origin", "int32")

    def make_body(i, payload, index, output):
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=base, scope=payload.scope())
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    functions = {}
    for name, func in mod.functions.items():
        functions[name] = tvm.tirx.PrimFunc([*func.params, base], func.body, func.ret_type, func.buffer_map, func.attrs)
    mod = tvm.IRModule(functions)
    with normalize_pto_target("pto"):
        verified = ascend_transform.VerifyParallelToPTO()(mod)
        converted = ascend_transform.VectorizeParallelToPTO()(verified)
    assert len(calls(converted, "tl.vmi.vgather")) == 1


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["floor_mod", "floor_div"])
def test_varying_coordinate_operation_still_requires_a_lowering(source_module, kind):
    def make_body(i, payload, index, output):
        offset = index[i] % 128 if kind == "floor_mod" else index[i] // 2
        return tvm.tirx.BufferStore(output, payload[offset], [i])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="unsupported indexed offset node"):
        ascend_transform.VerifyParallelToPTO()(mod)
