// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file vectorize_parallel_to_pto.cc
 * \brief VectorizeParallelToPTO: rewrite verified T.Parallel loops inside
 * PTO SIMD_VF blocks into logical VMI code.
 *
 * Consumes the pto_parallel_loop_layout fragment written by
 * LayoutInference and verified by VerifyParallelToPTO. Per parallel unit:
 *   1. invert the lane fragment through the shared helper (the inverse
 *      layout's own Forward result is the single source of truth);
 *   2. build one chunk loop q in [0, ceil(E/L)), mapping the logical
 *      index to q * L + lane;
 *   3. classify every access via the shared MemoryAccess analysis
 *      (continuous -> vload/vstore via tl.access_ptr(start, L, rw);
 *      lane-uniform load -> scalar BufferLoad; indexed load/store and
 *      lane-uniform store -> masked gather/scatter with element offsets);
 *   4. convert value expressions (value roles tracked by result dtype);
 *      every VMI with a return value is bound through EmitBind before use;
 *   5. delete the consumed layout annotation, keep the SIMD_VF block
 *      annotations, and leave no unconverted Parallel behind.
 *
 * Supported shapes: 1D parallel units and two-layer adjacent Parallel nests
 * (2D — the nest converts as one unit with the layout
 * annotation on its outermost For); sequential units allowed; static
 * extents with full or tail masks. A third parallel layer or any other
 * unsupported structure is rejected by Verify.
 *
 * Non-divisible extents use a per-chunk mask in the same loop. Continuous
 * vload is unmasked: the caller must provide initialized readable padding
 * for every complete logical vector. Indexed offsets are sanitized in every
 * iteration when the unit has a tail, and passed directly otherwise.
 */

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "buffer_version.h"
#include "parallel_to_pto_utils.h"
#include "transform/common/storage_size.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

DataType VectorDType(DataType elem, int64_t lanes) {
  return elem.with_lanes(static_cast<int>(lanes));
}

const Op &VmiOp(const std::string &tail) { return Op::Get("tl.vmi." + tail); }

// This version keeps local DCE in Vectorize so
// its directly emitted TIR/PTODSL is cleaner and easier to inspect.
// Cleanup may later move to PTOAS canonicalization/DCE to reduce
// TileLang-specific code and maintenance; clean-output checks would then
// target the optimized PTOAS IR. The emitted IR must be valid and preserve
// program semantics even before DCE.
//
// Dead vector-chain elimination on one converted chunk body. The
// conversion can leave Bind statements whose Var is never referenced:
// source binds consumed only through the address alias environment
// (bind_env_) never use their converted Var, and an unused source bind
// strands its whole value chain (vci/vbrc/vadd/vload).
//
// The traversal is a scope-aware *reverse liveness* pass — a
// dead Bind must not survive just because it was wrapped in an AttrStmt
// or appears as the only statement of a body. The rules are:
//   1. A SeqStmt is processed back to front; `live` always holds the
//      vars everything *after* the current statement needs.
//   2. A Bind's LHS var is a definition, never a use. When the result is
//      not live and the RHS is discardable, the Bind is deleted and its
//      RHS inputs do NOT become live (so the dead chain collapses in one
//      pass); otherwise it is kept and its RHS inputs join `live`.
//   3. An AttrStmt body is cleaned in its own nested scope with a fresh
//      liveness set; only the vars actually referenced inside propagate
//      outward. Attribute node/value references are always uses, and the
//      attribute structure and evaluation position are preserved.
//   4. A bare Bind follows the same rules; when everything dies the
//      remains are a legal no-op (Evaluate(0)).
// Nothing is merged, reordered or otherwise rescheduled;
// referenced loads stay exactly where they were (vecadd keeps its three
// vloads), and Bind-free (frontend-inlined) bodies are untouched.
class DeadBindElim {
public:
  Stmt operator()(const Stmt &stmt) {
    std::set<const VarNode *> live;
    Stmt out = Run(stmt, &live);
    return out.defined() ? out : NoOp();
  }

private:
  static Stmt NoOp() { return Evaluate(make_zero(DataType::Int(32))); }

  static void CollectExprVars(const PrimExpr &expr,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(expr, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static void CollectStmtVars(const Stmt &stmt,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(stmt, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static bool DiscardableRhs(const PrimExpr &rhs) {
    // Positive "allowed to discard" set: a Call may disappear
    // only when it is one of the tl.vmi.* ops this pass actually emits as
    // a Bind RHS and that is pure/read-only by construction — vload/vgather
    // (reads) and arithmetic, select or conversion ops (pure), enumerated from
    // the ConvertUnit/VmiOp call sites — or the pure tl.access_ptr address
    // descriptor nested inside a load operand (already whitelisted as a
    // pure index intrinsic by IsSupportedAddressExpr). vstore never
    // appears as a Bind RHS; any other Call — call_extern, other tl.* or
    // an unknown op — keeps its Bind. Plain UB BufferLoads are
    // discardable: they are ordinary reads with no write effect (the
    // scalar "sload" binds). That "ordinary" premise is enforced by
    // VerifyParallelToPTO, not assumed here: accesses to
    // tirx.volatile-annotated allocations and volatile_scope attributes
    // are rejected there, and opaque calls are already excluded
    // from addresses.
    static const std::set<std::string> kDiscardable = {
        "tl.vmi.vload",  "tl.vmi.vgather", "tl.vmi.vsel",
        "tl.vmi.vbrc",   "tl.vmi.vci",     "tl.vmi.vadd",
        "tl.vmi.vsub",   "tl.vmi.vmul",    "tl.vmi.vdiv",
        "tl.vmi.vabs",   "tl.vmi.vexp",    "tl.vmi.vln",
        "tl.vmi.vsqrt",  "tl.vmi.vcvt",    "tl.vmi.vinterpret_cast",
        "tl.access_ptr",
    };
    bool discardable = true;
    PostOrderVisit(rhs, [&](const ObjectRef &n) {
      if (const auto *call = n.as<CallNode>()) {
        const auto *op = call->op.as<OpNode>();
        const std::string name = op != nullptr ? op->name : "";
        if (kDiscardable.count(name) == 0) {
          discardable = false;
        }
      }
    });
    return discardable;
  }

  // Returns the kept statement, or an *undefined* Stmt when the statement
  // was deleted (only a Bind can be deleted). `live` is threaded from
  // later statements back to earlier ones.
  Stmt Run(const Stmt &stmt, std::set<const VarNode *> *live) {
    if (const auto *bind = stmt.as<BindNode>()) {
      if (live->count(bind->var.get()) == 0 && DiscardableRhs(bind->value)) {
        // Dead and safe to drop: the RHS inputs stay out of `live`, so a
        // chain feeding only this Bind collapses in the same pass.
        return Stmt();
      }
      // Kept: classic liveness — subtract the definition, add the uses.
      live->erase(bind->var.get());
      CollectExprVars(bind->value, live);
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> kept;
      for (auto it = seq->seq.rbegin(); it != seq->seq.rend(); ++it) {
        Stmt sub = Run(*it, live);
        if (sub.defined()) {
          kept.push_back(sub);
        }
      }
      if (kept.empty()) {
        return NoOp();
      }
      // std::reverse does not accept the ffi Array iterators on this
      // toolchain; re-append in reverse instead.
      Array<Stmt> forward;
      for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
        forward.push_back(*it);
      }
      return SeqStmt::Flatten(forward);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // Nested scope: clean the body with a fresh liveness set, then
      // propagate only the references that survive to its entry (external
      // uses; vars defined and used only inside never escape because the
      // converted Vars are unique objects). Attribute node/value
      // references are always uses and keep the structure.
      std::set<const VarNode *> body_live;
      Stmt body = Run(attr->body, &body_live);
      if (!body.defined()) {
        body = NoOp();
      }
      live->insert(body_live.begin(), body_live.end());
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          CollectExprVars(GetRef<PrimExpr>(pe), live);
        }
      }
      CollectExprVars(attr->value, live);
      if (body.same_as(attr->body)) {
        return stmt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    // Any other statement is always kept; all its Var occurrences are
    // uses.
    CollectStmtVars(stmt, live);
    return stmt;
  }
};

class VectorizeParallelToPTOImpl : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    VectorizeParallelToPTOImpl mutator;
    for (const Var &parameter : f->params) {
      mutator.parameters_.insert(parameter);
    }
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // AllocBuffer is a flat definition visible to later siblings. Every other
  // statement restores the owner environment on leaving its lexical scope.
  Stmt VisitStmt(const Stmt &stmt) final {
    if (stmt.as<AllocBufferNode>()) {
      return StmtExprMutator::VisitStmt(stmt);
    }
    const size_t allocation_mark = active_allocations_.size();
    Stmt result = StmtExprMutator::VisitStmt(stmt);
    active_allocations_.resize(allocation_mark);
    return result;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      active_allocations_.push_back(buffer);
    }
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Shared region scan: the same implementation Verify uses, so
    // skip/reject/convert decisions cannot drift apart.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel && !scan.has_direct_reduce && !scan.has_fill) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // by both passes; no lanes annotation needed.
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK(!scan.has_unsupported_wrapper)
        << "[VectorizeParallelToPTO] internal error: unsupported wrapper "
           "structure passed Verify: "
        << scan.issue;

    // Keep the same inverse for mask selection and conversion. The For
    // visitor reaches each unit with its lexical loop and allocation scopes.
    bool needs_full_mask = false;
    for (const For &unit : scan.units) {
      auto mapping = pto_analysis::InvertPtoLaneLayout(unit);
      needs_full_mask |= mapping.tail_lanes == 0;
      unit_mappings_.emplace(unit, std::move(mapping));
    }
    // A reduce-only VF has no Parallel layout from which to recover lanes.
    auto lanes_annotation = op->annotations.Get("tl.simdvf_lanes");
    ICHECK(lanes_annotation.has_value());
    int64_t lanes = Downcast<IntImm>(lanes_annotation.value())->value;
    shared_mask_ = Var();
    Stmt new_region = region;
    if (needs_full_mask) {
      shared_mask_ = Var("mask", DataType::Bool(static_cast<int>(lanes)));
      PrimExpr mask_value =
          Call(shared_mask_.dtype(), VmiOp("create_mask"),
               {IntImm(DataType::Int(32), static_cast<int>(lanes))},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes))}});
      // VPTO requires the shared mask and its uses to be in the same vecscope.
      Stmt mask_bind = Bind(shared_mask_, mask_value);
      if (const auto *scope = region.as<AttrStmtNode>();
          scope && scope->attr_key == "tl.simdvf_scope") {
        AttrStmt attr = GetRef<AttrStmt>(scope);
        attr.CopyOnWrite()->body = SeqStmt::Flatten(mask_bind, scope->body);
        new_region = std::move(attr);
      } else {
        new_region = SeqStmt::Flatten(mask_bind, region);
      }
    }

    current_vf_lanes_ = lanes;
    Stmt new_body = VisitStmt(new_region);
    current_vf_lanes_ = 0;
    unit_mappings_.clear();

    pto_analysis::PtoRegionScan residual =
        pto_analysis::ScanPtoRegion(new_body);
    ICHECK(!residual.has_direct_reduce && !residual.has_fill)
        << "[VectorizeParallelToPTO] direct tileops "
           "must be standalone Evaluate(Call) statements";

    // new_body is (mask bind + region with all units converted); the
    // scope AttrStmt wrapper, if any, is inside it already.
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                  op->init, op->alloc_buffers, op->match_buffers,
                  op->annotations);
  }

  Stmt LowerDirectReduce(const Call &call, int64_t lanes) {
    pto_analysis::PtoDirectReducePlan plan =
        pto_analysis::PlanPtoDirectReduce(call, analyzer_, buffer_versions_);
    bool absolute = plan.operation == "abssum" || plan.operation == "absmax";
    bool sum = plan.operation == "sum" || plan.operation == "abssum";
    bool maximum = plan.operation == "max" || plan.operation == "absmax";
    auto i32 = [](int64_t value) { return IntImm(DataType::Int(32), value); };
    DataType vector_dtype = plan.accumulator_dtype.with_lanes(lanes);
    DataType mask_dtype = DataType::Bool(lanes);
    Buffer accumulator =
        tirx::decl_buffer({i32(1)}, vector_dtype, "reduce_acc", "local.var");
    Var chunk("reduce_chunk", DataType::Int(32));
    Var output("reduce_output", DataType::Int(32));
    Var reduction("reduce_index", DataType::Int(32));
    Var full_mask("reduce_domain", mask_dtype);
    Var active_mask("reduce_active", mask_dtype);
    auto mask = [&](const PrimExpr &active) {
      return Call(mask_dtype, VmiOp("create_mask"), {active},
                  {{"size", i32(lanes)}});
    };
    auto pointer = [&](const BufferRegion &region,
                       const Array<PrimExpr> &indices, int64_t extent, int rw) {
      return Call(DataType::Handle(), tl::access_ptr(),
                  {BufferLoad(region->buffer, indices), i32(extent), i32(rw)});
    };
    auto source_indices = [&](const PrimExpr &out, const PrimExpr &red) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.source->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.source_prefix + plan.reduction_axis, red);
      if (plan.logical_rank == 2) {
        indices.Set(plan.source_prefix + plan.output_axis, out);
      }
      return indices;
    };
    auto destination_indices = [&](const PrimExpr &out) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.destination->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.destination_output_axis, out);
      return indices;
    };
    auto load = [&](const PrimExpr &out, const PrimExpr &red,
                    Array<Stmt> *statements) {
      DataType src_dtype = plan.source->buffer->dtype;
      int64_t footprint =
          ((lanes + 2048 / src_dtype.bits() - 1) / (2048 / src_dtype.bits())) *
          (2048 / src_dtype.bits());
      PrimExpr value = EmitBind(
          "reduce_load",
          Call(src_dtype.with_lanes(lanes), VmiOp("vload"),
               {pointer(plan.source, source_indices(out, red), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          statements);
      if (src_dtype != plan.accumulator_dtype) {
        value =
            EmitBind("reduce_cast",
                     Call(vector_dtype, VmiOp("vcvt"), {value},
                          {{"to_dtype",
                            StringImm(TvmDtypeName(plan.accumulator_dtype))}}),
                     statements);
      }
      if (absolute && !plan.accumulator_dtype.is_uint()) {
        value =
            EmitBind("reduce_abs", Call(vector_dtype, VmiOp("vabs"), {value}),
                     statements);
      }
      return value;
    };
    auto read_accumulator = [&]() { return BufferLoad(accumulator, {i32(0)}); };
    const char *fold_name = sum ? "vadd" : maximum ? "vmax" : "vmin";
    auto seed = [&](const PrimExpr &first, const PrimExpr &active,
                    Array<Stmt> *statements) -> PrimExpr {
      if (plan.operation != "absmax") {
        return first;
      }
      // Match the common Ascend/CUDA reducer's zero-initialized partial
      PrimExpr zero = EmitBind("reduce_zero",
                               Call(vector_dtype, VmiOp("vbrc"),
                                    {make_zero(plan.accumulator_dtype)},
                                    {{"size", i32(lanes)}}),
                               statements);
      return EmitBind("reduce_seed",
                      Call(vector_dtype, VmiOp("vmax"), {zero, first, active}),
                      statements);
    };
    Array<Stmt> body;
    if (plan.reduce_across_lanes) {
      PrimExpr first = load(output, i32(0), &body);
      first = seed(first, full_mask, &body);
      body.push_back(BufferStore(accumulator, first, {i32(0)}));
      auto accumulate = [&](const PrimExpr &red, const PrimExpr &active,
                            bool keep_inactive) {
        Array<Stmt> update;
        PrimExpr next = load(output, red, &update);
        PrimExpr folded = EmitBind("reduce_fold",
                                   Call(vector_dtype, VmiOp(fold_name),
                                        {read_accumulator(), next, active}),
                                   &update);
        if (keep_inactive) {
          folded = EmitBind("reduce_merge",
                            Call(vector_dtype, VmiOp("vsel"),
                                 {active, folded, read_accumulator()}),
                            &update);
        }
        update.push_back(BufferStore(accumulator, folded, {i32(0)}));
        return SeqStmt::Flatten(update);
      };
      int64_t full_chunks = plan.reduction_extent / lanes;
      int64_t tail_lanes = plan.reduction_extent % lanes;
      // The seed already covers the first chunk. Complete chunks need no
      // remaining-lane calculation or merge with the previous accumulator.
      if (full_chunks > 1) {
        body.push_back(For(chunk, i32(1), i32(full_chunks - 1),
                           ForKind::kSerial,
                           accumulate(chunk * i32(lanes), full_mask, false)));
      }
      if (full_chunks > 0 && tail_lanes != 0) {
        body.push_back(Bind(active_mask, mask(i32(tail_lanes))));
        body.push_back(accumulate(i32(full_chunks * lanes), active_mask, true));
      }
      ffi::Map<ffi::String, ObjectRef> annotations;
      if (sum && plan.accumulator_dtype.is_float()) {
        annotations.Set("reassoc", IntImm(DataType::Bool(), 1));
      }
      const char *reduce_name = sum ? "vcadd" : maximum ? "vcmax" : "vcmin";
      PrimExpr result =
          EmitBind("reduce_result",
                   Call(plan.accumulator_dtype, VmiOp(reduce_name),
                        {read_accumulator(), full_mask}, annotations),
                   &body);
      if (!plan.clear) {
        PrimExpr previous = EmitBind(
            "reduce_previous",
            Call(plan.accumulator_dtype, VmiOp("vload"),
                 {pointer(plan.destination, destination_indices(output), 1, 1),
                  i32(0)},
                 {{"size", i32(1)}}),
            &body);
        PrimExpr one = EmitBind("reduce_one",
                                Call(DataType::Bool(), VmiOp("create_mask"),
                                     {i32(1)}, {{"size", i32(1)}}),
                                &body);
        // Merge the previous output only after the cross-lane reduction;
        // seeding every lane with it would count it VL times for sum.
        result = EmitBind("reduce_update",
                          Call(plan.accumulator_dtype, VmiOp(fold_name),
                               {previous, result, one}),
                          &body);
      }
      body.push_back(Evaluate(Call(
          DataType::Void(), VmiOp("vstore"),
          {result, pointer(plan.destination, destination_indices(output), 1, 2),
           i32(0)})));
      Stmt output_loop = For(output, i32(0), i32(plan.output_extent),
                             ForKind::kSerial, SeqStmt::Flatten(body));
      Stmt compute = SeqStmt::Flatten(
          Bind(full_mask, mask(i32(std::min(plan.reduction_extent, lanes)))),
          output_loop);
      return SeqStmt::Flatten(AllocBuffer(accumulator),
                              AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                       StringImm("data"), compute));
    }
    body.push_back(
        Bind(active_mask, mask(i32(plan.output_extent) - chunk * i32(lanes))));
    PrimExpr first = load(chunk * i32(lanes), i32(0), &body);
    first = seed(first, active_mask, &body);
    body.push_back(BufferStore(accumulator, first, {i32(0)}));
    Array<Stmt> update;
    PrimExpr next = load(chunk * i32(lanes), reduction, &update);
    PrimExpr folded = EmitBind("reduce_fold",
                               Call(vector_dtype, VmiOp(fold_name),
                                    {read_accumulator(), next, active_mask}),
                               &update);
    update.push_back(BufferStore(accumulator, folded, {i32(0)}));
    if (plan.reduction_extent > 1) {
      body.push_back(For(reduction, i32(1), i32(plan.reduction_extent - 1),
                         ForKind::kSerial, SeqStmt::Flatten(update)));
    }
    if (!plan.clear) {
      int64_t physical_lanes = 2048 / plan.accumulator_dtype.bits();
      int64_t footprint =
          ((lanes + physical_lanes - 1) / physical_lanes) * physical_lanes;
      PrimExpr previous = EmitBind(
          "reduce_previous",
          Call(vector_dtype, VmiOp("vload"),
               {pointer(plan.destination,
                        destination_indices(chunk * i32(lanes)), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          &body);
      PrimExpr combined =
          EmitBind("reduce_update",
                   Call(vector_dtype, VmiOp(fold_name),
                        {previous, read_accumulator(), active_mask}),
                   &body);
      body.push_back(BufferStore(accumulator, combined, {i32(0)}));
    }
    body.push_back(Evaluate(
        Call(DataType::Void(), VmiOp("vstore"),
             {read_accumulator(),
              pointer(plan.destination, destination_indices(chunk * i32(lanes)),
                      lanes, 2),
              i32(0), active_mask})));
    // Full chunks and the tail use the same vector reduction. Loads may read
    // complete physical lanes; active_mask limits the fold and output store.
    int64_t chunks = (plan.output_extent + lanes - 1) / lanes;
    Stmt compute = For(chunk, i32(0), i32(chunks), ForKind::kSerial,
                       SeqStmt::Flatten(body));
    return SeqStmt::Flatten(AllocBuffer(accumulator),
                            AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                     StringImm("data"), compute));
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kBufferVersion) {
      BufferVersionMap saved = buffer_versions_;
      for (const auto &[storage, count] : op->node.cast<BufferVersionMap>()) {
        buffer_versions_.Set(storage, count);
      }
      Stmt result = StmtExprMutator::VisitStmt_(op);
      buffer_versions_ = std::move(saved);
      return result;
    }
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        With<arith::ConstraintContext> launch_bounds(
            analyzer_, iv->var >= 0 && iv->var < op->value);
        return StmtExprMutator::VisitStmt_(op);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    active_allocations_.push_back(op->buffer);
    return GetRef<Stmt>(op);
  }

  bool IsScalarFillParameter(const PrimExpr &value) const {
    if (value.dtype() != DataType::Float(16) &&
        value.dtype() != DataType::Float(32)) {
      return false;
    }
    if (const auto *parameter = value.as<VarNode>()) {
      return parameters_.count(GetRef<Var>(parameter)) != 0;
    }
    if (const auto *conversion = value.as<CastNode>()) {
      return IsScalarFillParameter(conversion->value);
    }
    return false;
  }

  Stmt LowerFill(const Call &call, int64_t lanes) {
    pto_analysis::PtoFillPlan plan =
        pto_analysis::PlanPtoFill(call, analyzer_, buffer_versions_);
    auto i32 = [](int64_t value) { return IntImm(DataType::Int(32), value); };
    Buffer dst = plan.destination->buffer;
    DataType dtype = dst->dtype;
    Array<Stmt> statements;
    // Capture once at the source statement, before any destination store.
    // Constants stay literals so the VMI bridge retains their exact dtype.
    PrimExpr value = plan.value;
    bool scalar_parameter = IsScalarFillParameter(value);
    PrimExpr load_value = value;
    std::vector<DataType> float_casts;
    if (!scalar_parameter) {
      while (const auto *conversion = load_value.as<CastNode>()) {
        if (!(conversion->dtype.is_float() ||
              conversion->dtype.is_bfloat16()) ||
            !(conversion->value.dtype().is_float() ||
              conversion->value.dtype().is_bfloat16())) {
          break;
        }
        float_casts.push_back(conversion->dtype);
        load_value = conversion->value;
      }
    }
    if (!load_value.as<BufferLoadNode>() && !load_value.as<VarNode>()) {
      // Integer casts and constants retain their scalar lowering. Floating
      // UB snapshots and parameters use vector conversions below, preserving
      // every intermediate rounding rather than collapsing a cast chain.
      load_value = value;
      float_casts.clear();
    }
    if (const auto *load = load_value.as<BufferLoadNode>()) {
      // Capture the scalar once before any store; a later store may write
      // the same element from which this uniform value was loaded.
      PrimExpr address = Call(DataType::Handle(), tl::access_ptr(),
                              {GetRef<BufferLoad>(load), i32(1), i32(1)});
      value = EmitBind("fill_value",
                       Call(load_value.dtype(), VmiOp("vload"),
                            {address, i32(0)}, {{"size", i32(1)}}),
                       &statements);
    } else if (!scalar_parameter && !load_value.as<IntImmNode>() &&
               !load_value.as<FloatImmNode>()) {
      value = EmitBind("fill_value", load_value, &statements);
    }
    DataType broadcast_dtype = float_casts.empty() ? dtype : load_value.dtype();
    PrimExpr vector =
        EmitBind("fill_vector",
                 Call(broadcast_dtype.with_lanes(lanes), VmiOp("vbrc"), {value},
                      {{"size", i32(lanes)}}),
                 &statements);
    for (auto conversion = float_casts.rbegin();
         conversion != float_casts.rend(); ++conversion) {
      // Keep the scalar snapshot in its loaded dtype. PTOAS has no complete
      // layout route for narrow one-lane vcvt -> broadcast -> wide store;
      // broadcasting the uniform input before vcvt preserves the same cast
      // per element and uses the validated full-VL conversion route.
      vector =
          EmitBind("fill_cast",
                   Call(conversion->with_lanes(lanes), VmiOp("vcvt"), {vector},
                        {{"to_dtype", StringImm(TvmDtypeName(*conversion))}}),
                   &statements);
    }
    PrimExpr mask = EmitBind("fill_mask",
                             Call(DataType::Bool(lanes), VmiOp("create_mask"),
                                  {i32(lanes)}, {{"size", i32(lanes)}}),
                             &statements);
    Var row("fill_row", DataType::Int(32));
    Var chunk("fill_chunk", DataType::Int(32));
    Array<PrimExpr> indices;
    for (const Range &range : plan.destination->region) {
      indices.push_back(range->min);
    }
    if (plan.outer_axis >= 0) {
      indices.Set(plan.outer_axis, row);
    }
    indices.Set(plan.continuous_axis, chunk * i32(lanes));
    PrimExpr pointer = Call(DataType::Handle(), tl::access_ptr(),
                            {BufferLoad(dst, indices), i32(lanes), i32(2)});
    Stmt body = Evaluate(Call(DataType::Void(), VmiOp("vstore"),
                              {vector, pointer, i32(0), mask}));
    body = For(chunk, i32(0), i32(plan.run_extent / lanes), ForKind::kSerial,
               body);
    if (plan.outer_axis >= 0) {
      body = For(row, i32(0), i32(plan.outer_extent), ForKind::kSerial, body);
    }
    statements.push_back(body);
    return SeqStmt::Flatten(statements);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      For loop = GetRef<For>(op);
      auto unit = unit_mappings_.find(loop);
      if (unit != unit_mappings_.end()) {
        return ConvertUnit(loop, unit->second);
      }
    }
    pto_analysis::ScopedPtoLoopBounds loop_bounds(analyzer_, GetRef<For>(op));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (current_vf_lanes_ != 0) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (pto_analysis::IsPtoDirectReduce(GetRef<Call>(call))) {
          return LowerDirectReduce(GetRef<Call>(call), current_vf_lanes_);
        }
        if (pto_analysis::IsPtoFill(GetRef<Call>(call))) {
          return LowerFill(GetRef<Call>(call), current_vf_lanes_);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // ------------------------------------------------------------------
  // Per-unit conversion
  // ------------------------------------------------------------------
  Stmt ConvertUnit(const For &loop,
                   const pto_analysis::PtoInverseMapping &mapping) {
    // The inverse mapping comes from the region entry: it is the shared
    // source of truth for lanes, the mask plan and the body conversion
    // (see InvertPtoLaneLayout).
    bind_env_.clear();
    value_map_.clear();
    emitted_value_info_.clear();
    tmp_counter_ = 0;
    unit_mapping_ = mapping;
    // Only address consumers may request algebraic scalarization of saved
    // integer coordinates. Ordinary computation uses value classification,
    // without applying the loaded-index dtype contract to its Cast inputs.
    address_vars_.clear();
    std::vector<Bind> definitions;
    std::unordered_set<BufferLoad, ObjectPtrHash, ObjectPtrEqual> address_loads;
    std::function<void(const PrimExpr &)> collect_address_vars =
        [&](const PrimExpr &expr) {
          PostOrderVisit(expr, [&](const ObjectRef &node) {
            if (auto var = node.as<Var>()) {
              address_vars_.insert(*var);
            } else if (auto load = node.as<BufferLoad>();
                       load && address_loads.insert(*load).second) {
              // The ordinary visitor skips view metadata at load sites.
              for (const PrimExpr &offset :
                   (*load)->buffer->ElemOffset((*load)->indices)) {
                collect_address_vars(offset);
              }
            }
          });
        };
    auto collect_access = [&](const Buffer &buffer,
                              const Array<PrimExpr> &indices) {
      for (const PrimExpr &index : indices) {
        collect_address_vars(index);
      }
      // View strides and origins can capture Bind values too.
      for (const PrimExpr &offset : buffer->ElemOffset(indices)) {
        collect_address_vars(offset);
      }
    };
    PostOrderVisit(loop->body, [&](const ObjectRef &node) {
      if (const auto *bind = node.as<BindNode>()) {
        definitions.push_back(GetRef<Bind>(bind));
      } else if (const auto *load = node.as<BufferLoadNode>()) {
        collect_access(load->buffer, load->indices);
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        collect_access(store->buffer, store->indices);
      }
    });
    // In verified SSA every dependency precedes its use.
    for (auto it = definitions.rbegin(); it != definitions.rend(); ++it) {
      if (address_vars_.count((*it)->var)) {
        collect_address_vars((*it)->value);
      }
    }
    Optional<For> inner;
    if (const auto *body_for = loop->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = GetRef<For>(body_for);
      }
    }
    loop_var_ = mapping.is_2d ? (mapping.select_inner ? mapping.inner_var
                                                      : mapping.outer_var)
                              : loop->loop_var;
    lanes_ = mapping.lanes;
    Var chunk("q", loop_var_.dtype());
    chunk_index_ = chunk;
    PrimExpr unit_bounds = chunk >= make_zero(chunk.dtype()) &&
                           chunk < IntImm(chunk.dtype(), mapping.chunk_count);
    const auto serial = mapping.SerialVar();
    if (serial) {
      const Var &serial_var = serial.value();
      unit_bounds = unit_bounds && serial_var >= make_zero(serial_var.dtype()) &&
                    serial_var < IntImm(serial_var.dtype(), mapping.SerialExtent());
    }
    With<arith::ConstraintContext> unit_scope(analyzer_, unit_bounds);

    // Decide once from the static extent. A tail-bearing unit keeps the
    // existing element-wise remaining-mask loop, including its full chunks.
    const bool has_tail = mapping.tail_lanes != 0;
    Buffer remaining;
    if (has_tail) {
      remaining =
          tirx::decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                            "remaining", "local.var");
      mask_var_ = Var("mask", DataType::Bool(static_cast<int>(lanes_)));
    } else {
      ICHECK(shared_mask_.defined());
      mask_var_ = shared_mask_;
    }

    Array<Stmt> statements;
    const Stmt &source = inner.defined() ? inner.value()->body : loop->body;
    ConvertBody(source, &statements);
    Stmt converted = DeadBindElim()(SeqStmt::Flatten(statements));
    Array<Stmt> chunk_body;
    if (has_tail) {
      PrimExpr count = BufferLoad(remaining, {make_zero(DataType::Int(32))});
      chunk_body.push_back(
          Bind(mask_var_, Call(mask_var_.dtype(), VmiOp("create_mask"), {count},
                               {{"size", IntImm(DataType::Int(32), lanes_)}})));
      chunk_body.push_back(converted);
      chunk_body.push_back(
          BufferStore(remaining, count - IntImm(DataType::Int(32), lanes_),
                      {make_zero(DataType::Int(32))}));
    } else {
      chunk_body.push_back(converted);
    }
    Stmt body = For(chunk, make_zero(chunk.dtype()),
                    IntImm(chunk.dtype(), mapping.chunk_count),
                    ForKind::kSerial, SeqStmt::Flatten(chunk_body));
    if (has_tail) {
      Stmt init = BufferStore(remaining,
                              IntImm(DataType::Int(32), mapping.logical_extent),
                              {make_zero(DataType::Int(32))});
      body = SeqStmt::Flatten(init, body);
    }
    if (serial) {
      // Reset remaining inside the kept loop so every row/column has the
      // same ceil(E/L) iterations and its own final partial mask.
      const Var &serial_var = serial.value();
      body = For(serial_var, make_zero(serial_var.dtype()),
                 IntImm(serial_var.dtype(), mapping.SerialExtent()),
                 ForKind::kSerial, body);
    }
    return has_tail ? SeqStmt::Flatten(AllocBuffer(remaining), body) : body;
  }

  // Substitute the analysis chunk placeholder with the unified loop's q.
  PrimExpr BindChunkToQ(const PrimExpr &expr) const {
    return pto_analysis::SubstituteVar(expr, unit_mapping_.chunk_var.get(),
                                       chunk_index_);
  }

  PrimExpr SubstituteLaneZero(const PrimExpr &expr) const {
    const Var &lane = unit_mapping_.lane_var;
    return pto_analysis::SubstituteVar(expr, lane.get(),
                                       make_zero(lane.dtype()));
  }

  PrimExpr Broadcast(PrimExpr value, Array<Stmt> *pre) {
    const auto adaptation = pto_analysis::PlanWidthAdaptation(
        pto_analysis::GetValueInfo(value, emitted_value_info_), lanes_);
    if (adaptation == pto_analysis::WidthAdaptation::kKeep)
      return value;
    Map<String, ObjectRef> attrs{{"size", IntImm(DataType::Int(32), lanes_)}};
    if (adaptation == pto_analysis::WidthAdaptation::kBroadcastOneLane) {
      attrs.Set("group", IntImm(DataType::Int(32), 1));
    }
    return EmitVmiDataBind(
        "broadcast",
        Call(VectorDType(value.dtype(), lanes_), VmiOp("vbrc"), {value}, attrs),
        pre);
  }

  bool IsLaneUniformAddressExpr(const PrimExpr &expr) {
    return pto_analysis::IsPtoLaneUniform(
        expr, loop_var_, unit_mapping_.index_expr, unit_mapping_.lane_var,
        unit_mapping_.chunk_var, lanes_, unit_mapping_.chunk_count,
        unit_mapping_.SerialVar(), unit_mapping_.SerialExtent(), bind_env_);
  }

  PrimExpr ConvertScalarExpr(const PrimExpr &expr) {
    class ScalarRemap : public ExprMutator {
    public:
      explicit ScalarRemap(VectorizeParallelToPTOImpl *parent)
          : parent_(parent) {}
      PrimExpr VisitExpr_(const BufferLoadNode *op) final {
        // A scalar index can itself be a read from a view whose stride or
        // origin uses the Parallel coordinate.
        return parent_->ScalarBufferLoad(op->buffer, op->indices);
      }
      PrimExpr VisitExpr_(const VarNode *op) final {
        auto it = parent_->value_map_.find(GetRef<Var>(op));
        if (it != parent_->value_map_.end()) {
          ICHECK(pto_analysis::GetValueInfo((*it).second,
                                            parent_->emitted_value_info_)
                     .kind == pto_analysis::ValueKind::kScalar);
          return (*it).second;
        }
        return GetRef<PrimExpr>(op);
      }

    private:
      VectorizeParallelToPTOImpl *parent_;
    } remap(this);
    PrimExpr scalar = pto_analysis::SubstituteVar(
        expr, loop_var_.get(), BindChunkToQ(unit_mapping_.index_expr));
    scalar = analyzer_->Simplify(SubstituteLaneZero(scalar));
    return analyzer_->Simplify(remap(scalar));
  }

  PrimExpr ScalarBufferLoad(const Buffer &buffer,
                            const Array<PrimExpr> &indices) {
    bool remap_metadata = false;
    auto inspect = [&](const PrimExpr &expr) {
      PostOrderVisit(expr, [&](const ObjectRef &node) {
        if (node.as<BufferLoadNode>()) {
          // The nested load's view may depend on the Parallel coordinate even
          // when its logical indices are constant.
          remap_metadata = true;
        } else if (const auto *var = node.as<VarNode>()) {
          remap_metadata |=
              var == loop_var_.get() || value_map_.count(GetRef<Var>(var));
        }
      });
    };
    for (const PrimExpr &shape : buffer->shape)
      inspect(shape);
    for (const PrimExpr &stride : buffer->strides)
      inspect(stride);
    inspect(buffer->elem_offset);
    bool has_non_scalar_index = false;
    for (const PrimExpr &index : indices) {
      PostOrderVisit(index, [&](const ObjectRef &node) {
        if (auto var = node.as<Var>()) {
          auto mapped = value_map_.find(*var);
          if (mapped != value_map_.end()) {
            auto info = pto_analysis::GetValueInfo((*mapped).second,
                                                   emitted_value_info_);
            has_non_scalar_index |=
                info.kind != pto_analysis::ValueKind::kScalar;
          }
        }
      });
    }
    if (remap_metadata || has_non_scalar_index) {
      // Simplify the complete address before remapping it. A zero stride
      // can make a lane-varying logical index irrelevant to a scalar address.
      for (const Buffer &owner : active_allocations_) {
        if (owner->data.same_as(buffer->data)) {
          Buffer flat = owner.GetFlattenedBuffer();
          PrimExpr offset = ConvertScalarExpr(buffer->ElemOffset(indices)[0]);
          if (flat->dtype != buffer->dtype) {
            // Scalar and continuous accesses keep the view's element units
            // when it reinterprets the allocation.
            PrimExpr extent = analyzer_->Simplify(floordiv(
                GetBufferStorageSizeBytes(flat), buffer->dtype.bytes()));
            auto *view = flat.CopyOnWrite();
            view->dtype = buffer->dtype;
            view->shape = {extent};
            view->elem_offset = make_zero(offset.dtype());
          }
          return BufferLoad(flat,
                            {analyzer_->Simplify(offset - flat->elem_offset)});
        }
      }
    }
    Array<PrimExpr> scalar_indices = indices.Map(
        [&](const PrimExpr &index) { return ConvertScalarExpr(index); });
    return BufferLoad(buffer, scalar_indices);
  }

  PrimExpr ConvertUint16Offset(const PrimExpr &expr, Array<Stmt> *pre) {
    const DataType dtype = DataType::UInt(16);
    const DataType vector_dtype = VectorDType(dtype, lanes_);
    if (const auto *integer = expr.as<IntImmNode>()) {
      return make_const(dtype, integer->value);
    }
    if (const auto *var = expr.as<VarNode>()) {
      if (var == loop_var_.get()) {
        PrimExpr base = analyzer_->Simplify(
            BindChunkToQ(SubstituteLaneZero(unit_mapping_.index_expr)));
        return EmitVmiDataBind(
            "offset_vci",
            Call(vector_dtype, VmiOp("vci"), {cast(dtype, base)},
                 {{"size", IntImm(DataType::Int(32), lanes_)}}),
            pre);
      }
      return cast(dtype, ConvertScalarExpr(expr));
    }
    const PrimExpr *lhs;
    const PrimExpr *rhs;
    std::string operation;
    if (const auto *add = expr.as<AddNode>()) {
      lhs = &add->a;
      rhs = &add->b;
      operation = "vadd";
    } else if (const auto *sub = expr.as<SubNode>()) {
      lhs = &sub->a;
      rhs = &sub->b;
      operation = "vsub";
    } else {
      const auto *mul = expr.as<MulNode>();
      ICHECK(mul) << "unsupported proved uint16 offset node "
                  << expr->GetTypeKey();
      lhs = &mul->a;
      rhs = &mul->b;
      operation = "vmul";
    }
    PrimExpr a = ConvertUint16Offset(*lhs, pre);
    PrimExpr b = ConvertUint16Offset(*rhs, pre);
    if (!pto_analysis::GetValueInfo(a, emitted_value_info_).IsVmi() &&
        !pto_analysis::GetValueInfo(b, emitted_value_info_).IsVmi()) {
      if (operation == "vadd") {
        return a + b;
      }
      if (operation == "vsub") {
        return a - b;
      }
      return a * b;
    }
    return EmitVmiDataBind(
        "offset",
        Call(vector_dtype, VmiOp(operation),
             {Broadcast(a, pre), Broadcast(b, pre), mask_var_}),
        pre);
  }

  struct IndexedOperands {
    PrimExpr pointer;
    PrimExpr base;
    PrimExpr offsets;
  };

  IndexedOperands LowerIndexed(const pto_analysis::MemoryAccess &access,
                               const PrimExpr &mask, Array<Stmt> *pre) {
    Buffer allocation = pto_analysis::FindAllocationOwner(
        active_allocations_, access.buffer, "VectorizeParallelToPTO");
    pto_analysis::IndexedMemoryOwner owner =
        pto_analysis::DescribeIndexedMemoryOwner(allocation);
    pto_analysis::PtoIndexedAccessPlan plan =
        pto_analysis::PlanPtoIndexedAccess(access, owner, loop_var_,
                                           unit_mapping_, analyzer_,
                                           buffer_versions_, bind_env_);
    PrimExpr base = plan.rebased ? ConvertScalarExpr(plan.base) : plan.base;
    PrimExpr offsets =
        Broadcast(plan.generate_uint16_offsets
                      ? ConvertUint16Offset(plan.offsets, pre)
                      : ConvertExpr(plan.offsets, pre),
                  pre);
    // A tail-bearing unit sanitizes offsets in every iteration. Divisible
    // units pass the loaded/calculated offsets directly, without cleaning.
    if (unit_mapping_.tail_lanes != 0) {
      PrimExpr zero =
          EmitVmiDataBind("offset_zero",
                          Call(offsets.dtype(), VmiOp("vbrc"),
                               {make_zero(offsets.dtype().element_of())},
                               {{"size", IntImm(DataType::Int(32), lanes_)}}),
                          pre);
      offsets = EmitVmiDataBind(
          "safe_offsets",
          Call(offsets.dtype(), VmiOp("vsel"), {mask, offsets, zero}), pre);
    }
    // PTOAS normalizes signedness and widens narrow offset carriers.
    Optional<PrimExpr> version_index;
    if (plan.version_index) {
      version_index = ConvertScalarExpr(plan.version_index.value());
    }
    return {pto_analysis::MakeIndexedMemoryDescriptor(owner, access.is_write,
                                                       version_index,
                                                       plan.descriptor_extent),
            base, offsets};
  }

  // ------------------------------------------------------------------
  // Statement conversion (pre-binds first, then the statement)
  // ------------------------------------------------------------------
  void ConvertBody(const Stmt &body, Array<Stmt> *out) {
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        ConvertBody(s, out);
      }
      return;
    }
    if (const auto *bind = body.as<BindNode>()) {
      Array<Stmt> pre;
      PrimExpr value =
          address_vars_.count(bind->var) &&
                  pto_analysis::IsPtoIndexDType(bind->value.dtype()) &&
                  IsLaneUniformAddressExpr(bind->value)
              ? ConvertScalarExpr(bind->value)
              : ConvertExpr(bind->value, &pre);
      // Pre-binds first, then the Bind itself.
      for (const auto &s : pre) {
        out->push_back(s);
      }
      Var new_var(bind->var->name_hint, value.dtype());
      bind_env_.Set(bind->var, bind->value); // analysis environment
      value_map_.Set(bind->var, new_var);    // value environment
      emitted_value_info_.emplace(
          new_var, pto_analysis::GetValueInfo(value, emitted_value_info_));
      out->push_back(Bind(new_var, value));
      return;
    }
    if (const auto *store = body.as<BufferStoreNode>()) {
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(store->value, &pre);
      auto access = pto_analysis::AnalyzeBufferAccess(
          store->buffer, store->indices, /*is_write=*/true, loop_var_,
          unit_mapping_, bind_env_);
      value = Broadcast(value, &pre);
      PrimExpr call;
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        Array<PrimExpr> starts = access.start_indices.Map(
            [&](const PrimExpr &index) { return BindChunkToQ(index); });
        PrimExpr ptr = Call(DataType::Handle(), tl::access_ptr(),
                            {ScalarBufferLoad(store->buffer, starts),
                             IntImm(DataType::Int(32), lanes_),
                             IntImm(DataType::Int(32), 2)});
        call = Call(DataType::Void(), VmiOp("vstore"),
                    {value, ptr, IntImm(DataType::Int(32), 0), mask_var_});
      } else {
        IndexedOperands operands = LowerIndexed(access, mask_var_, &pre);
        call = Call(DataType::Void(), VmiOp("vscatter"),
                    {value, operands.pointer, operands.base, operands.offsets,
                     mask_var_});
      }
      for (const Stmt &stmt : pre)
        out->push_back(stmt);
      out->push_back(Evaluate(call));
      return;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      // Verify guarantees the metadata has no lane-dependent semantics
      // (direct or via bind aliases). Rewrite its Var references through
      // value_map_ so already-converted binds are not left dangling
      // The body is converted underneath.
      Array<Stmt> stmts;
      // The body is a nested statement scope — keep the
      // source-bind and value maps scoped with it, so a bind defined
      // inside cannot remap a later sibling's reference (Verify rejects
      // that input; this keeps the transformation state consistent with
      // the unit visitor's scope handling).
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      ConvertBody(attr->body, &stmts);
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      Stmt inner = SeqStmt::Flatten(stmts);
      // Default to the *original* node. Only PrimExpr metadata is
      // remapped; a non-object Any (int/str atom metadata) must forward
      // unchanged — the old code wrote back a default-constructed (empty)
      // node when the try_cast failed.
      ffi::Any node = attr->node;
      if (auto expr = attr->node.try_cast<PrimExpr>()) {
        node = tirx::Substitute(expr.value(), value_map_);
      }
      PrimExpr value = tirx::Substitute(attr->value, value_map_);
      out->push_back(AttrStmt(node, attr->attr_key, value, inner));
      return;
    }
    if (const auto *eval = body.as<EvaluateNode>()) {
      // Convert the expression with the same machinery so no stale loop
      // variable survives; the whitelist is Verify's job.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(eval->value, &pre);
      for (const auto &s : pre) {
        out->push_back(s);
      }
      out->push_back(Evaluate(value));
      return;
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "statement passed Verify: "
               << body->GetTypeKey();
  }

  // ------------------------------------------------------------------
  // Expression conversion (index vars resolve through bind_env_)
  // ------------------------------------------------------------------
  PrimExpr ConvertExpr(const PrimExpr &expr, Array<Stmt> *pre) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      auto access = pto_analysis::AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, loop_var_,
          unit_mapping_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        Array<PrimExpr> starts = access.start_indices.Map(
            [&](const PrimExpr &index) { return BindChunkToQ(index); });
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {ScalarBufferLoad(load->buffer, starts),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 1)});
        return EmitVmiDataBind(
            "vload",
            Call(VectorDType(load->dtype, lanes_), VmiOp("vload"),
                 {ptr, IntImm(DataType::Int(32), 0)},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      if (access.pattern == pto_analysis::AccessPattern::kIndexed) {
        IndexedOperands operands = LowerIndexed(access, mask_var_, pre);
        const bool byte_payload = load->dtype.bits() == 8;
        const DataType result_dtype =
            byte_payload ? (load->dtype.is_int() ? DataType::Int(16)
                                                 : DataType::UInt(16))
                         : load->dtype;
        PrimExpr gathered = EmitVmiDataBind(
            "gather",
            Call(VectorDType(result_dtype, lanes_), VmiOp("vgather"),
                 {operands.pointer, operands.base, operands.offsets, mask_var_},
                 {{"pmode", StringImm("zero")}}),
            pre);
        if (byte_payload) {
          // Byte gathers zero-extend into B16 even for signed sources. Restore
          // source bits before any user Cast, arithmetic or saved Bind sees it.
          gathered = EmitVmiDataBind(
              "gather_byte",
              Call(VectorDType(load->dtype, lanes_), VmiOp("vcvt"), {gathered},
                   {{"to_dtype", StringImm(TvmDtypeName(load->dtype))},
                    {"saturate", StringImm("NOSAT")}}),
              pre);
        }
        return gathered;
      }
      // lane-uniform scalar load: keep the BufferLoad (lane bound to 0,
      // chunk bound to q — indices resolve through the bind env
      // inside the shared helper already) and bind it to a scalar Var in
      // the current chunk; later uses (vbrc broadcast, scalar arithmetic,
      // store values) reference that Var. Each textual occurrence binds
      // separately — no cross-access load reuse. The Var dtype is the
      // buffer's own scalar dtype.
      Array<PrimExpr> indices = access.start_indices.Map(
          [&](const PrimExpr &index) { return BindChunkToQ(index); });
      return EmitBind("sload", ScalarBufferLoad(load->buffer, indices), pre);
    }
    if (const auto *var = expr.as<VarNode>()) {
      auto it = value_map_.find(GetRef<Var>(var));
      if (it != value_map_.end()) {
        return (*it).second;
      }
      if (var == loop_var_.get()) {
        // The selected coordinate used as a value becomes a VMI vector.
        PrimExpr base = analyzer_->Simplify(
            BindChunkToQ(SubstituteLaneZero(unit_mapping_.index_expr)));
        return EmitVmiDataBind(
            "vci",
            Call(VectorDType(var->dtype, lanes_), VmiOp("vci"), {base},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      return GetRef<PrimExpr>(var);
    }
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
      return expr;
    }
    if (const auto *add = expr.as<AddNode>()) {
      return ConvertBinary(add->a, add->b, "vadd", pre);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      return ConvertBinary(sub->a, sub->b, "vsub", pre);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return ConvertBinary(mul->a, mul->b, "vmul", pre);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return ConvertBinary(div->a, div->b, "vdiv", pre);
    }
    if (expr.as<ModNode>() || expr.as<MinNode>() || expr.as<MaxNode>() ||
        expr.as<SelectNode>()) {
      ICHECK(IsLaneUniformAddressExpr(expr))
          << "[VectorizeParallelToPTO] internal error: unsupported "
             "lane-varying indexed coordinate passed Verify: "
          << expr->GetTypeKey();
      PrimExpr coordinate = pto_analysis::SubstituteVar(
          expr, loop_var_.get(), unit_mapping_.index_expr);
      return ConvertScalarExpr(BindChunkToQ(coordinate));
    }
    if (const auto *call = expr.as<CallNode>()) {
      // Hardware-backed unary math (fabs/exp/log/sqrt); the dtype matrix
      // mirrors the ISA documentation (see IsSupportedVectorUnaryOp).
      static const std::unordered_map<std::string, std::string> kUnaryMap = {
          {"tirx.fabs", "vabs"},
          {"tirx.exp", "vexp"},
          {"tirx.log", "vln"},
          {"tirx.sqrt", "vsqrt"}};
      auto call_op = call->op.as<Op>();
      ICHECK(call_op.has_value())
          << "[VectorizeParallelToPTO] internal error: non-Op call passed "
             "Verify: "
          << expr;
      auto it = kUnaryMap.find(call_op.value()->name);
      ICHECK(it != kUnaryMap.end())
          << "[VectorizeParallelToPTO] internal error: call passed Verify "
             "without a VMI mapping: "
          << call_op.value()->name;
      PrimExpr a = ConvertExpr(call->args[0], pre);
      if (!pto_analysis::GetValueInfo(a, emitted_value_info_).IsVmi()) {
        return Call(expr.dtype(), call->op, {a});
      }
      a = Broadcast(a, pre);
      return EmitVmiDataBind(
          it->second,
          Call(VectorDType(expr.dtype().element_of(), lanes_),
               VmiOp(it->second), {a, mask_var_}),
          pre);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      // Convert the input exactly once, then plan from the explicit shape of
      // that converted input (the original Cast carries the element types).
      PrimExpr value = ConvertExpr(cast->value, pre);
      pto_analysis::CastContext context;
      const auto value_info =
          pto_analysis::GetValueInfo(value, emitted_value_info_);
      if (value_info.kind == pto_analysis::ValueKind::kScalar) {
        context.value_kind = pto_analysis::CastValueKind::kScalar;
        context.logical_lanes = 1;
      } else {
        value = Broadcast(value, pre);
        context.value_kind = pto_analysis::CastValueKind::kVectorVmi;
        context.logical_lanes = lanes_;
      }
      ICHECK_EQ(value.dtype().element_of(), cast->value.dtype())
          << "[VectorizeParallelToPTO] internal error: converted Cast input "
             "element type "
          << value.dtype().element_of() << " does not match the original "
          << cast->value.dtype();
      std::string plan_why;
      auto plan = pto_analysis::PlanValueCast(cast, context, &plan_why);
      ICHECK(plan.has_value())
          << "[VectorizeParallelToPTO] internal error: Cast passed Verify "
             "without a conversion plan: "
          << plan_why;
      if (const auto *scalar =
              std::get_if<pto_analysis::ScalarCastPlan>(&plan.value())) {
        ICHECK_EQ(value.dtype().lanes(), 1)
            << "[VectorizeParallelToPTO] internal error: scalar Cast plan "
               "for a vector input";
        if (scalar->action == pto_analysis::ScalarCastAction::kKeepInput) {
          return value;
        }
        // Rebuild the ordinary scalar Cast with the validated annotations and
        // the original span; the scalar lowering itself is unchanged.
        return Cast(cast->dtype, value, scalar->annotations, cast->span);
      }
      const auto &vector = std::get<pto_analysis::VectorCastPlan>(plan.value());
      PrimExpr current = value;
      for (const auto &step : vector.steps) {
        ICHECK_EQ(current.dtype().element_of(), step.from)
            << "[VectorizeParallelToPTO] internal error: Cast step expects "
            << step.from << ", got " << current.dtype().element_of();
        Map<String, ObjectRef> attrs;
        attrs.Set("to_dtype", StringImm(TvmDtypeName(step.to)));
        Call call;
        if (step.op == pto_analysis::VectorCastOp::kIntegerReinterpret) {
          call = Call(VectorDType(step.to, lanes_), VmiOp("vinterpret_cast"),
                      {current}, attrs);
        } else {
          if (!step.saturate.empty()) {
            attrs.Set("saturate", StringImm(step.saturate));
          }
          if (!step.rounding.empty()) {
            attrs.Set("rounding", StringImm(step.rounding));
          }
          call = Call(VectorDType(step.to, lanes_), VmiOp("vcvt"), {current},
                      attrs);
        }
        current = EmitVmiDataBind("cast", call, pre);
      }
      // Zero steps is the proven identity; a final type check keeps every
      // multi-step chain honest (source convert once, dtype/VL continuous).
      ICHECK_EQ(current.dtype(), VectorDType(cast->dtype, lanes_))
          << "[VectorizeParallelToPTO] internal error: Cast plan ended at "
          << current.dtype() << ", expected "
          << VectorDType(cast->dtype, lanes_);
      return current;
    }
    if (pto_analysis::IsPtoScalarCoordinate(expr, loop_var_, bind_env_)) {
      return ConvertScalarExpr(expr);
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "expression passed Verify: "
               << expr->GetTypeKey();
  }

  PrimExpr ConvertBinary(const PrimExpr &lhs, const PrimExpr &rhs,
                         const std::string &vmi_name, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    if (!pto_analysis::GetValueInfo(a, emitted_value_info_).IsVmi() &&
        !pto_analysis::GetValueInfo(b, emitted_value_info_).IsVmi()) {
      // Lane-uniform folds stay scalar with the TIR semantics of the node.
      if (vmi_name == "vadd") {
        return a + b;
      }
      if (vmi_name == "vsub") {
        return a - b;
      }
      if (vmi_name == "vmul") {
        return a * b;
      }
      if (vmi_name == "vdiv") {
        return div(a, b);
      }
      ICHECK(false) << "[VectorizeParallelToPTO] internal error: unknown "
                       "binary op "
                    << vmi_name;
    }
    a = Broadcast(a, pre);
    b = Broadcast(b, pre);
    return EmitVmiDataBind(vmi_name,
                           Call(VectorDType(a.dtype().element_of(), lanes_),
                                VmiOp(vmi_name), {a, b, mask_var_}),
                           pre);
  }

  // TVM dtype name identical to Python str(dtype) (vmi.py ABI):
  // float32 -> "float32", bfloat16 -> "bfloat16", int8 -> "int8", ...
  static std::string TvmDtypeName(DataType dtype) {
    std::ostringstream os;
    if (dtype.is_bfloat16()) {
      return "bfloat16";
    }
    // The FP8 spellings the PTO dtype parser recognizes; both are the TIR
    // dtype names, not the PTO target names.
    if (dtype.is_float8_e4m3fn()) {
      return "float8_e4m3fn";
    }
    if (dtype.is_float8_e5m2()) {
      return "float8_e5m2";
    }
    if (dtype.is_float()) {
      os << "float" << dtype.bits();
    } else if (dtype.is_int()) {
      os << "int" << dtype.bits();
    } else if (dtype.is_uint()) {
      os << "uint" << dtype.bits();
    } else if (dtype.is_bool()) {
      return "bool";
    } else {
      os << dtype;
    }
    return os.str();
  }

  Var EmitBind(const std::string &name, const PrimExpr &value,
               Array<Stmt> *pre) {
    Var result(name + "_" + std::to_string(tmp_counter_++), value.dtype());
    pre->push_back(Bind(result, value));
    return result;
  }

  Var EmitVmiDataBind(const std::string &name, const Call &value,
                      Array<Stmt> *pre) {
    Var result = EmitBind(name, value, pre);
    emitted_value_info_.emplace(result,
                                pto_analysis::ValueInfo::Vmi(value.dtype()));
    return result;
  }

  arith::Analyzer root_analyzer_;
  arith::Analyzer *analyzer_ = &root_analyzer_;
  std::unordered_map<For, pto_analysis::PtoInverseMapping, ObjectPtrHash,
                     ObjectPtrEqual>
      unit_mappings_;
  Var shared_mask_; // one full mask per VF, shared by all units

  // Per-unit state.
  Var loop_var_;
  Var chunk_index_; // emitted chunk loop Var
  pto_analysis::PtoInverseMapping unit_mapping_;
  int64_t lanes_ = 0;
  Var mask_var_;
  int tmp_counter_ = 0;
  int64_t current_vf_lanes_ = 0;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> parameters_;
  std::vector<Buffer> active_allocations_;
  BufferVersionMap buffer_versions_;
  Map<Var, PrimExpr> bind_env_; // analysis env: original var -> def expr
  Map<Var, Var> value_map_;     // value env: original var -> converted var
  pto_analysis::ValueInfoMap emitted_value_info_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> address_vars_;
};

} // namespace

namespace transform {

tvm::transform::Pass VectorizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return VectorizeParallelToPTOImpl::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.VectorizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeParallelToPTO",
                        VectorizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm
