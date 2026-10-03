// Elementwise trace kernel as brass MIR (spirv_emit.h): a grid-stride loop
// over `n` slots of `lanes` elements; every operand addressed by its
// broadcast kind, every buffer at its own dtype.

#include "spirv_emit.h"
#include "spirv_emit_detail.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::jit::spirv {

using namespace detail;

namespace {

brass::target::SpirvKernel build_entry(const TraceDAG& dag, const plan::ElementwisePlan& plan,
                                       int lanes, const char* name) {
    brass::Module mod(std::string("mod_") + name);
    const std::size_t n_out = plan.outputs.size();
    const std::size_t n_in = plan.inputs.size();
    std::vector<Type> params(n_out + n_in, Type::ptr());
    params.push_back(Type::i32());
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));

    KernelBuilder kb(mod, fn);
    Ctx c(kb);
    Builder& b = c.b;
    const std::vector<Value*> p = entry_params(c, fn);
    Value* n = p.back();
    if (any_bf16(plan)) alloc_bitcast_slot(c, kEwBlock);

    Value* i0 = kb.global_tid_x();
    Value* stride = b.build_mul(kb.ntid_x(), kb.nctaid_x());

    kb.for_range(i0, n, stride, [&](Value* i) {
        // Element index of this thread's first lane, and its column when some
        // operand broadcasts across rows.
        Value* base = lanes > 1 ? b.build_shl(i, c.i32(log2_exact(lanes))) : i;
        Value* col = nullptr;
        if (plan.any_row_bcast) {
            col = is_pow2(plan.cols) ? b.build_and(base, c.i32(plan.cols - 1))
                                     : b.build_umod(base, c.i32(plan.cols));
        }
        // Byte offsets, one per (broadcast kind, element size) actually used.
        std::unordered_map<int, Value*> full_off, row_off;
        auto offset_for = [&](BroadcastKind bk, Dtype dt) -> Value* {
            const int esz = plan::elem_bytes(dt);
            auto& tbl = (bk == BroadcastKind::Row) ? row_off : full_off;
            auto it = tbl.find(esz);
            if (it != tbl.end()) return it->second;
            Value* off = byte_offset(c, bk == BroadcastKind::Row ? col : base, esz);
            tbl[esz] = off;
            return off;
        };

        ValueMap vals;
        for (std::size_t k = 0; k < n_in; ++k) {
            const plan::BufferSpec& s = plan.inputs[k];
            Value* ptr = p[n_out + k];
            if (s.bcast == BroadcastKind::Scalar) {
                // One load feeds every lane.
                Value* v = load_lanes(c, s.dtype, 1, ptr)[0];
                vals[s.node_id] = std::vector<Value*>(static_cast<std::size_t>(lanes), v);
            } else {
                vals[s.node_id] = load_lanes(c, s.dtype, lanes, b.build_add(ptr, offset_for(s.bcast, s.dtype)));
            }
        }

        vals = eval_chain(c, dag, lanes, std::move(vals));

        for (std::size_t m = 0; m < n_out; ++m) {
            const plan::BufferSpec& s = plan.outputs[m];
            Value* addr = b.build_add(p[m], offset_for(BroadcastKind::Full, s.dtype));
            store_lanes(c, s.dtype, lanes, addr, vals.at(s.node_id));
        }
    });
    b.build_ret_void();
    return compile(*fn, kEwBlock);
}

}  // namespace

bool ops_supported(const TraceDAG& dag) {
    for (const auto& n : dag.nodes()) {
        if (n.op == TraceOpKind::Input || n.op == TraceOpKind::RMSNorm ||
            n.op == TraceOpKind::LayerNorm) {
            continue;
        }
        if (!has_lane_form(n.op)) return false;
    }
    return true;
}

Kernels emit_elementwise(const TraceDAG& dag, const plan::ElementwisePlan& plan) {
    Kernels k;
    if (plan.vec > 1) k.vec = build_entry(dag, plan, plan.vec, kEntryVec);
    k.scalar = build_entry(dag, plan, 1, kEntryScalar);
    return k;
}

}  // namespace brotensor::jit::spirv
