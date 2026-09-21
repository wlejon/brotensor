#include "ptx_emit.h"
#include "ptx_emit_detail.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::jit::ptx {

using namespace brotensor::jit::ptx::detail;

namespace {

// One elementwise entry: `lanes` elements per thread, grid-stride.
std::string emit_ew_entry(const TraceDAG& dag, const ElementwisePlan& plan, int lanes,
                          const std::string& name) {
    Alloc a;
    std::stringstream b;

    const std::size_t n_out = plan.outputs.size();
    const std::size_t n_in = plan.inputs.size();

    const std::string r_n = a.R(), r_tid = a.R(), r_ntid = a.R();
    const std::string r_ctaid = a.R(), r_nctaid = a.R();
    const std::string r_i = a.R(), r_stride = a.R();
    const std::string p_oob = a.P();

    b << "    ld.param.u32 " << r_n << ", [" << name << "_n];\n";
    std::vector<std::string> out_ptr(n_out), in_ptr(n_in);
    for (std::size_t m = 0; m < n_out; ++m) {
        out_ptr[m] = a.RD();
        b << "    ld.param.u64 " << out_ptr[m] << ", [" << name << "_out_" << m << "];\n";
    }
    for (std::size_t k = 0; k < n_in; ++k) {
        in_ptr[k] = a.RD();
        b << "    ld.param.u64 " << in_ptr[k] << ", [" << name << "_in_" << k << "];\n";
    }
    b << "    mov.u32 " << r_tid << ", %tid.x;\n";
    b << "    mov.u32 " << r_ntid << ", %ntid.x;\n";
    b << "    mov.u32 " << r_ctaid << ", %ctaid.x;\n";
    b << "    mov.u32 " << r_nctaid << ", %nctaid.x;\n";
    b << "    mad.lo.u32 " << r_i << ", " << r_ctaid << ", " << r_ntid << ", " << r_tid << ";\n";
    b << "    mul.lo.u32 " << r_stride << ", " << r_ntid << ", " << r_nctaid << ";\n";

    b << "$L_" << name << "_head:\n";
    b << "    setp.ge.u32 " << p_oob << ", " << r_i << ", " << r_n << ";\n";
    b << "    @" << p_oob << " bra $L_" << name << "_exit;\n";

    // Element index of this thread's first lane.
    std::string r_base = r_i;
    if (lanes > 1) {
        r_base = a.R();
        b << "    shl.b32 " << r_base << ", " << r_i << ", "
          << (lanes == 2 ? 1 : (lanes == 4 ? 2 : 3)) << ";\n";
    }

    // Column index, only when some operand broadcasts across rows.
    std::string r_col;
    if (plan.any_row_bcast) {
        r_col = a.R();
        if (is_pow2(plan.cols)) {
            b << "    and.b32 " << r_col << ", " << r_base << ", " << (plan.cols - 1) << ";\n";
        } else {
            b << "    rem.u32 " << r_col << ", " << r_base << ", " << plan.cols << ";\n";
        }
    }

    // Byte offsets, one per (broadcast kind, element size) actually used.
    std::unordered_map<int, std::string> full_off, row_off;
    auto offset_for = [&](BroadcastKind bk, Dtype dt) -> std::string {
        const int esz = elem_bytes(dt);
        if (bk == BroadcastKind::Scalar) return std::string();
        auto& tbl = (bk == BroadcastKind::Row) ? row_off : full_off;
        auto it = tbl.find(esz);
        if (it != tbl.end()) return it->second;
        const std::string rd = a.RD();
        b << "    mul.wide.u32 " << rd << ", "
          << (bk == BroadcastKind::Row ? r_col : r_base) << ", " << esz << ";\n";
        tbl[esz] = rd;
        return rd;
    };

    ValueMap vals;
    for (std::size_t k = 0; k < n_in; ++k) {
        const BufferSpec& s = plan.inputs[k];
        const int ln = (s.bcast == BroadcastKind::Scalar) ? 1 : lanes;
        std::string addr;
        if (s.bcast == BroadcastKind::Scalar) {
            addr = in_ptr[k];
        } else {
            const std::string off = offset_for(s.bcast, s.dtype);
            addr = a.RD();
            b << "    add.u64 " << addr << ", " << in_ptr[k] << ", " << off << ";\n";
        }
        std::vector<std::string> v = emit_load(b, a, s.dtype, ln, addr);
        if (s.bcast == BroadcastKind::Scalar) {
            // One load feeds every lane. Copy the name out first: assign()
            // would clear the vector the reference points into.
            const std::string only = v[0];
            v.assign(static_cast<std::size_t>(lanes), only);
        }
        vals[s.node_id] = std::move(v);
    }

    vals = emit_chain(b, a, dag, lanes, std::move(vals));

    for (std::size_t m = 0; m < n_out; ++m) {
        const BufferSpec& s = plan.outputs[m];
        const std::string off = offset_for(BroadcastKind::Full, s.dtype);
        const std::string addr = a.RD();
        b << "    add.u64 " << addr << ", " << out_ptr[m] << ", " << off << ";\n";
        emit_store(b, a, s.dtype, lanes, addr, vals.at(s.node_id));
    }

    b << "    add.u32 " << r_i << ", " << r_i << ", " << r_stride << ";\n";
    b << "    bra $L_" << name << "_head;\n";
    b << "$L_" << name << "_exit:\n";
    b << "    ret;\n";

    std::stringstream ss;
    ss << ".visible .entry " << name << "(\n";
    for (std::size_t m = 0; m < n_out; ++m) {
        ss << "    .param .u64 " << name << "_out_" << m << ",\n";
    }
    for (std::size_t k = 0; k < n_in; ++k) {
        ss << "    .param .u64 " << name << "_in_" << k << ",\n";
    }
    ss << "    .param .u32 " << name << "_n\n";
    ss << ") {\n";
    ss << declarations(a);
    ss << b.str();
    ss << "}\n\n";
    return ss.str();
}

}  // namespace

int elem_bytes(Dtype d) {
    switch (d) {
        case Dtype::FP32: return 4;
        case Dtype::FP16:
        case Dtype::BF16: return 2;
        default: return 0;
    }
}

bool dtype_supported(Dtype d) { return elem_bytes(d) != 0; }

int vec_width_for(Dtype d) {
    const int e = elem_bytes(d);
    return e ? 16 / e : 1;
}

std::string emit_elementwise(const TraceDAG& dag, const ElementwisePlan& plan,
                             const std::string& arch) {
    std::stringstream ss;
    ss << ".version 7.8\n";
    ss << ".target " << arch << "\n";
    ss << ".address_size 64\n\n";
    ss << emit_ew_entry(dag, plan, plan.vec, kEntryVec);
    ss << emit_ew_entry(dag, plan, 1, kEntryScalar);
    return ss.str();
}

}  // namespace brotensor::jit::ptx
