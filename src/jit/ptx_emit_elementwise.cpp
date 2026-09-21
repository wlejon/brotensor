#include "ptx_emit.h"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace brotensor::jit::ptx {

namespace {

std::string fhex(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(float));
    std::stringstream ss;
    ss << "0f" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << bits;
    return ss.str();
}

// Virtual-register allocator. PTX register banks are declared up front, so the
// body is emitted into its own stream and the declarations are prepended once
// the high-water marks are known.
struct Alloc {
    int f = 0, r = 0, rd = 0, h = 0, p = 0;
    std::string F() { return "%f" + std::to_string(f++); }
    std::string R() { return "%r" + std::to_string(r++); }
    std::string RD() { return "%rd" + std::to_string(rd++); }
    std::string H() { return "%h" + std::to_string(h++); }
    std::string P() { return "%p" + std::to_string(p++); }
};

bool is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

// ── one lane of the op chain ────────────────────────────────────────────────

void emit_sigmoid(std::ostream& o, Alloc& a, const std::string& x, const std::string& d) {
    const std::string t0 = a.F(), t1 = a.F(), t2 = a.F();
    o << "    neg.f32 " << t0 << ", " << x << ";\n";
    o << "    mul.f32 " << t0 << ", " << t0 << ", 0f3FB8AA3B;\n";  // * log2(e)
    o << "    ex2.approx.f32 " << t1 << ", " << t0 << ";\n";
    o << "    add.f32 " << t2 << ", " << t1 << ", 0f3F800000;\n";
    o << "    rcp.approx.f32 " << d << ", " << t2 << ";\n";
}

void emit_lane_op(std::ostream& o, Alloc& a, const TraceNode& n,
                  const std::vector<std::string>& in, const std::string& d) {
    switch (n.op) {
        case TraceOpKind::Add:
            o << "    add.f32 " << d << ", " << in[0] << ", " << in[1] << ";\n";
            break;
        case TraceOpKind::Sub:
            o << "    sub.f32 " << d << ", " << in[0] << ", " << in[1] << ";\n";
            break;
        case TraceOpKind::Mul:
            o << "    mul.f32 " << d << ", " << in[0] << ", " << in[1] << ";\n";
            break;
        case TraceOpKind::Div:
            o << "    div.rn.f32 " << d << ", " << in[0] << ", " << in[1] << ";\n";
            break;
        case TraceOpKind::AddScalar:
            o << "    add.f32 " << d << ", " << in[0] << ", " << fhex(n.scalar) << ";\n";
            break;
        case TraceOpKind::SubScalar:
            o << "    sub.f32 " << d << ", " << in[0] << ", " << fhex(n.scalar) << ";\n";
            break;
        case TraceOpKind::MulScalar:
            o << "    mul.f32 " << d << ", " << in[0] << ", " << fhex(n.scalar) << ";\n";
            break;
        case TraceOpKind::DivScalar:
            o << "    div.rn.f32 " << d << ", " << in[0] << ", " << fhex(n.scalar) << ";\n";
            break;
        case TraceOpKind::ScalarSub:
            o << "    sub.f32 " << d << ", " << fhex(n.scalar) << ", " << in[0] << ";\n";
            break;
        case TraceOpKind::ScalarDiv:
            o << "    div.rn.f32 " << d << ", " << fhex(n.scalar) << ", " << in[0] << ";\n";
            break;
        case TraceOpKind::FMA:
            o << "    fma.rn.f32 " << d << ", " << in[0] << ", " << in[1] << ", " << in[2] << ";\n";
            break;
        case TraceOpKind::ReLU:
            o << "    max.f32 " << d << ", " << in[0] << ", 0f00000000;\n";
            break;
        case TraceOpKind::Sigmoid:
            emit_sigmoid(o, a, in[0], d);
            break;
        case TraceOpKind::SiLU: {
            const std::string s = a.F();
            emit_sigmoid(o, a, in[0], s);
            o << "    mul.f32 " << d << ", " << in[0] << ", " << s << ";\n";
            break;
        }
        case TraceOpKind::Tanh:
            o << "    tanh.approx.f32 " << d << ", " << in[0] << ";\n";
            break;
        case TraceOpKind::GELU: {
            // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) — the tanh
            // approximation nn.GELU(approximate="tanh") uses.
            const std::string x2 = a.F(), inner = a.F(), poly = a.F();
            const std::string arg = a.F(), th = a.F(), t1 = a.F(), hx = a.F();
            o << "    mul.f32 " << x2 << ", " << in[0] << ", " << in[0] << ";\n";
            o << "    fma.rn.f32 " << inner << ", " << x2 << ", " << fhex(0.044715f)
              << ", " << fhex(1.0f) << ";\n";
            o << "    mul.f32 " << poly << ", " << inner << ", " << in[0] << ";\n";
            o << "    mul.f32 " << arg << ", " << poly << ", " << fhex(0.7978845608f) << ";\n";
            o << "    tanh.approx.f32 " << th << ", " << arg << ";\n";
            o << "    add.f32 " << t1 << ", " << th << ", " << fhex(1.0f) << ";\n";
            o << "    mul.f32 " << hx << ", " << in[0] << ", " << fhex(0.5f) << ";\n";
            o << "    mul.f32 " << d << ", " << hx << ", " << t1 << ";\n";
            break;
        }
        case TraceOpKind::Modulate: {
            // x * (1 + scale) + shift, with scale/shift normally (1, D) rows
            // the loader has already resolved to this lane's column.
            const std::string s1 = a.F();
            o << "    add.f32 " << s1 << ", " << in[1] << ", " << fhex(1.0f) << ";\n";
            o << "    fma.rn.f32 " << d << ", " << in[0] << ", " << s1 << ", " << in[2] << ";\n";
            break;
        }
        default:
            throw std::runtime_error(
                std::string("brotensor::jit: op '") + op_kind_name(n.op) +
                "' has no elementwise PTX form");
    }
}

// ── typed load / store ──────────────────────────────────────────────────────

// Loads `lanes` consecutive elements at [base + off] into fresh FP32
// registers. `lanes` is 1, 2, 4 or 8 and the access is one instruction.
std::vector<std::string> emit_load(std::ostream& o, Alloc& a, Dtype dt, int lanes,
                                   const std::string& addr) {
    std::vector<std::string> out(static_cast<std::size_t>(lanes));
    for (int i = 0; i < lanes; ++i) out[static_cast<std::size_t>(i)] = a.F();

    if (dt == Dtype::FP32) {
        if (lanes == 1) {
            o << "    ld.global.f32 " << out[0] << ", [" << addr << "];\n";
        } else {
            o << "    ld.global.v" << lanes << ".f32 {";
            for (int i = 0; i < lanes; ++i) o << (i ? ", " : "") << out[static_cast<std::size_t>(i)];
            o << "}, [" << addr << "];\n";
        }
        return out;
    }

    // BF16 / FP16: move 16-bit words as packed b32 and widen in registers.
    if (lanes == 1) {
        const std::string h = a.H();
        o << "    ld.global.b16 " << h << ", [" << addr << "];\n";
        if (dt == Dtype::BF16) {
            const std::string t = a.R();
            o << "    cvt.u32.u16 " << t << ", " << h << ";\n";
            o << "    shl.b32 " << t << ", " << t << ", 16;\n";
            o << "    mov.b32 " << out[0] << ", " << t << ";\n";
        } else {
            o << "    cvt.f32.f16 " << out[0] << ", " << h << ";\n";
        }
        return out;
    }

    const int words = lanes / 2;
    std::vector<std::string> w(static_cast<std::size_t>(words));
    for (int i = 0; i < words; ++i) w[static_cast<std::size_t>(i)] = a.R();
    if (words == 1) {
        o << "    ld.global.u32 " << w[0] << ", [" << addr << "];\n";
    } else {
        o << "    ld.global.v" << words << ".u32 {";
        for (int i = 0; i < words; ++i) o << (i ? ", " : "") << w[static_cast<std::size_t>(i)];
        o << "}, [" << addr << "];\n";
    }
    for (int i = 0; i < words; ++i) {
        const std::string& pk = w[static_cast<std::size_t>(i)];
        const std::string& lo = out[static_cast<std::size_t>(2 * i)];
        const std::string& hi = out[static_cast<std::size_t>(2 * i + 1)];
        if (dt == Dtype::BF16) {
            const std::string t = a.R();
            o << "    shl.b32 " << t << ", " << pk << ", 16;\n";
            o << "    mov.b32 " << lo << ", " << t << ";\n";
            const std::string t2 = a.R();
            o << "    and.b32 " << t2 << ", " << pk << ", -65536;\n";
            o << "    mov.b32 " << hi << ", " << t2 << ";\n";
        } else {
            const std::string h0 = a.H(), h1 = a.H();
            o << "    mov.b32 {" << h0 << ", " << h1 << "}, " << pk << ";\n";
            o << "    cvt.f32.f16 " << lo << ", " << h0 << ";\n";
            o << "    cvt.f32.f16 " << hi << ", " << h1 << ";\n";
        }
    }
    return out;
}

void emit_store(std::ostream& o, Alloc& a, Dtype dt, int lanes,
                const std::string& addr, const std::vector<std::string>& vals) {
    if (dt == Dtype::FP32) {
        if (lanes == 1) {
            o << "    st.global.f32 [" << addr << "], " << vals[0] << ";\n";
        } else {
            o << "    st.global.v" << lanes << ".f32 [" << addr << "], {";
            for (int i = 0; i < lanes; ++i) o << (i ? ", " : "") << vals[static_cast<std::size_t>(i)];
            o << "};\n";
        }
        return;
    }

    const char* cvt = (dt == Dtype::BF16) ? "cvt.rn.bf16.f32" : "cvt.rn.f16.f32";
    if (lanes == 1) {
        const std::string h = a.H();
        o << "    " << cvt << " " << h << ", " << vals[0] << ";\n";
        o << "    st.global.b16 [" << addr << "], " << h << ";\n";
        return;
    }

    const int words = lanes / 2;
    std::vector<std::string> w(static_cast<std::size_t>(words));
    for (int i = 0; i < words; ++i) {
        const std::string h0 = a.H(), h1 = a.H();
        o << "    " << cvt << " " << h0 << ", " << vals[static_cast<std::size_t>(2 * i)] << ";\n";
        o << "    " << cvt << " " << h1 << ", " << vals[static_cast<std::size_t>(2 * i + 1)] << ";\n";
        w[static_cast<std::size_t>(i)] = a.R();
        o << "    mov.b32 " << w[static_cast<std::size_t>(i)] << ", {" << h0 << ", " << h1 << "};\n";
    }
    if (words == 1) {
        o << "    st.global.u32 [" << addr << "], " << w[0] << ";\n";
    } else {
        o << "    st.global.v" << words << ".u32 [" << addr << "], {";
        for (int i = 0; i < words; ++i) o << (i ? ", " : "") << w[static_cast<std::size_t>(i)];
        o << "};\n";
    }
}

// ── the chain, shared by the elementwise and row-norm emitters ──────────────

// Values already resolved for some node ids (the row-norm emitter seeds the
// normalised value); everything else is computed here. Returns the per-lane
// value of every node.
std::unordered_map<int, std::vector<std::string>> emit_chain(
    std::ostream& o, Alloc& a, const TraceDAG& dag, int lanes,
    std::unordered_map<int, std::vector<std::string>> vals) {
    for (const auto& n : dag.nodes()) {
        if (n.op == TraceOpKind::Input) continue;
        if (vals.count(n.id)) continue;

        std::vector<std::string> dst(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            std::vector<std::string> in;
            in.reserve(n.inputs.size());
            for (int src : n.inputs) {
                auto it = vals.find(src);
                if (it == vals.end()) {
                    throw std::runtime_error(
                        "brotensor::jit: trace node " + std::to_string(n.id) +
                        " reads an operand that was never materialised");
                }
                in.push_back(it->second[static_cast<std::size_t>(j)]);
            }
            dst[static_cast<std::size_t>(j)] = a.F();
            emit_lane_op(o, a, n, in, dst[static_cast<std::size_t>(j)]);
        }
        vals[n.id] = std::move(dst);
    }
    return vals;
}

std::string declarations(const Alloc& a) {
    std::stringstream ss;
    if (a.p) ss << "    .reg .pred %p<" << a.p << ">;\n";
    if (a.r) ss << "    .reg .b32 %r<" << a.r << ">;\n";
    if (a.rd) ss << "    .reg .b64 %rd<" << a.rd << ">;\n";
    if (a.h) ss << "    .reg .b16 %h<" << a.h << ">;\n";
    if (a.f) ss << "    .reg .f32 %f<" << a.f << ">;\n";
    return ss.str();
}

// ── one elementwise entry ───────────────────────────────────────────────────

std::string emit_ew_entry(const TraceDAG& dag, const ElementwisePlan& plan,
                          int lanes, const std::string& name) {
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

    // Loads.
    std::unordered_map<int, std::vector<std::string>> vals;
    for (std::size_t k = 0; k < n_in; ++k) {
        const BufferSpec& s = plan.inputs[k];
        const int ln = (s.bcast == BroadcastKind::Scalar) ? 1 : lanes;
        std::string addr;
        if (s.bcast == BroadcastKind::Scalar) {
            addr = in_ptr[k];
        } else {
            // offset_for may emit an instruction of its own, so resolve it
            // before opening the `add.u64` line.
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

    // Stores.
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

// ── row-norm ────────────────────────────────────────────────────────────────

std::string emit_row_norm(const TraceDAG& dag, const RowNormPlan& plan,
                          const std::string& arch) {
    const ElementwisePlan& ew = plan.ew;
    const int D = ew.cols;
    const int R = ew.rows;
    const bool mean = (plan.reduce == RowReduce::Mean);

    Alloc a;
    std::stringstream b;

    const std::size_t n_out = ew.outputs.size();
    const std::size_t n_in = ew.inputs.size();
    const BufferSpec& xs = ew.inputs[static_cast<std::size_t>(plan.x_input)];

    std::vector<std::string> out_ptr(n_out), in_ptr(n_in);
    for (std::size_t m = 0; m < n_out; ++m) {
        out_ptr[m] = a.RD();
        b << "    ld.param.u64 " << out_ptr[m] << ", [rn_out_" << m << "];\n";
    }
    for (std::size_t k = 0; k < n_in; ++k) {
        in_ptr[k] = a.RD();
        b << "    ld.param.u64 " << in_ptr[k] << ", [rn_in_" << k << "];\n";
    }

    const std::string r_tid = a.R(), r_row = a.R();
    b << "    mov.u32 " << r_tid << ", %tid.x;\n";
    b << "    mov.u32 " << r_row << ", %ctaid.x;\n";
    {
        const std::string p = a.P();
        b << "    setp.ge.u32 " << p << ", " << r_row << ", " << R << ";\n";
        b << "    @" << p << " bra $L_rn_done;\n";
    }

    // Byte offset of this row in each distinct element size in play.
    const std::string r_rowbase = a.R();
    b << "    mul.lo.u32 " << r_rowbase << ", " << r_row << ", " << D << ";\n";

    std::unordered_map<int, std::string> rowbyte;
    auto row_bytes = [&](Dtype dt) -> std::string {
        const int e = elem_bytes(dt);
        auto it = rowbyte.find(e);
        if (it != rowbyte.end()) return it->second;
        const std::string rd = a.RD();
        b << "    mul.wide.u32 " << rd << ", " << r_rowbase << ", " << e << ";\n";
        rowbyte[e] = rd;
        return rd;
    };

    const std::string x_rowb = row_bytes(xs.dtype);
    const std::string rd_xrow = a.RD();
    b << "    add.u64 " << rd_xrow << ", " << in_ptr[static_cast<std::size_t>(plan.x_input)]
      << ", " << x_rowb << ";\n";

    // ── pass 1: sum and sum of squares over the row ─────────────────────────
    const std::string f_sum = a.F(), f_sq = a.F();
    b << "    mov.f32 " << f_sum << ", 0f00000000;\n";
    b << "    mov.f32 " << f_sq << ", 0f00000000;\n";
    {
        const std::string r_c = a.R(), p = a.P();
        b << "    mov.u32 " << r_c << ", " << r_tid << ";\n";
        b << "$L_rn_sum:\n";
        b << "    setp.ge.u32 " << p << ", " << r_c << ", " << D << ";\n";
        b << "    @" << p << " bra $L_rn_sum_end;\n";
        const std::string rd_o = a.RD(), rd_p = a.RD();
        b << "    mul.wide.u32 " << rd_o << ", " << r_c << ", " << elem_bytes(xs.dtype) << ";\n";
        b << "    add.u64 " << rd_p << ", " << rd_xrow << ", " << rd_o << ";\n";
        std::vector<std::string> v = emit_load(b, a, xs.dtype, 1, rd_p);
        if (mean) b << "    add.f32 " << f_sum << ", " << f_sum << ", " << v[0] << ";\n";
        b << "    fma.rn.f32 " << f_sq << ", " << v[0] << ", " << v[0] << ", " << f_sq << ";\n";
        b << "    add.u32 " << r_c << ", " << r_c << ", " << kRowThreads << ";\n";
        b << "    bra $L_rn_sum;\n";
        b << "$L_rn_sum_end:\n";
    }

    // ── block reduction: butterfly inside the warp, shared memory across ────
    auto warp_reduce = [&](const std::string& acc) {
        const std::string rb = a.R();
        for (int m = 16; m >= 1; m >>= 1) {
            const std::string t = a.F();
            b << "    mov.b32 " << rb << ", " << acc << ";\n";
            b << "    shfl.sync.bfly.b32 " << rb << ", " << rb << ", " << m
              << ", 31, -1;\n";
            b << "    mov.b32 " << t << ", " << rb << ";\n";
            b << "    add.f32 " << acc << ", " << acc << ", " << t << ";\n";
        }
    };
    if (mean) warp_reduce(f_sum);
    warp_reduce(f_sq);

    {
        const int warps = kRowThreads / 32;
        const std::string r_warp = a.R(), r_lane = a.R(), r_sb = a.R(), r_sa = a.R();
        const std::string p_lane0 = a.P();
        b << "    shr.u32 " << r_warp << ", " << r_tid << ", 5;\n";
        b << "    and.b32 " << r_lane << ", " << r_tid << ", 31;\n";
        b << "    mov.u32 " << r_sb << ", rn_red;\n";
        b << "    setp.eq.u32 " << p_lane0 << ", " << r_lane << ", 0;\n";
        b << "    @!" << p_lane0 << " bra $L_rn_nostore;\n";
        {
            const std::string rt = a.R();
            b << "    mad.lo.u32 " << r_sa << ", " << r_warp << ", 4, " << r_sb << ";\n";
            b << "    mov.b32 " << rt << ", " << f_sq << ";\n";
            b << "    st.shared.b32 [" << r_sa << "], " << rt << ";\n";
            if (mean) {
                const std::string rt2 = a.R(), ra2 = a.R();
                b << "    add.u32 " << ra2 << ", " << r_sa << ", " << (warps * 4) << ";\n";
                b << "    mov.b32 " << rt2 << ", " << f_sum << ";\n";
                b << "    st.shared.b32 [" << ra2 << "], " << rt2 << ";\n";
            }
        }
        b << "$L_rn_nostore:\n";
        b << "    bar.sync 0;\n";

        // Every thread folds the per-warp partials, so no second broadcast is
        // needed: `warps` is 8, so this is 8 shared loads.
        b << "    mov.f32 " << f_sq << ", 0f00000000;\n";
        if (mean) b << "    mov.f32 " << f_sum << ", 0f00000000;\n";
        for (int w = 0; w < warps; ++w) {
            const std::string ra = a.R(), rv = a.R(), fv = a.F();
            b << "    add.u32 " << ra << ", " << r_sb << ", " << (w * 4) << ";\n";
            b << "    ld.shared.b32 " << rv << ", [" << ra << "];\n";
            b << "    mov.b32 " << fv << ", " << rv << ";\n";
            b << "    add.f32 " << f_sq << ", " << f_sq << ", " << fv << ";\n";
            if (mean) {
                const std::string ra2 = a.R(), rv2 = a.R(), fv2 = a.F();
                b << "    add.u32 " << ra2 << ", " << r_sb << ", " << (warps * 4 + w * 4) << ";\n";
                b << "    ld.shared.b32 " << rv2 << ", [" << ra2 << "];\n";
                b << "    mov.b32 " << fv2 << ", " << rv2 << ";\n";
                b << "    add.f32 " << f_sum << ", " << f_sum << ", " << fv2 << ";\n";
            }
        }
    }

    // ── scale and (for LayerNorm) shift ─────────────────────────────────────
    const std::string f_mean = a.F(), f_rstd = a.F();
    const float inv_d = 1.0f / static_cast<float>(D);
    if (mean) {
        // var = E[x^2] - mean^2
        const std::string var = a.F(), t = a.F();
        b << "    mul.f32 " << f_mean << ", " << f_sum << ", " << fhex(inv_d) << ";\n";
        b << "    mul.f32 " << t << ", " << f_mean << ", " << f_mean << ";\n";
        b << "    mul.f32 " << var << ", " << f_sq << ", " << fhex(inv_d) << ";\n";
        b << "    sub.f32 " << var << ", " << var << ", " << t << ";\n";
        b << "    add.f32 " << var << ", " << var << ", " << fhex(plan.eps) << ";\n";
        b << "    rsqrt.approx.f32 " << f_rstd << ", " << var << ";\n";
    } else {
        const std::string ms = a.F();
        b << "    mov.f32 " << f_mean << ", 0f00000000;\n";
        b << "    mul.f32 " << ms << ", " << f_sq << ", " << fhex(inv_d) << ";\n";
        b << "    add.f32 " << ms << ", " << ms << ", " << fhex(plan.eps) << ";\n";
        b << "    rsqrt.approx.f32 " << f_rstd << ", " << ms << ";\n";
    }

    // ── pass 2: normalise, run the chain, store ─────────────────────────────
    {
        const std::string r_c = a.R(), p = a.P();
        b << "    mov.u32 " << r_c << ", " << r_tid << ";\n";
        b << "$L_rn_apply:\n";
        b << "    setp.ge.u32 " << p << ", " << r_c << ", " << D << ";\n";
        b << "    @" << p << " bra $L_rn_apply_end;\n";

        // Column and row-major byte offsets for this element.
        std::unordered_map<int, std::string> coff, foff;
        auto col_off = [&](Dtype dt) -> std::string {
            const int e = elem_bytes(dt);
            auto it = coff.find(e);
            if (it != coff.end()) return it->second;
            const std::string rd = a.RD();
            b << "    mul.wide.u32 " << rd << ", " << r_c << ", " << e << ";\n";
            coff[e] = rd;
            return rd;
        };
        auto full_off = [&](Dtype dt) -> std::string {
            const int e = elem_bytes(dt);
            auto it = foff.find(e);
            if (it != foff.end()) return it->second;
            const std::string rb = row_bytes(dt);
            const std::string co = col_off(dt);
            const std::string rd = a.RD();
            b << "    add.u64 " << rd << ", " << rb << ", " << co << ";\n";
            foff[e] = rd;
            return rd;
        };

        std::unordered_map<int, std::vector<std::string>> vals;
        std::string f_x;
        for (std::size_t k = 0; k < n_in; ++k) {
            const BufferSpec& s = ew.inputs[k];
            std::string addr;
            if (s.bcast == BroadcastKind::Scalar) {
                addr = in_ptr[k];
            } else {
                const std::string off =
                    (s.bcast == BroadcastKind::Row) ? col_off(s.dtype) : full_off(s.dtype);
                addr = a.RD();
                b << "    add.u64 " << addr << ", " << in_ptr[k] << ", " << off << ";\n";
            }
            std::vector<std::string> v = emit_load(b, a, s.dtype, 1, addr);
            if (static_cast<int>(k) == plan.x_input) f_x = v[0];
            vals[s.node_id] = std::move(v);
        }

        // normalised = (x - mean) * rstd, then * gamma (+ beta).
        const std::string f_n = a.F();
        if (mean) {
            b << "    sub.f32 " << f_n << ", " << f_x << ", " << f_mean << ";\n";
            b << "    mul.f32 " << f_n << ", " << f_n << ", " << f_rstd << ";\n";
        } else {
            b << "    mul.f32 " << f_n << ", " << f_x << ", " << f_rstd << ";\n";
        }
        if (plan.gamma_input >= 0) {
            const std::string& g =
                vals.at(ew.inputs[static_cast<std::size_t>(plan.gamma_input)].node_id)[0];
            if (plan.beta_input >= 0) {
                const std::string& bt =
                    vals.at(ew.inputs[static_cast<std::size_t>(plan.beta_input)].node_id)[0];
                b << "    fma.rn.f32 " << f_n << ", " << f_n << ", " << g << ", " << bt << ";\n";
            } else {
                b << "    mul.f32 " << f_n << ", " << f_n << ", " << g << ";\n";
            }
        }
        vals[plan.norm_node] = {f_n};

        vals = emit_chain(b, a, dag, 1, std::move(vals));

        for (std::size_t m = 0; m < n_out; ++m) {
            const BufferSpec& s = ew.outputs[m];
            const std::string off = full_off(s.dtype);
            const std::string addr = a.RD();
            b << "    add.u64 " << addr << ", " << out_ptr[m] << ", " << off << ";\n";
            emit_store(b, a, s.dtype, 1, addr, vals.at(s.node_id));
        }

        b << "    add.u32 " << r_c << ", " << r_c << ", " << kRowThreads << ";\n";
        b << "    bra $L_rn_apply;\n";
        b << "$L_rn_apply_end:\n";
    }

    b << "$L_rn_done:\n";
    b << "    ret;\n";

    std::stringstream ss;
    ss << ".version 7.8\n";
    ss << ".target " << arch << "\n";
    ss << ".address_size 64\n\n";
    // Per-CTA partials: one slot per warp for sum-of-squares, and (LayerNorm
    // only) one more set for the plain sum.
    ss << ".shared .align 4 .b32 rn_red[" << (2 * (kRowThreads / 32)) << "];\n\n";
    ss << ".visible .entry " << kEntryRowNorm << "(\n";
    for (std::size_t m = 0; m < n_out; ++m) ss << "    .param .u64 rn_out_" << m << ",\n";
    for (std::size_t k = 0; k + 1 < n_in; ++k) ss << "    .param .u64 rn_in_" << k << ",\n";
    ss << "    .param .u64 rn_in_" << (n_in - 1) << "\n";
    ss << ") {\n";
    ss << declarations(a);
    ss << b.str();
    ss << "}\n";
    return ss.str();
}

}  // namespace brotensor::jit::ptx
