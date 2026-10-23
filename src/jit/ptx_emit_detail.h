#pragma once

// Shared PTX-emission helpers for the trace JIT's two CUDA emitters.
//
// One rule governs every function here: an emitter helper may write to the
// stream, so its result must be bound to a local before it is used inside
// another stream expression. `o << "add.u64 " << d << ", " << p << ", " <<
// offset(dt)` interleaves the two instructions and produces PTX that does not
// parse.

#include "ptx_emit.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brotensor::jit::ptx::detail {

inline std::string fhex(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(float));
    std::stringstream ss;
    ss << "0f" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << bits;
    return ss.str();
}

// Virtual-register allocator. PTX register banks are declared up front, so a
// body is emitted into its own stream and `declarations()` is prepended once
// the high-water marks are known.
struct Alloc {
    int f = 0, r = 0, rd = 0, h = 0, p = 0;
    std::string F() { return "%f" + std::to_string(f++); }
    std::string R() { return "%r" + std::to_string(r++); }
    std::string RD() { return "%rd" + std::to_string(rd++); }
    std::string H() { return "%h" + std::to_string(h++); }
    std::string P() { return "%p" + std::to_string(p++); }
};

inline std::string declarations(const Alloc& a) {
    std::stringstream ss;
    if (a.p) ss << "    .reg .pred %p<" << a.p << ">;\n";
    if (a.r) ss << "    .reg .b32 %r<" << a.r << ">;\n";
    if (a.rd) ss << "    .reg .b64 %rd<" << a.rd << ">;\n";
    if (a.h) ss << "    .reg .b16 %h<" << a.h << ">;\n";
    if (a.f) ss << "    .reg .f32 %f<" << a.f << ">;\n";
    return ss.str();
}

inline bool is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

inline int log2_exact(int v) {
    int l = 0;
    while ((1 << l) < v) ++l;
    return l;
}

inline bool is_exact_pow2_f32(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(float));
    const std::uint32_t exp = (bits >> 23) & 0xFFu;
    const std::uint32_t mant = bits & 0x007FFFFFu;
    return mant == 0u && exp > 1u && exp < 253u;
}

// Unsigned 32-bit Granlund-Montgomery magic-multiplier parameters for exact
// division by a runtime-constant divisor `d >= 1` using PTX `mul.hi.u32`.
struct MagicU32 {
    std::uint32_t mul = 1;
    std::uint32_t pre_shift = 0;
    std::uint32_t post_shift = 0;
    bool add = false;
};

inline MagicU32 magic_u32(std::uint32_t d) {
    MagicU32 m;
    if (d <= 1u) return m;
    while ((d & 1u) == 0u) {
        d >>= 1u;
        ++m.pre_shift;
    }
    if (d == 1u) return m;
    std::uint32_t l = 0;
    while ((1ULL << l) < d) ++l;
    const std::uint64_t p_lo = 1ULL << (32 + l - 1);
    const std::uint64_t m_lo = (p_lo + d - 1ULL) / d;
    const std::uint64_t e_lo = m_lo * d - p_lo;
    const std::uint64_t max_v = 0xFFFFFFFFULL >> m.pre_shift;
    const std::uint64_t nc = (max_v / d) * d - 1ULL;
    if (nc * e_lo < p_lo) {
        m.mul = static_cast<std::uint32_t>(m_lo);
        m.post_shift = l - 1u;
        m.add = false;
    } else {
        const std::uint64_t p_hi = 1ULL << (32 + l);
        const std::uint64_t m_hi = (p_hi + d - 1ULL) / d;
        m.mul = static_cast<std::uint32_t>(m_hi - (1ULL << 32));
        m.post_shift = l;
        m.add = true;
    }
    return m;
}

inline std::string emit_udiv_const(std::ostream& o, Alloc& a, const std::string& r_x, int d) {
    if (d <= 1) return r_x;
    if (is_pow2(d)) {
        const std::string r_q = a.R();
        o << "    shr.u32 " << r_q << ", " << r_x << ", " << log2_exact(d) << ";\n";
        return r_q;
    }
    const MagicU32 m = magic_u32(static_cast<std::uint32_t>(d));
    std::string r_xp = r_x;
    if (m.pre_shift > 0u) {
        r_xp = a.R();
        o << "    shr.u32 " << r_xp << ", " << r_x << ", " << m.pre_shift << ";\n";
    }
    const std::string r_hi = a.R();
    o << "    mul.hi.u32 " << r_hi << ", " << r_xp << ", " << m.mul << ";\n";
    if (!m.add) {
        if (m.post_shift == 0u) return r_hi;
        const std::string r_q = a.R();
        o << "    shr.u32 " << r_q << ", " << r_hi << ", " << m.post_shift << ";\n";
        return r_q;
    }
    const std::string r_diff = a.R(), r_half = a.R(), r_sum = a.R();
    o << "    sub.u32 " << r_diff << ", " << r_xp << ", " << r_hi << ";\n";
    o << "    shr.u32 " << r_half << ", " << r_diff << ", 1;\n";
    o << "    add.u32 " << r_sum << ", " << r_half << ", " << r_hi << ";\n";
    if (m.post_shift <= 1u) return r_sum;
    const std::string r_q = a.R();
    o << "    shr.u32 " << r_q << ", " << r_sum << ", " << (m.post_shift - 1u) << ";\n";
    return r_q;
}

inline void emit_udiv_urem_const(std::ostream& o, Alloc& a, const std::string& r_x, int d,
                                 std::string& r_q, std::string& r_r) {
    r_q = emit_udiv_const(o, a, r_x, d);
    r_r = a.R();
    if (d <= 1) {
        o << "    mov.u32 " << r_r << ", 0;\n";
    } else if (is_pow2(d)) {
        o << "    and.b32 " << r_r << ", " << r_x << ", " << (d - 1) << ";\n";
    } else {
        const std::string r_prod = a.R();
        o << "    mul.lo.u32 " << r_prod << ", " << r_q << ", " << d << ";\n";
        o << "    sub.u32 " << r_r << ", " << r_x << ", " << r_prod << ";\n";
    }
}

// ── one lane of the op chain ────────────────────────────────────────────────

inline void emit_sigmoid(std::ostream& o, Alloc& a, const std::string& x,
                         const std::string& d) {
    const std::string t0 = a.F(), t1 = a.F(), t2 = a.F();
    o << "    neg.f32 " << t0 << ", " << x << ";\n";
    o << "    mul.f32 " << t0 << ", " << t0 << ", 0f3FB8AA3B;\n";  // * log2(e)
    o << "    ex2.approx.f32 " << t1 << ", " << t0 << ";\n";
    o << "    add.f32 " << t2 << ", " << t1 << ", 0f3F800000;\n";
    o << "    rcp.approx.f32 " << d << ", " << t2 << ";\n";
}

inline void emit_lane_op(std::ostream& o, Alloc& a, const TraceNode& n,
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
            if (is_exact_pow2_f32(n.scalar)) {
                o << "    mul.f32 " << d << ", " << in[0] << ", " << fhex(1.0f / n.scalar) << ";\n";
            } else {
                o << "    div.rn.f32 " << d << ", " << in[0] << ", " << fhex(n.scalar) << ";\n";
            }
            break;
        case TraceOpKind::ScalarSub:
            o << "    sub.f32 " << d << ", " << fhex(n.scalar) << ", " << in[0] << ";\n";
            break;
        case TraceOpKind::ScalarDiv:
            o << "    div.rn.f32 " << d << ", " << fhex(n.scalar) << ", " << in[0] << ";\n";
            break;
        case TraceOpKind::FMA:
            o << "    fma.rn.f32 " << d << ", " << in[0] << ", " << in[1] << ", " << in[2]
              << ";\n";
            break;
        case TraceOpKind::ReLU:
            o << "    max.f32 " << d << ", " << in[0] << ", 0f00000000;\n";
            break;
        case TraceOpKind::Clamp: {
            const std::string t = a.F();
            o << "    max.f32 " << t << ", " << in[0] << ", " << fhex(n.scalar) << ";\n";
            o << "    min.f32 " << d << ", " << t << ", " << fhex(n.scalar2) << ";\n";
            break;
        }
        case TraceOpKind::Where: {
            const std::string p = a.P();
            o << "    setp.ne.f32 " << p << ", " << in[0] << ", 0f00000000;\n";
            o << "    selp.f32 " << d << ", " << in[1] << ", " << in[2] << ", " << p << ";\n";
            break;
        }
        case TraceOpKind::Cast: {
            if (n.dtype == Dtype::FP16) {
                const std::string h = a.H();
                o << "    cvt.rn.f16.f32 " << h << ", " << in[0] << ";\n";
                o << "    cvt.f32.f16 " << d << ", " << h << ";\n";
            } else if (n.dtype == Dtype::BF16) {
                const std::string h = a.H(), t = a.R();
                o << "    cvt.rn.bf16.f32 " << h << ", " << in[0] << ";\n";
                o << "    cvt.u32.u16 " << t << ", " << h << ";\n";
                o << "    shl.b32 " << t << ", " << t << ", 16;\n";
                o << "    mov.b32 " << d << ", " << t << ";\n";
            } else {
                o << "    mov.f32 " << d << ", " << in[0] << ";\n";
            }
            break;
        }
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
        case TraceOpKind::Copy:
            // The value already sits in a register; the node exists so the
            // store below lands in the caller's buffer.
            o << "    mov.f32 " << d << ", " << in[0] << ";\n";
            break;
        case TraceOpKind::GELU: {
            // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) — the tanh
            // approximation nn.GELU(approximate="tanh") uses.
            const std::string x2 = a.F(), inner = a.F(), poly = a.F();
            const std::string arg = a.F(), th = a.F(), t1 = a.F(), hx = a.F();
            o << "    mul.f32 " << x2 << ", " << in[0] << ", " << in[0] << ";\n";
            o << "    fma.rn.f32 " << inner << ", " << x2 << ", " << fhex(0.044715f) << ", "
              << fhex(1.0f) << ";\n";
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
            throw std::runtime_error(std::string("brotensor::jit: op '") + op_kind_name(n.op) +
                                     "' has no elementwise PTX form");
    }
}

// ── typed load / store ──────────────────────────────────────────────────────

// Loads `lanes` consecutive elements at [addr] into fresh FP32 registers.
// `lanes` is 1, 2, 4 or 8 and the access is one instruction.
inline std::vector<std::string> emit_load(std::ostream& o, Alloc& a, Dtype dt, int lanes,
                                          const std::string& addr) {
    std::vector<std::string> out(static_cast<std::size_t>(lanes));
    for (int i = 0; i < lanes; ++i) out[static_cast<std::size_t>(i)] = a.F();

    if (dt == Dtype::FP32) {
        if (lanes == 1) {
            o << "    ld.global.f32 " << out[0] << ", [" << addr << "];\n";
        } else {
            o << "    ld.global.v" << lanes << ".f32 {";
            for (int i = 0; i < lanes; ++i)
                o << (i ? ", " : "") << out[static_cast<std::size_t>(i)];
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

inline void emit_store(std::ostream& o, Alloc& a, Dtype dt, int lanes,
                       const std::string& addr, const std::vector<std::string>& vals) {
    if (dt == Dtype::FP32) {
        if (lanes == 1) {
            o << "    st.global.f32 [" << addr << "], " << vals[0] << ";\n";
        } else {
            o << "    st.global.v" << lanes << ".f32 [" << addr << "], {";
            for (int i = 0; i < lanes; ++i)
                o << (i ? ", " : "") << vals[static_cast<std::size_t>(i)];
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
        o << "    " << cvt << " " << h1 << ", " << vals[static_cast<std::size_t>(2 * i + 1)]
          << ";\n";
        w[static_cast<std::size_t>(i)] = a.R();
        o << "    mov.b32 " << w[static_cast<std::size_t>(i)] << ", {" << h0 << ", " << h1
          << "};\n";
    }
    if (words == 1) {
        o << "    st.global.u32 [" << addr << "], " << w[0] << ";\n";
    } else {
        o << "    st.global.v" << words << ".u32 [" << addr << "], {";
        for (int i = 0; i < words; ++i) o << (i ? ", " : "") << w[static_cast<std::size_t>(i)];
        o << "};\n";
    }
}

// ── the op chain ────────────────────────────────────────────────────────────

using ValueMap = std::unordered_map<int, std::vector<std::string>>;

inline bool same_node_expr(const TraceNode& a, const TraceNode& b) {
    if (a.op != b.op || a.dtype != b.dtype || a.inputs != b.inputs) return false;
    std::uint32_t as1 = 0, bs1 = 0, as2 = 0, bs2 = 0;
    std::memcpy(&as1, &a.scalar, sizeof(float));
    std::memcpy(&bs1, &b.scalar, sizeof(float));
    std::memcpy(&as2, &a.scalar2, sizeof(float));
    std::memcpy(&bs2, &b.scalar2, sizeof(float));
    return as1 == bs1 && as2 == bs2;
}

// Evaluates every node in `order` (default: the whole DAG, in creation order,
// which is topological) that does not already have a value. Nodes seeded by
// the caller — a loaded input, or the row-norm emitter's normalised value —
// are left alone.
inline ValueMap emit_chain(std::ostream& o, Alloc& a, const TraceDAG& dag, int lanes,
                           ValueMap vals, const std::vector<int>* order = nullptr) {
    std::vector<int> ids;
    if (order) {
        ids = *order;
    } else {
        ids.reserve(dag.node_count());
        for (const auto& n : dag.nodes()) ids.push_back(n.id);
    }

    // Identify single-use Mul nodes that feed directly into an Add in this chain.
    std::unordered_map<int, int> use_count;
    for (const auto& n : dag.nodes()) {
        for (int src : n.inputs) ++use_count[src];
    }
    std::unordered_set<int> in_chain(ids.begin(), ids.end());
    std::unordered_set<int> fused_mul;
    for (int id : ids) {
        const TraceNode& n = dag.node(id);
        if (n.op != TraceOpKind::Add || n.inputs.size() < 2) continue;
        for (int k = 0; k < 2; ++k) {
            const int src = n.inputs[static_cast<std::size_t>(k)];
            if (!in_chain.count(src) || vals.count(src)) continue;
            const TraceNode& m = dag.node(src);
            if (m.op == TraceOpKind::Mul && !m.is_live_at_end && use_count[src] == 1 &&
                m.inputs.size() >= 2) {
                fused_mul.insert(src);
                break;
            }
        }
    }

    std::vector<int> evaluated;
    evaluated.reserve(ids.size());

    for (int id : ids) {
        const TraceNode& n = dag.node(id);
        if (n.op == TraceOpKind::Input) continue;
        if (vals.count(n.id)) continue;
        if (fused_mul.count(n.id)) continue;

        if (n.op == TraceOpKind::Cast && !n.inputs.empty()) {
            const int src = n.inputs[0];
            if (n.dtype == Dtype::FP32 || dag.node(src).dtype == n.dtype) {
                vals[n.id] = vals.at(src);
                continue;
            }
        }
        if (n.op == TraceOpKind::Copy && !n.inputs.empty()) {
            vals[n.id] = vals.at(n.inputs[0]);
            continue;
        }

        int cse_match = -1;
        for (int prev_id : evaluated) {
            if (same_node_expr(n, dag.node(prev_id))) {
                cse_match = prev_id;
                break;
            }
        }
        if (cse_match >= 0) {
            vals[n.id] = vals.at(cse_match);
            continue;
        }

        int mul_input_idx = -1;
        if (n.op == TraceOpKind::Add && n.inputs.size() >= 2) {
            if (fused_mul.count(n.inputs[0])) mul_input_idx = 0;
            else if (fused_mul.count(n.inputs[1])) mul_input_idx = 1;
        }

        std::vector<std::string> dst(static_cast<std::size_t>(lanes));
        if (mul_input_idx >= 0) {
            const TraceNode& m = dag.node(n.inputs[static_cast<std::size_t>(mul_input_idx)]);
            const int addend_id = n.inputs[static_cast<std::size_t>(1 - mul_input_idx)];
            const auto& va = vals.at(m.inputs[0]);
            const auto& vb = vals.at(m.inputs[1]);
            const auto& vd = vals.at(addend_id);
            for (int j = 0; j < lanes; ++j) {
                const std::size_t js = static_cast<std::size_t>(j);
                if (j > 0 && va[js] == va[0] && vb[js] == vb[0] && vd[js] == vd[0]) {
                    dst[js] = dst[0];
                } else {
                    dst[js] = a.F();
                    o << "    fma.rn.f32 " << dst[js] << ", " << va[js] << ", " << vb[js] << ", "
                      << vd[js] << ";\n";
                }
            }
            vals[n.id] = std::move(dst);
            evaluated.push_back(n.id);
            continue;
        }

        std::vector<std::string> in0;
        in0.reserve(n.inputs.size());
        for (int j = 0; j < lanes; ++j) {
            std::vector<std::string> in;
            in.reserve(n.inputs.size());
            bool same_as_lane0 = (j > 0);
            for (std::size_t k = 0; k < n.inputs.size(); ++k) {
                const int src = n.inputs[k];
                auto it = vals.find(src);
                if (it == vals.end()) {
                    throw std::runtime_error("brotensor::jit: trace node " +
                                             std::to_string(n.id) +
                                             " reads an operand that was never materialised");
                }
                const std::string& v = it->second[static_cast<std::size_t>(j)];
                in.push_back(v);
                if (j > 0 && v != in0[k]) same_as_lane0 = false;
            }
            if (j == 0) in0 = in;
            if (same_as_lane0) {
                dst[static_cast<std::size_t>(j)] = dst[0];
            } else {
                dst[static_cast<std::size_t>(j)] = a.F();
                emit_lane_op(o, a, n, in, dst[static_cast<std::size_t>(j)]);
            }
        }
        vals[n.id] = std::move(dst);
        evaluated.push_back(n.id);
    }
    return vals;
}

}  // namespace brotensor::jit::ptx::detail
