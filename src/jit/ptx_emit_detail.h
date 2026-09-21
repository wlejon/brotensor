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
            o << "    div.rn.f32 " << d << ", " << in[0] << ", " << fhex(n.scalar) << ";\n";
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
    for (int id : ids) {
        const TraceNode& n = dag.node(id);
        if (n.op == TraceOpKind::Input) continue;
        if (vals.count(n.id)) continue;

        std::vector<std::string> dst(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            std::vector<std::string> in;
            in.reserve(n.inputs.size());
            for (int src : n.inputs) {
                auto it = vals.find(src);
                if (it == vals.end()) {
                    throw std::runtime_error("brotensor::jit: trace node " +
                                             std::to_string(n.id) +
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

}  // namespace brotensor::jit::ptx::detail
