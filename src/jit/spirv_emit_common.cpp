// Shared MIR helpers for the Vulkan trace emitters: typed lane loads and
// stores, BF16/FP16 packing, the per-lane op table. Notes: spirv_emit_detail.h.

#include "spirv_emit_detail.h"

#include <cstdint>
#include <stdexcept>

namespace brotensor::jit::spirv::detail {

namespace {

constexpr float kLog2e = 1.4426950408889634f;

// 1 / (1 + 2^(-x log2 e)), with the exponent clamped so ex2 stays finite: the
// module does not declare SignedZeroInfNanPreserve, so an infinity in the
// middle of the expression is undefined rather than "rounds to 0".
Value* sigmoid(Ctx& c, Value* x) {
    Value* t = c.kb.fmin(c.b.build_mul(x, c.f32(-kLog2e)), c.f32(88.0f));
    Value* e = c.kb.ex2_approx(t);
    return c.kb.rcp_approx(c.b.build_add(e, c.f32(1.0f)));
}

// tanh(x) = sign(x) (1 - e) / (1 + e), e = 2^(-2|x| log2 e) in (0, 1]: no
// overflow anywhere, and no cancellation worse than the absolute error of e.
Value* tanh_(Ctx& c, Value* x) {
    Builder& b = c.b;
    Value* e = c.kb.ex2_approx(b.build_mul(c.kb.fabs(x), c.f32(-2.0f * kLog2e)));
    Value* t = c.kb.div_approx(b.build_sub(c.f32(1.0f), e), b.build_add(c.f32(1.0f), e));
    return b.build_select(b.build_slt(x, c.f32(0.0f)), b.build_neg(t), t);
}

}  // namespace

std::vector<Value*> entry_params(Ctx& c, Function* fn) {
    brass::BasicBlock* entry = c.b.append_block("entry");
    c.b.position_at_end(entry);
    std::vector<Value*> out;
    for (Type t : fn->param_types()) out.push_back(c.b.add_block_param(entry, t));
    return out;
}

Value* bits_to_f32(Ctx& c, Value* bits) { return c.b.build_bitcast_f32_i32(bits); }

Value* f32_to_bits(Ctx& c, Value* x) { return c.b.build_bitcast_i32_f32(x); }

Value* byte_offset(Ctx& c, Value* elem, int esz) {
    Value* e = c.b.build_zext_i64(elem);
    return esz == 1 ? e : c.b.build_shl(e, c.kb.const_i64(log2_exact(esz)));
}

namespace {

// The two halves of a packed word as FP32.
void unpack_word(Ctx& c, Dtype dt, Value* w, std::vector<Value*>& out) {
    Builder& b = c.b;
    if (dt == Dtype::FP16) {
        // UnpackHalf2x16(x).x reads the low 16 bits only.
        out.push_back(c.kb.f16_to_f32(w));
        out.push_back(c.kb.f16_to_f32(b.build_lshr(w, c.i32(16))));
    } else {
        out.push_back(bits_to_f32(c, b.build_shl(w, c.i32(16))));
        out.push_back(bits_to_f32(c, b.build_and(w, c.i32(static_cast<int>(0xFFFF0000u)))));
    }
}

// One FP32 value as the 16 bits of `dt` (in the low half of an i32, zeros
// above). BF16 rounds to nearest even and keeps NaNs quiet — bit-identical to
// f32_to_bf16 in shaders/common.glsl and to brotensor::fp32_to_bf16_bits.
Value* pack_half(Ctx& c, Dtype dt, Value* x) {
    Builder& b = c.b;
    if (dt == Dtype::FP16) return c.kb.f32_to_f16(x);
    Value* u = f32_to_bits(c, x);
    Value* hi = b.build_lshr(u, c.i32(16));
    Value* rounded = b.build_lshr(
        b.build_add(b.build_add(u, c.i32(0x7FFF)), b.build_and(hi, c.i32(1))), c.i32(16));
    Value* is_nan = b.build_and(
        b.build_eq(b.build_and(u, c.i32(0x7F800000)), c.i32(0x7F800000)),
        b.build_ne(b.build_and(u, c.i32(0x007FFFFF)), c.i32(0)));
    Value* quiet = b.build_or(hi, c.i32(0x40));
    return b.build_select(b.build_ne(is_nan, c.i32(0)), quiet, rounded);
}

}  // namespace

std::vector<Value*> load_lanes(Ctx& c, Dtype dt, int lanes, Value* addr) {
    Builder& b = c.b;
    std::vector<Value*> out;
    out.reserve(static_cast<std::size_t>(lanes));
    if (dt == Dtype::FP32) {
        if (lanes == 1) {
            out.push_back(b.build_load(Type::f32(), addr, 0));
        } else if (lanes == 4) {
            Value* v = c.kb.vload_f32x4(addr);
            for (uint32_t j = 0; j < 4; ++j) out.push_back(b.build_vextract_lane(v, j));
        } else {
            throw std::logic_error("brotensor::jit: FP32 load of " + std::to_string(lanes) + " lanes");
        }
        return out;
    }
    if (lanes == 1) {
        Value* h = c.kb.load_u16(addr);
        if (dt == Dtype::FP16) {
            out.push_back(c.kb.f16_to_f32(h));
        } else {
            out.push_back(bits_to_f32(c, b.build_shl(h, c.i32(16))));
        }
        return out;
    }
    if (lanes == 4) {
        Value* w = c.kb.load_i64(addr);
        unpack_word(c, dt, b.build_trunc_i32(w), out);
        unpack_word(c, dt, b.build_trunc_i32(b.build_lshr(w, c.kb.const_i64(32))), out);
        return out;
    }
    if (lanes == 8) {
        Value* v = c.kb.vload(Type::i32x4(), addr);
        for (uint32_t j = 0; j < 4; ++j) unpack_word(c, dt, b.build_vextract_lane(v, j), out);
        return out;
    }
    throw std::logic_error("brotensor::jit: 16-bit load of " + std::to_string(lanes) + " lanes");
}

void store_lanes(Ctx& c, Dtype dt, int lanes, Value* addr, const std::vector<Value*>& vals) {
    Builder& b = c.b;
    if (dt == Dtype::FP32) {
        if (lanes == 1) {
            b.build_store(Type::f32(), addr, 0, vals[0]);
        } else {
            Value* v = b.build_vzero(Type::f32x4());
            for (uint32_t j = 0; j < 4; ++j) v = b.build_vinsert_lane(v, vals[j], j);
            c.kb.vstore_f32x4(addr, v);
        }
        return;
    }
    if (lanes == 1) {
        c.kb.store_u16(addr, pack_half(c, dt, vals[0]));
        return;
    }
    std::vector<Value*> words;
    for (int j = 0; j < lanes; j += 2) {
        Value* lo = pack_half(c, dt, vals[static_cast<std::size_t>(j)]);
        Value* hi = pack_half(c, dt, vals[static_cast<std::size_t>(j + 1)]);
        words.push_back(b.build_or(lo, b.build_shl(hi, c.i32(16))));
    }
    if (lanes == 4) {
        Value* w = b.build_or(b.build_zext_i64(words[0]),
                              b.build_shl(b.build_zext_i64(words[1]), c.kb.const_i64(32)));
        c.kb.store_i64(addr, w);
        return;
    }
    Value* v = b.build_vzero(Type::i32x4());
    for (uint32_t j = 0; j < 4; ++j) v = b.build_vinsert_lane(v, words[j], j);
    c.kb.vstore(Type::i32x4(), addr, v);
}

bool has_lane_form(TraceOpKind op) {
    switch (op) {
        case TraceOpKind::Add: case TraceOpKind::Sub: case TraceOpKind::Mul: case TraceOpKind::Div:
        case TraceOpKind::AddScalar: case TraceOpKind::SubScalar: case TraceOpKind::MulScalar:
        case TraceOpKind::DivScalar: case TraceOpKind::ScalarSub: case TraceOpKind::ScalarDiv:
        case TraceOpKind::FMA: case TraceOpKind::SiLU: case TraceOpKind::GELU: case TraceOpKind::ReLU:
        case TraceOpKind::Tanh: case TraceOpKind::Sigmoid: case TraceOpKind::Modulate:
        case TraceOpKind::Copy:
            return true;
        default:
            return false;
    }
}

Value* lane_op(Ctx& c, const TraceNode& n, const std::vector<Value*>& in) {
    Builder& b = c.b;
    auto need = [&](std::size_t k) {
        if (in.size() < k) {
            throw std::runtime_error(std::string("brotensor::jit: ") + op_kind_name(n.op) +
                                     " is missing an operand");
        }
    };
    switch (n.op) {
        case TraceOpKind::Add: need(2); return b.build_add(in[0], in[1]);
        case TraceOpKind::Sub: need(2); return b.build_sub(in[0], in[1]);
        case TraceOpKind::Mul: need(2); return b.build_mul(in[0], in[1]);
        case TraceOpKind::Div: need(2); return b.build_sdiv(in[0], in[1]);
        case TraceOpKind::AddScalar: need(1); return b.build_add(in[0], c.f32(n.scalar));
        case TraceOpKind::SubScalar: need(1); return b.build_sub(in[0], c.f32(n.scalar));
        case TraceOpKind::MulScalar: need(1); return b.build_mul(in[0], c.f32(n.scalar));
        case TraceOpKind::DivScalar: need(1); return b.build_sdiv(in[0], c.f32(n.scalar));
        case TraceOpKind::ScalarSub: need(1); return b.build_sub(c.f32(n.scalar), in[0]);
        case TraceOpKind::ScalarDiv: need(1); return b.build_sdiv(c.f32(n.scalar), in[0]);
        case TraceOpKind::FMA: need(3); return b.build_fma_f32(in[0], in[1], in[2]);
        case TraceOpKind::ReLU: need(1); return c.kb.fmax(in[0], c.f32(0.0f));
        case TraceOpKind::Sigmoid: need(1); return sigmoid(c, in[0]);
        case TraceOpKind::SiLU: need(1); return b.build_mul(in[0], sigmoid(c, in[0]));
        case TraceOpKind::Tanh: need(1); return tanh_(c, in[0]);
        case TraceOpKind::Copy: need(1); return in[0];
        case TraceOpKind::GELU: {
            // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) — the tanh
            // approximation nn.GELU(approximate="tanh") uses, as the PTX form.
            need(1);
            Value* x = in[0];
            Value* x2 = b.build_mul(x, x);
            Value* inner = b.build_fma_f32(x2, c.f32(0.044715f), c.f32(1.0f));
            Value* arg = b.build_mul(b.build_mul(inner, x), c.f32(0.7978845608f));
            Value* t1 = b.build_add(tanh_(c, arg), c.f32(1.0f));
            return b.build_mul(b.build_mul(x, c.f32(0.5f)), t1);
        }
        case TraceOpKind::Modulate:
            // x * (1 + scale) + shift; scale/shift are normally (1, D) rows
            // the loader has already resolved to this lane's column.
            need(3);
            return b.build_fma_f32(in[0], b.build_add(in[1], c.f32(1.0f)), in[2]);
        default:
            throw std::runtime_error(std::string("brotensor::jit: op '") + op_kind_name(n.op) +
                                     "' has no elementwise SPIR-V form");
    }
}

ValueMap eval_chain(Ctx& c, const TraceDAG& dag, int lanes, ValueMap vals,
                    const std::vector<int>* order) {
    std::vector<int> ids;
    if (order) {
        ids = *order;
    } else {
        for (const auto& n : dag.nodes()) ids.push_back(n.id);
    }
    for (int id : ids) {
        const TraceNode& n = dag.node(id);
        if (n.op == TraceOpKind::Input || vals.count(n.id)) continue;
        std::vector<Value*> dst(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            std::vector<Value*> in;
            for (int src : n.inputs) {
                auto it = vals.find(src);
                if (it == vals.end()) {
                    throw std::runtime_error("brotensor::jit: trace node " + std::to_string(n.id) +
                                             " reads an operand that was never materialised");
                }
                in.push_back(it->second[static_cast<std::size_t>(j)]);
            }
            dst[static_cast<std::size_t>(j)] = lane_op(c, n, in);
        }
        vals[n.id] = std::move(dst);
    }
    return vals;
}

brass::target::SpirvKernel compile(const Function& fn, std::uint32_t block) {
    brass::target::SpirvOptions opts;
    opts.local_size_x = block;
    return brass::target::SpirvTarget::compile(fn, opts);
}

}  // namespace brotensor::jit::spirv::detail
