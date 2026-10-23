// Shared MIR helpers for the Vulkan trace emitters: typed lane loads and
// stores, BF16/FP16 packing, the per-lane op table. Notes: spirv_emit_detail.h.

#include "spirv_emit_detail.h"

#include <brass/mir/fma_opt.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

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

Value* udiv_const(Ctx& c, Value* x, int d) {
    if (d <= 1) return x;
    Builder& b = c.b;
    if (is_pow2(d)) return b.build_lshr(x, c.i32(log2_exact(d)));
    const MagicU32 m = magic_u32(static_cast<std::uint32_t>(d));
    Value* xp = m.pre_shift ? b.build_lshr(x, c.i32(static_cast<int>(m.pre_shift))) : x;
    Value* prod64 = b.build_mul(b.build_zext_i64(xp), c.kb.const_i64(static_cast<std::int64_t>(m.mul)));
    if (!m.add) {
        Value* sh = b.build_lshr(prod64, c.kb.const_i64(static_cast<std::int64_t>(32u + m.post_shift)));
        return b.build_trunc_i32(sh);
    }
    Value* q = b.build_trunc_i32(b.build_lshr(prod64, c.kb.const_i64(32)));
    Value* diff = b.build_sub(xp, q);
    Value* half = b.build_lshr(diff, c.i32(1));
    Value* sum = b.build_add(half, q);
    return m.post_shift > 1 ? b.build_lshr(sum, c.i32(static_cast<int>(m.post_shift - 1))) : sum;
}

void udiv_urem_const(Ctx& c, Value* x, int d, Value*& q, Value*& r) {
    Builder& b = c.b;
    q = udiv_const(c, x, d);
    if (d <= 1) {
        r = c.i32(0);
    } else if (is_pow2(d)) {
        r = b.build_and(x, c.i32(d - 1));
    } else {
        r = b.build_sub(x, b.build_mul(q, c.i32(d)));
    }
}

namespace {

bool is_exact_pow2_f32(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(float));
    const std::uint32_t exp = (bits >> 23) & 0xFFu;
    const std::uint32_t mant = bits & 0x007FFFFFu;
    return mant == 0u && exp > 1u && exp < 253u;
}

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
        case TraceOpKind::Copy: case TraceOpKind::Where: case TraceOpKind::Clamp:
        case TraceOpKind::Cast:
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
        case TraceOpKind::DivScalar:
            need(1);
            if (is_exact_pow2_f32(n.scalar)) return b.build_mul(in[0], c.f32(1.0f / n.scalar));
            return b.build_sdiv(in[0], c.f32(n.scalar));
        case TraceOpKind::ScalarSub: need(1); return b.build_sub(c.f32(n.scalar), in[0]);
        case TraceOpKind::ScalarDiv: need(1); return b.build_sdiv(c.f32(n.scalar), in[0]);
        case TraceOpKind::FMA: need(3); return b.build_fma_f32(in[0], in[1], in[2]);
        case TraceOpKind::ReLU: need(1); return c.kb.fmax(in[0], c.f32(0.0f));
        case TraceOpKind::Clamp: need(1); return c.kb.fmin(c.kb.fmax(in[0], c.f32(n.scalar)), c.f32(n.scalar2));
        case TraceOpKind::Where: {
            need(3);
            Value* bits = b.build_and(f32_to_bits(c, in[0]), c.i32(0x7FFFFFFF));
            return b.build_select(b.build_ne(bits, c.i32(0)), in[1], in[2]);
        }
        case TraceOpKind::Cast: {
            need(1);
            if (n.dtype == Dtype::FP16) return c.kb.f16_to_f32(c.kb.f32_to_f16(in[0]));
            if (n.dtype == Dtype::BF16) {
                return bits_to_f32(c, b.build_shl(pack_half(c, Dtype::BF16, in[0]), c.i32(16)));
            }
            return in[0];
        }
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

namespace {

bool same_node_expr(const TraceNode& a, const TraceNode& b) {
    if (a.op != b.op || a.dtype != b.dtype || a.inputs != b.inputs) return false;
    std::uint32_t as1 = 0, bs1 = 0, as2 = 0, bs2 = 0;
    std::memcpy(&as1, &a.scalar, sizeof(float));
    std::memcpy(&bs1, &b.scalar, sizeof(float));
    std::memcpy(&as2, &a.scalar2, sizeof(float));
    std::memcpy(&bs2, &b.scalar2, sizeof(float));
    return as1 == bs1 && as2 == bs2;
}

}  // namespace

ValueMap eval_chain(Ctx& c, const TraceDAG& dag, int lanes, ValueMap vals,
                    const std::vector<int>* order) {
    std::vector<int> ids;
    if (order) {
        ids = *order;
    } else {
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
        if (n.op == TraceOpKind::Input || vals.count(n.id)) continue;
        if (fused_mul.count(n.id)) continue;

        // Elide redundant Cast when input node already carries the target dtype
        // or when casting to FP32 (since registers are already FP32).
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

        // DAG-level CSE across identical expressions in the same chain.
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

        // Fused Add(Mul(a, b), d) or Add(d, Mul(a, b)) -> FMA(a, b, d).
        int mul_input_idx = -1;
        if (n.op == TraceOpKind::Add && n.inputs.size() >= 2) {
            if (fused_mul.count(n.inputs[0])) mul_input_idx = 0;
            else if (fused_mul.count(n.inputs[1])) mul_input_idx = 1;
        }

        std::vector<Value*> dst(static_cast<std::size_t>(lanes));
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
                    dst[js] = c.b.build_fma_f32(va[js], vb[js], vd[js]);
                }
            }
            vals[n.id] = std::move(dst);
            evaluated.push_back(n.id);
            continue;
        }

        std::vector<Value*> in0;
        in0.reserve(n.inputs.size());
        for (int j = 0; j < lanes; ++j) {
            std::vector<Value*> in;
            in.reserve(n.inputs.size());
            bool same_as_lane0 = (j > 0);
            for (std::size_t k = 0; k < n.inputs.size(); ++k) {
                const int src = n.inputs[k];
                auto it = vals.find(src);
                if (it == vals.end()) {
                    throw std::runtime_error("brotensor::jit: trace node " + std::to_string(n.id) +
                                             " reads an operand that was never materialised");
                }
                Value* v = it->second[static_cast<std::size_t>(j)];
                in.push_back(v);
                if (j > 0 && v != in0[k]) same_as_lane0 = false;
            }
            if (j == 0) in0 = in;
            if (same_as_lane0) {
                dst[static_cast<std::size_t>(j)] = dst[0];
            } else {
                dst[static_cast<std::size_t>(j)] = lane_op(c, n, in);
            }
        }
        vals[n.id] = std::move(dst);
        evaluated.push_back(n.id);
    }
    return vals;
}

void optimize_function(Function& fn) {
    brass::fold_constants(fn);
    brass::dominator_cse(fn);
    brass::GvnOptions gvn_opts;
    gvn_opts.enable_cse = true;
    gvn_opts.enable_rle = true;
    gvn_opts.enable_dse = true;
    brass::gvn_function(fn, gvn_opts);
    if (fn.parent()) {
        brass::SmtGuardElimOptions ge_opts;
        ge_opts.fold_branches = false;
        brass::run_smt_guard_elimination(fn, *fn.parent(), ge_opts);
        brass::run_mir_superopt(fn, *fn.parent());
    }
    brass::scalar_cleanup(fn);
}

brass::target::SpirvKernel compile(Function& fn, std::uint32_t block, bool optimize) {
    if (optimize) optimize_function(fn);
    brass::target::SpirvOptions opts;
    opts.local_size_x = block;
    return brass::target::SpirvTarget::compile(fn, opts);
}

brass::target::SpirvKernel compile(const Function& fn, std::uint32_t block) {
    brass::target::SpirvOptions opts;
    opts.local_size_x = block;
    return brass::target::SpirvTarget::compile(fn, opts);
}

}  // namespace brotensor::jit::spirv::detail
