#include "inst_opt_dataset.hpp"

#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/egraph.hpp>
#include <brass/mir/inst_opt_model.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/verify/smt_equiv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(__x86_64__))
#include <intrin.h>
#define BROTENSOR_HAVE_RDTSC 1
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#define BROTENSOR_HAVE_RDTSC 1
#else
#define BROTENSOR_HAVE_RDTSC 0
#endif

namespace brotensor::inst_opt_train {

using namespace brass;

namespace {

uint64_t splitmix64(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint64_t read_cycles_now() noexcept {
#if BROTENSOR_HAVE_RDTSC
    return __rdtsc();
#else
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

struct CandSpec {
    ExtractedExpr expr;
    uint32_t width = 64;
    uint32_t action_label = 0;
    bool has_guards = false;
    uint32_t group_id = 0;
};

// Materializes `expr` inside MIR builder using `vars[0..3]` mapped by `var_id % 4`.
Value* emit_extracted_mir(Builder& b,
                          const ExtractedExpr& expr,
                          uint32_t width,
                          const std::vector<Value*>& vars) {
    const Type t = (width == 32) ? Type::i32() : Type::i64();
    if (expr.node.op == EOp::Var) {
        const size_t idx = static_cast<size_t>(expr.node.imm) % vars.size();
        return vars[idx];
    }
    if (expr.node.op == EOp::Const) {
        return (width == 32) ? b.build_iconst_i32(static_cast<int32_t>(expr.node.imm))
                             : b.build_iconst_i64(static_cast<int64_t>(expr.node.imm));
    }
    const uint32_t ar = eop_arity(expr.node.op);
    if (expr.children.size() < ar) {
        throw std::runtime_error("emit_extracted_mir: missing child operands for EOp " +
                                 std::to_string(static_cast<uint32_t>(expr.node.op)) +
                                 " (arity=" + std::to_string(ar) +
                                 ", children=" + std::to_string(expr.children.size()) + ")");
    }
    Value* v0 = (ar >= 1) ? emit_extracted_mir(b, expr.children[0], width, vars) : nullptr;
    Value* v1 = (ar >= 2) ? emit_extracted_mir(b, expr.children[1], width, vars) : nullptr;
    Value* v2 = (ar >= 3) ? emit_extracted_mir(b, expr.children[2], width, vars) : nullptr;
    if ((ar >= 1 && !v0) || (ar >= 2 && !v1) || (ar >= 3 && !v2)) {
        throw std::runtime_error("emit_extracted_mir: null operand for EOp " +
                                 std::to_string(static_cast<uint32_t>(expr.node.op)));
    }
    auto coerce = [&](Value* v, Type dst) -> Value* {
        if (!v) return nullptr;
        if (v->type() == dst) return v;
        if (dst == Type::i64() && v->type() == Type::i32()) return b.build_zext_i64(v);
        if (dst == Type::i32() && v->type() == Type::i64()) return b.build_trunc_i32(v);
        return v;
    };
    const EOp op = expr.node.op;
    switch (op) {
        case EOp::Add:     return b.build_add(v0, v1);
        case EOp::Sub:     return b.build_sub(v0, v1);
        case EOp::Mul:     return b.build_mul(v0, v1);
        case EOp::SDiv:    return b.build_sdiv(v0, v1);
        case EOp::UDiv:    return b.build_udiv(v0, v1);
        case EOp::SMod:    return b.build_smod(v0, v1);
        case EOp::UMod:    return b.build_umod(v0, v1);
        case EOp::And:     return b.build_and(v0, v1);
        case EOp::Or:      return b.build_or(v0, v1);
        case EOp::Xor:     return b.build_xor(v0, v1);
        case EOp::Shl:     return b.build_shl(v0, v1);
        case EOp::LShr:    return b.build_lshr(v0, v1);
        case EOp::AShr:    return b.build_ashr(v0, v1);
        case EOp::Neg:     return b.build_neg(v0);
        case EOp::Not:     return b.build_not(v0);
        case EOp::SExt64:  return b.build_sext_i64(coerce(v0, Type::i32()));
        case EOp::ZExt64:  return b.build_zext_i64(coerce(v0, Type::i32()));
        case EOp::Trunc32: return b.build_trunc_i32(coerce(v0, Type::i64()));
        case EOp::Eq:      return coerce(b.build_eq(v0, v1), t);
        case EOp::Ne:      return coerce(b.build_ne(v0, v1), t);
        case EOp::Slt:     return coerce(b.build_slt(v0, v1), t);
        case EOp::Ult:     return coerce(b.build_ult(v0, v1), t);
        case EOp::Sle:     return coerce(b.build_sle(v0, v1), t);
        case EOp::Ule:     return coerce(b.build_ule(v0, v1), t);
        case EOp::Select:  return b.build_select(coerce(v0, Type::i32()), v1, v2);
        default:           return v0 ? v0 : vars[0];
    }
}

ExtractedExpr make_var(uint32_t id, uint32_t w) {
    return ExtractedExpr{kInvalidEClass, 0, ENode{EOp::Var, w, kInvalidEClass, kInvalidEClass, kInvalidEClass, id}};
}

ExtractedExpr make_const(uint64_t c, uint32_t w) {
    return ExtractedExpr{kInvalidEClass, 0, ENode{EOp::Const, w, kInvalidEClass, kInvalidEClass, kInvalidEClass, c}};
}

ExtractedExpr make_unary(EOp op, uint32_t w, ExtractedExpr a) {
    ExtractedExpr e{kInvalidEClass, eop_cost(op) + a.cost, ENode{op, w}};
    e.children.push_back(std::move(a));
    return e;
}

ExtractedExpr make_bin(EOp op, uint32_t w, ExtractedExpr a, ExtractedExpr b) {
    ExtractedExpr e{kInvalidEClass, eop_cost(op) + a.cost + b.cost, ENode{op, w}};
    e.children.push_back(std::move(a));
    e.children.push_back(std::move(b));
    return e;
}

ExtractedExpr make_sel(uint32_t w, ExtractedExpr c, ExtractedExpr t, ExtractedExpr f) {
    ExtractedExpr e{kInvalidEClass, eop_cost(EOp::Select) + c.cost + t.cost + f.cost, ENode{EOp::Select, w}};
    e.children.push_back(std::move(c));
    e.children.push_back(std::move(t));
    e.children.push_back(std::move(f));
    return e;
}

std::string expr_key(const ExtractedExpr& e, uint32_t d = 0) {
    if (d > 20) return "?";
    if (e.node.op == EOp::Var) return "V" + std::to_string(e.node.imm);
    if (e.node.op == EOp::Const) return "C" + std::to_string(e.node.imm);
    std::string s = "(" + std::to_string(static_cast<uint32_t>(e.node.op)) + ":" + std::to_string(e.node.width);
    for (const auto& ch : e.children) s += " " + expr_key(ch, d + 1);
    return s + ")";
}

bool contains_div_or_shift_var0(const ExtractedExpr& e) noexcept {
    if (e.node.op == EOp::SDiv || e.node.op == EOp::UDiv || e.node.op == EOp::SMod || e.node.op == EOp::UMod) {
        if (e.children.size() >= 2 && e.children[1].node.op != EOp::Const) return true;
    }
    for (const auto& ch : e.children) {
        if (contains_div_or_shift_var0(ch)) return true;
    }
    return false;
}

bool expr_all_same_width(const ExtractedExpr& e, uint32_t w) noexcept {
    if (e.node.width != w) return false;
    if (e.node.op == EOp::SExt64 || e.node.op == EOp::ZExt64 || e.node.op == EOp::Trunc32) return false;
    for (const auto& ch : e.children) {
        if (!expr_all_same_width(ch, w)) return false;
    }
    return true;
}

EClassId insert_into_egraph(EGraph& eg, const ExtractedExpr& e) {
    if (e.node.op == EOp::Var) return eg.add_var(static_cast<uint32_t>(e.node.imm), e.node.width);
    if (e.node.op == EOp::Const) return eg.add_const(e.node.imm, e.node.width);
    const uint32_t ar = eop_arity(e.node.op);
    EClassId ch[3] = {kInvalidEClass, kInvalidEClass, kInvalidEClass};
    for (uint32_t i = 0; i < ar && i < e.children.size(); ++i) {
        ch[i] = insert_into_egraph(eg, e.children[i]);
    }
    return eg.add_node(ENode{e.node.op, e.node.width, ch[0], ch[1], ch[2], 0});
}

// Verifies that `a` and `b` are bit-exact equivalent using Brass MIR verifier.
bool prove_expr_pair_equiv(const ExtractedExpr& a, const ExtractedExpr& b, uint32_t w, bool nonneg_v0) {
    if (!expr_all_same_width(a, w) || !expr_all_same_width(b, w)) return false;
    Module m1("eq1"), m2("eq2");
    const Type t = (w == 32) ? Type::i32() : Type::i64();
    auto build_fn = [&](Module& mod, const ExtractedExpr& ex) -> Function* {
        Function* fn = mod.create_function("f", t, {t, t, t, t});
        Builder mb(mod);
        mb.set_function(fn);
        BasicBlock* entry = mb.append_block("entry");
        std::vector<Value*> vs{
            mb.add_block_param(entry, t),
            mb.add_block_param(entry, t),
            mb.add_block_param(entry, t),
            mb.add_block_param(entry, t)
        };
        mb.position_at_end(entry);
        if (nonneg_v0) {
            Value* z = (w == 32) ? mb.build_iconst_i32(0) : mb.build_iconst_i64(0);
            mb.build_guard(mb.build_sge(vs[0], z), "deopt");
        }
        mb.build_ret(emit_extracted_mir(mb, ex, w, vs));
        fn->rebuild_cfg_predecessors();
        return fn;
    };
    Function* f1 = build_fn(m1, a);
    Function* f2 = build_fn(m2, b);
    return verify::verify_mir_equivalence(*f1, *f2).status == verify::EquivStatus::ProvenEquivalent;
}

uint32_t infer_action_for_expr(const ExtractedExpr& e) noexcept {
    switch (e.node.op) {
        case EOp::SDiv: case EOp::UDiv: case EOp::SMod: case EOp::UMod:
            return static_cast<uint32_t>(InstOptAction::DivModReduce);
        case EOp::Mul:
            return static_cast<uint32_t>(InstOptAction::MulStrengthReduce);
        case EOp::Neg:
            return static_cast<uint32_t>(InstOptAction::TwosComplementFold);
        case EOp::Not:
            return static_cast<uint32_t>(InstOptAction::BitwiseDeMorganDist);
        case EOp::Eq: case EOp::Ne: case EOp::Slt: case EOp::Ult: case EOp::Sle: case EOp::Ule: case EOp::Select:
            return static_cast<uint32_t>(InstOptAction::CmpSelectCanon);
        case EOp::SExt64: case EOp::ZExt64: case EOp::Trunc32:
            return static_cast<uint32_t>(InstOptAction::GuardExtTruncNarrow);
        case EOp::Shl: case EOp::LShr: case EOp::AShr:
            return static_cast<uint32_t>(InstOptAction::ShiftMaskFuse);
        default:
            return static_cast<uint32_t>(InstOptAction::NoOp);
    }
}

// Adds a group of expressions with optional EGraph saturation & BVSolver-proven extras.
void emit_group(std::vector<CandSpec>& out,
                uint32_t& next_gid,
                const ExtractedExpr& root,
                uint32_t action_on_root,
                bool has_guards,
                const std::vector<ExtractedExpr>& extra_equivs = {},
                bool prove_extras = true) {
    const uint32_t gid = next_gid++;
    const uint32_t w = root.node.width;
    std::set<std::string> seen;

    auto push_cand = [&](const ExtractedExpr& ex, uint32_t act) {
        std::string k = expr_key(ex);
        if (seen.insert(std::move(k)).second) {
            out.push_back(CandSpec{ex, w, act, has_guards, gid});
        }
    };

    push_cand(root, action_on_root);

    if (expr_all_same_width(root, w)) {
        EGraph eg;
        const EClassId rid = insert_into_egraph(eg, root);
        EGraphLimits lim;
        lim.max_enodes = 384;
        lim.max_eclasses = 192;
        lim.max_iters = 4;
        eg.saturate(lim);
        std::vector<ExtractedExpr> cands;
        if (extract_top_k_candidates(eg, rid, 8, cands)) {
            for (const auto& c : cands) {
                const uint32_t act = (c.cost + 0 < root.cost) ? 0u : infer_action_for_expr(c);
                push_cand(c, act);
            }
        }
    }

    for (const auto& ex : extra_equivs) {
        if (!prove_extras || prove_expr_pair_equiv(root, ex, w, has_guards)) {
            push_cand(ex, 0u);
        }
    }
}

// Builds a JIT-executable loop function evaluating `spec.expr` 8x unrolled * 256 = 2048 dependent ops.
void add_bench_function(Module& mod, const std::string& name, const CandSpec& spec) {
    const uint32_t w = spec.width;
    const Type t = (w == 32) ? Type::i32() : Type::i64();
    Function* fn = mod.create_function(name, Type::i64(), {Type::i64(), Type::i64(), Type::i64(), Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* hdr = b.append_block("loop_hdr");
    BasicBlock* body = b.append_block("loop_body");
    BasicBlock* exit = b.append_block("exit");

    Value* n_iters = b.add_block_param(entry, Type::i64());
    Value* p0 = b.add_block_param(entry, Type::i64());
    Value* p1 = b.add_block_param(entry, Type::i64());
    Value* p2 = b.add_block_param(entry, Type::i64());
    Value* p3 = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);
    b.build_br(hdr, {b.build_iconst_i64(0), p0});

    Value* idx = b.add_block_param(hdr, Type::i64());
    Value* acc = b.add_block_param(hdr, Type::i64());
    b.position_at_end(hdr);
    Value* cond = b.build_slt(idx, n_iters);
    b.build_br_if(cond, body, {}, exit, {});

    b.position_at_end(body);
    Value* a = (w == 32) ? b.build_trunc_i32(acc) : acc;
    Value* v1 = (w == 32) ? b.build_trunc_i32(p1) : p1;
    Value* v2 = (w == 32) ? b.build_trunc_i32(p2) : p2;
    Value* v3 = (w == 32) ? b.build_trunc_i32(p3) : p3;

    // Force positive domain when evaluating guarded expressions (prevents accidental neg shifts or signed div trap)
    Value* mask_pos = (w == 32) ? b.build_iconst_i32(0x7FFFFFFF) : b.build_iconst_i64(0x7FFFFFFFFFFFFFFFLL);
    for (int u = 0; u < 8; ++u) {
        Value* cur_in = spec.has_guards ? b.build_and(a, mask_pos) : a;
        std::vector<Value*> vs{cur_in, v1, v2, v3};
        a = emit_extracted_mir(b, spec.expr, w, vs);
        if (contains_div_or_shift_var0(spec.expr)) {
            a = b.build_xor(a, v1);
        }
    }
    Value* next_acc = (w == 32) ? b.build_sext_i64(a) : a;
    Value* next_idx = b.build_add(idx, b.build_iconst_i64(1));
    b.build_br(hdr, {next_idx, next_acc});

    b.position_at_end(exit);
    b.build_ret(acc);
    fn->rebuild_cfg_predecessors();
}

// Estimates GPU PTX cycle cost by lowering `spec.expr` with PtxTarget and scanning PTX ops.
float estimate_ptx_cost(const CandSpec& spec) {
    if (!expr_all_same_width(spec.expr, spec.width)) {
        return (std::max)(0.5f, static_cast<float>(spec.expr.cost));
    }
    Module mod("ptx_cost");
    const Type t = (spec.width == 32) ? Type::i32() : Type::i64();
    Function* fn = mod.create_function("k", Type::void_type(), {Type::ptr(), t, t, t, t});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    std::vector<Value*> vs{
        b.add_block_param(entry, t),
        b.add_block_param(entry, t),
        b.add_block_param(entry, t),
        b.add_block_param(entry, t)
    };
    b.position_at_end(entry);
    b.build_store(t, out_ptr, emit_extracted_mir(b, spec.expr, spec.width, vs));
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    target::PtxOptions popts;
    popts.cleanup = true;
    const std::string ptx = target::PtxTarget::emit_function(*fn, popts);

    float cost = 0.0f;
    size_t pos = 0;
    while (pos < ptx.size()) {
        size_t eol = ptx.find('\n', pos);
        if (eol == std::string::npos) eol = ptx.size();
        const std::string_view line(ptx.data() + pos, eol - pos);
        pos = eol + 1;
        if (line.find('\t') == std::string_view::npos) continue;
        if (line.find("ret;") != std::string_view::npos ||
            line.find("ld.param") != std::string_view::npos ||
            line.find("st.") != std::string_view::npos) {
            continue;
        }
        if (line.find("div.") != std::string_view::npos || line.find("rem.") != std::string_view::npos) {
            cost += (spec.width == 64) ? 26.0f : 18.0f;
        } else if (line.find("mul.") != std::string_view::npos || line.find("mad.") != std::string_view::npos) {
            cost += (spec.width == 64) ? 6.0f : 3.5f;
        } else if (line.find("selp.") != std::string_view::npos || line.find("setp.") != std::string_view::npos) {
            cost += 1.5f;
        } else {
            cost += (spec.width == 64) ? 1.25f : 1.0f;
        }
    }
    return (std::max)(0.4f, cost);
}

} // namespace

InstOptDataset generate_inst_opt_dataset(uint64_t seed, bool verbose) {
    uint64_t rng = seed;
    std::vector<CandSpec> specs;
    uint32_t next_gid = 0;

    for (uint32_t w : {32u, 64u}) {
        const ExtractedExpr v0 = make_var(0, w);
        const ExtractedExpr v1 = make_var(1, w);
        const ExtractedExpr v2 = make_var(2, w);
        const ExtractedExpr v3 = make_var(3, w);

        // 1. Associative serial chains vs height-balanced trees (TreeHeightBalance = 12)
        for (EOp op : {EOp::Add, EOp::And, EOp::Or, EOp::Xor, EOp::Mul}) {
            // Length 4 serial vs balanced
            ExtractedExpr s4 = make_bin(op, w, make_bin(op, w, make_bin(op, w, v0, v1), v2), v3);
            ExtractedExpr b4 = make_bin(op, w, make_bin(op, w, v0, v1), make_bin(op, w, v2, v3));
            emit_group(specs, next_gid, s4, static_cast<uint32_t>(InstOptAction::TreeHeightBalance), false, {b4});

            // Length 6 serial vs balanced
            ExtractedExpr s6 = make_bin(op, w, make_bin(op, w, s4, v0), v1);
            ExtractedExpr b6 = make_bin(op, w, b4, make_bin(op, w, v0, v1));
            emit_group(specs, next_gid, s6, static_cast<uint32_t>(InstOptAction::TreeHeightBalance), false, {b6});

            // Length 8 serial vs depth-3 balanced
            ExtractedExpr s8 = make_bin(op, w, make_bin(op, w, s6, v2), v3);
            ExtractedExpr b8 = make_bin(op, w, b4, make_bin(op, w, make_bin(op, w, v0, v1), make_bin(op, w, v2, v3)));
            emit_group(specs, next_gid, s8, static_cast<uint32_t>(InstOptAction::TreeHeightBalance), false, {b8});
        }

        // 2. Bitwise carry-add identities (BitwiseCarryAdd = 3)
        emit_group(specs, next_gid,
                   make_bin(EOp::Add, w, make_bin(EOp::Or, w, v0, v1), make_bin(EOp::And, w, v0, v1)),
                   static_cast<uint32_t>(InstOptAction::BitwiseCarryAdd), false,
                   {make_bin(EOp::Add, w, v0, v1)});
        emit_group(specs, next_gid,
                   make_bin(EOp::Add, w, make_bin(EOp::Xor, w, v0, v1),
                            make_bin(EOp::Shl, w, make_bin(EOp::And, w, v0, v1), make_const(1, w))),
                   static_cast<uint32_t>(InstOptAction::BitwiseCarryAdd), false,
                   {make_bin(EOp::Add, w, v0, v1)});
        emit_group(specs, next_gid,
                   make_bin(EOp::Sub, w, make_bin(EOp::Or, w, v0, v1), make_bin(EOp::And, w, v0, v1)),
                   static_cast<uint32_t>(InstOptAction::BitwiseCarryAdd), false,
                   {make_bin(EOp::Xor, w, v0, v1)});

        // 3. De Morgan & distributive factoring (BitwiseDeMorganDist = 2)
        emit_group(specs, next_gid,
                   make_bin(EOp::And, w, make_unary(EOp::Not, w, v0), make_unary(EOp::Not, w, v1)),
                   static_cast<uint32_t>(InstOptAction::BitwiseDeMorganDist), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Or, w, make_unary(EOp::Not, w, v0), make_unary(EOp::Not, w, v1)),
                   static_cast<uint32_t>(InstOptAction::BitwiseDeMorganDist), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Or, w, make_bin(EOp::And, w, v0, v1), make_bin(EOp::And, w, v0, v2)),
                   static_cast<uint32_t>(InstOptAction::BitwiseDeMorganDist), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::And, w, make_bin(EOp::Or, w, v0, v1), make_bin(EOp::Or, w, v0, v2)),
                   static_cast<uint32_t>(InstOptAction::BitwiseDeMorganDist), false);

        // 4. Mul strength reduction & shift-add decompositions (MulStrengthReduce = 4)
        for (uint64_t k : {2ULL, 4ULL, 8ULL, 16ULL, 32ULL, 64ULL}) {
            const uint32_t sh = static_cast<uint32_t>(std::countr_zero(k));
            emit_group(specs, next_gid,
                       make_bin(EOp::Mul, w, v0, make_const(k, w)),
                       static_cast<uint32_t>(InstOptAction::MulStrengthReduce), false,
                       {make_bin(EOp::Shl, w, v0, make_const(sh, w))});
        }
        struct ShiftAddMul { uint64_t c; uint32_t s1; uint32_t s2; bool is_sub; };
        for (const auto& sm : std::vector<ShiftAddMul>{
                {3, 1, 0, false}, {5, 2, 0, false}, {6, 2, 1, false}, {7, 3, 0, true},
                {9, 3, 0, false}, {10, 3, 1, false}, {12, 3, 2, false}, {15, 4, 0, true},
                {17, 4, 0, false}, {24, 4, 3, false}, {31, 5, 0, true}, {33, 5, 0, false}}) {
            ExtractedExpr lhs = make_bin(EOp::Shl, w, v0, make_const(sm.s1, w));
            ExtractedExpr rhs = (sm.s2 == 0) ? v0 : make_bin(EOp::Shl, w, v0, make_const(sm.s2, w));
            ExtractedExpr sa = make_bin(sm.is_sub ? EOp::Sub : EOp::Add, w, lhs, rhs);
            emit_group(specs, next_gid,
                       make_bin(EOp::Mul, w, v0, make_const(sm.c, w)),
                       static_cast<uint32_t>(InstOptAction::MulStrengthReduce), false, {sa});
        }

        // 5. Mul-Add factoring (MulAddFactor = 11)
        emit_group(specs, next_gid,
                   make_bin(EOp::Add, w, make_bin(EOp::Mul, w, v0, v1), make_bin(EOp::Mul, w, v0, v2)),
                   static_cast<uint32_t>(InstOptAction::MulAddFactor), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Sub, w, make_bin(EOp::Mul, w, v0, v1), make_bin(EOp::Mul, w, v0, v2)),
                   static_cast<uint32_t>(InstOptAction::MulAddFactor), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Add, w, make_bin(EOp::Mul, w, v0, make_const(3, w)),
                            make_bin(EOp::Mul, w, v0, make_const(5, w))),
                   static_cast<uint32_t>(InstOptAction::MulAddFactor), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Add, w, make_bin(EOp::Mul, w, v0, make_const(7, w)),
                            make_bin(EOp::Mul, w, v0, make_const(9, w))),
                   static_cast<uint32_t>(InstOptAction::MulAddFactor), false);

        // 6. Div & Mod reduction (DivModReduce = 5) — guarded non-neg vs unguarded
        for (uint32_t sh : {1u, 2u, 3u, 4u, 5u, 8u}) {
            const uint64_t p2 = 1ULL << sh;
            emit_group(specs, next_gid,
                       make_bin(EOp::UDiv, w, v0, make_const(p2, w)),
                       static_cast<uint32_t>(InstOptAction::DivModReduce), false,
                       {make_bin(EOp::LShr, w, v0, make_const(sh, w))});
            emit_group(specs, next_gid,
                       make_bin(EOp::UMod, w, v0, make_const(p2, w)),
                       static_cast<uint32_t>(InstOptAction::DivModReduce), false,
                       {make_bin(EOp::And, w, v0, make_const(p2 - 1ULL, w))});
            emit_group(specs, next_gid,
                       make_bin(EOp::SDiv, w, v0, make_const(p2, w)),
                       static_cast<uint32_t>(InstOptAction::DivModReduce), true,
                       {make_bin(EOp::LShr, w, v0, make_const(sh, w))});
            emit_group(specs, next_gid,
                       make_bin(EOp::SMod, w, v0, make_const(p2, w)),
                       static_cast<uint32_t>(InstOptAction::DivModReduce), true,
                       {make_bin(EOp::And, w, v0, make_const(p2 - 1ULL, w))});
        }
        // Non-power-of-2 constant divisions -> NoOp (stays DivMod or magic div baseline)
        for (uint64_t d : {3ULL, 5ULL, 7ULL, 10ULL, 13ULL}) {
            emit_group(specs, next_gid,
                       make_bin(EOp::UDiv, w, v0, make_const(d, w)),
                       static_cast<uint32_t>(InstOptAction::DivModReduce), false);
            emit_group(specs, next_gid,
                       make_bin(EOp::SDiv, w, v0, make_const(d, w)),
                       static_cast<uint32_t>(InstOptAction::DivModReduce), true);
        }

        // 7. Two's complement & Algebraic identities (6 and 1)
        emit_group(specs, next_gid,
                   make_bin(EOp::Add, w, make_unary(EOp::Not, w, v0), make_const(1, w)),
                   static_cast<uint32_t>(InstOptAction::TwosComplementFold), false);
        emit_group(specs, next_gid,
                   make_unary(EOp::Neg, w, make_bin(EOp::Add, w, v0, make_const(1, w))),
                   static_cast<uint32_t>(InstOptAction::TwosComplementFold), false);
        emit_group(specs, next_gid,
                   make_unary(EOp::Neg, w, make_unary(EOp::Neg, w, v0)),
                   static_cast<uint32_t>(InstOptAction::TwosComplementFold), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Sub, w, make_bin(EOp::Add, w, v0, v1), v1),
                   static_cast<uint32_t>(InstOptAction::AlgebraicIdentity), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Xor, w, make_bin(EOp::Xor, w, v0, v1), v1),
                   static_cast<uint32_t>(InstOptAction::AlgebraicIdentity), false);
        emit_group(specs, next_gid,
                   make_bin(EOp::Or, w, make_bin(EOp::And, w, v0, v1), make_bin(EOp::And, w, v0, make_unary(EOp::Not, w, v1))),
                   static_cast<uint32_t>(InstOptAction::AlgebraicIdentity), false);

        // 8. Assoc constant combine (10), Guard mask elide (8), Shift-mask fuse (13), Cmp-select (7)
        for (EOp op : {EOp::Add, EOp::And, EOp::Or, EOp::Xor}) {
            emit_group(specs, next_gid,
                       make_bin(op, w, make_bin(op, w, v0, make_const(13, w)), make_const(29, w)),
                       static_cast<uint32_t>(InstOptAction::AssocConstCombine), false);
        }
        emit_group(specs, next_gid,
                   make_bin(EOp::And, w, make_bin(EOp::LShr, w, v0, make_const(w - 8u, w)), make_const(0xFFULL, w)),
                   static_cast<uint32_t>(InstOptAction::GuardMaskElide), true);
        emit_group(specs, next_gid,
                   make_bin(EOp::And, w, make_bin(EOp::And, w, v0, make_const(0xFFULL, w)), make_const(0x0FULL, w)),
                   static_cast<uint32_t>(InstOptAction::GuardMaskElide), true);
        emit_group(specs, next_gid,
                   make_bin(EOp::Shl, w, make_bin(EOp::And, w, make_bin(EOp::LShr, w, v0, make_const(3, w)), make_const(0xFFULL, w)), make_const(3, w)),
                   static_cast<uint32_t>(InstOptAction::ShiftMaskFuse), false);
        emit_group(specs, next_gid,
                   make_sel(w, make_bin(EOp::Eq, w, v0, v1), v2, v2),
                   static_cast<uint32_t>(InstOptAction::CmpSelectCanon), false);
        emit_group(specs, next_gid,
                   make_sel(w, make_bin(EOp::Slt, w, v0, v0), v1, v2),
                   static_cast<uint32_t>(InstOptAction::CmpSelectCanon), false);

        // 9. Clean unsimplifiable random arithmetic/bitwise expressions (NoOp = 0 ground truth + random DAG variety)
        const std::vector<EOp> rand_ops = {EOp::Add, EOp::Sub, EOp::Mul, EOp::And, EOp::Or, EOp::Xor, EOp::Shl, EOp::LShr};
        for (int n = 0; n < 180; ++n) {
            const uint32_t depth = 1 + static_cast<uint32_t>(splitmix64(rng) % 5);
            auto gen = [&](auto& self, uint32_t d) -> ExtractedExpr {
                if (d == 0 || (splitmix64(rng) % 4 == 0 && d < depth)) {
                    if (splitmix64(rng) % 5 == 0) {
                        return make_const((splitmix64(rng) % 15) + 1, w);
                    }
                    return make_var(static_cast<uint32_t>(splitmix64(rng) % 4), w);
                }
                const EOp rop = rand_ops[splitmix64(rng) % rand_ops.size()];
                ExtractedExpr l = self(self, d - 1);
                if (rop == EOp::Shl || rop == EOp::LShr) {
                    return make_bin(rop, w, std::move(l), make_const((splitmix64(rng) % 7) + 1, w));
                }
                ExtractedExpr r = self(self, d - 1);
                return make_bin(rop, w, std::move(l), std::move(r));
            };
            ExtractedExpr rx = gen(gen, depth);
            if (rx.cost == 0) rx = make_bin(EOp::Add, w, v0, v1);
            emit_group(specs, next_gid, rx, static_cast<uint32_t>(InstOptAction::NoOp), false);
        }
    }

    if (verbose) {
        std::printf("[dataset] Generated %zu expression specs across %u groups\n",
                    specs.size(), next_gid);
        std::fflush(stdout);
    }

    // Hardware x64 JIT benchmarking in batches of 48 functions per Brass Module
    InstOptDataset ds;
    ds.num_groups = next_gid;
    ds.samples.reserve(specs.size() * 2);

    std::vector<int> x64_idx_for_spec(specs.size(), -1);
    std::vector<int> ptx_idx_for_spec(specs.size(), -1);

    constexpr size_t kBatchSize = 48;
    constexpr int64_t kIters = 256; // * 8 unrolled = 2048 evaluations per call
    using JitFn = int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t);

    volatile int64_t sink_acc = 0;
    for (size_t base = 0; base < specs.size(); base += kBatchSize) {
        const size_t end = (std::min)(specs.size(), base + kBatchSize);
        Module mod("bench_batch");
        std::vector<std::string> names;
        names.reserve(end - base);
        for (size_t i = base; i < end; ++i) {
            std::string nm = "fn_" + std::to_string(i);
            add_bench_function(mod, nm, specs[i]);
            names.push_back(std::move(nm));
        }
        DiagnosticReporter diag;
        if (!verify_module(mod, &diag)) {
            throw std::runtime_error("verify_module failed on bench_batch: " + diag.format_all());
        }

        codegen::JitExecutionEngine jit;
        const bool compiled = jit.compile_and_load(mod);
        if (!compiled) {
            throw std::runtime_error("JitExecutionEngine::compile_and_load failed on benchmark batch");
        }

        for (size_t k = 0; k < names.size(); ++k) {
            const size_t idx = base + k;
            JitFn fn_ptr = jit.get_function_ptr<JitFn>(names[k]);
            if (!fn_ptr) {
                throw std::runtime_error("Missing JIT function pointer: " + names[k]);
            }

            // Warmup call
            int64_t res = fn_ptr(kIters, 0x13579BDF2468ACE1LL, 0x01020305070B0D11LL,
                                 0x1F2E3D4C5B6A7981LL, 0x0000123400005679LL);
            sink_acc ^= res;

            uint64_t timings[5] = {};
            for (int rep = 0; rep < 5; ++rep) {
                const uint64_t t0 = read_cycles_now();
                const int64_t r = fn_ptr(kIters, 0x13579BDF2468ACE1LL + rep, 0x01020305070B0D11LL,
                                         0x1F2E3D4C5B6A7981LL, 0x0000123400005679LL);
                const uint64_t t1 = read_cycles_now();
                sink_acc ^= r;
                timings[rep] = (t1 >= t0) ? (t1 - t0) : 1ULL;
            }
            std::sort(std::begin(timings), std::end(timings));
            const float cycles_per_eval = std::clamp(
                static_cast<float>(timings[2]) / static_cast<float>(kIters * 8),
                0.2f, 40.0f);

            // Emit x64 sample
            InstOptSample sx;
            sx.features = extract_inst_opt_features(specs[idx].expr, InstOptTarget::X64, specs[idx].has_guards);
            sx.measured_cost = cycles_per_eval;
            sx.static_cost = specs[idx].expr.cost;
            sx.action_label = specs[idx].action_label;
            sx.group_id = specs[idx].group_id;
            sx.is_gpu = false;
            x64_idx_for_spec[idx] = static_cast<int>(ds.samples.size());
            ds.samples.push_back(sx);
            ds.x64_benchmarked++;

            // Emit corresponding GPU PTX sample
            InstOptSample sg;
            sg.features = extract_inst_opt_features(specs[idx].expr, InstOptTarget::GPU_PTX, specs[idx].has_guards);
            sg.measured_cost = estimate_ptx_cost(specs[idx]);
            sg.static_cost = specs[idx].expr.cost;
            sg.action_label = specs[idx].action_label;
            sg.group_id = specs[idx].group_id;
            sg.is_gpu = true;
            ptx_idx_for_spec[idx] = static_cast<int>(ds.samples.size());
            ds.samples.push_back(sg);
            ds.ptx_estimated++;
        }
    }
    (void)sink_acc;

    // Build equivalent-group `(fast, slow)` pairs
    std::map<uint32_t, std::vector<size_t>> x64_by_group;
    std::map<uint32_t, std::vector<size_t>> ptx_by_group;
    for (size_t i = 0; i < specs.size(); ++i) {
        if (x64_idx_for_spec[i] >= 0) x64_by_group[specs[i].group_id].push_back(static_cast<size_t>(x64_idx_for_spec[i]));
        if (ptx_idx_for_spec[i] >= 0) ptx_by_group[specs[i].group_id].push_back(static_cast<size_t>(ptx_idx_for_spec[i]));
    }
    auto add_group_pairs = [&](const std::map<uint32_t, std::vector<size_t>>& by_grp) {
        for (const auto& [gid, idxs] : by_grp) {
            for (size_t a = 0; a < idxs.size(); ++a) {
                for (size_t b = a + 1; b < idxs.size(); ++b) {
                    const size_t ia = idxs[a];
                    const size_t ib = idxs[b];
                    const float ca = ds.samples[ia].measured_cost;
                    const float cb = ds.samples[ib].measured_cost;
                    if (ca < cb * 0.94f) {
                        ds.pairs.push_back(InstOptPair{ia, ib, gid});
                    } else if (cb < ca * 0.94f) {
                        ds.pairs.push_back(InstOptPair{ib, ia, gid});
                    }
                }
            }
        }
    };
    add_group_pairs(x64_by_group);
    add_group_pairs(ptx_by_group);

    // Deterministic 85% / 15% split by group_id (zero leakage between equivalent forms)
    std::vector<uint32_t> gids(next_gid);
    for (uint32_t g = 0; g < next_gid; ++g) gids[g] = g;
    for (uint32_t g = next_gid; g > 1; --g) {
        const uint32_t j = static_cast<uint32_t>(splitmix64(rng) % g);
        std::swap(gids[g - 1], gids[j]);
    }
    const uint32_t n_val_groups = (std::max)(1u, next_gid * 15u / 100u);
    std::vector<uint8_t> is_val_group(next_gid, 0);
    for (uint32_t i = 0; i < n_val_groups; ++i) is_val_group[gids[i]] = 1;

    for (size_t i = 0; i < ds.samples.size(); ++i) {
        if (is_val_group[ds.samples[i].group_id]) ds.val_indices.push_back(i);
        else ds.train_indices.push_back(i);
    }
    for (size_t i = 0; i < ds.pairs.size(); ++i) {
        if (is_val_group[ds.pairs[i].group_id]) ds.val_pairs.push_back(i);
        else ds.train_pairs.push_back(i);
    }

    if (verbose) {
        std::printf("[dataset] Benchmarked %u x64 + %u PTX = %zu total samples, %zu equiv pairs "
                    "(train=%zu/%zu pairs, val=%zu/%zu pairs)\n",
                    ds.x64_benchmarked, ds.ptx_estimated, ds.samples.size(), ds.pairs.size(),
                    ds.train_indices.size(), ds.train_pairs.size(),
                    ds.val_indices.size(), ds.val_pairs.size());
    }
    return ds;
}

} // namespace brotensor::inst_opt_train
