// ─── Trace JIT Optimizer, Broadcast Decomposition & SMT Equivalence Suite ────
//
// Verifies the issue #88 optimizations in brotensor's PTX and SPIR-V trace JIT:
//   1. PTX broadcast index decomposition (power-of-two shift/mask and non-power-
//      of-two 32-bit unsigned magic-multiplier mul.hi.u32 + multiply-subtract
//      remainder, sharing `row` when both row and col broadcasts are active) —
//      zero `div.u32` / `rem.u32` instructions in vector kernels.
//   2. PTX & SPIR-V chain optimizations: single-use Mul+Add -> FMA fusion,
//      DAG-level CSE, broadcast-lane CSE across Scalar/Col inputs, exact
//      power-of-two DivScalar -> Mul, and redundant Cast/Copy elision.
//   3. SPIR-V MIR optimization pipeline (`spirv::detail::optimize_function`)
//      verified for formal semantic equivalence via
//      `brass::verify::verify_mir_equivalence`, plus SPIR-V opcode inspection
//      (zero OpUDiv / OpUMod) and MIR / SPIR-V size reduction checks.
//   4. End-to-end numerical parity against CPU reference on non-power-of-two
//      and power-of-two broadcast shapes across available backends.

#include "../src/jit/ptx_emit.h"
#include "../src/jit/ptx_emit_detail.h"
#include "../src/jit/trace_dag.h"
#include "../src/jit/trace_plan.h"

#if BROTENSOR_HAS_BRASS_JIT
#include "../src/jit/spirv_emit.h"
#include "../src/jit/spirv_emit_detail.h"
#include <brass/verify/smt_equiv.hpp>
#endif

#include <brotensor/jit/trace.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace brotensor;
using namespace brotensor::jit;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    for (std::string l; std::getline(ss, l);) out.push_back(l);
    return out;
}

std::vector<std::vector<std::string>> entries_of(const std::string& ptx) {
    std::vector<std::vector<std::string>> out;
    for (const std::string& l : lines_of(ptx)) {
        if (l.rfind(".visible .entry", 0) == 0) out.emplace_back();
        if (!out.empty()) out.back().push_back(l);
    }
    return out;
}

std::size_t count_substr(const std::vector<std::string>& lines, const std::string& needle) {
    std::size_t n = 0;
    for (const std::string& l : lines) {
        if (l.find(needle) != std::string::npos) ++n;
    }
    return n;
}

// ── Test 1: MagicU32 exactness across boundary and non-power-of-two divisors ─

void test_magic_u32_exactness() {
    std::printf("\n--- Test 1: 32-bit Unsigned Magic-Multiplier Exactness ---\n");
    const std::uint32_t divisors[] = {3, 5, 7, 13, 64, 127, 128, 255, 256, 300, 384, 512, 768, 1000, 1536, 1920, 4095};
    bool all_ok = true;
    for (std::uint32_t d : divisors) {
        const auto m = ptx::detail::magic_u32(d);
        auto eval_div = [&](std::uint32_t v) -> std::uint32_t {
            if ((d & (d - 1u)) == 0u) {
                unsigned sh = 0;
                for (std::uint32_t t = d; t > 1u; t >>= 1u) ++sh;
                return v >> sh;
            }
            const std::uint32_t vp = (m.pre_shift > 0u) ? (v >> m.pre_shift) : v;
            const std::uint32_t q = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(vp) * m.mul) >> 32);
            if (!m.add) return (m.post_shift > 0u) ? (q >> m.post_shift) : q;
            const std::uint32_t t = ((vp - q) >> 1u) + q;
            return (m.post_shift > 1u) ? (t >> (m.post_shift - 1u)) : t;
        };

        for (std::uint32_t v = 0; v < 4096; ++v) {
            const std::uint32_t q = eval_div(v);
            const std::uint32_t r = v - q * d;
            if (q != v / d || r != v % d) {
                all_ok = false;
                break;
            }
        }
        for (std::uint64_t k = 1; k <= 100000; k = k * 3 + 1) {
            for (int delta = -2; delta <= 2; ++delta) {
                const std::int64_t cand = static_cast<std::int64_t>(k * d) + delta;
                if (cand < 0 || cand > 0x7FFFFFFFLL) continue;
                const std::uint32_t v = static_cast<std::uint32_t>(cand);
                const std::uint32_t q = eval_div(v);
                const std::uint32_t r = v - q * d;
                if (q != v / d || r != v % d) {
                    all_ok = false;
                    break;
                }
            }
        }
    }
    check(all_ok, "magic_u32 matches exact v / d and v % d across all test divisors and boundary inputs");
}

// ── Test 2: PTX Broadcast Index Decomposition & Chain Optimizations ─────────

void test_ptx_optimizations() {
    std::printf("\n--- Test 2: PTX Broadcast Decomposition, FMA, CSE & Broadcast-Lane Dedup ---\n");
    static float dummy[4];

    auto test_shape = [&](int rows, int cols, bool pow2) {
        TraceDAG dag;
        const int x = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, rows, cols);
        const int r = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 1, cols);
        const int c = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, rows, 1);
        const int s = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 1, 1);

        const int col_sc  = dag.add_node(TraceOpKind::Add, {c, s}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, 1);
        const int col_act = dag.add_node(TraceOpKind::SiLU, {col_sc}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, 1);
        const int mul_xr  = dag.add_node(TraceOpKind::Mul, {x, r}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int fma_val = dag.add_node(TraceOpKind::Add, {mul_xr, col_act}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int cse_a   = dag.add_node(TraceOpKind::Add, {x, r}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int cse_b   = dag.add_node(TraceOpKind::Add, {x, r}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int sum1    = dag.add_node(TraceOpKind::Add, {fma_val, cse_a}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int sum2    = dag.add_node(TraceOpKind::Add, {sum1, cse_b}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int div_p2  = dag.add_node(TraceOpKind::DivScalar, {sum2}, 8.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int cast_id = dag.add_node(TraceOpKind::Cast, {div_p2}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        const int out     = dag.add_node(TraceOpKind::Copy, {cast_id}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, rows, cols);
        dag.nodes()[static_cast<std::size_t>(out)].is_live_at_end = true;

        plan::ElementwisePlan ew;
        const bool planned = plan::build_plan(dag, ew);
        const std::string label = "PTX elementwise " + std::to_string(rows) + "x" + std::to_string(cols);
        check(planned && ew.vec == 4, label + ": planned with vec=4");
        if (!planned) return;

        const std::string ptx = ptx::emit_elementwise(dag, ew, "sm_80");
        const auto entries = entries_of(ptx);
        check(entries.size() == 2, label + ": has vector and scalar entries");
        if (entries.size() < 2) return;

        const auto& vec_e = entries[0];
        // 1. Zero hardware integer div/rem in vector entry
        check(count_substr(vec_e, "div.u32") == 0 && count_substr(vec_e, "rem.u32") == 0,
              label + ": zero div.u32 / rem.u32 in vector entry");
        if (pow2) {
            check(count_substr(vec_e, "and.b32") >= 1 && count_substr(vec_e, "shr.u32") >= 1,
                  label + ": power-of-two cols uses bitwise shift/mask");
        } else {
            check(count_substr(vec_e, "mul.hi.u32") == 1,
                  label + ": non-power-of-two cols shares single mul.hi.u32 for row & col broadcasts");
        }

        // 2. Broadcast-lane CSE: `silu(col + sc)` is broadcast-invariant across
        // the 4 vector lanes, so `ex2.approx.f32` is emitted ONCE in vec_e
        // instead of 4 times!
        check(count_substr(vec_e, "ex2.approx.f32") == 1,
              label + ": broadcast-lane CSE evaluates silu(col + sc) once across 4 lanes");

        // 3. Single-use Mul+Add -> fma.rn.f32 (4 lanes -> 4 FMAs)
        check(count_substr(vec_e, "fma.rn.f32") == 4,
              label + ": single-use Mul+Add fused into 4 fma.rn.f32 instructions");

        // 4. Exact power-of-two DivScalar(8.0f) -> mul.f32 by 0.125f
        check(count_substr(vec_e, "div.rn.f32") == 0 && count_substr(vec_e, "rcp.approx.f32") == 1,
              label + ": DivScalar(8.0) lowered to mul.f32 with zero div.rn.f32");

        // 5. Redundant Cast + Copy elided -> zero mov.f32 in vector entry
        check(count_substr(vec_e, "mov.f32") == 0,
              label + ": redundant Cast/Copy elided (zero mov.f32)");
    };

    test_shape(64, 256, true);
    test_shape(64, 300, false);
}

// ── Test 3: SPIR-V MIR Optimization Pipeline & SMT Equivalence Verification ─

#if BROTENSOR_HAS_BRASS_JIT
void test_spirv_and_smt_equivalence() {
    std::printf("\n--- Test 3: SPIR-V Optimizer & Brass Z3 SMT Equivalence Verification ---\n");

    // 3a. Verify udiv_urem_const power-of-two equivalence against hardware udiv/urem via Z3
    {
        brass::Module mod_hw("m_hw");
        const std::vector<brass::Type> params = {brass::Type::i32()};
        brass::Function* fn_hw = mod_hw.create_function("udiv_hw", brass::Type::i32(),
                                                        brass::Span<const brass::Type>(params));
        {
            brass::codegen::KernelBuilder kb(mod_hw, fn_hw);
            spirv::detail::Ctx c(kb);
            const auto p = spirv::detail::entry_params(c, fn_hw);
            brass::Value* v = p[0];
            brass::Value* d = c.i32(256);
            brass::Value* q = c.b.build_udiv(v, d);
            brass::Value* r = c.b.build_umod(v, d);
            c.b.build_ret(c.b.build_add(c.b.build_mul(q, c.i32(1000)), r));
        }

        brass::Module mod_fast("m_fast");
        brass::Function* fn_fast = mod_fast.create_function("udiv_fast", brass::Type::i32(),
                                                            brass::Span<const brass::Type>(params));
        {
            brass::codegen::KernelBuilder kb(mod_fast, fn_fast);
            spirv::detail::Ctx c(kb);
            const auto p = spirv::detail::entry_params(c, fn_fast);
            brass::Value* v = p[0];
            brass::Value* q = nullptr;
            brass::Value* r = nullptr;
            spirv::detail::udiv_urem_const(c, v, 256, q, r);
            c.b.build_ret(c.b.build_add(c.b.build_mul(q, c.i32(1000)), r));
        }

        const auto res = brass::verify::verify_mir_equivalence(*fn_hw, *fn_fast);
        check(res.is_equivalent(), "SMT proves udiv_urem_const(v, 256) === (v / 256, v % 256)");
    }

    // 3b. Build two identical scalar MIR functions representing a trace JIT
    // broadcast + chain evaluation, run `spirv::detail::optimize_function` on
    // the second, and prove semantic equivalence before vs after optimization
    // using `brass::verify::verify_mir_equivalence`.
    {
        static float dummy[4];
        TraceDAG dag;
        const int x = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 64, 300);
        const int r = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 1, 300);
        const int c = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 64, 1);

        const int mul_xr = dag.add_node(TraceOpKind::Mul, {x, r}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int fma_v  = dag.add_node(TraceOpKind::Add, {mul_xr, c}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int cse1   = dag.add_node(TraceOpKind::Sub, {mul_xr, r}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int cse2   = dag.add_node(TraceOpKind::Sub, {mul_xr, r}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int sum_c  = dag.add_node(TraceOpKind::Add, {cse1, cse2}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int comb   = dag.add_node(TraceOpKind::Add, {fma_v, sum_c}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int scaled = dag.add_node(TraceOpKind::DivScalar, {comb}, 4.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int out    = dag.add_node(TraceOpKind::Copy, {scaled}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        dag.nodes()[static_cast<std::size_t>(out)].is_live_at_end = true;

        plan::ElementwisePlan ew;
        const bool planned = plan::build_plan(dag, ew);
        check(planned, "SPIR-V elementwise 64x300 planned");

        auto build_trace_fn = [&](brass::Module& mod) -> brass::Function* {
            const std::vector<brass::Type> params = {
                brass::Type::i32(), brass::Type::f32(), brass::Type::f32(), brass::Type::f32()
            };
            brass::Function* fn = mod.create_function(
                "trace_scalar_chain", brass::Type::f32(), brass::Span<const brass::Type>(params));
            brass::codegen::KernelBuilder kb(mod, fn);
            spirv::detail::Ctx ctx(kb);
            const auto p = spirv::detail::entry_params(ctx, fn);
            brass::Value* idx = p[0];
            brass::Value* row = nullptr;
            brass::Value* col = nullptr;
            spirv::detail::udiv_urem_const(ctx, idx, 300, row, col);
            // Duplicate integer index computation + redundant integer algebraic identity
            // (`col + (idx_check - dup_idx)`) that `optimize_function` folds and CSEs.
            brass::Value* idx_check = ctx.b.build_sub(
                idx, ctx.b.build_add(ctx.b.build_mul(row, ctx.i32(300)), col));
            brass::Value* dup_idx = ctx.b.build_sub(
                idx, ctx.b.build_add(ctx.b.build_mul(row, ctx.i32(300)), col));
            brass::Value* adj_idx = ctx.b.build_add(col, ctx.b.build_sub(idx_check, dup_idx));
            brass::Value* z_f = spirv::detail::bits_to_f32(ctx, adj_idx);

            // Also include a duplicate FP subexpression that dominator_cse / GVN eliminates
            brass::Value* fp_dup1 = ctx.b.build_mul(p[1], z_f);
            brass::Value* fp_dup2 = ctx.b.build_mul(p[1], z_f);

            spirv::detail::ValueMap vals;
            vals[x] = {ctx.b.build_add(fp_dup1, fp_dup2)};
            vals[r] = {p[2]};
            vals[c] = {p[3]};
            vals = spirv::detail::eval_chain(ctx, dag, 1, std::move(vals));
            ctx.b.build_ret(vals.at(out)[0]);
            return fn;
        };

        brass::Module mod_unopt("m_unopt");
        brass::Function* fn_unopt = build_trace_fn(mod_unopt);

        brass::Module mod_opt("m_opt");
        brass::Function* fn_opt = build_trace_fn(mod_opt);

        const std::size_t insts_before = fn_unopt->entry_block()->instruction_count();
        spirv::detail::optimize_function(*fn_opt);
        const std::size_t insts_after = fn_opt->entry_block()->instruction_count();

        std::printf("    MIR instructions before optimize_function: %zu, after: %zu\n",
                    insts_before, insts_after);
        check(insts_after < insts_before,
              "optimize_function reduces MIR instruction count (" +
                  std::to_string(insts_before) + " -> " + std::to_string(insts_after) + ")");

        brass::verify::EquivOptions opts;
        opts.max_sat_conflicts = 50000;
        const auto eq = brass::verify::verify_mir_equivalence(*fn_unopt, *fn_opt, opts);
        check(eq.is_equivalent(),
              "SMT proves unoptimized vs optimized trace MIR equivalent (" + eq.diagnostic + ")");

        // Also build a DAG with single-use Mul+Add to verify GLSL.std.450 Fma & zero OpUDiv/OpUMod
        TraceDAG fma_dag;
        const int fx = fma_dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 64, 300);
        const int fr = fma_dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 1, 300);
        const int fc = fma_dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CUDA, Dtype::FP32, 64, 1);
        const int fmul = fma_dag.add_node(TraceOpKind::Mul, {fx, fr}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        const int fout = fma_dag.add_node(TraceOpKind::Add, {fmul, fc}, 0.0f, nullptr, Device::CUDA, Dtype::FP32, 64, 300);
        fma_dag.nodes()[static_cast<std::size_t>(fout)].is_live_at_end = true;
        plan::ElementwisePlan fma_ew;
        check(plan::build_plan(fma_dag, fma_ew), "SPIR-V FMA elementwise 64x300 planned");

        // Inspect emitted SPIR-V module for 64x300
        const auto spv = spirv::emit_elementwise(fma_dag, fma_ew);
        check(spv.vec.has_value() && !spv.vec->words.empty() && spv.vec->words[0] == 0x07230203u,
              "SPIR-V vec module has valid magic header");

        // Scan SPIR-V opcodes: verify zero OpUDiv (134) and zero OpUMod (138)
        std::size_t op_udiv = 0, op_umod = 0, op_ext_fma = 0;
        if (spv.vec.has_value()) {
            const auto& words = spv.vec->words;
            std::size_t pos = 5; // skip 5-word SPIR-V header
            while (pos < words.size()) {
                const std::uint32_t w0 = words[pos];
                const std::uint32_t wc = w0 >> 16;
                const std::uint32_t op = w0 & 0xFFFFu;
                if (wc == 0 || pos + wc > words.size()) break;
                if (op == 134u) ++op_udiv; // OpUDiv
                if (op == 138u) ++op_umod; // OpUMod
                if (op == 12u && wc >= 5 && words[pos + 4] == 50u) ++op_ext_fma; // GLSL.std.450 Fma
                pos += wc;
            }
        }
        check(op_udiv == 0 && op_umod == 0,
              "SPIR-V 64x300 broadcast module has zero OpUDiv and zero OpUMod");
        check(op_ext_fma >= 1,
              "SPIR-V 64x300 module emits GLSL.std.450 Fma for fused Mul+Add");
    }
}
#endif

// ── Test 4: Numerical Parity & Execution across Devices ─────────────────────

void test_end_to_end_parity(Device dev) {
    const char* dev_name = dev.is_cuda() ? "CUDA" : (dev.is_vulkan() ? "Vulkan" : "CPU");
    std::printf("\n--- Test 4: End-to-End Trace JIT Numerical Parity on %s ---\n", dev_name);

    // Non-power-of-two cols (300) with simultaneous Row (1x300), Col (48x1),
    // and Scalar (1x1) broadcasts, FMA, DAG CSE, and power-of-two DivScalar
    const int R = 48;
    const int C = 300;
    std::vector<float> hx(R * C), hr(C), hc(R), hs(1, 0.35f);
    for (int i = 0; i < R * C; ++i) hx[i] = 0.01f * static_cast<float>((i % 97) - 48);
    for (int i = 0; i < C; ++i)     hr[i] = 0.5f + 0.002f * static_cast<float>(i);
    for (int i = 0; i < R; ++i)     hc[i] = -0.25f + 0.01f * static_cast<float>(i);

    std::vector<float> ref(R * C);
    for (int row = 0; row < R; ++row) {
        const float cv = hc[row] + hs[0];
        const float col_silu = cv / (1.0f + std::exp(-cv));
        for (int col = 0; col < C; ++col) {
            const float xv = hx[row * C + col];
            const float rv = hr[col];
            const float fma_v = xv * rv + col_silu;
            const float cse_v = xv + rv;
            ref[row * C + col] = (fma_v + cse_v + cse_v) / 8.0f;
        }
    }

    Tensor x_dev = Tensor::from_host_on(dev, hx.data(), R, C);
    Tensor r_dev = Tensor::from_host_on(dev, hr.data(), 1, C);
    Tensor c_dev = Tensor::from_host_on(dev, hc.data(), R, 1);
    Tensor s_dev = Tensor::from_host_on(dev, hs.data(), 1, 1);

    begin_trace();
    Tensor col_act = silu(c_dev + s_dev);
    Tensor fma_v   = (x_dev * r_dev) + col_act;
    Tensor cse_a   = x_dev + r_dev;
    Tensor cse_b   = x_dev + r_dev;
    Tensor out_dev = (fma_v + cse_a + cse_b) / 8.0f;
    TraceHandle handle = end_trace();

    brotensor::sync(dev);
    std::vector<float> got = out_dev.to_host_vector();

    float max_err = 0.0f;
    for (int i = 0; i < R * C; ++i) {
        max_err = std::max(max_err, std::abs(got[i] - ref[i]));
    }
    check(max_err <= 1e-4f, std::string(dev_name) + " 48x300 broadcast + FMA + CSE parity (max_err=" +
                                std::to_string(max_err) + ")");

    // Replay verification
    handle.execute();
    brotensor::sync(dev);
    got = out_dev.to_host_vector();
    max_err = 0.0f;
    for (int i = 0; i < R * C; ++i) {
        max_err = std::max(max_err, std::abs(got[i] - ref[i]));
    }
    check(max_err <= 1e-4f, std::string(dev_name) + " 48x300 replay parity (max_err=" +
                                std::to_string(max_err) + ")");
}

}  // namespace

int main() {
    brotensor::init();
    test_magic_u32_exactness();
    test_ptx_optimizations();
#if BROTENSOR_HAS_BRASS_JIT
    test_spirv_and_smt_equivalence();
#endif

    test_end_to_end_parity(Device::CPU);
    if (brotensor::is_available(Device::CUDA)) {
        test_end_to_end_parity(Device::CUDA);
    }
    if (brotensor::is_available(Device::VULKAN)) {
        test_end_to_end_parity(Device::VULKAN);
    }

    if (g_failures > 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nAll trace JIT optimization & SMT equivalence checks PASSED.\n");
    return 0;
}
