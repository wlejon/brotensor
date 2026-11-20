// Cross-target verification of the unified Brass GPU MIR trace emitters
// (SPIR-V, PTX, Metal MSL) over shared gpu_mir_emit_* builders.

#include "../src/jit/gpu_mir_emit_detail.h"
#include "../src/jit/kernel_disk_cache.h"
#include "../src/jit/msl_emit.h"
#include "../src/jit/ptx_emit.h"
#include "../src/jit/spirv_emit.h"
#include "../src/jit/trace_dag.h"
#include "../src/jit/trace_plan.h"

#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/verify/gpu_equiv.hpp>
#include <brotensor/runtime.h>

#include <cstdio>
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

bool has(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

TraceDAG make_elementwise_dag(Dtype dt, int rows, int cols) {
    static float dummy[4];
    TraceDAG dag;
    const int x = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, dt, rows, cols);
    const int r = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, dt, 1, cols);
    const int c = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, dt, rows, 1);

    const int mul_xr = dag.add_node(TraceOpKind::Mul, {x, r}, 0.0f, nullptr, Device::CPU, dt, rows, cols);
    const int add_c = dag.add_node(TraceOpKind::Add, {mul_xr, c}, 0.0f, nullptr, Device::CPU, dt, rows, cols);
    const int t = dag.add_node(TraceOpKind::Tanh, {add_c}, 0.0f, nullptr, Device::CPU, dt, rows, cols);
    const int s = dag.add_node(TraceOpKind::SiLU, {t}, 0.0f, nullptr, Device::CPU, dt, rows, cols);
    const int out = dag.add_node(TraceOpKind::Copy, {s}, 0.0f, nullptr, Device::CPU, dt, rows, cols);
    dag.nodes()[static_cast<std::size_t>(out)].is_live_at_end = true;
    return dag;
}

TraceDAG make_rownorm_dag(bool layernorm, Dtype dt, int rows, int cols) {
    static float dummy[4];
    TraceDAG dag;
    const int x = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, dt, rows, cols);
    const int g = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, dt, 1, cols);
    std::vector<int> ins = {x, g};
    if (layernorm) {
        const int b = dag.add_node(TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, dt, 1, cols);
        ins.push_back(b);
    }
    const TraceOpKind kind = layernorm ? TraceOpKind::LayerNorm : TraceOpKind::RMSNorm;
    const int n = dag.add_node(kind, ins, 1e-5f, nullptr, Device::CPU, dt, rows, cols);
    dag.nodes()[static_cast<std::size_t>(n)].is_live_at_end = true;
    return dag;
}

// Straight-line 1-element-per-thread MIR kernel built from the shared gpu_mir
// primitives (udiv_urem_const + load_lanes + eval_chain + store_lanes) so
// bounded symbolic execution proves MIR == SPIR-V == PTX without loop-bound truncation.
brass::Function* build_straightline_ew_mir(brass::Module& mod,
                                           const TraceDAG& dag,
                                           const plan::ElementwisePlan& ew,
                                           Dtype dt,
                                           const char* name) {
    const std::size_t n_out = ew.outputs.size();
    const std::size_t n_in = ew.inputs.size();
    std::vector<brass::Type> params(n_out + n_in, brass::Type::ptr());
    brass::Function* fn = mod.create_function(
        name, brass::Type::void_type(), brass::Span<const brass::Type>(params));

    brass::codegen::KernelBuilder kb(mod, fn);
    gpu_mir::Ctx c(kb);
    brass::Builder& b = c.b;
    const auto p = gpu_mir::entry_params(c, fn);

    brass::Value* tid = kb.global_tid_x();
    brass::Value* row = nullptr;
    brass::Value* col = nullptr;
    gpu_mir::udiv_urem_const(c, tid, ew.cols, row, col);

    const int esz = plan::elem_bytes(dt);
    gpu_mir::ValueMap vals;
    for (std::size_t k = 0; k < n_in; ++k) {
        const auto& s = ew.inputs[k];
        brass::Value* ptr = p[n_out + k];
        if (s.bcast == BroadcastKind::Scalar) {
            vals[s.node_id] = {gpu_mir::load_lanes(c, s.dtype, 1, ptr)[0]};
        } else {
            brass::Value* idx = (s.bcast == BroadcastKind::Row) ? col
                              : (s.bcast == BroadcastKind::Col) ? row
                              : tid;
            brass::Value* off = gpu_mir::byte_offset(c, idx, esz);
            vals[s.node_id] = gpu_mir::load_lanes(c, s.dtype, 1, b.build_add(ptr, off));
        }
    }
    vals = gpu_mir::eval_chain(c, dag, 1, std::move(vals));
    for (std::size_t m = 0; m < n_out; ++m) {
        const auto& s = ew.outputs[m];
        brass::Value* off = gpu_mir::byte_offset(c, tid, esz);
        gpu_mir::store_lanes(c, s.dtype, 1, b.build_add(p[m], off), vals.at(s.node_id));
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    gpu_mir::optimize_function(*fn);
    return fn;
}

void test_elementwise_equivalence_and_msl() {
    std::printf("\n--- Test 1: Unified Elementwise MIR -> SPIR-V == PTX + MSL ---\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        const std::string dt_name = (dt == Dtype::FP32) ? "f32" : (dt == Dtype::FP16 ? "f16" : "bf16");
        TraceDAG dag = make_elementwise_dag(dt, 16, 240);
        plan::ElementwisePlan ew;
        check(plan::build_plan(dag, ew), "ew plan built for " + dt_name);

        brass::Module mod("test_ew_sl_" + dt_name);
        brass::Function* fn_sl = build_straightline_ew_mir(mod, dag, ew, dt, ("sl_ew_" + dt_name).c_str());

        brass::verify::EquivOptions opts;
        opts.max_sat_conflicts = 4000;
        std::string err;
        auto eq = brass::verify::verify_spirv_ptx_equivalence(*fn_sl, &err, opts);
        check(eq.is_equivalent(),
              "SPIR-V == PTX equivalence on shared gpu_mir elementwise (" + dt_name + "): " + err);

        spirv::Kernels spv = spirv::emit_elementwise(dag, ew);
        check(spv.vec.has_value() && !spv.vec->words.empty() && !spv.scalar.words.empty(),
              "SPIR-V ew vec+scalar emitted (" + dt_name + ")");
        if (spv.vec.has_value()) {
            brass::spirv::Module parsed_spv;
            std::string parse_err;
            check(brass::spirv::parse_binary(spv.vec->words, parsed_spv, &parse_err),
                  "SPIR-V ew vec parses cleanly (" + dt_name + "): " + parse_err);
            check(brass::spirv::verify(parsed_spv).empty(),
                  "SPIR-V ew vec validates cleanly (" + dt_name + ")");
        }

        std::string ptx_src = ptx::emit_elementwise(dag, ew, "sm_80");
        check(has(ptx_src, ".version 7.8") && has(ptx_src, "tanh.approx.f32") &&
                  has(ptx_src, "mul.hi.u32") &&
                  has(ptx_src, ".visible .entry trace_ew_vec") &&
                  has(ptx_src, ".visible .entry trace_ew_scalar"),
              "PTX ew has expected intrinsics and dual entries (" + dt_name + ")");

        msl::Kernels msl = msl::emit_elementwise(dag, ew);
        check(msl.vec.has_value() && !msl.vec->source.empty() && !msl.scalar.source.empty(),
              "MSL ew emitted vec+scalar (" + dt_name + ")");
        if (msl.vec.has_value()) {
            const std::string& src = msl.vec->source;
            check(has(src, "device uint8_t* _b0 [[buffer(0)]]") &&
                      has(src, "constant trace_ew_vec_Params& _params") &&
                      has(src, "tanh(") && has(src, "mulhi(") &&
                      has(src, "while (true)") && !has(src, "goto"),
                  "MSL ew has valid buffer/param/intrinsic/structured-loop output (" + dt_name + ")");
            if (dt == Dtype::BF16) {
                check(has(src, "_brass_f32_to_bf16("), "MSL BF16 ew uses _brass_f32_to_bf16 helper");
            }

            const std::string cache_sig = "test_gpu_mir_emit/msl/" + dt_name + "/" + dag.key_material();
            check(KernelDiskCache::instance().store_msl_kernel(cache_sig, *msl.vec),
                  "KernelDiskCache::store_msl_kernel succeeds (" + dt_name + ")");
            brass::target::MslKernel loaded;
            check(KernelDiskCache::instance().load_msl_kernel(cache_sig, loaded) &&
                      loaded.source == msl.vec->source &&
                      loaded.entry_name == msl.vec->entry_name &&
                      loaded.num_buffers == msl.vec->num_buffers &&
                      loaded.scalar_params_bytes == msl.vec->scalar_params_bytes &&
                      loaded.scalar_param_offsets == msl.vec->scalar_param_offsets,
                  "KernelDiskCache::load_msl_kernel round-trips bit-identically (" + dt_name + ")");
        }
    }
}

void test_rownorm_equivalence_and_msl() {
    std::printf("\n--- Test 2: Unified RowNorm MIR -> SPIR-V / PTX / MSL ---\n");
    // Straight-line warp-shuffle + shared-memory + rsqrt + bf16 reduction kernel
    // exercising every RowNorm intrinsic across MIR == SPIR-V == PTX.
    {
        brass::Module mod("test_rn_equiv");
        brass::Function* fn = mod.create_function(
            "rn_reduce_step", brass::Type::void_type(), {brass::Type::ptr(), brass::Type::ptr()});
        brass::codegen::KernelBuilder kb(mod, fn);
        gpu_mir::Ctx c(kb);
        brass::Builder& b = c.b;
        const auto p = gpu_mir::entry_params(c, fn);

        brass::Value* tid = kb.tid_x();
        brass::Value* smem = kb.shared_alloc_f32(32);
        brass::Value* idx = gpu_mir::byte_offset(c, tid, 4);
        brass::Value* x = kb.load_f32(b.build_add(p[0], idx));
        brass::Value* sq = b.build_mul(x, x);
        brass::Value* sh = b.build_add(sq, kb.shfl_bfly_f32(sq, 1u));
        kb.shared_store_f32_indexed(smem, b.build_and(tid, c.i32(31)), sh);
        kb.sync();
        brass::Value* rstd = kb.rsqrt_approx(
            b.build_add(kb.shared_load_f32_indexed(smem, b.build_and(tid, c.i32(31))), c.f32(1e-5f)));
        brass::Value* y = b.build_mul(x, rstd);
        kb.store_u16(b.build_add(p[1], gpu_mir::byte_offset(c, tid, 2)), kb.f32_to_bf16(y));
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        gpu_mir::optimize_function(*fn);

        brass::verify::EquivOptions opts;
        opts.max_sat_conflicts = 4000;
        std::string err;
        auto eq = brass::verify::verify_spirv_ptx_equivalence(*fn, &err, opts);
        check(eq.is_equivalent(), "SPIR-V == PTX equivalence on RowNorm intrinsic pipeline: " + err);
    }

    for (bool ln : {false, true}) {
        const std::string tag = ln ? "layernorm" : "rmsnorm";
        TraceDAG dag = make_rownorm_dag(ln, Dtype::BF16, 16, 1024);
        plan::RowNormPlan rn;
        check(plan::build_row_norm_plan(dag, rn), tag + " multi-warp plan built");

        spirv::Kernels spv = spirv::emit_row_norm(dag, rn);
        check(spv.vec.has_value() && !spv.vec->words.empty() && !spv.scalar.words.empty(),
              "SPIR-V " + tag + " vec+scalar emitted");
        if (spv.vec.has_value()) {
            brass::spirv::Module parsed_spv;
            std::string parse_err;
            check(brass::spirv::parse_binary(spv.vec->words, parsed_spv, &parse_err),
                  "SPIR-V " + tag + " vec parses: " + parse_err);
            check(brass::spirv::verify(parsed_spv).empty(), "SPIR-V " + tag + " vec validates cleanly");
        }

        std::string ptx_src = ptx::emit_row_norm(dag, rn, "sm_80");
        check(has(ptx_src, ".visible .entry trace_row_norm_vec") &&
                  has(ptx_src, ".visible .entry trace_row_norm_scalar") &&
                  has(ptx_src, "bar.sync") &&
                  has(ptx_src, "shfl.sync.bfly.b32"),
              "PTX " + tag + " emits valid multi-warp reduction + barrier + shfl");

        msl::Kernels msl = msl::emit_row_norm(dag, rn);
        check(msl.vec.has_value() && msl.vec->uses_shared_memory && msl.vec->shared_memory_bytes > 0 &&
                  has(msl.vec->source, "threadgroup float _smem_") &&
                  has(msl.vec->source, "threadgroup_barrier(mem_flags::mem_threadgroup);") &&
                  has(msl.vec->source, "simd_shuffle_xor(") &&
                  has(msl.vec->source, "_brass_f32_to_bf16(") &&
                  has(msl.vec->source, "while (true)") &&
                  !has(msl.vec->source, "goto"),
              "MSL " + tag + " emits threadgroup smem, barrier, simd_shuffle_xor, bf16 helper");
    }
}

}  // namespace

int main() {
    brotensor::init();
    test_elementwise_equivalence_and_msl();
    test_rownorm_equivalence_and_msl();
    if (g_failures > 0) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nAll unified GPU MIR emitter tests PASSED.\n");
    return 0;
}
