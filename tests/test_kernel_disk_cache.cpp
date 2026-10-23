// ─── Persistent On-Disk Superoptimized ML Kernel Cache Test Suite ────────────
//
// Validates:
//   1. Cold-start compilation, SMT guard elimination, MIR superoptimization,
//      and atomic .btkc persistence across all fixed CPU SIMD kernels
//      (cpu_jit.cpp, cpu_jit_fused.cpp, cpu_jit_quant.cpp) and CPU trace JIT.
//   2. Warm-cache reload via brass::target::DynamicLibrary after resetting
//      in-memory caches, verifying 0 misses and bit-exact output parity.
//   3. Vulkan SPIR-V (SpirvKernel + raw word stream) and CUDA PTX persistence.
//   4. Corruption detection (payload bit-flip and header truncation) with
//      automatic unlink and recompilation recovery.

#include <brotensor/jit/trace.h>
#include <brotensor/ops.h>
#include <brotensor/ops/fused.h>
#include <brotensor/tensor.h>

#include "../src/cpu/cpu_jit.h"
#include "../src/jit/kernel_disk_cache.h"
#include "../src/jit/ptx_emit.h"
#include "../src/jit/spirv_emit.h"
#include "../src/jit/trace_cache.h"
#include "../src/jit/trace_dag.h"
#include "../src/jit/trace_plan.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;
using brotensor::jit::KernelDiskCache;
using brotensor::jit::TraceCache;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  [FAIL] %s (line %d)\n", (msg), __LINE__);           \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

struct Q8Block {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(Q8Block) == 34, "Q8Block size");

struct Q4KBlock {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "Q4KBlock size");

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9f;
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

size_t count_btkc_files(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return 0;
    size_t n = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".btkc") {
            ++n;
        }
    }
    return n;
}

struct AllKernelOutputs {
    std::vector<float> rms;
    std::vector<float> ln;
    std::vector<float> mod_out;
    std::vector<float> bmul_out;
    std::vector<float> swiglu_out;
    std::vector<float> res_rms_x;
    std::vector<float> res_rms_y;
    std::vector<float> res_ln_x;
    std::vector<float> res_ln_y;
    std::vector<float> ln_mod_y;
    std::vector<float> q8_y;
    std::vector<float> q4k_y;
    std::vector<float> trace_y;
};

AllKernelOutputs run_all_cpu_kernels() {
    namespace cj = brotensor::detail::cpu::jit;
    AllKernelOutputs out;

    const int B = 4;
    const int D = 64;
    std::vector<float> X(B * D), res(B * D), gamma(D), beta(D), scale(D), shift(D);
    for (int i = 0; i < B * D; ++i) {
        X[i]   = 0.1f * static_cast<float>((i % 17) - 8);
        res[i] = 0.05f * static_cast<float>((i % 13) - 6);
    }
    for (int d = 0; d < D; ++d) {
        gamma[d] = 0.8f + 0.01f * static_cast<float>(d);
        beta[d]  = -0.1f + 0.005f * static_cast<float>(d);
        scale[d] = 0.05f * static_cast<float>((d % 7) - 3);
        shift[d] = 0.02f * static_cast<float>((d % 5) - 2);
    }

    out.rms.resize(B * D, 0.0f);
    cj::rms_norm_forward(X.data(), gamma.data(), 1e-5f, out.rms.data(), B, D);

    out.ln.resize(B * D, 0.0f);
    cj::layernorm_forward_inference_batched(X.data(), gamma.data(), beta.data(), 1e-5f, out.ln.data(), B, D);

    out.mod_out.resize(B * D, 0.0f);
    cj::modulate(X.data(), scale.data(), shift.data(), out.mod_out.data(), B, D);

    out.bmul_out.resize(B * D, 0.0f);
    cj::broadcast_mul(X.data(), gamma.data(), out.bmul_out.data(), B, D);

    std::vector<float> X_swiglu(B * 2 * D);
    for (int i = 0; i < B * 2 * D; ++i) X_swiglu[i] = 0.1f * static_cast<float>((i % 19) - 9);
    out.swiglu_out.resize(B * D, 0.0f);
    cj::swiglu_forward(X_swiglu.data(), out.swiglu_out.data(), B, D);

    out.res_rms_x = X;
    out.res_rms_y.resize(B * D, 0.0f);
    cj::fused_residual_rmsnorm(out.res_rms_x.data(), res.data(), gamma.data(), 1e-5f, out.res_rms_y.data(), B, D);

    out.res_ln_x = X;
    out.res_ln_y.resize(B * D, 0.0f);
    cj::fused_residual_layernorm(out.res_ln_x.data(), res.data(), gamma.data(), beta.data(), 1e-5f, out.res_ln_y.data(), B, D);

    out.ln_mod_y.resize(B * D, 0.0f);
    cj::fused_layernorm_modulate(X.data(), gamma.data(), beta.data(), scale.data(), shift.data(), 1e-5f, out.ln_mod_y.data(), B, D);

    // Quantized Q8_0 (N=4, K=256)
    const int N = 4;
    const int K = 256;
    std::vector<Q8Block> w_q8(N * (K / 32));
    for (size_t b = 0; b < w_q8.size(); ++b) {
        w_q8[b].d = brotensor::fp32_to_fp16_bits(0.02f);
        for (int i = 0; i < 32; ++i) w_q8[b].qs[i] = static_cast<int8_t>((i % 15) - 7);
    }
    std::vector<float> q_x(K);
    for (int i = 0; i < K; ++i) q_x[i] = 0.1f * static_cast<float>((i % 11) - 5);
    out.q8_y.resize(N, 0.0f);
    cj::gemv_q8_0(w_q8.data(), q_x.data(), out.q8_y.data(), N, K);

    // Quantized Q4_K (N=4, K=256)
    std::vector<Q4KBlock> w_q4k(N * (K / 256));
    for (size_t b = 0; b < w_q4k.size(); ++b) {
        w_q4k[b].d    = brotensor::fp32_to_fp16_bits(0.05f);
        w_q4k[b].dmin = brotensor::fp32_to_fp16_bits(0.01f);
        for (int i = 0; i < 12; ++i) w_q4k[b].scales[i] = static_cast<uint8_t>(2 + (i % 5));
        for (int i = 0; i < 128; ++i) w_q4k[b].qs[i] = static_cast<uint8_t>((i & 0x0F) | (((i + 3) & 0x0F) << 4));
    }
    out.q4k_y.resize(N, 0.0f);
    cj::gemv_q4_k(w_q4k.data(), q_x.data(), out.q4k_y.data(), N, K);

    // Traced CPU elementwise graph
    Tensor tA = Tensor::mat(B, D);
    Tensor tB = Tensor::mat(B, D);
    std::memcpy(tA.host_f32_mut(), X.data(), B * D * sizeof(float));
    std::memcpy(tB.host_f32_mut(), res.data(), B * D * sizeof(float));
    brotensor::jit::begin_trace();
    Tensor tY = brotensor::jit::silu(tA + tB);
    brotensor::jit::TraceHandle th = brotensor::jit::end_trace();
    (void)th;
    out.trace_y = tY.to_host_vector();

    return out;
}

void test_cpu_cold_warm_and_corruption(const std::filesystem::path& cache_dir) {
    std::cout << "--- 1. Testing CPU SIMD Kernel Disk Cache (Cold -> Warm -> Corruption) ---\n";
    auto& cache = KernelDiskCache::instance();
    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
    cache.set_cache_dir(cache_dir);
    cache.reset_stats();
    brotensor::detail::cpu::jit::reset_cpu_jit_caches_for_testing();
    TraceCache::instance().clear();

    // 1. Cold run: compiles, superoptimizes, links .so, and writes .btkc files
    auto t_cold0 = std::chrono::steady_clock::now();
    AllKernelOutputs cold_out = run_all_cpu_kernels();
    double cold_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_cold0).count();

    auto s_cold = cache.stats();
    size_t files_on_disk = count_btkc_files(cache_dir);
    std::printf("  Cold run: %.2f ms | hits=%zu misses=%zu writes=%zu files=%zu\n",
                cold_ms, s_cold.hits, s_cold.misses, s_cold.writes, files_on_disk);

    CHECK(s_cold.hits == 0, "Cold cache should have 0 hits");
    CHECK(s_cold.misses >= 10, "Cold cache should record misses for all 9 fixed kernels + trace kernel");
    CHECK(s_cold.writes >= 10, "Cold cache should write .btkc files for all 9 fixed kernels + trace kernel");
    CHECK(files_on_disk >= 10, "Expected at least 10 .btkc files on disk");

    // 2. Warm run: clear in-memory function pointers & TraceCache, reload from .btkc via DynamicLibrary
    cache.reset_stats();
    brotensor::detail::cpu::jit::reset_cpu_jit_caches_for_testing();
    TraceCache::instance().clear();

    auto t_warm0 = std::chrono::steady_clock::now();
    AllKernelOutputs warm_out = run_all_cpu_kernels();
    double warm_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_warm0).count();

    auto s_warm = cache.stats();
    std::printf("  Warm run: %.2f ms | hits=%zu misses=%zu writes=%zu corruptions=%zu\n",
                warm_ms, s_warm.hits, s_warm.misses, s_warm.writes, s_warm.corruptions);

    CHECK(s_warm.hits >= 10, "Warm cache should hit on all 10 kernels");
    CHECK(s_warm.misses == 0, "Warm cache should have 0 misses");
    CHECK(s_warm.writes == 0, "Warm cache should perform 0 writes");
    CHECK(s_warm.corruptions == 0, "Warm cache should have 0 corruptions");

    // Verify bit-exact parity between cold and warm runs
    CHECK(max_abs_diff(cold_out.rms, warm_out.rms) == 0.0f, "RMSNorm cold vs warm bit-exact parity");
    CHECK(max_abs_diff(cold_out.ln, warm_out.ln) == 0.0f, "LayerNorm cold vs warm bit-exact parity");
    CHECK(max_abs_diff(cold_out.mod_out, warm_out.mod_out) == 0.0f, "Modulate cold vs warm bit-exact parity");
    CHECK(max_abs_diff(cold_out.bmul_out, warm_out.bmul_out) == 0.0f, "BroadcastMul cold vs warm bit-exact parity");
    CHECK(max_abs_diff(cold_out.swiglu_out, warm_out.swiglu_out) == 0.0f, "SwiGLU cold vs warm bit-exact parity");
    CHECK(max_abs_diff(cold_out.res_rms_x, warm_out.res_rms_x) == 0.0f, "FusedResRMSNorm X cold vs warm parity");
    CHECK(max_abs_diff(cold_out.res_rms_y, warm_out.res_rms_y) == 0.0f, "FusedResRMSNorm Y cold vs warm parity");
    CHECK(max_abs_diff(cold_out.res_ln_x, warm_out.res_ln_x) == 0.0f, "FusedResLayerNorm X cold vs warm parity");
    CHECK(max_abs_diff(cold_out.res_ln_y, warm_out.res_ln_y) == 0.0f, "FusedResLayerNorm Y cold vs warm parity");
    CHECK(max_abs_diff(cold_out.ln_mod_y, warm_out.ln_mod_y) == 0.0f, "FusedLayerNormModulate cold vs warm parity");
    CHECK(max_abs_diff(cold_out.q8_y, warm_out.q8_y) == 0.0f, "GEMV Q8_0 cold vs warm bit-exact parity");
    CHECK(max_abs_diff(cold_out.q4k_y, warm_out.q4k_y) == 0.0f, "GEMV Q4_K cold vs warm bit-exact parity");
    CHECK(max_abs_diff(cold_out.trace_y, warm_out.trace_y) == 0.0f, "CPU Trace elementwise cold vs warm parity");

    // 3. Corruption detection & self-healing recovery
    int corrupted_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".btkc") continue;
        if (corrupted_files == 0) {
            // Flip payload byte to trigger FNV-1a checksum mismatch
            std::fstream fs(entry.path(), std::ios::in | std::ios::out | std::ios::binary);
            fs.seekp(40, std::ios::beg);
            char b = 0;
            fs.read(&b, 1);
            b ^= static_cast<char>(0xFF);
            fs.seekp(40, std::ios::beg);
            fs.write(&b, 1);
            ++corrupted_files;
        } else if (corrupted_files == 1) {
            // Truncate header to trigger size/header corruption
            std::ofstream ofs(entry.path(), std::ios::binary | std::ios::trunc);
            ofs.write("BAD", 3);
            ++corrupted_files;
            break;
        }
    }
    CHECK(corrupted_files == 2, "Should have corrupted 2 .btkc files on disk");

    cache.reset_stats();
    brotensor::detail::cpu::jit::reset_cpu_jit_caches_for_testing();
    TraceCache::instance().clear();

    AllKernelOutputs recovered_out = run_all_cpu_kernels();
    auto s_rec = cache.stats();
    std::printf("  Recovery run: hits=%zu misses=%zu writes=%zu corruptions=%zu\n",
                s_rec.hits, s_rec.misses, s_rec.writes, s_rec.corruptions);

    CHECK(s_rec.corruptions == 2, "Should detect both corrupted .btkc files");
    CHECK(s_rec.writes == 2, "Should recompile and rewrite both corrupted .btkc files");
    CHECK(s_rec.hits >= 8, "Remaining uncorrupted kernels should still hit");
    CHECK(max_abs_diff(cold_out.rms, recovered_out.rms) == 0.0f, "Recovered RMSNorm parity");
    CHECK(max_abs_diff(cold_out.q8_y, recovered_out.q8_y) == 0.0f, "Recovered GEMV Q8_0 parity");
}

void test_spirv_and_ptx_disk_cache(const std::filesystem::path& cache_dir) {
    std::cout << "--- 2. Testing Vulkan SPIR-V & CUDA PTX Disk Cache ---\n";
    auto& cache = KernelDiskCache::instance();
    cache.set_cache_dir(cache_dir);
    cache.reset_stats();

    // Build a small synthetic TraceDAG for elementwise + row-norm emission
    Tensor t0 = Tensor::mat(4, 64);
    Tensor t1 = Tensor::mat(4, 64);
    Tensor tOut = Tensor::mat(4, 64);
    brotensor::jit::TraceDAG dag;
    int in0 = dag.register_input(t0);
    int in1 = dag.register_input(t1);
    int add_id = dag.add_node(brotensor::jit::TraceOpKind::Add, {in0, in1}, 0.0f,
                              nullptr, Device::CPU, Dtype::FP32, 4, 64);
    int silu_id = dag.add_node(brotensor::jit::TraceOpKind::SiLU, {add_id}, 0.0f,
                               tOut.data, Device::CPU, Dtype::FP32, 4, 64);
    dag.analyze_liveness();
    (void)silu_id;

    brotensor::jit::plan::ElementwisePlan ew;
    bool built_ew = brotensor::jit::plan::build_plan(dag, ew);
    CHECK(built_ew, "Should build ElementwisePlan for synthetic DAG");

    // 1. Test SPIR-V kernel emission + disk cache round-trip
    auto spv_kernels = brotensor::jit::spirv::emit_elementwise(dag, ew);
    CHECK(!spv_kernels.scalar.words.empty(), "SPIR-V scalar kernel should be non-empty");
    CHECK(spv_kernels.scalar.words[0] == 0x07230203u, "SPIR-V magic word should match");

    bool stored_spv = cache.store_spirv_kernel("test_spirv_ew_scalar", spv_kernels.scalar);
    CHECK(stored_spv, "store_spirv_kernel should succeed");

    brass::target::SpirvKernel loaded_spv;
    bool hit_spv = cache.load_spirv_kernel("test_spirv_ew_scalar", loaded_spv);
    CHECK(hit_spv, "load_spirv_kernel should hit");
    CHECK(loaded_spv.words == spv_kernels.scalar.words, "Loaded SPIR-V words must match bit-for-bit");
    CHECK(loaded_spv.entry == spv_kernels.scalar.entry, "Loaded SPIR-V entry name must match");
    CHECK(loaded_spv.push_constant_bytes == spv_kernels.scalar.push_constant_bytes, "Loaded SPIR-V push bytes must match");
    CHECK(loaded_spv.params.size() == spv_kernels.scalar.params.size(), "Loaded SPIR-V param count must match");

    // Raw SPIR-V words API
    CHECK(cache.store_spirv("test_raw_spirv_words", spv_kernels.scalar.words), "store_spirv should succeed");
    std::vector<uint32_t> raw_words;
    CHECK(cache.load_spirv("test_raw_spirv_words", raw_words), "load_spirv should hit");
    CHECK(raw_words == spv_kernels.scalar.words, "Raw SPIR-V words round-trip match");

    // 2. Test CUDA PTX emission + disk cache round-trip
    std::string ptx_src = brotensor::jit::ptx::emit_elementwise(dag, ew, "sm_89");
    CHECK(!ptx_src.empty(), "Emitted PTX should be non-empty");
    CHECK(cache.store_ptx("test_cuda_ptx_ew", ptx_src), "store_ptx should succeed");

    std::string loaded_ptx;
    CHECK(cache.load_ptx("test_cuda_ptx_ew", loaded_ptx), "load_ptx should hit");
    CHECK(loaded_ptx == ptx_src, "Loaded PTX must match emitted PTX character-for-character");

    auto s = cache.stats();
    std::printf("  SPIR-V & PTX cache: hits=%zu misses=%zu writes=%zu\n",
                s.hits, s.misses, s.writes);
    CHECK(s.hits == 3, "Expected 3 hits (spirv_meta, raw spirv, ptx)");
    CHECK(s.writes == 3, "Expected 3 writes");
}

} // namespace

int main() {
    std::filesystem::path tmp_dir =
        std::filesystem::temp_directory_path() / ("brotensor_test_kcache_" + std::to_string(12345));

    test_cpu_cold_warm_and_corruption(tmp_dir);
    test_spirv_and_ptx_disk_cache(tmp_dir);

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
    KernelDiskCache::instance().clear_cache_dir_override();

    if (g_failures == 0) {
        std::cout << "[PASS] All KernelDiskCache tests passed.\n";
        return 0;
    }
    std::cout << "[FAIL] " << g_failures << " test(s) failed.\n";
    return 1;
}
