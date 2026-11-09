// ─── brotensor's On-Disk Kernel Cache (over brass's artifact cache) ─────────
//
//   1. Every fixed CPU kernel (cpu_jit*.cpp) and the CPU trace kernel: a cold
//      run builds and stores them, a warm run (in-memory tables reset, images
//      unloaded) loads every one from disk with bit-identical output.
//   2. Damaged entries (a flipped bit, a truncation, a foreign file at the
//      entry's path) are misses, rebuilt and stored over.
//   3. Keys: every CpuCompileOptions field, the entry, the signature and the
//      kind make a different entry; two option sets are two files on disk.
//   4. Old-format files (*.btkc, .rt/) are never read and are cleaned up.
//   5. Eviction keeps the directory within its budget, and a loaded kernel
//      keeps running after its entry is evicted.
//   6. SPIR-V kernels (with launch metadata) and PTX round-trip; damage is a miss.

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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;
using brotensor::jit::KernelDiskCache;
using brotensor::jit::TraceCache;
using CpuOpts = KernelDiskCache::CpuCompileOptions;

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

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<fs::path> entries(const fs::path& dir) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".bac") out.push_back(e.path());
    }
    return out;
}

uint64_t entry_bytes_on_disk(const fs::path& dir) {
    uint64_t total = 0;
    for (const auto& p : entries(dir)) {
        std::error_code ec;
        const uint64_t n = fs::file_size(p, ec);
        total += (n + 4095) / 4096 * 4096;
    }
    return total;
}

void make_old(const fs::path& p) {
    std::error_code ec;
    fs::last_write_time(p, fs::file_time_type::clock::now() - std::chrono::hours(1), ec);
}

void reset_in_memory() {
    brotensor::detail::cpu::jit::reset_cpu_jit_caches_for_testing();   // also unloads the images
    TraceCache::instance().clear();
}

struct AllKernelOutputs {
    std::vector<std::vector<float>> v;
};

AllKernelOutputs run_all_cpu_kernels() {
    namespace cj = brotensor::detail::cpu::jit;
    AllKernelOutputs out;
    out.v.reserve(32);   // slot() hands out references into it
    auto slot = [&out](size_t n) -> std::vector<float>& {
        out.v.emplace_back(n, 0.0f);
        return out.v.back();
    };

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

    cj::rms_norm_forward(X.data(), gamma.data(), 1e-5f, slot(B * D).data(), B, D);
    cj::layernorm_forward_inference_batched(X.data(), gamma.data(), beta.data(), 1e-5f, slot(B * D).data(), B, D);
    cj::modulate(X.data(), scale.data(), shift.data(), slot(B * D).data(), B, D);
    cj::broadcast_mul(X.data(), gamma.data(), slot(B * D).data(), B, D);

    std::vector<float> X_swiglu(B * 2 * D);
    for (int i = 0; i < B * 2 * D; ++i) X_swiglu[i] = 0.1f * static_cast<float>((i % 19) - 9);
    cj::swiglu_forward(X_swiglu.data(), slot(B * D).data(), B, D);

    std::vector<float>& res_rms_x = slot(0);
    res_rms_x = X;
    cj::fused_residual_rmsnorm(res_rms_x.data(), res.data(), gamma.data(), 1e-5f, slot(B * D).data(), B, D);
    std::vector<float> res_ln_x = X;
    cj::fused_residual_layernorm(res_ln_x.data(), res.data(), gamma.data(), beta.data(), 1e-5f,
                                 slot(B * D).data(), B, D);
    out.v.push_back(res_ln_x);
    cj::fused_layernorm_modulate(X.data(), gamma.data(), beta.data(), scale.data(), shift.data(), 1e-5f,
                                 slot(B * D).data(), B, D);

    const int N = 4;
    const int K = 256;
    std::vector<Q8Block> w_q8(N * (K / 32));
    for (auto& b : w_q8) {
        b.d = brotensor::fp32_to_fp16_bits(0.02f);
        for (int i = 0; i < 32; ++i) b.qs[i] = static_cast<int8_t>((i % 15) - 7);
    }
    std::vector<float> q_x(K);
    for (int i = 0; i < K; ++i) q_x[i] = 0.1f * static_cast<float>((i % 11) - 5);
    cj::gemv_q8_0(w_q8.data(), q_x.data(), slot(N).data(), N, K);

    std::vector<Q4KBlock> w_q4k(N * (K / 256));
    for (auto& b : w_q4k) {
        b.d    = brotensor::fp32_to_fp16_bits(0.05f);
        b.dmin = brotensor::fp32_to_fp16_bits(0.01f);
        for (int i = 0; i < 12; ++i) b.scales[i] = static_cast<uint8_t>(2 + (i % 5));
        for (int i = 0; i < 128; ++i) b.qs[i] = static_cast<uint8_t>((i & 0x0F) | (((i + 3) & 0x0F) << 4));
    }
    cj::gemv_q4_k(w_q4k.data(), q_x.data(), slot(N).data(), N, K);

    Tensor tA = Tensor::mat(B, D);
    Tensor tB = Tensor::mat(B, D);
    std::memcpy(tA.host_f32_mut(), X.data(), B * D * sizeof(float));
    std::memcpy(tB.host_f32_mut(), res.data(), B * D * sizeof(float));
    brotensor::jit::begin_trace();
    Tensor tY = brotensor::jit::silu(tA + tB);
    brotensor::jit::TraceHandle th = brotensor::jit::end_trace();
    (void)th;
    out.v.push_back(tY.to_host_vector());
    return out;
}

bool same_outputs(const AllKernelOutputs& a, const AllKernelOutputs& b) {
    if (a.v.size() != b.v.size()) return false;
    for (size_t i = 0; i < a.v.size(); ++i) {
        if (!same(a.v[i], b.v[i])) return false;
    }
    return true;
}

// The fixed kernels and the trace kernel: ten CPU entries.
constexpr size_t kCpuKernels = 10;

void test_cpu_cold_warm_and_damage(const fs::path& dir, AllKernelOutputs& reference) {
    std::cout << "--- 1/2. CPU kernels: cold, warm, damaged entries ---\n";
    auto& cache = KernelDiskCache::instance();
    cache.set_cache_dir(dir);
    cache.reset_stats();
    reset_in_memory();

    reference = run_all_cpu_kernels();
    auto s = cache.stats();
    const size_t files = entries(dir).size();
    std::printf("  cold: hits=%zu misses=%zu writes=%zu entries=%zu\n", s.hits, s.misses, s.writes, files);
    CHECK(s.hits == 0, "a cold cache has no hits");
    CHECK(s.misses == kCpuKernels, "a cold cache builds every CPU kernel");
    CHECK(s.writes == kCpuKernels, "a cold cache stores every CPU kernel");
    CHECK(files == kCpuKernels, "one entry per CPU kernel");

    cache.reset_stats();
    reset_in_memory();
    AllKernelOutputs warm = run_all_cpu_kernels();
    s = cache.stats();
    std::printf("  warm: hits=%zu misses=%zu writes=%zu\n", s.hits, s.misses, s.writes);
    CHECK(s.hits == kCpuKernels, "a warm cache loads every CPU kernel");
    CHECK(s.misses == 0 && s.writes == 0, "a warm cache builds nothing");
    CHECK(same_outputs(reference, warm), "warm kernels are bit-identical to cold ones");

    // A flipped bit, a truncation and a foreign file at an entry's path.
    std::vector<fs::path> files_now = entries(dir);
    CHECK(files_now.size() >= 3, "three entries to damage");
    if (files_now.size() >= 3) {
        {
            std::fstream f(files_now[0], std::ios::in | std::ios::out | std::ios::binary);
            f.seekg(100);
            char b = 0;
            f.read(&b, 1);
            b = static_cast<char>(b ^ 0x5A);
            f.seekp(100);
            f.write(&b, 1);
        }
        fs::resize_file(files_now[1], fs::file_size(files_now[1]) / 2);
        {
            std::ofstream f(files_now[2], std::ios::binary | std::ios::trunc);
            f << "BTKC\x01\x00\x00\x00 not an artifact";
        }
    }
    cache.reset_stats();
    reset_in_memory();
    AllKernelOutputs healed = run_all_cpu_kernels();
    s = cache.stats();
    std::printf("  damaged: hits=%zu misses=%zu writes=%zu\n", s.hits, s.misses, s.writes);
    CHECK(s.misses == 3 && s.writes == 3, "each damaged entry is a miss and is rebuilt");
    CHECK(s.hits == kCpuKernels - 3, "the undamaged entries still hit");
    CHECK(same_outputs(reference, healed), "rebuilt kernels are bit-identical");

    cache.reset_stats();
    reset_in_memory();
    (void)run_all_cpu_kernels();
    CHECK(cache.stats().hits == kCpuKernels, "the rebuilt entries were stored over the damaged ones");
}

void test_keys(const fs::path& dir) {
    std::cout << "--- 3. Keys ---\n";
    const CpuOpts base;
    const auto k0 = KernelDiskCache::cpu_key("sig", "entry", base);
    CHECK(k0.valid, "the test binary can be identified, so keys are valid");
    CHECK(k0 == KernelDiskCache::cpu_key("sig", "entry", base), "a key is deterministic");

    std::vector<CpuOpts> variants(9, base);
    variants[0].enable_loop_opt = false;
    variants[1].enable_avx2 = false;
    variants[2].enable_fma = false;
    variants[3].enable_vectorize = false;
    variants[4].enable_unroll = false;
    variants[5].unroll_factor = 8;
    variants[6].enable_fp_reassociation = false;
    variants[7].enable_smt_guard_elim = false;
    variants[8].enable_mir_superopt = false;
    for (size_t i = 0; i < variants.size(); ++i) {
        CHECK(KernelDiskCache::cpu_key("sig", "entry", variants[i]) != k0,
              "every CpuCompileOptions field is in the key");
    }
    CHECK(KernelDiskCache::cpu_key("sig", "entry2", base) != k0, "the entry name is in the key");
    CHECK(KernelDiskCache::cpu_key("sig2", "entry", base) != k0, "the signature is in the key");
    CHECK(KernelDiskCache::blob_key("ptx", "s") != KernelDiskCache::blob_key("spirv", "s"),
          "the kind is in the key");

    // On disk: one build per option set; each then hits its own entry.
    auto& cache = KernelDiskCache::instance();
    cache.set_cache_dir(dir);
    cache.reset_stats();
    CpuOpts no_unroll = base;
    no_unroll.enable_unroll = false;
    const fs::path pa = cache.cpu_entry_path("cpu_core_broadcast_mul_v1", "broadcast_mul", base);
    const fs::path pb = cache.cpu_entry_path("cpu_core_broadcast_mul_v1", "broadcast_mul", no_unroll);
    CHECK(pa != pb, "two option sets are two entry paths");
    CHECK(fs::exists(pa), "the default options' entry exists from part 1");
    CHECK(!fs::exists(pb), "the other option set has no entry yet");
}

void test_old_format(const fs::path& dir) {
    std::cout << "--- 4. Old-format files ---\n";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / ".rt", ec);
    const fs::path old_entry = dir / "0123456789abcdef0123456789abcdef.btkc";
    const fs::path old_tmp = dir / "0123456789abcdef0123456789abcdef.btkc.tmp.1.2.3";
    const fs::path old_rt = dir / ".rt" / "0123_1_1.so";
    const fs::path keep = dir / "unrelated.txt";
    for (const fs::path& p : {old_entry, old_tmp, old_rt, keep}) {
        std::ofstream f(p, std::ios::binary);
        f << "BTKC\x01\x00\x00\x00 old";
    }

    auto& cache = KernelDiskCache::instance();
    cache.set_cache_dir(dir);
    // An old file where the new entry goes reads as a foreign file: a miss.
    const fs::path ptx_path = cache.blob_entry_path("ptx", "old-format-probe");
    {
        std::ofstream f(ptx_path, std::ios::binary);
        f << "BTKC\x01\x00\x00\x00\x03\x00\x00\x00 .version 7.8 .entry x";
    }
    cache.reset_stats();
    std::string ptx;
    CHECK(!cache.load_ptx("old-format-probe", ptx), "an old-format file is never read");
    CHECK(cache.stats().corruptions == 1, "and is counted as an unusable entry");
    CHECK(!fs::exists(old_entry) && !fs::exists(old_tmp), "old .btkc files are cleaned up");
    CHECK(!fs::exists(dir / ".rt"), "the old .rt load copies are cleaned up");
    CHECK(fs::exists(keep), "files that are not the old cache's are left alone");
    CHECK(cache.store_ptx("old-format-probe", ".version 7.8\n.entry x {}\n"), "a store replaces it");
    CHECK(cache.load_ptx("old-format-probe", ptx) && ptx == ".version 7.8\n.entry x {}\n", "and hits");
}

void test_eviction(const fs::path& dir) {
    std::cout << "--- 5. Eviction, and a loaded kernel outliving its entry ---\n";
    namespace cj = brotensor::detail::cpu::jit;
    auto& cache = KernelDiskCache::instance();
    std::error_code ec;
    fs::remove_all(dir, ec);
    cache.set_cache_dir(dir);
    cache.set_max_bytes(0);
    reset_in_memory();

    const int B = 2, D = 32;
    std::vector<float> X(B * D), gamma(D, 1.25f), y0(B * D), y1(B * D);
    for (int i = 0; i < B * D; ++i) X[i] = 0.03f * static_cast<float>(i - 20);
    cj::rms_norm_forward(X.data(), gamma.data(), 1e-5f, y0.data(), B, D);
    const fs::path rms_entry = cache.cpu_entry_path("cpu_core_rms_norm_row_v1", "rms_norm_row", CpuOpts{});
    CHECK(fs::exists(rms_entry), "the rms_norm kernel is stored");
    for (const fs::path& p : entries(dir)) make_old(p);

    // 40 entries of ~20 KiB under a 128 KiB budget.
    const uint64_t budget = 128u << 10;
    cache.set_max_bytes(budget);
    std::string ptx = ".version 7.8\n.entry k {}\n";
    ptx.resize(20u << 10, ' ');
    for (int i = 0; i < 40; ++i) {
        CHECK(cache.store_ptx("evict-" + std::to_string(i), ptx), "store a filler entry");
    }
    const uint64_t after = cache.trim();
    const uint64_t on_disk = entry_bytes_on_disk(dir);
    std::printf("  budget=%llu trim=%llu on_disk=%llu entries=%zu\n",
                static_cast<unsigned long long>(budget), static_cast<unsigned long long>(after),
                static_cast<unsigned long long>(on_disk), entries(dir).size());
    CHECK(on_disk <= budget, "the directory is within its budget after a trim");
    CHECK(!fs::exists(rms_entry), "the least recently used entry (the CPU kernel) was evicted");

    // The function pointer the dispatch table holds still runs.
    cj::rms_norm_forward(X.data(), gamma.data(), 1e-5f, y1.data(), B, D);
    CHECK(same(y0, y1), "a loaded kernel keeps running after its entry is evicted");
    cache.set_max_bytes(0);
}

void test_spirv_and_ptx(const fs::path& dir) {
    std::cout << "--- 6. SPIR-V and PTX ---\n";
    auto& cache = KernelDiskCache::instance();
    cache.set_cache_dir(dir);
    cache.reset_stats();

    Tensor t0 = Tensor::mat(4, 64);
    Tensor t1 = Tensor::mat(4, 64);
    Tensor tOut = Tensor::mat(4, 64);
    brotensor::jit::TraceDAG dag;
    int in0 = dag.register_input(t0);
    int in1 = dag.register_input(t1);
    int add_id = dag.add_node(brotensor::jit::TraceOpKind::Add, {in0, in1}, 0.0f,
                              nullptr, Device::CPU, Dtype::FP32, 4, 64);
    dag.add_node(brotensor::jit::TraceOpKind::SiLU, {add_id}, 0.0f,
                 tOut.data, Device::CPU, Dtype::FP32, 4, 64);
    dag.analyze_liveness();

    brotensor::jit::plan::ElementwisePlan ew;
    CHECK(brotensor::jit::plan::build_plan(dag, ew), "an ElementwisePlan for the DAG");

    const std::string sig = "test_ew/" + dag.key_material();
    auto spv = brotensor::jit::spirv::emit_elementwise(dag, ew);
    CHECK(!spv.scalar.words.empty() && spv.scalar.words[0] == 0x07230203u, "SPIR-V is emitted");
    CHECK(cache.store_spirv_kernel(sig, spv.scalar), "store_spirv_kernel");
    brass::target::SpirvKernel got;
    CHECK(cache.load_spirv_kernel(sig, got), "load_spirv_kernel hits");
    const auto& k = spv.scalar;
    CHECK(got.words == k.words && got.entry == k.entry && got.stage == k.stage, "words, entry, stage");
    CHECK(got.push_constant_bytes == k.push_constant_bytes && got.shared_bytes == k.shared_bytes,
          "push constant and shared bytes");
    CHECK(std::memcmp(got.local_size, k.local_size, sizeof(k.local_size)) == 0 &&
              got.local_size_spec_constants == k.local_size_spec_constants, "local size");
    CHECK(got.params.size() == k.params.size() && got.descriptors == k.descriptors &&
              got.capabilities == k.capabilities && got.extensions == k.extensions, "launch metadata");
    for (size_t i = 0; i < got.params.size() && i < k.params.size(); ++i) {
        CHECK(got.params[i].type.kind() == k.params[i].type.kind() && got.params[i].offset == k.params[i].offset &&
                  got.params[i].size == k.params[i].size, "a parameter");
    }

    // A DAG that differs in one scalar is a different entry.
    brotensor::jit::TraceDAG dag2 = dag;
    dag2.nodes()[3].scalar = 0.5f;
    CHECK(!cache.load_spirv_kernel("test_ew/" + dag2.key_material(), got), "another DAG misses");

    const std::string ptx_src = brotensor::jit::ptx::emit_elementwise(dag, ew, "sm_89");
    CHECK(cache.store_ptx(sig, ptx_src), "store_ptx");
    std::string ptx;
    CHECK(cache.load_ptx(sig, ptx) && ptx == ptx_src, "load_ptx returns the text stored");

    // A flipped bit in the PTX entry is a miss, never different text.
    const fs::path p = cache.blob_entry_path("ptx", sig);
    {
        std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(static_cast<std::streamoff>(fs::file_size(p) - 1));
        f.put('!');
    }
    cache.reset_stats();
    CHECK(!cache.load_ptx(sig, ptx), "a damaged PTX entry misses");
    CHECK(cache.stats().corruptions == 1, "and is counted");
}

} // namespace

int main() {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(::getpid());
#endif
    const fs::path root = fs::temp_directory_path() / ("brotensor_test_kcache_" + std::to_string(pid));
    std::error_code ec;
    fs::remove_all(root, ec);

    AllKernelOutputs reference;
    test_cpu_cold_warm_and_damage(root / "main", reference);
    test_keys(root / "main");
    test_old_format(root / "old");
    test_eviction(root / "evict");
    test_spirv_and_ptx(root / "blobs");

    reset_in_memory();
    KernelDiskCache::instance().clear_cache_dir_override();
    fs::remove_all(root, ec);

    if (g_failures == 0) {
        std::cout << "[PASS] All KernelDiskCache tests passed.\n";
        return 0;
    }
    std::cout << "[FAIL] " << g_failures << " test(s) failed.\n";
    return 1;
}
