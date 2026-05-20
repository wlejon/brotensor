#pragma once

// Shared helpers for the per-op CPU↔GPU parity suite.
//
// Header-only — each parity test executable includes this and gets its own
// copy of the inline helpers. Keeps the build wiring trivial (no extra .cpp).
// Complements the monolithic test_cpu_gpu_parity.cpp: this suite has one
// executable per op group so ctest surfaces failures individually.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace bt_parity {

using brotensor::Tensor;

// ─── GPU backend selection ─────────────────────────────────────────────────
//
// The parity suite is backend-neutral: it runs the same device-neutral op on
// the CPU and on whichever GPU backend this binary was built with. Prefer
// CUDA when present, else Metal. Returns Device::CPU as a sentinel meaning
// "no GPU backend" — run_all() checks this up front and skips the suite.
//
// First call must happen after brotensor::init() (run_all guarantees this);
// the result is cached.
inline brotensor::Device gpu_device() {
    using brotensor::Device;
    static const Device d = [] {
        if (brotensor::is_available(Device::CUDA))  return Device::CUDA;
        if (brotensor::is_available(Device::Metal)) return Device::Metal;
        return Device::CPU;
    }();
    return d;
}

// ─── Test registry ─────────────────────────────────────────────────────────

struct TestEntry {
    const char* name;
    void (*fn)();
};

inline std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> r;
    return r;
}

#define BT_PARITY_TEST(name)                                                   \
    static void name();                                                        \
    namespace {                                                                \
    struct Reg_##name {                                                        \
        Reg_##name() { ::bt_parity::registry().push_back({#name, name}); }      \
    } reg_##name;                                                               \
    }                                                                          \
    static void name()

inline void check(bool cond, const char* msg, const char* file, int line) {
    if (!cond) {
        std::printf("    assertion failed at %s:%d: %s\n", file, line, msg);
        throw 0;
    }
}

#define BT_CHECK(cond) ::bt_parity::check((cond), #cond, __FILE__, __LINE__)

// ─── Deterministic RNG (splitmix64 → uniform float in [-1, 1]) ─────────────

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float next_f01() {
        // Map top 24 bits to [0, 1).
        return static_cast<float>(next_u64() >> 40) / 16777216.0f;
    }
    float next_unit() { return next_f01() * 2.0f - 1.0f; }
};

inline void fill_random(Tensor& t, SplitMix64& rng, float scale = 1.0f) {
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = rng.next_unit() * scale;
}

// ─── Tolerance comparison ─────────────────────────────────────────────────

inline void compare_tensors(const Tensor& cpu, const Tensor& gpu,
                            const char* tag,
                            float atol = 1e-5f, float rtol = 1e-4f) {
    if (cpu.rows != gpu.rows || cpu.cols != gpu.cols) {
        std::printf("    [%s] shape mismatch: cpu (%d,%d) vs gpu (%d,%d)\n",
                    tag, cpu.rows, cpu.cols, gpu.rows, gpu.cols);
        throw 0;
    }
    const int n = cpu.size();
    int worst_idx = -1;
    float worst_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = cpu[i];
        const float b = gpu[i];
        const float d = std::fabs(a - b);
        const float tol = atol + rtol * std::fabs(a);
        if (d > tol) {
            if (d > worst_diff) { worst_diff = d; worst_idx = i; }
        }
    }
    if (worst_idx >= 0) {
        std::printf("    [%s] mismatch at i=%d  cpu=%.7g gpu=%.7g  diff=%.3g\n",
                    tag, worst_idx,
                    cpu[worst_idx], gpu[worst_idx], worst_diff);
        throw 0;
    }
}

// Bring a (possibly device-resident) tensor back to a CPU tensor for
// host-side inspection.
inline Tensor download_to_host(const Tensor& g) {
    brotensor::sync_all();
    return g.to(brotensor::Device::CPU);
}

// ─── BF16 parity helpers ───────────────────────────────────────────────────
//
// BF16 ops are GPU-only (the CPU backend is FP32-only). A BF16 parity test
// rounds its FP32 inputs to BF16, runs the op on CUDA, widens the BF16 result
// back to FP32, and compares against the FP32 CPU reference with a loose
// tolerance — BF16 carries only 8 mantissa bits (~2-3 decimal digits), so use
// atol/rtol around 2e-2 (looser still for long reductions: matmul, attention).

// Round an FP32 host tensor to a BF16 host tensor (same shape, Device::CPU).
inline Tensor to_bf16_host(const Tensor& f32cpu) {
    Tensor out = Tensor::zeros_on(brotensor::Device::CPU, f32cpu.rows,
                                  f32cpu.cols, brotensor::Dtype::BF16);
    const float* s = f32cpu.host_f32();
    uint16_t* d = out.host_bf16_mut();
    for (int i = 0; i < f32cpu.size(); ++i) d[i] = brotensor::fp32_to_bf16_bits(s[i]);
    return out;
}

// Widen a BF16 host tensor back to an FP32 host tensor for compare_tensors().
inline Tensor bf16_host_to_f32(const Tensor& bf16cpu) {
    Tensor out = Tensor::zeros_on(brotensor::Device::CPU, bf16cpu.rows,
                                  bf16cpu.cols, brotensor::Dtype::FP32);
    const uint16_t* s = bf16cpu.host_bf16();
    float* d = out.host_f32_mut();
    for (int i = 0; i < bf16cpu.size(); ++i) d[i] = brotensor::bf16_bits_to_fp32(s[i]);
    return out;
}

// Convenience: round an FP32 host tensor to BF16 and place it on CUDA.
inline Tensor to_bf16_cuda(const Tensor& f32cpu) {
    return to_bf16_host(f32cpu).to(brotensor::Device::CUDA);
}

// ─── Backend-neutral mask / index buffer helpers ──────────────────────────

// Build a GPU-resident float mask buffer from a host float mask vector. If
// `mask` is null, returns a default-constructed (empty) Tensor whose `.data`
// is null — matches the "no mask" sentinel used by the op APIs.
inline Tensor upload_mask(const std::vector<float>* mask) {
    if (!mask) return Tensor{};
    const int n = static_cast<int>(mask->size());
    Tensor h = Tensor::vec(n);
    for (int i = 0; i < n; ++i) h.ptr()[i] = (*mask)[i];
    return h.to(gpu_device());
}

// Same for an int32 index vector (embedding lookup tests). Returns a
// GPU-resident INT32 tensor.
inline Tensor upload_indices(const std::vector<int32_t>& idx) {
    const int n = static_cast<int>(idx.size());
    Tensor h = Tensor::zeros_on(brotensor::Device::CPU, n, 1,
                                brotensor::Dtype::INT32);
    auto* p = static_cast<int32_t*>(h.host_raw_mut());
    for (int i = 0; i < n; ++i) p[i] = idx[i];
    return h.to(gpu_device());
}

// Same for an int offsets array (head_offsets in batched softmax-xent).
inline Tensor upload_offsets(const std::vector<int>& off) {
    const int n = static_cast<int>(off.size());
    Tensor h = Tensor::zeros_on(brotensor::Device::CPU, n, 1,
                                brotensor::Dtype::INT32);
    auto* p = static_cast<int32_t*>(h.host_raw_mut());
    for (int i = 0; i < n; ++i) p[i] = static_cast<int32_t>(off[i]);
    return h.to(gpu_device());
}

// ─── Test runner ──────────────────────────────────────────────────────────

inline int run_all(const char* banner) {
    std::printf("%s\n", banner);
    for (size_t i = 0; i < std::strlen(banner); ++i) std::putchar('=');
    std::putchar('\n');

    brotensor::init();
    if (gpu_device() == brotensor::Device::CPU) {
        std::printf("no GPU backend available - skipping\n");
        return 0;
    }

    int passed = 0;
    int total = static_cast<int>(registry().size());
    for (const auto& t : registry()) {
        try {
            t.fn();
            ++passed;
            std::printf("  PASS  %s\n", t.name);
        } catch (const std::exception& e) {
            std::printf("  FAIL  %s  (exception: %s)\n", t.name, e.what());
        } catch (...) {
            std::printf("  FAIL  %s\n", t.name);
        }
    }
    std::printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

} // namespace bt_parity
