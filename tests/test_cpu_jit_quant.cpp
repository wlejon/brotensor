// ─── Brass JIT CPU Quantized GEMV (Q8_0 & Q4_K) Benchmark & Parity ──────────
//
// Verifies numerical parity (<= 1e-4 max error) against double-precision reference
// and benchmarks execution latency (us) and effective memory bandwidth (GB/s)
// of Brass AVX2/FMA JIT quantized GEMV on CPU across shapes (K=4096, N in {1, 8, 32, 128}).

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/detail/cpu/thread_pool.h>
#include "../src/cpu/cpu_jit.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>

using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

int g_failures = 0;

#define CHECK_PARITY(err, tol, name) do {                                            \
    if ((err) > (tol)) {                                                              \
        std::printf("  [FAIL] %s: max err = %g > tol = %g\n", (name), (err), (tol));  \
        ++g_failures;                                                                 \
    }                                                                                 \
} while (0)

#define CHECK(cond, msg) do {                                                         \
    if (!(cond)) {                                                                    \
        std::printf("  [FAIL] %s\n", (msg));                                          \
        ++g_failures;                                                                 \
    }                                                                                 \
} while (0)

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
        return static_cast<float>(next_u64() >> 40) / 16777216.0f;
    }
    float next_unit() { return next_f01() * 2.0f - 1.0f; }
};

void fill_random(std::vector<float>& vec, uint64_t seed, float scale = 1.0f) {
    SplitMix64 rng(seed);
    for (auto& val : vec) {
        val = rng.next_unit() * scale;
    }
}

float max_abs_error(const float* a, const float* b, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

template <typename Fn>
double time_cpu_us(Fn&& fn, int warmup = 5, int iters = 25) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    return elapsed.count() / iters;
}

// ─── GGUF Q8_0 Layout & Helpers ─────────────────────────────────────────────

constexpr int Q8_BLOCK = 32;
constexpr int Q8_BYTES = 34;

struct Q8Block {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(Q8Block) == 34, "Q8Block must be 34 bytes");

void quantize_q8_0_block(const float* src, Q8Block& out) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float a = std::fabs(src[i]);
        if (a > amax) amax = a;
    }
    const float d = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    out.d = brotensor::fp32_to_fp16_bits(d);
    const float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
    for (int i = 0; i < 32; ++i) {
        int q = static_cast<int>(std::lround(src[i] * inv_d));
        q = std::clamp(q, -128, 127);
        out.qs[i] = static_cast<int8_t>(q);
    }
}

void dequant_q8_0_block(const Q8Block& blk, float* dst) {
    const float d = brotensor::fp16_bits_to_fp32(blk.d);
    for (int i = 0; i < 32; ++i) {
        dst[i] = d * static_cast<float>(blk.qs[i]);
    }
}

// ─── GGUF Q4_K Layout & Helpers ─────────────────────────────────────────────

constexpr int Q4K_BLOCK = 256;
constexpr int Q4K_BYTES = 144;

struct Q4KBlock {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "Q4KBlock must be 144 bytes");

void quantize_q4k_block(const float* src, Q4KBlock& out) {
    float lo[8], hi[8];
    for (int is = 0; is < 8; ++is) {
        lo[is] = src[is * 32];
        hi[is] = src[is * 32];
        for (int l = 1; l < 32; ++l) {
            const float v = src[is * 32 + l];
            if (v < lo[is]) lo[is] = v;
            if (v > hi[is]) hi[is] = v;
        }
    }
    float max_range = 0.0f;
    for (int is = 0; is < 8; ++is) {
        const float r = hi[is] - lo[is];
        if (r > max_range) max_range = r;
    }
    const float d = (max_range > 0.0f) ? (max_range / (15.0f * 63.0f)) : 1.0f;
    float max_neg_lo = 0.0f;
    for (int is = 0; is < 8; ++is) {
        const float v = -lo[is];
        if (v > max_neg_lo) max_neg_lo = v;
    }
    const float dmin = (max_neg_lo > 0.0f) ? (max_neg_lo / 63.0f) : 1.0f;

    uint8_t sc[8], m[8];
    for (int is = 0; is < 8; ++is) {
        const float r = hi[is] - lo[is];
        int sc_i = (d > 0.0f) ? static_cast<int>(std::lround(r / (15.0f * d))) : 0;
        sc_i = std::clamp(sc_i, 0, 63);
        sc[is] = static_cast<uint8_t>(sc_i);

        int m_i = (dmin > 0.0f) ? static_cast<int>(std::lround(-lo[is] / dmin)) : 0;
        m_i = std::clamp(m_i, 0, 63);
        m[is] = static_cast<uint8_t>(m_i);
    }

    std::memset(out.scales, 0, 12);
    for (int j = 0; j < 4; ++j) {
        out.scales[j]     = sc[j] & 0x3F;
        out.scales[j + 4] = m[j]  & 0x3F;
    }
    for (int j = 4; j < 8; ++j) {
        const uint8_t sc_lo = sc[j] & 0x0F;
        const uint8_t sc_hi = (sc[j] >> 4) & 0x03;
        const uint8_t m_lo  = m[j]  & 0x0F;
        const uint8_t m_hi  = (m[j]  >> 4) & 0x03;
        out.scales[j + 4] = static_cast<uint8_t>(sc_lo | (m_lo << 4));
        out.scales[j - 4] |= static_cast<uint8_t>(sc_hi << 6);
        out.scales[j]     |= static_cast<uint8_t>(m_hi  << 6);
    }

    std::memset(out.qs, 0, 128);
    for (int p = 0; p < 4; ++p) {
        const int is_lo = 2 * p;
        const int is_hi = 2 * p + 1;
        const float w_lo = static_cast<float>(sc[is_lo]) * d;
        const float w_hi = static_cast<float>(sc[is_hi]) * d;
        const float b_lo = static_cast<float>(m [is_lo]) * dmin;
        const float b_hi = static_cast<float>(m [is_hi]) * dmin;
        for (int l = 0; l < 32; ++l) {
            int n_lo = 0;
            if (w_lo > 0.0f) {
                n_lo = static_cast<int>(std::lround((src[is_lo * 32 + l] + b_lo) / w_lo));
            }
            n_lo = std::clamp(n_lo, 0, 15);

            int n_hi = 0;
            if (w_hi > 0.0f) {
                n_hi = static_cast<int>(std::lround((src[is_hi * 32 + l] + b_hi) / w_hi));
            }
            n_hi = std::clamp(n_hi, 0, 15);

            out.qs[p * 32 + l] = static_cast<uint8_t>((n_lo & 0x0F) | ((n_hi & 0x0F) << 4));
        }
    }

    out.d    = brotensor::fp32_to_fp16_bits(d);
    out.dmin = brotensor::fp32_to_fp16_bits(dmin);
}

void dequant_q4k_block(const Q4KBlock& blk, float* dst) {
    const float d    = brotensor::fp16_bits_to_fp32(blk.d);
    const float dmin = brotensor::fp16_bits_to_fp32(blk.dmin);
    uint8_t sc[8], m[8];
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = blk.scales[j]     & 0x3F;
            m[j]  = blk.scales[j + 4] & 0x3F;
        } else {
            sc[j] = (blk.scales[j + 4] & 0x0F) | ((blk.scales[j - 4] >> 6) << 4);
            m[j]  = (blk.scales[j + 4] >> 4)   | ((blk.scales[j - 0] >> 6) << 4);
        }
    }
    for (int p = 0; p < 4; ++p) {
        const int is_lo = 2 * p;
        const int is_hi = 2 * p + 1;
        const float w_lo = static_cast<float>(sc[is_lo]) * d;
        const float w_hi = static_cast<float>(sc[is_hi]) * d;
        const float b_lo = static_cast<float>(m [is_lo]) * dmin;
        const float b_hi = static_cast<float>(m [is_hi]) * dmin;
        for (int l = 0; l < 32; ++l) {
            const uint8_t byte = blk.qs[p * 32 + l];
            const int n_lo = byte & 0x0F;
            const int n_hi = (byte >> 4) & 0x0F;
            dst[is_lo * 32 + l] = w_lo * static_cast<float>(n_lo) - b_lo;
            dst[is_hi * 32 + l] = w_hi * static_cast<float>(n_hi) - b_hi;
        }
    }
}

// ─── Tests & Benchmarks ──────────────────────────────────────────────────────

void test_q8_0_cpu() {
    std::cout << "\n================================================================================\n";
    std::cout << "  1. BRASS JIT CPU QUANTIZED GEMV (Q8_0) PARITY & BENCHMARK\n";
    std::cout << "================================================================================\n";

    const std::vector<int> n_values = {1, 8, 32, 128};
    const int K = 4096;
    const int BPR = K / Q8_BLOCK;

    for (int N : n_values) {
        std::vector<float> h_wf(static_cast<size_t>(N) * K);
        fill_random(h_wf, 1001 + N, 0.05f);

        std::vector<Q8Block> h_wq(static_cast<size_t>(N) * BPR);
        std::vector<float>   h_wd(static_cast<size_t>(N) * K);

        for (int r = 0; r < N; ++r) {
            for (int sb = 0; sb < BPR; ++sb) {
                const size_t elem_off = static_cast<size_t>(r) * K + sb * Q8_BLOCK;
                const size_t blk_off  = static_cast<size_t>(r) * BPR + sb;
                quantize_q8_0_block(&h_wf[elem_off], h_wq[blk_off]);
                dequant_q8_0_block(h_wq[blk_off], &h_wd[elem_off]);
            }
        }

        std::vector<float> h_x(K);
        fill_random(h_x, 2001 + N, 0.5f);

        std::vector<float> h_ref(N);
        std::vector<float> h_jit(N, 0.0f);

        // Double-precision host reference
        for (int r = 0; r < N; ++r) {
            double acc = 0.0;
            const size_t row_off = static_cast<size_t>(r) * K;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(h_wd[row_off + k]) * static_cast<double>(h_x[k]);
            }
            h_ref[r] = static_cast<float>(acc);
        }

        // 1. Direct JIT gemv_q8_0
        brotensor::detail::cpu::jit::gemv_q8_0(h_wq.data(), h_x.data(), h_jit.data(), N, K);

        float err = max_abs_error(h_ref.data(), h_jit.data(), N);
        CHECK_PARITY(err, 1e-4f, "gemv_q8_0");

        // 2. High-level op dispatcher linear_forward_q8_0_fp16
        {
            Tensor W_tensor = Tensor::empty_on(Device::CPU, N, K, Dtype::Q8_0);
            std::memcpy(W_tensor.data, h_wq.data(), static_cast<size_t>(N) * BPR * Q8_BYTES);
            std::vector<uint16_t> h_x_fp16(K);
            for (int i = 0; i < K; ++i) h_x_fp16[i] = brotensor::fp32_to_fp16_bits(h_x[i]);
            Tensor x_tensor = Tensor::from_host_fp16_on(Device::CPU, h_x_fp16.data(), K, 1);
            Tensor y_tensor = Tensor::zeros_on(Device::CPU, N, 1, Dtype::FP16);

            brotensor::linear_forward_q8_0_fp16(W_tensor, nullptr, x_tensor, y_tensor);

            const uint16_t* y_ptr = static_cast<const uint16_t*>(y_tensor.data);
            std::vector<float> h_y_op(N);
            for (int i = 0; i < N; ++i) h_y_op[i] = brotensor::fp16_bits_to_fp32(y_ptr[i]);

            float op_err = max_abs_error(h_ref.data(), h_y_op.data(), N);
            CHECK_PARITY(op_err, 2e-3f, "linear_forward_q8_0_fp16");
        }

        // Benchmark
        const size_t w_bytes = static_cast<size_t>(N) * BPR * Q8_BYTES;
        double t_us = time_cpu_us([&]() {
            brotensor::detail::cpu::jit::gemv_q8_0(h_wq.data(), h_x.data(), h_jit.data(), N, K);
        });

        const double total_bytes = static_cast<double>(w_bytes + K * sizeof(float) + N * sizeof(float));
        const double bw_gbs = (total_bytes / (t_us * 1e-6)) / 1e9;

        std::cout << "  N=" << std::setw(4) << N << " K=" << std::setw(5) << K
                  << " | Weight: " << std::setw(6) << std::fixed << std::setprecision(2) << (w_bytes / 1e6) << " MB"
                  << " | Latency: " << std::setw(8) << std::fixed << std::setprecision(2) << t_us << " us"
                  << " | Bandwidth: " << std::setw(6) << std::fixed << std::setprecision(1) << bw_gbs << " GB/s"
                  << " | MaxErr: " << std::scientific << std::setprecision(2) << err
                  << " | Status: " << (err <= 1e-4f ? "PASS" : "FAIL")
                  << std::endl;
    }
}

void test_q4_k_cpu() {
    std::cout << "\n================================================================================\n";
    std::cout << "  2. BRASS JIT CPU QUANTIZED GEMV (Q4_K) PARITY & BENCHMARK\n";
    std::cout << "================================================================================\n";

    const std::vector<int> n_values = {1, 8, 32, 128};
    const int K = 4096;
    const int BPR = K / Q4K_BLOCK;

    for (int N : n_values) {
        std::vector<float> h_wf(static_cast<size_t>(N) * K);
        fill_random(h_wf, 3001 + N, 0.05f);

        std::vector<Q4KBlock> h_wq(static_cast<size_t>(N) * BPR);
        std::vector<float>    h_wd(static_cast<size_t>(N) * K);

        for (int r = 0; r < N; ++r) {
            for (int sb = 0; sb < BPR; ++sb) {
                const size_t elem_off = static_cast<size_t>(r) * K + sb * Q4K_BLOCK;
                const size_t blk_off  = static_cast<size_t>(r) * BPR + sb;
                quantize_q4k_block(&h_wf[elem_off], h_wq[blk_off]);
                dequant_q4k_block(h_wq[blk_off], &h_wd[elem_off]);
            }
        }

        std::vector<float> h_x(K);
        fill_random(h_x, 4001 + N, 0.5f);

        std::vector<float> h_ref(N);
        std::vector<float> h_jit(N, 0.0f);

        // Double-precision host reference
        for (int r = 0; r < N; ++r) {
            double acc = 0.0;
            const size_t row_off = static_cast<size_t>(r) * K;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(h_wd[row_off + k]) * static_cast<double>(h_x[k]);
            }
            h_ref[r] = static_cast<float>(acc);
        }

        // 1. Direct JIT gemv_q4_k
        brotensor::detail::cpu::jit::gemv_q4_k(h_wq.data(), h_x.data(), h_jit.data(), N, K);

        float err = max_abs_error(h_ref.data(), h_jit.data(), N);
        CHECK_PARITY(err, 1e-4f, "gemv_q4_k");

        // 2. High-level op dispatcher linear_forward_q4k_fp16
        {
            Tensor W_tensor = Tensor::empty_on(Device::CPU, N, K, Dtype::Q4_K);
            std::memcpy(W_tensor.data, h_wq.data(), static_cast<size_t>(N) * BPR * Q4K_BYTES);
            std::vector<uint16_t> h_x_fp16(K);
            for (int i = 0; i < K; ++i) h_x_fp16[i] = brotensor::fp32_to_fp16_bits(h_x[i]);
            Tensor x_tensor = Tensor::from_host_fp16_on(Device::CPU, h_x_fp16.data(), K, 1);
            Tensor y_tensor = Tensor::zeros_on(Device::CPU, N, 1, Dtype::FP16);

            brotensor::linear_forward_q4k_fp16(W_tensor, nullptr, x_tensor, y_tensor);

            const uint16_t* y_ptr = static_cast<const uint16_t*>(y_tensor.data);
            std::vector<float> h_y_op(N);
            for (int i = 0; i < N; ++i) h_y_op[i] = brotensor::fp16_bits_to_fp32(y_ptr[i]);

            float op_err = max_abs_error(h_ref.data(), h_y_op.data(), N);
            CHECK_PARITY(op_err, 2e-3f, "linear_forward_q4k_fp16");
        }

        // Benchmark
        const size_t w_bytes = static_cast<size_t>(N) * BPR * Q4K_BYTES;
        double t_us = time_cpu_us([&]() {
            brotensor::detail::cpu::jit::gemv_q4_k(h_wq.data(), h_x.data(), h_jit.data(), N, K);
        });

        const double total_bytes = static_cast<double>(w_bytes + K * sizeof(float) + N * sizeof(float));
        const double bw_gbs = (total_bytes / (t_us * 1e-6)) / 1e9;

        std::cout << "  N=" << std::setw(4) << N << " K=" << std::setw(5) << K
                  << " | Weight: " << std::setw(6) << std::fixed << std::setprecision(2) << (w_bytes / 1e6) << " MB"
                  << " | Latency: " << std::setw(8) << std::fixed << std::setprecision(2) << t_us << " us"
                  << " | Bandwidth: " << std::setw(6) << std::fixed << std::setprecision(1) << bw_gbs << " GB/s"
                  << " | MaxErr: " << std::scientific << std::setprecision(2) << err
                  << " | Status: " << (err <= 1e-4f ? "PASS" : "FAIL")
                  << std::endl;
    }
}

void test_dequant_cpu() {
    std::cout << "\n================================================================================\n";
    std::cout << "  3. CPU DEQUANTIZATION (Q8_0 & Q4_K -> FP16) PARITY\n";
    std::cout << "================================================================================\n";

    constexpr int N = 8;
    constexpr int K = 512;

    // Q8_0 dequant
    {
        const int BPR = K / Q8_BLOCK;
        std::vector<float> h_wf(N * K);
        fill_random(h_wf, 7001, 0.1f);
        std::vector<Q8Block> h_wq(N * BPR);
        std::vector<float> h_wd(N * K);
        for (int r = 0; r < N; ++r) {
            for (int b = 0; b < BPR; ++b) {
                quantize_q8_0_block(&h_wf[r * K + b * Q8_BLOCK], h_wq[r * BPR + b]);
                dequant_q8_0_block(h_wq[r * BPR + b], &h_wd[r * K + b * Q8_BLOCK]);
            }
        }

        Tensor W_q8 = Tensor::empty_on(Device::CPU, N, K, Dtype::Q8_0);
        std::memcpy(W_q8.data, h_wq.data(), static_cast<size_t>(N) * BPR * Q8_BYTES);
        Tensor W_fp16 = Tensor::zeros_on(Device::CPU, N, K, Dtype::FP16);
        brotensor::dequant_q8_0_to_fp16(W_q8, W_fp16);

        const uint16_t* ptr = static_cast<const uint16_t*>(W_fp16.data);
        float max_err = 0.0f;
        for (int i = 0; i < N * K; ++i) {
            float f = brotensor::fp16_bits_to_fp32(ptr[i]);
            float diff = std::fabs(f - h_wd[i]);
            if (diff > max_err) max_err = diff;
        }
        std::cout << "  dequant_q8_0_to_fp16: max error = " << max_err << "\n";
        CHECK_PARITY(max_err, 1e-3f, "dequant_q8_0_to_fp16");
    }

    // Q4_K dequant
    {
        const int BPR = K / Q4K_BLOCK;
        std::vector<float> h_wf(N * K);
        fill_random(h_wf, 8001, 0.1f);
        std::vector<Q4KBlock> h_wq(N * BPR);
        std::vector<float> h_wd(N * K);
        for (int r = 0; r < N; ++r) {
            for (int b = 0; b < BPR; ++b) {
                quantize_q4k_block(&h_wf[r * K + b * Q4K_BLOCK], h_wq[r * BPR + b]);
                dequant_q4k_block(h_wq[r * BPR + b], &h_wd[r * K + b * Q4K_BLOCK]);
            }
        }

        Tensor W_q4k = Tensor::empty_on(Device::CPU, N, K, Dtype::Q4_K);
        std::memcpy(W_q4k.data, h_wq.data(), static_cast<size_t>(N) * BPR * Q4K_BYTES);
        Tensor W_fp16 = Tensor::zeros_on(Device::CPU, N, K, Dtype::FP16);
        brotensor::dequant_q4k_to_fp16(W_q4k, W_fp16);

        const uint16_t* ptr = static_cast<const uint16_t*>(W_fp16.data);
        float max_err = 0.0f;
        for (int i = 0; i < N * K; ++i) {
            float f = brotensor::fp16_bits_to_fp32(ptr[i]);
            float diff = std::fabs(f - h_wd[i]);
            if (diff > max_err) max_err = diff;
        }
        std::cout << "  dequant_q4k_to_fp16: max error = " << max_err << "\n";
        CHECK_PARITY(max_err, 1e-3f, "dequant_q4k_to_fp16");

        // Also test FP32 activations & default output dtype on resize for Q4_K and Q8_0
        std::vector<float> h_x(K);
        fill_random(h_x, 3001, 0.25f);
        Tensor x_f32 = Tensor::from_host_on(Device::CPU, h_x.data(), K, 1);
        Tensor y_f32;
        brotensor::linear_forward_q4k_fp16(W_q4k, nullptr, x_f32, y_f32);
        if (y_f32.dtype != Dtype::FP32 || y_f32.rows != N || y_f32.cols != 1) {
            std::cerr << "  [FAIL] linear_forward_q4k_fp16 did not resize y to FP32\n";
            ++g_failures;
        }
        Tensor X_bd_f32 = Tensor::from_host_on(Device::CPU, h_x.data(), 1, K);
        Tensor Y_bd_f32;
        brotensor::linear_forward_batched_q4k_fp16(W_q4k, nullptr, X_bd_f32, Y_bd_f32);
        if (Y_bd_f32.dtype != Dtype::FP32 || Y_bd_f32.rows != 1 || Y_bd_f32.cols != N) {
            std::cerr << "  [FAIL] linear_forward_batched_q4k_fp16 did not resize Y_BD to FP32\n";
            ++g_failures;
        }
        std::vector<float> h_y1 = y_f32.to_host_vector();
        std::vector<float> h_y2 = Y_bd_f32.to_host_vector();
        CHECK_PARITY(max_abs_error(h_y1.data(), h_y2.data(), N), 1e-5f, "q4k_fp32_vec_vs_batched");
    }
}

// ─── Q6_K Reference Helpers & Tests ──────────────────────────────────────────

constexpr int Q6K_BLOCK = 256;
constexpr int Q6K_BYTES = 210;

struct Q6KBlock {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(Q6KBlock) == Q6K_BYTES, "Q6KBlock size must be 210 bytes");

void pack_q6k_block(float d_f32, const int8_t sc[16], const int q[256], Q6KBlock& out) {
    std::memset(&out, 0, sizeof(out));
    out.d = brotensor::fp32_to_fp16_bits(d_f32);
    std::memcpy(out.scales, sc, 16);
    for (int e = 0; e < 256; ++e) {
        int val6 = std::clamp(q[e], -32, 31) + 32;
        int raw4 = val6 & 0x0F;
        int high2 = (val6 >> 4) & 0x03;

        int group = e >> 7;
        int local = e & 127;
        int quad  = local >> 5;
        int l     = local & 31;

        int ql_idx = group * 64 + (quad & 1) * 32 + l;
        if (quad < 2) {
            out.ql[ql_idx] |= static_cast<uint8_t>(raw4);
        } else {
            out.ql[ql_idx] |= static_cast<uint8_t>(raw4 << 4);
        }
        int qh_idx = group * 32 + l;
        out.qh[qh_idx] |= static_cast<uint8_t>(high2 << (quad * 2));
    }
}

void dequant_q6k_block(const Q6KBlock& blk, float* dst) {
    const float d = brotensor::fp16_bits_to_fp32(blk.d);
    for (int e = 0; e < 256; ++e) {
        const int group = e >> 7;
        const int local = e & 127;
        const int quad  = local >> 5;
        const int l     = local & 31;
        const int sb    = (group << 3) + (quad << 1) + (l >> 4);

        const uint8_t ql_b = blk.ql[group * 64 + (quad & 1) * 32 + l];
        const uint8_t qh_b = blk.qh[group * 32 + l];
        const int raw4  = (quad < 2) ? (ql_b & 0x0F) : (ql_b >> 4);
        const int high2 = (qh_b >> (quad * 2)) & 0x03;
        const int val6  = (raw4 | (high2 << 4)) - 32;
        dst[e] = d * static_cast<float>(blk.scales[sb]) * static_cast<float>(val6);
    }
}

void test_q6_k_cpu() {
    std::cout << "\n================================================================================\n";
    std::cout << "  4. BRASS JIT CPU QUANTIZED (Q6_K) DEQUANT & LINEAR TESTS\n";
    std::cout << "================================================================================\n";

    const int N = 16;
    const int K = 512;
    const int BPR = K / Q6K_BLOCK;

    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> q_dist(-32, 31);
    std::uniform_int_distribution<int> sc_dist(-12, 12);

    std::vector<Q6KBlock> h_wq(static_cast<size_t>(N) * BPR);
    std::vector<float> h_wd(static_cast<size_t>(N) * K);

    for (int r = 0; r < N; ++r) {
        for (int b = 0; b < BPR; ++b) {
            int8_t sc[16];
            int q[256];
            for (int i = 0; i < 16; ++i) sc[i] = static_cast<int8_t>(sc_dist(rng));
            for (int i = 0; i < 256; ++i) q[i] = q_dist(rng);
            const size_t blk_off = static_cast<size_t>(r) * BPR + b;
            pack_q6k_block(0.015625f, sc, q, h_wq[blk_off]);
            dequant_q6k_block(h_wq[blk_off], &h_wd[static_cast<size_t>(r) * K + b * Q6K_BLOCK]);
        }
    }

    Tensor W_q6k = Tensor::empty_on(Device::CPU, N, K, Dtype::Q6_K);
    std::memcpy(W_q6k.data, h_wq.data(), static_cast<size_t>(N) * BPR * Q6K_BYTES);

    // 1. dequant_q6k_to_fp16
    Tensor W_fp16;
    brotensor::dequant_q6k_to_fp16(W_q6k, W_fp16);
    const uint16_t* w_fp16_ptr = static_cast<const uint16_t*>(W_fp16.data);
    float deq_err = 0.0f;
    for (int i = 0; i < N * K; ++i) {
        float diff = std::fabs(brotensor::fp16_bits_to_fp32(w_fp16_ptr[i]) - h_wd[i]);
        if (diff > deq_err) deq_err = diff;
    }
    std::cout << "  dequant_q6k_to_fp16: max error = " << deq_err << "\n";
    CHECK_PARITY(deq_err, 1e-3f, "dequant_q6k_to_fp16");

    // 2. gemv_q6_k + linear_forward_q6k_fp16 (FP32 & FP16) + linear_forward_batched_q6k_fp16 (FP32 & FP16)
    std::vector<float> h_x(K);
    fill_random(h_x, 5001, 0.5f);
    std::vector<float> h_bias(N);
    fill_random(h_bias, 5002, 0.2f);

    std::vector<float> h_ref_nobias(N, 0.0f);
    std::vector<float> h_ref_bias(N, 0.0f);
    for (int r = 0; r < N; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            acc += static_cast<double>(h_wd[static_cast<size_t>(r) * K + k]) * static_cast<double>(h_x[k]);
        }
        h_ref_nobias[r] = static_cast<float>(acc);
        h_ref_bias[r] = static_cast<float>(acc) + h_bias[r];
    }

    std::vector<float> h_jit(N, 0.0f);
    brotensor::detail::cpu::jit::gemv_q6_k(h_wq.data(), h_x.data(), h_jit.data(), N, K);
    CHECK_PARITY(max_abs_error(h_ref_nobias.data(), h_jit.data(), N), 1e-4f, "gemv_q6_k");

    // FP32 activations + bias
    Tensor x_f32 = Tensor::from_host_on(Device::CPU, h_x.data(), K, 1);
    Tensor bias_f32 = Tensor::from_host_on(Device::CPU, h_bias.data(), N, 1);
    Tensor y_f32;
    brotensor::linear_forward_q6k_fp16(W_q6k, &bias_f32, x_f32, y_f32);
    if (y_f32.dtype != Dtype::FP32) {
        std::cerr << "  [FAIL] linear_forward_q6k_fp16 did not default y.dtype to FP32\n";
        ++g_failures;
    }
    std::vector<float> h_y_f32 = y_f32.to_host_vector();
    CHECK_PARITY(max_abs_error(h_ref_bias.data(), h_y_f32.data(), N), 1e-4f, "linear_forward_q6k_fp16 (FP32)");

    // Batched FP32 (B=2)
    std::vector<float> h_X_bd(2 * K);
    std::memcpy(h_X_bd.data(), h_x.data(), K * sizeof(float));
    std::memcpy(h_X_bd.data() + K, h_x.data(), K * sizeof(float));
    Tensor X_bd_f32 = Tensor::from_host_on(Device::CPU, h_X_bd.data(), 2, K);
    Tensor Y_bd_f32;
    brotensor::linear_forward_batched_q6k_fp16(W_q6k, &bias_f32, X_bd_f32, Y_bd_f32);
    if (Y_bd_f32.dtype != Dtype::FP32 || Y_bd_f32.rows != 2 || Y_bd_f32.cols != N) {
        std::cerr << "  [FAIL] linear_forward_batched_q6k_fp16 did not default Y_BD.dtype to FP32\n";
        ++g_failures;
    }
    std::vector<float> h_Y_bd = Y_bd_f32.to_host_vector();
    CHECK_PARITY(max_abs_error(h_ref_bias.data(), h_Y_bd.data(), N), 1e-4f, "linear_forward_batched_q6k_fp16 (FP32 row 0)");
    CHECK_PARITY(max_abs_error(h_ref_bias.data(), h_Y_bd.data() + N, N), 1e-4f, "linear_forward_batched_q6k_fp16 (FP32 row 1)");

    // FP16 activations
    std::vector<uint16_t> h_x_fp16(K);
    for (int i = 0; i < K; ++i) h_x_fp16[i] = brotensor::fp32_to_fp16_bits(h_x[i]);
    Tensor x_fp16 = Tensor::from_host_fp16_on(Device::CPU, h_x_fp16.data(), K, 1);
    Tensor y_fp16;
    brotensor::linear_forward_q6k_fp16(W_q6k, nullptr, x_fp16, y_fp16);
    if (y_fp16.dtype != Dtype::FP16) {
        std::cerr << "  [FAIL] linear_forward_q6k_fp16 did not default y.dtype to FP16\n";
        ++g_failures;
    }
    const uint16_t* yp16 = static_cast<const uint16_t*>(y_fp16.data);
    std::vector<float> h_y_fp16(N);
    for (int i = 0; i < N; ++i) h_y_fp16[i] = brotensor::fp16_bits_to_fp32(yp16[i]);
    CHECK_PARITY(max_abs_error(h_ref_nobias.data(), h_y_fp16.data(), N), 5e-2f, "linear_forward_q6k_fp16 (FP16)");
}

// ─── W8A16 (int8w_fp16) CPU Tests ────────────────────────────────────────────

void test_int8w_fp16_cpu() {
    std::cout << "\n================================================================================\n";
    std::cout << "  5. CPU W8A16 (matmul_int8w_fp16 & linear_forward_batched_int8w_fp16) TESTS\n";
    std::cout << "================================================================================\n";

    const int N = 16;
    const int K = 128;

    std::vector<float> h_wf(N * K);
    fill_random(h_wf, 6001, 0.5f);
    std::vector<uint16_t> h_wf16(N * K);
    for (int i = 0; i < N * K; ++i) h_wf16[i] = brotensor::fp32_to_fp16_bits(h_wf[i]);

    std::vector<int8_t> h_wi8(N * K);
    std::vector<float> h_scales(N);
    brotensor::quantize_int8_per_row_host(h_wf16.data(), N, K, h_wi8.data(), h_scales.data());

    Tensor W_int8 = Tensor::empty_on(Device::CPU, N, K, Dtype::INT8);
    std::memcpy(W_int8.data, h_wi8.data(), N * K * sizeof(int8_t));
    Tensor scales = Tensor::from_host_on(Device::CPU, h_scales.data(), N, 1);

    std::vector<float> h_x(K);
    fill_random(h_x, 6002, 0.5f);
    std::vector<float> h_bias(N);
    fill_random(h_bias, 6003, 0.1f);

    std::vector<float> h_ref_nobias(N, 0.0f);
    std::vector<float> h_ref_bias(N, 0.0f);
    for (int r = 0; r < N; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            acc += static_cast<double>(h_wi8[r * K + k]) * static_cast<double>(h_x[k]);
        }
        float v = static_cast<float>(acc) * h_scales[r];
        h_ref_nobias[r] = v;
        h_ref_bias[r] = v + h_bias[r];
    }

    // 1. matmul_int8w_fp16 (FP32)
    Tensor x_f32 = Tensor::from_host_on(Device::CPU, h_x.data(), K, 1);
    Tensor y_f32;
    brotensor::matmul_int8w_fp16(W_int8, scales, x_f32, y_f32);
    if (y_f32.dtype != Dtype::FP32 || y_f32.rows != N || y_f32.cols != 1) {
        std::cerr << "  [FAIL] matmul_int8w_fp16 did not default Y.dtype to FP32\n";
        ++g_failures;
    }
    std::vector<float> h_y_f32 = y_f32.to_host_vector();
    CHECK_PARITY(max_abs_error(h_ref_nobias.data(), h_y_f32.data(), N), 1e-4f, "matmul_int8w_fp16 (FP32)");

    // 2. matmul_int8w_fp16 (FP16)
    std::vector<uint16_t> h_x16(K);
    for (int i = 0; i < K; ++i) h_x16[i] = brotensor::fp32_to_fp16_bits(h_x[i]);
    Tensor x_fp16 = Tensor::from_host_fp16_on(Device::CPU, h_x16.data(), K, 1);
    Tensor y_fp16;
    brotensor::matmul_int8w_fp16(W_int8, scales, x_fp16, y_fp16);
    if (y_fp16.dtype != Dtype::FP16) {
        std::cerr << "  [FAIL] matmul_int8w_fp16 did not default Y.dtype to FP16\n";
        ++g_failures;
    }
    const uint16_t* yp16 = static_cast<const uint16_t*>(y_fp16.data);
    std::vector<float> h_y_fp16(N);
    for (int i = 0; i < N; ++i) h_y_fp16[i] = brotensor::fp16_bits_to_fp32(yp16[i]);
    CHECK_PARITY(max_abs_error(h_ref_nobias.data(), h_y_fp16.data(), N), 2e-2f, "matmul_int8w_fp16 (FP16)");

    // 3. linear_forward_batched_int8w_fp16 (FP32 with bias)
    Tensor X_bd_f32 = Tensor::from_host_on(Device::CPU, h_x.data(), 1, K);
    Tensor bias_f32 = Tensor::from_host_on(Device::CPU, h_bias.data(), N, 1);
    Tensor Y_bd_f32;
    brotensor::linear_forward_batched_int8w_fp16(W_int8, scales, &bias_f32, X_bd_f32, Y_bd_f32);
    if (Y_bd_f32.dtype != Dtype::FP32 || Y_bd_f32.rows != 1 || Y_bd_f32.cols != N) {
        std::cerr << "  [FAIL] linear_forward_batched_int8w_fp16 did not default Y_BD.dtype to FP32\n";
        ++g_failures;
    }
    std::vector<float> h_Y_bd = Y_bd_f32.to_host_vector();
    CHECK_PARITY(max_abs_error(h_ref_bias.data(), h_Y_bd.data(), N), 1e-4f, "linear_forward_batched_int8w_fp16 (FP32)");
    std::cout << "  matmul_int8w_fp16 & linear_forward_batched_int8w_fp16: PASS\n";
}

} // namespace

int main() {
    brotensor::init();
    brotensor::set_default_device(Device::CPU);

    std::cout << "================================================================================\n";
    std::cout << "  BRASS JIT CPU QUANTIZED GEMV (Q8_0, Q4_K, Q6_K, INT8) TEST & BENCHMARK SUITE\n";
    std::cout << "  Threads: " << brotensor::detail::cpu::ThreadPool::instance().num_threads() << "\n";
    std::cout << "================================================================================\n";

#if !BROTENSOR_HAS_BRASS_JIT
    std::cout << "[SKIP] Brass JIT is not enabled in this build.\n";
    return 0;
#else
    test_q8_0_cpu();
    test_q4_k_cpu();
    test_dequant_cpu();
    test_q6_k_cpu();
    test_int8w_fp16_cpu();

    std::cout << "\n================================================================================\n";
    if (g_failures == 0) {
        std::cout << "  [SUCCESS] All Quantized CPU JIT (Q8_0, Q4_K, Q6_K, INT8) tests PASSED!\n";
    } else {
        std::cout << "  [FAILURE] " << g_failures << " tests failed!\n";
    }
    std::cout << "================================================================================\n";

    return g_failures == 0 ? 0 : 1;
#endif
}
