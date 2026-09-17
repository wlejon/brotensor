// ─── Brass CUDA Driver JIT Quantized GEMV (Q8_0 & Q4_K) Benchmark & Parity ───
//
// Verifies numerical parity (<= 1e-4 max error) and benchmarks execution latency
// and effective memory bandwidth (GB/s) of Brass JIT quantized GEMV kernels
// on NVIDIA RTX GPUs (sm_89) across decode shapes (M=1 at N x K).

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include "../src/cuda/cuda_jit.h"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <cassert>
#include <cstring>

using brotensor::Tensor;
using brotensor::Device;

namespace {

int g_failures = 0;

#define CHECK_PARITY(err, tol, name) do {                                            \
    if ((err) > (tol)) {                                                              \
        std::printf("  [FAIL] %s: max err = %g > tol = %g\n", (name), (err), (tol));  \
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
double time_cuda_us(Fn&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (static_cast<double>(ms) * 1000.0) / iters;
}

// ─── GGUF Q8_0 CPU Layout & Quant/Dequant ────────────────────────────────────

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

// ─── GGUF Q4_K CPU Layout & Quant/Dequant ────────────────────────────────────

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

    // Pack scales[12]
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

    // Pack qs[128]
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

// ─── Benchmarks ──────────────────────────────────────────────────────────────

void bench_q8_0_gemv() {
    std::cout << "\n─── 1. Fused Q8_0 GEMV (In-Register Dequant + FP32 Accumulation) ────────\n";

    struct Shape { int N; int K; };
    std::vector<Shape> shapes = {
        {4096, 4096},   // Square LLM Decode
        {11008, 4096},  // LLaMA-7B Up/Gate Projection
    };

    for (const auto& sh : shapes) {
        const int N = sh.N;
        const int K = sh.K;
        const int BPR = K / Q8_BLOCK;

        std::vector<float> h_wf(static_cast<size_t>(N) * K);
        fill_random(h_wf, 4001, 0.05f);

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
        fill_random(h_x, 4002, 0.5f);

        std::vector<float> h_ref(N);
        std::vector<float> h_jit(N);

        // Host reference
        for (int r = 0; r < N; ++r) {
            float acc = 0.0f;
            const size_t row_off = static_cast<size_t>(r) * K;
            for (int k = 0; k < K; ++k) {
                acc += h_wd[row_off + k] * h_x[k];
            }
            h_ref[r] = acc;
        }

        // Allocate GPU memory
        const size_t w_bytes = static_cast<size_t>(N) * BPR * Q8_BYTES;
        void* d_w = nullptr;
        float* d_x = nullptr;
        float* d_y = nullptr;
        cudaMalloc(&d_w, w_bytes);
        cudaMalloc(&d_x, K * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));

        cudaMemcpy(d_w, h_wq.data(), w_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), K * sizeof(float), cudaMemcpyHostToDevice);

        // Run JIT
        brotensor::detail::cuda::jit::launch_fused_gemv_q8_0_ptx(
            d_w, d_x, d_y, N, K, nullptr
        );
        cudaDeviceSynchronize();

        cudaMemcpy(h_jit.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        float err = max_abs_error(h_ref.data(), h_jit.data(), N);
        CHECK_PARITY(err, 1e-4f, "fused_gemv_q8_0");

        // Benchmark
        double t_jit_us = time_cuda_us([&]() {
            brotensor::detail::cuda::jit::launch_fused_gemv_q8_0_ptx(
                d_w, d_x, d_y, N, K, nullptr
            );
        });

        const double total_bytes = static_cast<double>(w_bytes + K * sizeof(float) + N * sizeof(float));
        const double bw_gbs = (total_bytes / (t_jit_us * 1e-6)) / 1e9;

        std::cout << "  N=" << std::setw(5) << N << " K=" << std::setw(5) << K
                  << " | Weight: " << std::setw(5) << std::fixed << std::setprecision(1) << (w_bytes / 1e6) << " MB"
                  << " | Latency: " << std::setw(7) << std::fixed << std::setprecision(2) << t_jit_us << " us"
                  << " | Bandwidth: " << std::setw(6) << std::fixed << std::setprecision(1) << bw_gbs << " GB/s"
                  << " | MaxErr: " << std::scientific << std::setprecision(2) << err
                  << std::endl;

        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }
}

void bench_q4_k_gemv() {
    std::cout << "\n─── 2. Fused Q4_K GEMV (In-Register 4-bit Unpack & Affine Scaling) ─────\n";

    struct Shape { int N; int K; };
    std::vector<Shape> shapes = {
        {4096, 4096},   // Square LLM Decode
        {11008, 4096},  // LLaMA-7B Up/Gate Projection
    };

    for (const auto& sh : shapes) {
        const int N = sh.N;
        const int K = sh.K;
        const int BPR = K / Q4K_BLOCK;

        std::vector<float> h_wf(static_cast<size_t>(N) * K);
        fill_random(h_wf, 5001, 0.05f);

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
        fill_random(h_x, 5002, 0.5f);

        std::vector<float> h_ref(N);
        std::vector<float> h_jit(N);

        // Host reference
        for (int r = 0; r < N; ++r) {
            float acc = 0.0f;
            const size_t row_off = static_cast<size_t>(r) * K;
            for (int k = 0; k < K; ++k) {
                acc += h_wd[row_off + k] * h_x[k];
            }
            h_ref[r] = acc;
        }

        // Allocate GPU memory
        const size_t w_bytes = static_cast<size_t>(N) * BPR * Q4K_BYTES;
        void* d_w = nullptr;
        float* d_x = nullptr;
        float* d_y = nullptr;
        cudaMalloc(&d_w, w_bytes);
        cudaMalloc(&d_x, K * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));

        cudaMemcpy(d_w, h_wq.data(), w_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), K * sizeof(float), cudaMemcpyHostToDevice);

        // Run JIT
        brotensor::detail::cuda::jit::launch_fused_gemv_q4_k_ptx(
            d_w, d_x, d_y, N, K, nullptr
        );
        cudaDeviceSynchronize();

        cudaMemcpy(h_jit.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        float err = max_abs_error(h_ref.data(), h_jit.data(), N);
        CHECK_PARITY(err, 1e-4f, "fused_gemv_q4_k");

        // Benchmark
        double t_jit_us = time_cuda_us([&]() {
            brotensor::detail::cuda::jit::launch_fused_gemv_q4_k_ptx(
                d_w, d_x, d_y, N, K, nullptr
            );
        });

        const double total_bytes = static_cast<double>(w_bytes + K * sizeof(float) + N * sizeof(float));
        const double bw_gbs = (total_bytes / (t_jit_us * 1e-6)) / 1e9;

        std::cout << "  N=" << std::setw(5) << N << " K=" << std::setw(5) << K
                  << " | Weight: " << std::setw(5) << std::fixed << std::setprecision(1) << (w_bytes / 1e6) << " MB"
                  << " | Latency: " << std::setw(7) << std::fixed << std::setprecision(2) << t_jit_us << " us"
                  << " | Bandwidth: " << std::setw(6) << std::fixed << std::setprecision(1) << bw_gbs << " GB/s"
                  << " | MaxErr: " << std::scientific << std::setprecision(2) << err
                  << std::endl;

        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }
}

} // namespace

int main() {
    brotensor::init();

    if (!brotensor::is_available(Device::CUDA)) {
        std::cout << "CUDA device is not available — skipping test_cuda_jit_quant\n";
        return 0;
    }

    if (!brotensor::detail::cuda::jit::is_cuda_jit_available()) {
        std::cout << "CUDA Driver JIT is not available on this system. Skipping.\n";
        return 0;
    }

    std::cout << "================================================================================\n";
    std::cout << "  BRASS CUDA DRIVER JIT QUANTIZED GEMV (Q8_0 & Q4_K) BENCHMARK SUITE (RTX SM_89)\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  Device: " << prop.name
              << " (SMs: " << prop.multiProcessorCount
              << ", Global Mem: " << std::fixed << std::setprecision(1) << (prop.totalGlobalMem / 1e9) << " GB)\n";
    std::cout << "================================================================================\n";

    // A kernel the JIT cannot find or launch (a brass launch contract that
    // moved under this file) throws; caught, it is a failure that names the
    // kernel instead of a fast-fail exit with the buffered output lost.
    try {
        bench_q8_0_gemv();
        bench_q4_k_gemv();
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] " << e.what() << "\n";
        ++g_failures;
    }

    std::cout << "================================================================================\n";
    if (g_failures == 0) {
        std::cout << "  [SUCCESS] All Quantized GEMV (Q8_0 & Q4_K) tests and benchmarks PASSED!\n";
    } else {
        std::cout << "  [FAILURE] " << g_failures << " tests failed!\n";
    }
    std::cout << "================================================================================\n";

    return g_failures == 0 ? 0 : 1;
}
