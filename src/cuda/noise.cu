// ─── CUDA counter-based noise generation ────────────────────────────────────
//
// Philox 4x32-10 implementations of randn / rand_uniform / rand_bernoulli /
// randn_truncated. One thread per output element; substream (counter + i) per
// element. Byte-identical Philox construction to the CPU reference
// (src/cpu/sample_logits.cpp, src/cpu/noise.cpp), so a given (key, counter)
// yields the same draws across backends.
//
// All four ops require Y FP32 and pre-sized; the op fills rows*cols elements
// in row-major linear order. See ops.h for the full ABI contract.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int NOISE_BLOCK = 256;

inline int noise_grid(long long n) {
    long long blocks = (n + NOISE_BLOCK - 1) / NOISE_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

__device__ inline void mulhilo32(uint32_t a, uint32_t b,
                                 uint32_t& hi, uint32_t& lo) {
    const uint64_t p = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(p >> 32);
    lo = static_cast<uint32_t>(p);
}

__device__ inline void philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, ctr[0], hi0, lo0);
    mulhilo32(0xCD9E8D57u, ctr[2], hi1, lo1);
    const uint32_t n0 = hi1 ^ ctr[1] ^ key[0];
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ ctr[3] ^ key[1];
    const uint32_t n3 = lo0;
    ctr[0] = n0; ctr[1] = n1; ctr[2] = n2; ctr[3] = n3;
}

__device__ inline void philox4x32(uint64_t key64, uint64_t substream,
                                  uint32_t out[4]) {
    uint32_t key[2] = {
        static_cast<uint32_t>(key64 & 0xFFFFFFFFull),
        static_cast<uint32_t>(key64 >> 32),
    };
    uint32_t ctr[4] = {
        static_cast<uint32_t>(substream & 0xFFFFFFFFull),
        static_cast<uint32_t>(substream >> 32),
        0u, 0u,
    };
    for (int r = 0; r < 10; ++r) {
        philox_round(ctr, key);
        if (r < 9) {
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
        }
    }
    out[0] = ctr[0]; out[1] = ctr[1]; out[2] = ctr[2]; out[3] = ctr[3];
}

__device__ inline float u01_from(uint32_t w) {
    return static_cast<float>(w >> 8) * (1.0f / 16777216.0f);
}

__device__ inline float philox_uniform(uint64_t key64, uint64_t substream) {
    uint32_t ctr[4];
    philox4x32(key64, substream, ctr);
    return u01_from(ctr[0]);
}

// Box-Muller cosine branch on (1 - u1, u2); matches the CPU reference.
__device__ inline float philox_normal(uint64_t key64, uint64_t substream) {
    uint32_t ctr[4];
    philox4x32(key64, substream, ctr);
    const float u1 = 1.0f - u01_from(ctr[0]);
    const float u2 = u01_from(ctr[1]);
    constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
    const float radius = sqrtf(-2.0f * logf(u1));
    const float theta =
        static_cast<float>(kTwoPi * static_cast<double>(u2));
    return radius * cosf(theta);
}

__global__ void k_randn(float* __restrict__ y, std::size_t n,
                        uint64_t key, uint64_t counter) {
    for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
         i < n; i += (std::size_t)blockDim.x * gridDim.x) {
        y[i] = philox_normal(key, counter + i);
    }
}

__global__ void k_rand_uniform(float* __restrict__ y, std::size_t n,
                               uint64_t key, uint64_t counter) {
    for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
         i < n; i += (std::size_t)blockDim.x * gridDim.x) {
        y[i] = philox_uniform(key, counter + i);
    }
}

__global__ void k_rand_bernoulli(float* __restrict__ y, std::size_t n,
                                 float p, uint64_t key, uint64_t counter) {
    for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
         i < n; i += (std::size_t)blockDim.x * gridDim.x) {
        const float u = philox_uniform(key, counter + i);
        y[i] = (u < p) ? 1.0f : 0.0f;
    }
}

__global__ void k_randn_truncated(float* __restrict__ y, std::size_t n,
                                  float lo, float hi,
                                  uint64_t key, uint64_t counter) {
    constexpr int kMaxRetries = 64;
    for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
         i < n; i += (std::size_t)blockDim.x * gridDim.x) {
        float z = 0.0f;
        for (int r = 0; r < kMaxRetries; ++r) {
            const uint64_t sub = counter + i +
                                 static_cast<uint64_t>(r) * n;
            z = philox_normal(key, sub);
            if (z >= lo && z <= hi) break;
        }
        if (z < lo) z = lo;
        if (z > hi) z = hi;
        y[i] = z;
    }
}

inline std::size_t check_y(const char* op, const ::brotensor::Tensor& Y) {
    if (Y.dtype != ::brotensor::Dtype::FP32) fail(op, "Y must be FP32");
    if (Y.rows < 0 || Y.cols < 0) fail(op, "Y has negative dimension");
    const std::size_t n = static_cast<std::size_t>(Y.rows) *
                          static_cast<std::size_t>(Y.cols);
    if (n != 0 && Y.data == nullptr) {
        fail(op, "Y is uncommitted; pre-allocate before calling");
    }
    return n;
}

} // namespace

// ─── randn ──────────────────────────────────────────────────────────────────
void randn(uint64_t key, uint64_t counter, ::brotensor::Tensor& Y) {
    const std::size_t n = check_y("randn", Y);
    if (n == 0) return;
    k_randn<<<noise_grid((long long)n), NOISE_BLOCK, 0, cur_stream()>>>(
        static_cast<float*>(Y.data), n, key, counter);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── rand_uniform ───────────────────────────────────────────────────────────
void rand_uniform(uint64_t key, uint64_t counter, ::brotensor::Tensor& Y) {
    const std::size_t n = check_y("rand_uniform", Y);
    if (n == 0) return;
    k_rand_uniform<<<noise_grid((long long)n), NOISE_BLOCK, 0, cur_stream()>>>(
        static_cast<float*>(Y.data), n, key, counter);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── rand_bernoulli ─────────────────────────────────────────────────────────
void rand_bernoulli(float p, uint64_t key, uint64_t counter,
                    ::brotensor::Tensor& Y) {
    const char* op = "rand_bernoulli";
    if (!(p >= 0.0f && p <= 1.0f)) fail(op, "p must be in [0, 1]");
    const std::size_t n = check_y(op, Y);
    if (n == 0) return;
    k_rand_bernoulli<<<noise_grid((long long)n), NOISE_BLOCK, 0, cur_stream()>>>(
        static_cast<float*>(Y.data), n, p, key, counter);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── randn_truncated ────────────────────────────────────────────────────────
void randn_truncated(float lo, float hi, uint64_t key, uint64_t counter,
                     ::brotensor::Tensor& Y) {
    const char* op = "randn_truncated";
    if (!(lo < hi)) fail(op, "lo must be < hi");
    const std::size_t n = check_y(op, Y);
    if (n == 0) return;
    k_randn_truncated<<<noise_grid((long long)n), NOISE_BLOCK, 0, cur_stream()>>>(
        static_cast<float*>(Y.data), n, lo, hi, key, counter);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────
void fill_cuda_vtable_noise(::brotensor::detail::OpsVTable& v) {
    v.randn            = &randn;
    v.rand_uniform     = &rand_uniform;
    v.rand_bernoulli   = &rand_bernoulli;
    v.randn_truncated  = &randn_truncated;
}

} // namespace brotensor::detail::cuda
