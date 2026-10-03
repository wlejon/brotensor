#pragma once

// Shared by self_attention_bias.cu (the general-bias self-attention) and
// self_attention_rel_pos.cu (SAM's decomposed rel-pos attention): the dtype
// load / store helpers, the FP32 register-tiled GEMM and sardp_gemm (WMMA for
// FP16 / BF16), and the head split / merge kernels both pipelines run.
// Everything is in an anonymous namespace: each including TU gets its own
// copy, as when it all lived in one file.

#include <brotensor/tensor.h>

#include "detail/cuda_check.h"
#include "fp16_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor {
namespace detail::cuda {

namespace {

constexpr int SAB_SM_BLOCK = 256;

__device__ inline float sab_ld(const float& x)         { return x; }
__device__ inline float sab_ld(const __half& x)        { return __half2float(x); }
__device__ inline float sab_ld(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ inline void  sab_st(float& d, float v)         { d = v; }
__device__ inline void  sab_st(__half& d, float v)        { d = __float2half(v); }
__device__ inline void  sab_st(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

template <typename T> struct sardp_dtype;
template <> struct sardp_dtype<float>         { static constexpr ::brotensor::Dtype value = ::brotensor::Dtype::FP32; };
template <> struct sardp_dtype<__half>        { static constexpr ::brotensor::Dtype value = ::brotensor::Dtype::FP16; };
template <> struct sardp_dtype<__nv_bfloat16> { static constexpr ::brotensor::Dtype value = ::brotensor::Dtype::BF16; };

constexpr int SG_BM = 128;  // CTA tile rows
constexpr int SG_BN = 64;   // CTA tile cols
constexpr int SG_BK = 16;   // K chunk
constexpr int SG_TM = 8;    // per-thread micro-tile rows (16 thread rows)
constexpr int SG_TN = 4;    // per-thread micro-tile cols (16 thread cols)

// FP32 register-tiled GEMM: C(M, N) = A(M, K) @ B(N, K)^T (+ optional per-N
// bias). Batched via grid.z at element offsets z*strideA / z*strideB /
// z*strideC. 16x16 threads, each owning an 8x4 micro-tile of the 128x64 CTA
// tile; A/B chunks are staged through shared memory K-major so the inner loop
// is an outer product with broadcast/conflict-light reads.
__global__ void sardp_gemm_f32_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K,
                                      size_t strideA, size_t strideB,
                                      size_t strideC,
                                      const float* __restrict__ bias) {
    __shared__ float As[SG_BK][SG_BM];
    __shared__ float Bs[SG_BK][SG_BN];
    A += blockIdx.z * strideA;
    B += blockIdx.z * strideB;
    C += blockIdx.z * strideC;
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int m0  = blockIdx.y * SG_BM;
    const int n0  = blockIdx.x * SG_BN;

    float acc[SG_TM][SG_TN] = {};
    for (int k0 = 0; k0 < K; k0 += SG_BK) {
        // Stage A (SG_BM x SG_BK) and B (SG_BN x SG_BK) K-major, zero-padding
        // the edges; 16 consecutive K columns per row keep global reads
        // coalesced.
        #pragma unroll
        for (int l = 0; l < (SG_BM * SG_BK) / 256; ++l) {
            const int lin = tid + l * 256;
            const int r   = lin >> 4;
            const int c   = lin & 15;
            const int gm  = m0 + r;
            const int gk  = k0 + c;
            As[c][r] = (gm < M && gk < K) ? A[static_cast<size_t>(gm) * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int l = 0; l < (SG_BN * SG_BK) / 256; ++l) {
            const int lin = tid + l * 256;
            const int r   = lin >> 4;
            const int c   = lin & 15;
            const int gn  = n0 + r;
            const int gk  = k0 + c;
            Bs[c][r] = (gn < N && gk < K) ? B[static_cast<size_t>(gn) * K + gk] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < SG_BK; ++k) {
            float av[SG_TM], bv[SG_TN];
            #pragma unroll
            for (int i = 0; i < SG_TM; ++i) av[i] = As[k][threadIdx.y * SG_TM + i];
            #pragma unroll
            for (int j = 0; j < SG_TN; ++j) bv[j] = Bs[k][threadIdx.x * SG_TN + j];
            #pragma unroll
            for (int i = 0; i < SG_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < SG_TN; ++j) acc[i][j] += av[i] * bv[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < SG_TM; ++i) {
        const int gm = m0 + threadIdx.y * SG_TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < SG_TN; ++j) {
            const int gn = n0 + threadIdx.x * SG_TN + j;
            if (gn >= N) continue;
            float v = acc[i][j];
            if (bias) v += bias[gn];
            C[static_cast<size_t>(gm) * N + gn] = v;
        }
    }
}

// Strided-batched C_b = A_b @ B_b^T (+ shared per-N bias), FP32 accumulation,
// dispatched per dtype: FP32 takes the register-tiled kernel above, FP16/BF16
// the shared WMMA tensor-core matmul.
inline void sardp_gemm(const float* A, const float* B, float* C,
                       int batch, int M, int N, int K,
                       size_t sA, size_t sB, size_t sC, const float* bias) {
    if (batch == 0 || M == 0 || N == 0) return;
    constexpr int kMaxZ = 65535;  // grid.z cap
    const dim3 block(16, 16);
    for (int b0 = 0; b0 < batch; b0 += kMaxZ) {
        const int nb = batch - b0 < kMaxZ ? batch - b0 : kMaxZ;
        dim3 grid((N + SG_BN - 1) / SG_BN, (M + SG_BM - 1) / SG_BM, nb);
        sardp_gemm_f32_kernel<<<grid, block, 0, cur_stream()>>>(
            A + static_cast<size_t>(b0) * sA,
            B + static_cast<size_t>(b0) * sB,
            C + static_cast<size_t>(b0) * sC,
            M, N, K, sA, sB, sC, bias);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}
inline void sardp_gemm(const __half* A, const __half* B, __half* C,
                       int batch, int M, int N, int K,
                       size_t sA, size_t sB, size_t sC, const __half* bias) {
    ::brotensor::fp16_internal::launch_matmul_ABT_batched(
        A, B, C, batch, M, N, K, sA, sB, sC, bias);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}
inline void sardp_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B,
                       __nv_bfloat16* C,
                       int batch, int M, int N, int K,
                       size_t sA, size_t sB, size_t sC,
                       const __nv_bfloat16* bias) {
    ::brotensor::fp16_internal::launch_matmul_ABT_batched(
        A, B, C, batch, M, N, K, sA, sB, sC, bias);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

inline int sardp_flat_grid(long long total) {
    const long long blocks = (total + 255) / 256;
    return static_cast<int>(blocks < (1 << 20) ? blocks : (1 << 20));
}

// Split the row-major (nWin*Lw, D) projection into per-(unit, head) panels
// padded to Lp rows:  Out[(w*H+h)*Lp + i, c] = In[w*Lw + i, h*dh + c], zero
// for i >= Lw. total = batch * Lp * dh, grid-stride, c fastest so reads and
// writes both coalesce.
template <typename T>
__global__ void sardp_split_heads_kernel(const T* __restrict__ In,
                                         T* __restrict__ Out,
                                         int H, int Lw, int Lp, int dh,
                                         long long total) {
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < total; t += step) {
        const int c = static_cast<int>(t % dh);
        const int i = static_cast<int>((t / dh) % Lp);
        const int b = static_cast<int>(t / (static_cast<long long>(dh) * Lp));
        const int w = b / H, h = b % H;
        T v;
        if (i < Lw) {
            v = In[(static_cast<size_t>(w) * Lw + i) * (static_cast<size_t>(H) * dh)
                   + static_cast<size_t>(h) * dh + c];
        } else {
            sab_st(v, 0.0f);
        }
        Out[t] = v;
    }
}

// As above for V, but writing transposed (dh, Lp) panels so A@V can run as an
// A@B^T GEMM:  Out[(w*H+h)*dh + c, j] = In[w*Lw + j, h*dh + c]. j fastest so
// the writes coalesce (the strided reads ride the L2).
template <typename T>
__global__ void sardp_split_heads_t_kernel(const T* __restrict__ In,
                                           T* __restrict__ Out,
                                           int H, int Lw, int Lp, int dh,
                                           long long total) {
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < total; t += step) {
        const int j = static_cast<int>(t % Lp);
        const int c = static_cast<int>((t / Lp) % dh);
        const int b = static_cast<int>(t / (static_cast<long long>(dh) * Lp));
        const int w = b / H, h = b % H;
        T v;
        if (j < Lw) {
            v = In[(static_cast<size_t>(w) * Lw + j) * (static_cast<size_t>(H) * dh)
                   + static_cast<size_t>(h) * dh + c];
        } else {
            sab_st(v, 0.0f);
        }
        Out[t] = v;
    }
}

// Merge the per-(unit, head) Y panels back to row-major (nWin*Lw, D), applying
// the deferred softmax row normalisation:
//   Out[w*Lw + i, h*dh + c] = Y[(w*H+h)*Lp + i, c] / sums[(w*H+h)*Lp + i].
// total = nWin*Lw*D, grid-stride, (h, c) fastest so the writes coalesce.
template <typename T>
__global__ void sardp_merge_heads_kernel(const T* __restrict__ Y,
                                         const float* __restrict__ sums,
                                         T* __restrict__ Out,
                                         int H, int Lw, int Lp, int dh,
                                         long long total) {
    const int D = H * dh;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < total; t += step) {
        const int c2  = static_cast<int>(t % D);
        const long long r = t / D;
        const int i = static_cast<int>(r % Lw);
        const int w = static_cast<int>(r / Lw);
        const int h = c2 / dh, c = c2 % dh;
        const size_t prow = (static_cast<size_t>(w) * H + h) * Lp + i;
        sab_st(Out[t], sab_ld(Y[prow * dh + c]) / sums[prow]);
    }
}

} // namespace

} // namespace detail::cuda
} // namespace brotensor
