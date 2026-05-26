// Backward of flash_attention_forward_gpu — bare attention core, no
// projection weights. Recompute-based: reproduces the per-head softmax to
// obtain P, then runs the standard FlashAttention-2 backward (dV = P^T·dO,
// dP = dO·V^T, dS = P*(dP - D_q)*inv_sqrt, dQ = dS·K, dK = dS^T·Q) per head
// and packs results back into (Lq, D) / (Lk, D).
//
// Numerically this is the same per-head sweep used inside
// flash_attention_qkvo_backward_gpu (src/cuda/flash_attention.cu) — the
// helper kernels (extract_head_LD, pack_head_LD, extract_head_DL,
// fa_scale_mask_causal_softmax_rows_kernel, fa_dP_kernel, fa_dS_from_P_dP_kernel,
// fa_dVh_kernel, fa_dQh_kernel, fa_dKh_kernel) live in flash_attention.cu.
// We forward-declare them here at namespace scope so we can call them
// without duplicating code. They have internal linkage in flash_attention.cu;
// to keep this file self-contained, we re-define equivalent helpers in our
// own anonymous namespace.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "fp16_internal.cuh"
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brotensor {

void* cuda_current_stream();

namespace {

constexpr int FAB_BLOCK = 128;

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return b;
}

// ───── Per-head extract / pack kernels (mirror flash_attention.cu) ─────────

__global__ void fab_extract_head_LD_kernel(const __half* __restrict__ X,
                                           __half* __restrict__ Y,
                                           int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[l * head_dim + d] = X[l * D + head_off + d];
}

__global__ void fab_extract_head_DL_kernel(const __half* __restrict__ X,
                                           __half* __restrict__ Y,
                                           int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[d * L + l] = X[l * D + head_off + d];
}

__global__ void fab_pack_head_LD_kernel(const __half* __restrict__ Y,
                                        __half* __restrict__ Out,
                                        int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Out[l * D + head_off + d] = Y[l * head_dim + d];
}

// Row-wise scaled, optionally masked, optionally causal softmax. One block
// per query row. Equivalent to fa_scale_mask_causal_softmax_rows_kernel.
__global__ void fab_softmax_rows_kernel(__half* __restrict__ S,
                                        int Lq, int Lk,
                                        float scale,
                                        const float* __restrict__ mask,
                                        int causal) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __half* row = S + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local_max = -1e30f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __half2float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        if (v > local_max) local_max = v;
    }
    ssm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float o = ssm[tid + s];
            if (o > ssm[tid]) ssm[tid] = o;
        }
        __syncthreads();
    }
    const float rmax = ssm[0];
    const bool empty = (rmax <= -1e29f);

    float local_sum = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __half2float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        const float e = empty ? 0.0f : __expf(v - rmax);
        row[k] = __float2half(e);
        local_sum += e;
    }
    ssm[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float rsum = ssm[0];
    const float inv = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        const float e = __half2float(row[k]);
        row[k] = __float2half(e * inv);
    }
}

// dP[q, k] = sum_d dO_h[q, d] * V_h[k, d]   (Lq, Lk), FP32 accumulation
__global__ void fab_dP_kernel(const __half* __restrict__ dOh,
                              const __half* __restrict__ Vh,
                              __half* __restrict__ dP,
                              int Lq, int Lk, int hd) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || k >= Lk) return;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) {
        acc += __half2float(dOh[q * hd + d]) * __half2float(Vh[k * hd + d]);
    }
    dP[q * Lk + k] = __float2half(acc);
}

// In-place dS = P * (dP - D_q) * scale  where D_q = sum_k P[q,k]*dP[q,k].
__global__ void fab_dS_from_P_dP_kernel(__half* __restrict__ P_dS,
                                        const __half* __restrict__ dP,
                                        int Lq, int Lk,
                                        float scale) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __half* prow = P_dS + static_cast<size_t>(q) * static_cast<size_t>(Lk);
    const __half* dprow = dP + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        local += __half2float(prow[k]) * __half2float(dprow[k]);
    }
    ssm[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float Dq = ssm[0];

    for (int k = tid; k < Lk; k += blockDim.x) {
        const float p  = __half2float(prow[k]);
        const float dp = __half2float(dprow[k]);
        prow[k] = __float2half(p * (dp - Dq) * scale);
    }
}

// dV_h[k, d] = sum_q P[q, k] * dO_h[q, d]
__global__ void fab_dVh_kernel(const __half* __restrict__ P,
                               const __half* __restrict__ dOh,
                               __half* __restrict__ dVh,
                               int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += __half2float(P[q * Lk + k]) * __half2float(dOh[q * hd + d]);
    }
    dVh[k * hd + d] = __float2half(acc);
}

// dQ_h[q, d] = sum_k dS[q, k] * K_h[k, d]
__global__ void fab_dQh_kernel(const __half* __restrict__ dS,
                               const __half* __restrict__ Kh,
                               __half* __restrict__ dQh,
                               int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || d >= hd) return;
    float acc = 0.0f;
    for (int k = 0; k < Lk; ++k) {
        acc += __half2float(dS[q * Lk + k]) * __half2float(Kh[k * hd + d]);
    }
    dQh[q * hd + d] = __float2half(acc);
}

// dK_h[k, d] = sum_q dS[q, k] * Q_h[q, d]
__global__ void fab_dKh_kernel(const __half* __restrict__ dS,
                               const __half* __restrict__ Qh,
                               __half* __restrict__ dKh,
                               int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += __half2float(dS[q * Lk + k]) * __half2float(Qh[q * hd + d]);
    }
    dKh[k * hd + d] = __float2half(acc);
}

// ─── BF16 kernels (verbatim copies of the FP16 kernels above, with ──────────
//     __half→__nv_bfloat16 / __half2float→__bfloat162float /
//     __float2half→__float2bfloat16). All real math stays in float. ──────────
//
// fp16_internal::launch_matmul_ABT is FP16-only (tensor cores), so the BF16
// path uses a self-contained naive BF16 matmul (FP32 accumulation) instead.

__global__ void fab_extract_head_LD_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                                __nv_bfloat16* __restrict__ Y,
                                                int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[l * head_dim + d] = X[l * D + head_off + d];
}

__global__ void fab_pack_head_LD_bf16_kernel(const __nv_bfloat16* __restrict__ Y,
                                             __nv_bfloat16* __restrict__ Out,
                                             int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Out[l * D + head_off + d] = Y[l * head_dim + d];
}

// Naive BF16 matmul: C(M, N) = A(M, K) @ B(N, K)^T, FP32 accumulation.
__global__ void fab_matmul_ABT_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                                           const __nv_bfloat16* __restrict__ B,
                                           __nv_bfloat16* __restrict__ C,
                                           int M, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;
    const int m = idx / N;
    const int n = idx % N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(A[m * K + k]) * __bfloat162float(B[n * K + k]);
    }
    C[idx] = __float2bfloat16(acc);
}

inline void launch_matmul_ABT_bf16(const __nv_bfloat16* A,
                                   const __nv_bfloat16* B,
                                   __nv_bfloat16* C, int M, int N, int K,
                                   cudaStream_t stream) {
    if (M == 0 || N == 0) return;
    const int total = M * N;
    const int block = 128;
    const int grid  = (total + block - 1) / block;
    fab_matmul_ABT_bf16_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

__global__ void fab_softmax_rows_bf16_kernel(__nv_bfloat16* __restrict__ S,
                                             int Lq, int Lk,
                                             float scale,
                                             const float* __restrict__ mask,
                                             int causal) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __nv_bfloat16* row = S + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local_max = -1e30f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __bfloat162float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        if (v > local_max) local_max = v;
    }
    ssm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float o = ssm[tid + s];
            if (o > ssm[tid]) ssm[tid] = o;
        }
        __syncthreads();
    }
    const float rmax = ssm[0];
    const bool empty = (rmax <= -1e29f);

    float local_sum = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __bfloat162float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        const float e = empty ? 0.0f : __expf(v - rmax);
        row[k] = __float2bfloat16(e);
        local_sum += e;
    }
    ssm[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float rsum = ssm[0];
    const float inv = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        const float e = __bfloat162float(row[k]);
        row[k] = __float2bfloat16(e * inv);
    }
}

__global__ void fab_dP_bf16_kernel(const __nv_bfloat16* __restrict__ dOh,
                                   const __nv_bfloat16* __restrict__ Vh,
                                   __nv_bfloat16* __restrict__ dP,
                                   int Lq, int Lk, int hd) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || k >= Lk) return;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) {
        acc += __bfloat162float(dOh[q * hd + d]) * __bfloat162float(Vh[k * hd + d]);
    }
    dP[q * Lk + k] = __float2bfloat16(acc);
}

__global__ void fab_dS_from_P_dP_bf16_kernel(__nv_bfloat16* __restrict__ P_dS,
                                             const __nv_bfloat16* __restrict__ dP,
                                             int Lq, int Lk,
                                             float scale) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __nv_bfloat16* prow = P_dS + static_cast<size_t>(q) * static_cast<size_t>(Lk);
    const __nv_bfloat16* dprow = dP + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        local += __bfloat162float(prow[k]) * __bfloat162float(dprow[k]);
    }
    ssm[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float Dq = ssm[0];

    for (int k = tid; k < Lk; k += blockDim.x) {
        const float p  = __bfloat162float(prow[k]);
        const float dp = __bfloat162float(dprow[k]);
        prow[k] = __float2bfloat16(p * (dp - Dq) * scale);
    }
}

__global__ void fab_dVh_bf16_kernel(const __nv_bfloat16* __restrict__ P,
                                    const __nv_bfloat16* __restrict__ dOh,
                                    __nv_bfloat16* __restrict__ dVh,
                                    int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += __bfloat162float(P[q * Lk + k]) * __bfloat162float(dOh[q * hd + d]);
    }
    dVh[k * hd + d] = __float2bfloat16(acc);
}

__global__ void fab_dQh_bf16_kernel(const __nv_bfloat16* __restrict__ dS,
                                    const __nv_bfloat16* __restrict__ Kh,
                                    __nv_bfloat16* __restrict__ dQh,
                                    int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || d >= hd) return;
    float acc = 0.0f;
    for (int k = 0; k < Lk; ++k) {
        acc += __bfloat162float(dS[q * Lk + k]) * __bfloat162float(Kh[k * hd + d]);
    }
    dQh[q * hd + d] = __float2bfloat16(acc);
}

__global__ void fab_dKh_bf16_kernel(const __nv_bfloat16* __restrict__ dS,
                                    const __nv_bfloat16* __restrict__ Qh,
                                    __nv_bfloat16* __restrict__ dKh,
                                    int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += __bfloat162float(dS[q * Lk + k]) * __bfloat162float(Qh[q * hd + d]);
    }
    dKh[k * hd + d] = __float2bfloat16(acc);
}

// ─── FP32 kernels (mirror the BF16 block with no half↔float conversions) ───
//
// Used only by flash_attention_varlen_backward's FP32 path. Naive matmuls,
// scalar softmax — slower than the FP16 tensor-core route but eliminates the
// cast hop for FP32 trainers (per brosoundml's profiling, the cast pairs
// around varlen calls were a measurable cost). FP32 storage throughout.

__global__ void fab_extract_head_LD_fp32_kernel(const float* __restrict__ X,
                                                float* __restrict__ Y,
                                                int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[l * head_dim + d] = X[l * D + head_off + d];
}

__global__ void fab_pack_head_LD_fp32_kernel(const float* __restrict__ Y,
                                             float* __restrict__ Out,
                                             int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Out[l * D + head_off + d] = Y[l * head_dim + d];
}

// Naive FP32 matmul: C(M, N) = A(M, K) @ B(N, K)^T.
__global__ void fab_matmul_ABT_fp32_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;
    const int m = idx / N;
    const int n = idx % N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[n * K + k];
    }
    C[idx] = acc;
}

inline void launch_matmul_ABT_fp32(const float* A, const float* B, float* C,
                                   int M, int N, int K, cudaStream_t stream) {
    if (M == 0 || N == 0) return;
    const int total = M * N;
    const int block = 128;
    const int grid  = (total + block - 1) / block;
    fab_matmul_ABT_fp32_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

__global__ void fab_softmax_rows_fp32_kernel(float* __restrict__ S,
                                             int Lq, int Lk,
                                             float scale,
                                             const float* __restrict__ mask,
                                             int causal) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    float* row = S + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local_max = -1e30f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = row[k] * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        if (v > local_max) local_max = v;
    }
    ssm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float o = ssm[tid + s];
            if (o > ssm[tid]) ssm[tid] = o;
        }
        __syncthreads();
    }
    const float rmax = ssm[0];
    const bool empty = (rmax <= -1e29f);

    float local_sum = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = row[k] * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        const float e = empty ? 0.0f : __expf(v - rmax);
        row[k] = e;
        local_sum += e;
    }
    ssm[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float rsum = ssm[0];
    const float inv = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        row[k] *= inv;
    }
}

__global__ void fab_dP_fp32_kernel(const float* __restrict__ dOh,
                                   const float* __restrict__ Vh,
                                   float* __restrict__ dP,
                                   int Lq, int Lk, int hd) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || k >= Lk) return;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) {
        acc += dOh[q * hd + d] * Vh[k * hd + d];
    }
    dP[q * Lk + k] = acc;
}

__global__ void fab_dS_from_P_dP_fp32_kernel(float* __restrict__ P_dS,
                                             const float* __restrict__ dP,
                                             int Lq, int Lk,
                                             float scale) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    float* prow = P_dS + static_cast<size_t>(q) * static_cast<size_t>(Lk);
    const float* dprow = dP + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        local += prow[k] * dprow[k];
    }
    ssm[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float Dq = ssm[0];

    for (int k = tid; k < Lk; k += blockDim.x) {
        prow[k] = prow[k] * (dprow[k] - Dq) * scale;
    }
}

__global__ void fab_dVh_fp32_kernel(const float* __restrict__ P,
                                    const float* __restrict__ dOh,
                                    float* __restrict__ dVh,
                                    int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += P[q * Lk + k] * dOh[q * hd + d];
    }
    dVh[k * hd + d] = acc;
}

__global__ void fab_dQh_fp32_kernel(const float* __restrict__ dS,
                                    const float* __restrict__ Kh,
                                    float* __restrict__ dQh,
                                    int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || d >= hd) return;
    float acc = 0.0f;
    for (int k = 0; k < Lk; ++k) {
        acc += dS[q * Lk + k] * Kh[k * hd + d];
    }
    dQh[q * hd + d] = acc;
}

__global__ void fab_dKh_fp32_kernel(const float* __restrict__ dS,
                                    const float* __restrict__ Qh,
                                    float* __restrict__ dKh,
                                    int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += dS[q * Lk + k] * Qh[q * hd + d];
    }
    dKh[k * hd + d] = acc;
}

} // namespace

namespace detail::cuda {

// O is consumed by the qkvo backward only via Wo backward; here we don't have
// Wo. O is not actually needed for the recompute path — we re-derive P from
// (Q, K) + softmax. We accept O in the signature to mirror standard
// flash-attn backward APIs and to allow a future caller-supplied-cache path,
// but the parameter is currently unused.
void flash_attention_backward(const Tensor& Q,
                              const Tensor& K,
                              const Tensor& V,
                              const Tensor& O,
                              const Tensor& dO,
                              const float* d_mask,
                              int num_heads,
                              bool causal,
                              Tensor& dQ,
                              Tensor& dK,
                              Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry.

    // Dtype-dispatched: FP16 or BF16; Q/K/V/dO (and dQ/dK/dV) share one dtype.
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO dtype must match");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_backward: Q/K/V shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_backward: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dQ.rows != Lq || dQ.cols != D || dQ.dtype != dt) {
        dQ.resize(Lq, D, dt);
    }
    if (dK.rows != Lk || dK.cols != D || dK.dtype != dt) {
        dK.resize(Lk, D, dt);
    }
    if (dV.rows != Lk || dV.cols != D || dV.dtype != dt) {
        dV.resize(Lk, D, dt);
    }
    dQ.zero();
    dK.zero();
    dV.zero();

    if (Lq == 0 || Lk == 0 || D == 0) return;

    constexpr int CP_BLOCK = 256;
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    if (sm_block > 1024) sm_block = 1024;
    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(hd));

    Tensor Qh = Tensor::empty_on(Device::CUDA, Lq, hd, dt);
    Tensor Kh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);
    Tensor Vh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);
    Tensor dOh = Tensor::empty_on(Device::CUDA, Lq, hd, dt);
    Tensor P = Tensor::empty_on(Device::CUDA, Lq, Lk, dt);
    Tensor dP = Tensor::empty_on(Device::CUDA, Lq, Lk, dt);
    Tensor dQh = Tensor::empty_on(Device::CUDA, Lq, hd, dt);
    Tensor dKh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);
    Tensor dVh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    (void)FAB_BLOCK;

    for (int h = 0; h < num_heads; ++h) {
        const int head_off = h * hd;
        const int total_q = Lq * hd;
        const int total_k = Lk * hd;
        const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);

        if (bf16) {
            // BF16 path — fp16_internal::launch_matmul_ABT is FP16-only, so
            // use the file-local naive BF16 matmul. Math stays in float.
            fab_extract_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Q.data),
                reinterpret_cast<__nv_bfloat16*>(Qh.data),
                Lq, D, head_off, hd);
            fab_extract_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(K.data),
                reinterpret_cast<__nv_bfloat16*>(Kh.data),
                Lk, D, head_off, hd);
            fab_extract_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(V.data),
                reinterpret_cast<__nv_bfloat16*>(Vh.data),
                Lk, D, head_off, hd);
            fab_extract_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(dO.data),
                reinterpret_cast<__nv_bfloat16*>(dOh.data),
                Lq, D, head_off, hd);

            launch_matmul_ABT_bf16(
                reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                reinterpret_cast<__nv_bfloat16*>(P.data),
                Lq, Lk, hd, stream);
            fab_softmax_rows_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(P.data),
                Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);

            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fab_dVh_bf16_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(P.data),
                    reinterpret_cast<const __nv_bfloat16*>(dOh.data),
                    reinterpret_cast<__nv_bfloat16*>(dVh.data),
                    Lq, Lk, hd);
            }
            {
                dim3 block(16, 16);
                dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
                fab_dP_bf16_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dOh.data),
                    reinterpret_cast<const __nv_bfloat16*>(Vh.data),
                    reinterpret_cast<__nv_bfloat16*>(dP.data),
                    Lq, Lk, hd);
            }
            {
                fab_dS_from_P_dP_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(P.data),
                    reinterpret_cast<const __nv_bfloat16*>(dP.data),
                    Lq, Lk, inv_sqrt);
            }
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
                fab_dQh_bf16_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(P.data),
                    reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                    reinterpret_cast<__nv_bfloat16*>(dQh.data),
                    Lq, Lk, hd);
            }
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fab_dKh_bf16_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(P.data),
                    reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                    reinterpret_cast<__nv_bfloat16*>(dKh.data),
                    Lq, Lk, hd);
            }
            fab_pack_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(dQh.data),
                reinterpret_cast<__nv_bfloat16*>(dQ.data),
                Lq, D, head_off, hd);
            fab_pack_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(dKh.data),
                reinterpret_cast<__nv_bfloat16*>(dK.data),
                Lk, D, head_off, hd);
            fab_pack_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(dVh.data),
                reinterpret_cast<__nv_bfloat16*>(dV.data),
                Lk, D, head_off, hd);
            continue;
        }

        // Extract per-head buffers (Q_h, K_h, V_h, dO_h).
        fab_extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(Q.data),
            reinterpret_cast<__half*>(Qh.data),
            Lq, D, head_off, hd);
        fab_extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(K.data),
            reinterpret_cast<__half*>(Kh.data),
            Lk, D, head_off, hd);
        fab_extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(V.data),
            reinterpret_cast<__half*>(Vh.data),
            Lk, D, head_off, hd);
        fab_extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dO.data),
            reinterpret_cast<__half*>(dOh.data),
            Lq, D, head_off, hd);

        // Recompute P: S(Lq, Lk) = Qh · Kh^T then row-softmax(scale, mask, causal).
        fp16_internal::launch_matmul_ABT(
            reinterpret_cast<const __half*>(Qh.data),
            reinterpret_cast<const __half*>(Kh.data),
            reinterpret_cast<__half*>(P.data),
            Lq, Lk, hd);
        {
            fab_softmax_rows_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(P.data),
                Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);
        }

        // dV_h = P^T · dO_h
        {
            dim3 block(16, 16);
            dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
            fab_dVh_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(P.data),
                reinterpret_cast<const __half*>(dOh.data),
                reinterpret_cast<__half*>(dVh.data),
                Lq, Lk, hd);
        }

        // dP = dO_h · V_h^T
        {
            dim3 block(16, 16);
            dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
            fab_dP_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(dOh.data),
                reinterpret_cast<const __half*>(Vh.data),
                reinterpret_cast<__half*>(dP.data),
                Lq, Lk, hd);
        }

        // dS = P * (dP - D_q) * inv_sqrt  (in-place over P)
        {
            fab_dS_from_P_dP_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(P.data),
                reinterpret_cast<const __half*>(dP.data),
                Lq, Lk, inv_sqrt);
        }

        // dQ_h = dS · K_h
        {
            dim3 block(16, 16);
            dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
            fab_dQh_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(P.data),
                reinterpret_cast<const __half*>(Kh.data),
                reinterpret_cast<__half*>(dQh.data),
                Lq, Lk, hd);
        }

        // dK_h = dS^T · Q_h
        {
            dim3 block(16, 16);
            dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
            fab_dKh_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(P.data),
                reinterpret_cast<const __half*>(Qh.data),
                reinterpret_cast<__half*>(dKh.data),
                Lq, Lk, hd);
        }

        // Pack per-head grads back into the (L, D) slot for this head.
        fab_pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dQh.data),
            reinterpret_cast<__half*>(dQ.data),
            Lq, D, head_off, hd);
        fab_pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dKh.data),
            reinterpret_cast<__half*>(dK.data),
            Lk, D, head_off, hd);
        fab_pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dVh.data),
            reinterpret_cast<__half*>(dV.data),
            Lk, D, head_off, hd);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── flash_attention_varlen_backward ────────────────────────────────────────
//
// Packed variable-length backward. Per-sequence offset arithmetic into the
// (total_tokens, D) Q/K/V/dO/dQ/dK/dV buffers, then the same 8-kernel
// per-head sweep as `flash_attention_backward` above (extract → matmul →
// row-softmax → dV, dP, dS, dQ_h, dK_h → pack).
//
// cu_seqlens_q/k are DEVICE pointers (matching the varlen forward); we copy
// them host-side once per call (B+1 INT32s) so the host can drive the
// per-sequence loop. B is small for visual / token workloads — a few dozen
// at most — so the D→H copy + B kernel-launch tier overhead is negligible
// next to the per-sequence kernel work.
//
// Scratch (Qh/Kh/Vh/dOh/dQh/dKh/dVh + P + dP) is sized to the observed
// max-per-sequence dims and reused across sequences. The kernels take
// explicit (Lq, Lk) bounds so the unused tail of each scratch buffer is
// never read.
void flash_attention_varlen_backward(const Tensor& Q,
                                     const Tensor& K,
                                     const Tensor& V,
                                     const Tensor& O,
                                     const Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size,
                                     int max_seqlen_q,
                                     int max_seqlen_k,
                                     int num_heads,
                                     int head_dim,
                                     bool causal,
                                     Tensor& dQ,
                                     Tensor& dK,
                                     Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry.

    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO dtype must match");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const bool fp32 = (dt == Dtype::FP32);
    const int total_q = Q.rows;
    const int total_k = K.rows;
    const int D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k) {
        throw std::runtime_error("flash_attention_varlen_backward: shape mismatch");
    }
    if (dO.rows != total_q || dO.cols != D) {
        throw std::runtime_error("flash_attention_varlen_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_varlen_backward: num_heads/head_dim must be positive");
    }
    if (batch_size < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: batch_size must be non-negative");
    }
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) {
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q/k required when batch_size > 0");
    }
    if (max_seqlen_q < 0 || max_seqlen_k < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: max_seqlen_q/k must be non-negative");
    }

    if (dQ.rows != total_q || dQ.cols != D || dQ.dtype != dt) dQ.resize(total_q, D, dt);
    if (dK.rows != total_k || dK.cols != D || dK.dtype != dt) dK.resize(total_k, D, dt);
    if (dV.rows != total_k || dV.cols != D || dV.dtype != dt) dV.resize(total_k, D, dt);
    dQ.zero();
    dK.zero();
    dV.zero();

    if (D == 0 || batch_size == 0) return;
    if (total_q == 0 && total_k == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    // Copy cu_seqlens device→host once. (batch_size+1)*4 bytes — negligible.
    std::vector<int32_t> cq(batch_size + 1);
    std::vector<int32_t> ck(batch_size + 1);
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(cq.data(), cu_seqlens_q,
                                         sizeof(int32_t) * (batch_size + 1),
                                         cudaMemcpyDeviceToHost, stream));
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(ck.data(), cu_seqlens_k,
                                         sizeof(int32_t) * (batch_size + 1),
                                         cudaMemcpyDeviceToHost, stream));
    BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (cq[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[0] must be 0");
    if (ck[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[0] must be 0");
    if (cq[batch_size] != total_q)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[B] != total_tokens_q");
    if (ck[batch_size] != total_k)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[B] != total_tokens_k");

    // Determine effective max-per-sequence for scratch allocation. The caller's
    // max_seqlen_* are advisory; we honour the observed maximum so an
    // under-reported max doesn't cause out-of-bounds writes.
    int max_lq = max_seqlen_q, max_lk = max_seqlen_k;
    for (int b = 0; b < batch_size; ++b) {
        const int Lq_b = cq[b + 1] - cq[b];
        const int Lk_b = ck[b + 1] - ck[b];
        if (Lq_b < 0 || Lk_b < 0)
            throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens must be non-decreasing");
        if (causal && Lq_b != Lk_b)
            throw std::runtime_error("flash_attention_varlen_backward: causal requires per-sequence Lq == Lk");
        if (Lq_b > max_lq) max_lq = Lq_b;
        if (Lk_b > max_lk) max_lk = Lk_b;
    }
    if (max_lq <= 0 || max_lk <= 0) return;  // every sequence empty.

    const int hd = head_dim;
    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(hd));
    constexpr int CP_BLOCK = 256;
    int sm_block = 32;
    while (sm_block < max_lk && sm_block < 1024) sm_block *= 2;
    if (sm_block > 1024) sm_block = 1024;

    // Per-head scratch, max-sized once.
    Tensor Qh  = Tensor::empty_on(Device::CUDA, max_lq, hd, dt);
    Tensor Kh  = Tensor::empty_on(Device::CUDA, max_lk, hd, dt);
    Tensor Vh  = Tensor::empty_on(Device::CUDA, max_lk, hd, dt);
    Tensor dOh = Tensor::empty_on(Device::CUDA, max_lq, hd, dt);
    Tensor P   = Tensor::empty_on(Device::CUDA, max_lq, max_lk, dt);
    Tensor dP  = Tensor::empty_on(Device::CUDA, max_lq, max_lk, dt);
    Tensor dQh = Tensor::empty_on(Device::CUDA, max_lq, hd, dt);
    Tensor dKh = Tensor::empty_on(Device::CUDA, max_lk, hd, dt);
    Tensor dVh = Tensor::empty_on(Device::CUDA, max_lk, hd, dt);

    const std::size_t dtsz = static_cast<std::size_t>(dtype_size_bytes(dt));

    for (int b = 0; b < batch_size; ++b) {
        const int q_beg = cq[b];
        const int k_beg = ck[b];
        const int Lq    = cq[b + 1] - q_beg;
        const int Lk    = ck[b + 1] - k_beg;
        if (Lq == 0 || Lk == 0) continue;  // grad rows already zero.

        const std::size_t q_off = static_cast<std::size_t>(q_beg) * D * dtsz;
        const std::size_t k_off = static_cast<std::size_t>(k_beg) * D * dtsz;

        const void* Qp_b  = static_cast<const char*>(Q.data)  + q_off;
        const void* Kp_b  = static_cast<const char*>(K.data)  + k_off;
        const void* Vp_b  = static_cast<const char*>(V.data)  + k_off;
        const void* dOp_b = static_cast<const char*>(dO.data) + q_off;
        void* dQp_b = static_cast<char*>(dQ.data) + q_off;
        void* dKp_b = static_cast<char*>(dK.data) + k_off;
        void* dVp_b = static_cast<char*>(dV.data) + k_off;

        const std::size_t shmem = static_cast<std::size_t>(sm_block) * sizeof(float);
        const int total_qh = Lq * hd;
        const int total_kh = Lk * hd;

        for (int h = 0; h < num_heads; ++h) {
            const int head_off = h * hd;

            if (bf16) {
                fab_extract_head_LD_bf16_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Qp_b),
                    reinterpret_cast<__nv_bfloat16*>(Qh.data),
                    Lq, D, head_off, hd);
                fab_extract_head_LD_bf16_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Kp_b),
                    reinterpret_cast<__nv_bfloat16*>(Kh.data),
                    Lk, D, head_off, hd);
                fab_extract_head_LD_bf16_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Vp_b),
                    reinterpret_cast<__nv_bfloat16*>(Vh.data),
                    Lk, D, head_off, hd);
                fab_extract_head_LD_bf16_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dOp_b),
                    reinterpret_cast<__nv_bfloat16*>(dOh.data),
                    Lq, D, head_off, hd);

                launch_matmul_ABT_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                    reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                    reinterpret_cast<__nv_bfloat16*>(P.data),
                    Lq, Lk, hd, stream);
                fab_softmax_rows_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(P.data),
                    Lq, Lk, inv_sqrt, /*mask=*/nullptr, causal ? 1 : 0);

                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                    fab_dVh_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(P.data),
                        reinterpret_cast<const __nv_bfloat16*>(dOh.data),
                        reinterpret_cast<__nv_bfloat16*>(dVh.data),
                        Lq, Lk, hd);
                }
                {
                    dim3 block(16, 16);
                    dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
                    fab_dP_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(dOh.data),
                        reinterpret_cast<const __nv_bfloat16*>(Vh.data),
                        reinterpret_cast<__nv_bfloat16*>(dP.data),
                        Lq, Lk, hd);
                }
                fab_dS_from_P_dP_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(P.data),
                    reinterpret_cast<const __nv_bfloat16*>(dP.data),
                    Lq, Lk, inv_sqrt);
                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
                    fab_dQh_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(P.data),
                        reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                        reinterpret_cast<__nv_bfloat16*>(dQh.data),
                        Lq, Lk, hd);
                }
                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                    fab_dKh_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(P.data),
                        reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                        reinterpret_cast<__nv_bfloat16*>(dKh.data),
                        Lq, Lk, hd);
                }
                fab_pack_head_LD_bf16_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dQh.data),
                    reinterpret_cast<__nv_bfloat16*>(dQp_b),
                    Lq, D, head_off, hd);
                fab_pack_head_LD_bf16_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dKh.data),
                    reinterpret_cast<__nv_bfloat16*>(dKp_b),
                    Lk, D, head_off, hd);
                fab_pack_head_LD_bf16_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dVh.data),
                    reinterpret_cast<__nv_bfloat16*>(dVp_b),
                    Lk, D, head_off, hd);
                continue;
            }

            if (fp32) {
                fab_extract_head_LD_fp32_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(Qp_b),
                    reinterpret_cast<float*>(Qh.data),
                    Lq, D, head_off, hd);
                fab_extract_head_LD_fp32_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(Kp_b),
                    reinterpret_cast<float*>(Kh.data),
                    Lk, D, head_off, hd);
                fab_extract_head_LD_fp32_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(Vp_b),
                    reinterpret_cast<float*>(Vh.data),
                    Lk, D, head_off, hd);
                fab_extract_head_LD_fp32_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(dOp_b),
                    reinterpret_cast<float*>(dOh.data),
                    Lq, D, head_off, hd);

                launch_matmul_ABT_fp32(
                    reinterpret_cast<const float*>(Qh.data),
                    reinterpret_cast<const float*>(Kh.data),
                    reinterpret_cast<float*>(P.data),
                    Lq, Lk, hd, stream);
                fab_softmax_rows_fp32_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<float*>(P.data),
                    Lq, Lk, inv_sqrt, /*mask=*/nullptr, causal ? 1 : 0);

                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                    fab_dVh_fp32_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const float*>(P.data),
                        reinterpret_cast<const float*>(dOh.data),
                        reinterpret_cast<float*>(dVh.data),
                        Lq, Lk, hd);
                }
                {
                    dim3 block(16, 16);
                    dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
                    fab_dP_fp32_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const float*>(dOh.data),
                        reinterpret_cast<const float*>(Vh.data),
                        reinterpret_cast<float*>(dP.data),
                        Lq, Lk, hd);
                }
                fab_dS_from_P_dP_fp32_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<float*>(P.data),
                    reinterpret_cast<const float*>(dP.data),
                    Lq, Lk, inv_sqrt);
                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
                    fab_dQh_fp32_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const float*>(P.data),
                        reinterpret_cast<const float*>(Kh.data),
                        reinterpret_cast<float*>(dQh.data),
                        Lq, Lk, hd);
                }
                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                    fab_dKh_fp32_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const float*>(P.data),
                        reinterpret_cast<const float*>(Qh.data),
                        reinterpret_cast<float*>(dKh.data),
                        Lq, Lk, hd);
                }
                fab_pack_head_LD_fp32_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(dQh.data),
                    reinterpret_cast<float*>(dQp_b),
                    Lq, D, head_off, hd);
                fab_pack_head_LD_fp32_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(dKh.data),
                    reinterpret_cast<float*>(dKp_b),
                    Lk, D, head_off, hd);
                fab_pack_head_LD_fp32_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(dVh.data),
                    reinterpret_cast<float*>(dVp_b),
                    Lk, D, head_off, hd);
                continue;
            }

            // FP16 path
            fab_extract_head_LD_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(Qp_b),
                reinterpret_cast<__half*>(Qh.data),
                Lq, D, head_off, hd);
            fab_extract_head_LD_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(Kp_b),
                reinterpret_cast<__half*>(Kh.data),
                Lk, D, head_off, hd);
            fab_extract_head_LD_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(Vp_b),
                reinterpret_cast<__half*>(Vh.data),
                Lk, D, head_off, hd);
            fab_extract_head_LD_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dOp_b),
                reinterpret_cast<__half*>(dOh.data),
                Lq, D, head_off, hd);

            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __half*>(Qh.data),
                reinterpret_cast<const __half*>(Kh.data),
                reinterpret_cast<__half*>(P.data),
                Lq, Lk, hd);
            fab_softmax_rows_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(P.data),
                Lq, Lk, inv_sqrt, /*mask=*/nullptr, causal ? 1 : 0);

            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fab_dVh_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(P.data),
                    reinterpret_cast<const __half*>(dOh.data),
                    reinterpret_cast<__half*>(dVh.data),
                    Lq, Lk, hd);
            }
            {
                dim3 block(16, 16);
                dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
                fab_dP_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(dOh.data),
                    reinterpret_cast<const __half*>(Vh.data),
                    reinterpret_cast<__half*>(dP.data),
                    Lq, Lk, hd);
            }
            fab_dS_from_P_dP_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(P.data),
                reinterpret_cast<const __half*>(dP.data),
                Lq, Lk, inv_sqrt);
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
                fab_dQh_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(P.data),
                    reinterpret_cast<const __half*>(Kh.data),
                    reinterpret_cast<__half*>(dQh.data),
                    Lq, Lk, hd);
            }
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fab_dKh_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(P.data),
                    reinterpret_cast<const __half*>(Qh.data),
                    reinterpret_cast<__half*>(dKh.data),
                    Lq, Lk, hd);
            }
            fab_pack_head_LD_kernel<<<grid_for(total_qh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dQh.data),
                reinterpret_cast<__half*>(dQp_b),
                Lq, D, head_off, hd);
            fab_pack_head_LD_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dKh.data),
                reinterpret_cast<__half*>(dKp_b),
                Lk, D, head_off, hd);
            fab_pack_head_LD_kernel<<<grid_for(total_kh, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dVh.data),
                reinterpret_cast<__half*>(dVp_b),
                Lk, D, head_off, hd);
        }
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda

} // namespace brotensor
