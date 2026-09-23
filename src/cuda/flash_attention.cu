#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include "fp16_internal.cuh"
#include "flash_fused_internal.cuh"
#include "gemm_mma.cuh"
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

// Forward decls of CUDA helpers from sibling files. They all live in
// brotensor::detail::cuda.
namespace detail::cuda {
void linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                 const ::brotensor::Tensor* bias,
                                 const ::brotensor::Tensor& X_BD,
                                 ::brotensor::Tensor& Y_BD);
void linear_backward_batched(const ::brotensor::Tensor& W,
                             const ::brotensor::Tensor& X_BD,
                             const ::brotensor::Tensor& dY_BD,
                             ::brotensor::Tensor& dX_BD,
                             ::brotensor::Tensor& dW,
                             ::brotensor::Tensor& dB);
void linear_forward_batched_int8w_fp16(const ::brotensor::Tensor& W_int8,
                                       const ::brotensor::Tensor& scales,
                                       const ::brotensor::Tensor* bias,
                                       const ::brotensor::Tensor& X_BD,
                                       ::brotensor::Tensor& Y_BD);
// flash_attention_backward.cu
void flash_attention_backward(const ::brotensor::Tensor& Q,
                              const ::brotensor::Tensor& K,
                              const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& O,
                              const ::brotensor::Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              ::brotensor::Tensor& dQ,
                              ::brotensor::Tensor& dK,
                              ::brotensor::Tensor& dV);
} // namespace detail::cuda

void* cuda_current_stream();

namespace {

constexpr int FA_BLOCK = 128;
constexpr int FA_KTILE = 64;

// ─── Per-head fallback kernels (flash_attention_forward, uncovered head_dim) ─

template <typename T> struct fa_cvt;
template <> struct fa_cvt<__half> {
    __device__ static __half from_f32(float v) { return __float2half(v); }
};
template <> struct fa_cvt<__nv_bfloat16> {
    __device__ static __nv_bfloat16 from_f32(float v) { return __float2bfloat16(v); }
};

// Y[l, d] = X[l, head_off + d] for d < hd, row stride ldY >= hd. Columns
// [hd, ldY) are left alone (the caller zeroes them once).
template <typename T>
__global__ void fa_gather_head_kernel(const T* __restrict__ X, T* __restrict__ Y,
                                      int L, int D, int head_off, int hd, int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * hd) return;
    const int l = idx / hd, d = idx % hd;
    Y[static_cast<size_t>(l) * ldY + d] = X[static_cast<size_t>(l) * D + head_off + d];
}

// The transposed gather: Y[d, l] = X[l, head_off + d], row stride ldY >= L.
template <typename T>
__global__ void fa_gather_head_T_kernel(const T* __restrict__ X, T* __restrict__ Y,
                                        int L, int D, int head_off, int hd, int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * hd) return;
    const int l = idx / hd, d = idx % hd;
    Y[static_cast<size_t>(d) * ldY + l] = X[static_cast<size_t>(l) * D + head_off + d];
}

// Out[l, head_off + d] = Y[l, d] for d < hd (Y row stride ldY).
template <typename T>
__global__ void fa_scatter_head_kernel(const T* __restrict__ Y, T* __restrict__ Out,
                                       int L, int D, int head_off, int hd, int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * hd) return;
    const int l = idx / hd, d = idx % hd;
    Out[static_cast<size_t>(l) * D + head_off + d] = Y[static_cast<size_t>(l) * ldY + d];
}

// Block-wide max / sum over blockDim.x (a multiple of 32, <= 1024) threads;
// every thread gets the result. `red` holds 32 floats.
__device__ __forceinline__ float fa_block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : -1e30f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();   // red is reused by the next reduction
    return v;
}
__device__ __forceinline__ float fa_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    return v;
}

// P[q, :] = softmax(S[q, :Lk] * scale, mask) for one query row per block.
// S holds the raw FP32 Q@K^T scores; only the normalised probabilities are
// narrowed to 16 bits (they are the P@V GEMM's operand). Positions with
// mask[k] <= 0.5 drop out; a row with no valid key comes out all zero. Pad
// columns [Lk, ld) are written as exact zeros, so the padded P@V adds nothing.
template <typename T>
__global__ void fa_softmax_rows_f32_kernel(const float* __restrict__ S, T* __restrict__ P,
                                           int Lk, int ld, float scale,
                                           const float* __restrict__ mask) {
    __shared__ float red[32];
    const float* srow = S + static_cast<size_t>(blockIdx.x) * ld;
    T* prow = P + static_cast<size_t>(blockIdx.x) * ld;

    float mx = -1e30f;
    for (int k = threadIdx.x; k < Lk; k += blockDim.x) {
        if (!mask || mask[k] > 0.5f) mx = fmaxf(mx, srow[k] * scale);
    }
    mx = fa_block_max(mx, red);
    const bool empty = mx <= -1e29f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < Lk; k += blockDim.x) {
        if (!mask || mask[k] > 0.5f) sum += __expf(srow[k] * scale - mx);
    }
    sum = fa_block_sum(sum, red);
    const float inv = (!empty && sum > 0.0f) ? 1.0f / sum : 0.0f;

    for (int k = threadIdx.x; k < ld; k += blockDim.x) {
        const bool valid = k < Lk && (!mask || mask[k] > 0.5f);
        prow[k] = fa_cvt<T>::from_f32(valid ? __expf(srow[k] * scale - mx) * inv : 0.0f);
    }
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return b;
}

// Flash-attention-style online-softmax kernel. One block per (q, head) tile.
// Tiles over Lk so the scores row never lives in shared/global memory in full
// — the algorithm carries a running max `m` and running normaliser `l` plus a
// partial output row, rescaling them when a new tile produces a larger max.
//
//   - m_new   = max(m_old, max_t s_t)
//   - alpha   = exp(m_old - m_new)
//   - l_new   = alpha * l_old + sum_t exp(s_t - m_new)
//   - O_new   = alpha * O_old + sum_t exp(s_t - m_new) * V[t]
//
// At the end we divide O by l. Shared memory holds the K/V tile and a small
// reduction scratch; nothing scales with Lk.
__global__ void flash_attention_kernel(
        const __half* __restrict__ Q,    // (Lq, D)
        const __half* __restrict__ K,    // (Lk, D)
        const __half* __restrict__ V,    // (Lk, D)
        const float*  __restrict__ mask, // (Lk,) may be null
        __half* __restrict__ Out,        // (Lq, D)
        int Lq, int Lk, int D, int head_dim,
        int causal) {
    extern __shared__ float s_smem[];
    // Layout: scores[FA_KTILE], red[blockDim.x]
    float* scores = s_smem;
    float* red    = s_smem + FA_KTILE;

    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int head_off = h * head_dim;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    // Each thread keeps a slice of the partial output for `d` indices it owns.
    // We parallelise over d (head_dim) with a strided assignment.
    float run_max = -1e30f;
    float run_sum = 0.0f;

    // Partial output in registers (slice owned by this thread). head_dim up to
    // 128 in practice; pre-compute thread's owned indices on the fly.
    // We store the partial output back to a shared array between tiles to
    // accumulate across threads' Sum-over-tiles step. Actually each thread
    // owns its own d-stripe across the WHOLE tile loop, so the partial output
    // for THIS thread's d's lives in registers / a small local array.
    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    // Per-thread cached Q values for this thread's d slice (used in score
    // computation as well — but Q is accessed by all threads, so we cache
    // each d once for cleanliness).
    // For score computation, every thread does the full head_dim dot product
    // for its assigned k. We keep that approach (already cheap; head_dim ≤
    // 128) so Q is read directly from global.

    for (int k0 = 0; k0 < Lk; k0 += FA_KTILE) {
        // Causal: skip tiles entirely beyond the query's diagonal. For the
        // boundary tile, shrink the effective klen so masked positions don't
        // even enter the score loop.
        if (causal && k0 > q) break;
        int klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;
        if (causal && k0 + klen - 1 > q) klen = q - k0 + 1;

        // 1. Compute scores[t] for t in [0, klen). Each thread strides.
        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += __half2float(Q[q * D + head_off + d]) *
                       __half2float(K[kg * D + head_off + d]);
            }
            float s = dot * inv_sqrt;
            if (mask && mask[kg] <= 0.5f) s = -1e30f;
            scores[t] = s;
        }
        __syncthreads();

        // 2. Find tile max.
        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            __syncthreads();
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;

        // 3. Exponentiate scores against m_new, sum. If m_new is still
        //    -inf (whole-tile mask AND no prior valid scores), force all
        //    exponentiated values to 0 — otherwise exp(-inf - -inf) = exp(0)
        //    would pollute the sum.
        const bool tile_empty = (m_new <= -1e29f);
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) local_sum += scores[t];
        red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const float tile_sum = red[0];

        // 4. Rescale running state. If run_max is still -inf (first non-empty
        // tile), alpha collapses to 0 cleanly.
        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = __expf(run_max - m_new);
        }

        // 5. Update partial output for this thread's d's.
        //    partial[i] = alpha * partial[i] + sum_t scores[t] * V[k0+t, head_off+d_i]
        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] *
                       __half2float(V[(k0 + t) * D + head_off + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;

        __syncthreads();
    }

    // 6. Normalise and write out.
    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * D + head_off + d] = __float2half(partial[slot] * inv);
    }
}

// ─── qkvo-backward helpers ─────────────────────────────────────────────────

// FP16 in-place add: dst[i] += src[i] (FP32 sum, written back as FP16).
__global__ void fa_fp16_add_inplace_kernel(__half* __restrict__ dst,
                                           const __half* __restrict__ src,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

// ─── BF16 kernels ──────────────────────────────────────────────────────────

__global__ void flash_attention_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,    // (Lq, D)
        const __nv_bfloat16* __restrict__ K,    // (Lk, D)
        const __nv_bfloat16* __restrict__ V,    // (Lk, D)
        const float*  __restrict__ mask, // (Lk,) may be null
        __nv_bfloat16* __restrict__ Out,        // (Lq, D)
        int Lq, int Lk, int D, int head_dim,
        int causal) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FA_KTILE;

    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int head_off = h * head_dim;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    float run_max = -1e30f;
    float run_sum = 0.0f;

    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (int k0 = 0; k0 < Lk; k0 += FA_KTILE) {
        if (causal && k0 > q) break;
        int klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;
        if (causal && k0 + klen - 1 > q) klen = q - k0 + 1;

        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += __bfloat162float(Q[q * D + head_off + d]) *
                       __bfloat162float(K[kg * D + head_off + d]);
            }
            float s = dot * inv_sqrt;
            if (mask && mask[kg] <= 0.5f) s = -1e30f;
            scores[t] = s;
        }
        __syncthreads();

        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            __syncthreads();
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;

        const bool tile_empty = (m_new <= -1e29f);
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) local_sum += scores[t];
        red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const float tile_sum = red[0];

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = __expf(run_max - m_new);
        }

        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] *
                       __bfloat162float(V[(k0 + t) * D + head_off + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;

        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * D + head_off + d] = __float2bfloat16(partial[slot] * inv);
    }
}

// BF16 in-place add: dst[i] += src[i] (FP32 sum, written back as BF16).
__global__ void fa_bf16_add_inplace_kernel(__nv_bfloat16* __restrict__ dst,
                                           const __nv_bfloat16* __restrict__ src,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

// ─── BF16 batched-linear backward helpers ───────────────────────────────────
//
// The qkvo projections' forward runs on linear_forward_batched_fp16 (gemm.cu),
// which takes FP16 and BF16 alike: tensor-core GEMM, FP32 accumulation, bias
// and activation added in FP32 ahead of the one narrowing. The backward's
// linear_backward_batched (batched_ops.cu) is FP16-or-FP32 only, so the BF16
// projection grads are carried here:
//   dX_BD = dY·W ; dW += dY^T·X (FP32 scratch) ; dB += colsum(dY).

__global__ void fa_lbb_dx_bf16_kernel(const __nv_bfloat16* __restrict__ W,
                                      const __nv_bfloat16* __restrict__ dY,
                                      __nv_bfloat16* __restrict__ dX,
                                      int B, int out_dim, int in_dim) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= in_dim) return;
    const __nv_bfloat16* dY_row = dY + static_cast<size_t>(b) * out_dim;
    float acc = 0.0f;
    for (int i = 0; i < out_dim; ++i) {
        acc += __bfloat162float(W[static_cast<size_t>(i) * in_dim + j]) *
               __bfloat162float(dY_row[i]);
    }
    dX[static_cast<size_t>(b) * in_dim + j] = __float2bfloat16(acc);
}

__global__ void fa_lbb_dw_bf16_kernel(const __nv_bfloat16* __restrict__ dY,
                                      const __nv_bfloat16* __restrict__ X,
                                      float* __restrict__ dW_scratch,
                                      int B, int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += __bfloat162float(dY[static_cast<size_t>(b) * out_dim + i]) *
               __bfloat162float(X [static_cast<size_t>(b) * in_dim  + j]);
    }
    dW_scratch[static_cast<size_t>(i) * in_dim + j] = acc;
}

__global__ void fa_lbb_db_bf16_kernel(const __nv_bfloat16* __restrict__ dY,
                                      float* __restrict__ dB_scratch,
                                      int B, int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += __bfloat162float(dY[static_cast<size_t>(b) * out_dim + i]);
    }
    dB_scratch[i] = acc;
}

__global__ void fa_add_fp32_into_bf16_kernel(const float* __restrict__ src,
                                             __nv_bfloat16* __restrict__ dst,
                                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
}

} // namespace

namespace detail::cuda {

// ─── BF16 batched-linear backward host wrapper (file-local) ─────────────────
//
// BF16 twin of linear_backward_batched on the FP32-scratch fold kernels above.
// Used by the BF16 path of flash_attention_qkvo_backward.

namespace {

// dX_BD overwritten; dW / dB accumulate (+=), matching linear_backward_batched.
void fa_linear_backward_batched_bf16(const Tensor& W, const Tensor& X_BD,
                                     const Tensor& dY_BD,
                                     Tensor& dX_BD, Tensor& dW, Tensor& dB) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (dX_BD.rows != B || dX_BD.cols != in_dim || dX_BD.dtype != Dtype::BF16) {
        dX_BD.resize(B, in_dim, Dtype::BF16);
    }
    if (B == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    if (in_dim > 0 && out_dim > 0) {
        dim3 block(64, 1);
        // CUDA caps gridDim.y at 65535; chunk B to stay within the limit.
        constexpr int kMaxGridY = 65535;
        const int grid_x = (in_dim + 63) / 64;
        const auto* dY_p = static_cast<const __nv_bfloat16*>(dY_BD.data);
        auto*       dX_p = static_cast<__nv_bfloat16*>(dX_BD.data);
        for (int b0 = 0; b0 < B; b0 += kMaxGridY) {
            const int b_chunk = (B - b0) < kMaxGridY ? (B - b0) : kMaxGridY;
            dim3 grid(grid_x, b_chunk);
            fa_lbb_dx_bf16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(W.data),
                dY_p + static_cast<size_t>(b0) * out_dim,
                dX_p + static_cast<size_t>(b0) * in_dim,
                b_chunk, out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    if (out_dim > 0 && in_dim > 0) {
        const int dw_n = out_dim * in_dim;
        float* d_dw_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dw_scratch),
                                        dw_n * sizeof(float)));
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        fa_lbb_dw_bf16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_BD.data),
            static_cast<const __nv_bfloat16*>(X_BD.data),
            d_dw_scratch, B, out_dim, in_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (dw_n + 255) / 256;
        fa_add_fp32_into_bf16_kernel<<<blocks_fold, 256, 0, stream>>>(
            d_dw_scratch, static_cast<__nv_bfloat16*>(dW.data), dw_n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dw_scratch);
    }
    if (out_dim > 0) {
        float* d_db_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_db_scratch),
                                        out_dim * sizeof(float)));
        const int blocks = (out_dim + 255) / 256;
        fa_lbb_db_bf16_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_BD.data),
            d_db_scratch, B, out_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (out_dim + 255) / 256;
        fa_add_fp32_into_bf16_kernel<<<blocks_fold, 256, 0, stream>>>(
            d_db_scratch, static_cast<__nv_bfloat16*>(dB.data), out_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_db_scratch);
    }
}

// ── Per-head fallback for head_dims the fused kernel does not instantiate ──
// For each head h (non-causal only):
//   1. Gather Qh(Lq, hdp), Kh(Lk_pad, hdp) and Vth(hdp, Lk_pad) out of the
//      interleaved (L, D) inputs.
//   2. S(rows, Lk_pad) = Qh @ Kh^T, stored as FP32 (mma_gemm::launch_f32out).
//   3. P = softmax_row(S * 1/sqrt(hd), mask), FP32 math, P narrowed to T.
//   4. Oh(rows, hdp) = P @ Vth^T (the 16-bit tensor-core GEMM).
//   5. Scatter Oh back into O's slot [h*hd, (h+1)*hd).
//
// The scores never exist at 16-bit precision. Holding the unscaled Q@K^T in
// an FP16 buffer ahead of the scale and max subtraction (what this path used
// to do) put the score's rounding into every exp(): ~1-2% on each probability
// of a peaked softmax in FP16 and ~8x that in BF16, where the fused kernel is
// exact to the output rounding.
//
// Both GEMMs want 8-element row strides (the int4 / cp.async loads), so hd
// pads to hdp = round8(hd) and Lk to Lk_pad = round8(Lk). The pads are zeroed
// once per call and the gathers never write them: a zero pad column of Qh / Kh
// adds nothing to a score, pad rows of Kh give scores the softmax overwrites
// with exact zero probabilities, and zero pad rows / columns of Vth add
// nothing to Oh — so padding does not change the result.
//
// S is FP32 now, so the query axis is processed in chunks that keep S under
// kFaScoreBudget floats (128 MB): an SD VAE mid-block at 1024 px (one head,
// Lq = Lk = 16384) would otherwise want a 1 GB score buffer.
//
// Returns false, having written nothing to O, when the FP32-score GEMM cannot
// run (pre-sm_80); the caller then takes the online-softmax scalar kernel.
constexpr size_t kFaScoreBudget = size_t(32) << 20;

template <typename T>
bool fa_per_head_forward(const T* Q, const T* K, const T* V, const float* mask, T* O,
                         Dtype dt, int Lq, int Lk, int D, int num_heads, int hd,
                         cudaStream_t stream) {
    const int hdp = (hd + 7) & ~7;
    const int Lk_pad = (Lk + 7) & ~7;
    int rows = Lq;
    if (static_cast<size_t>(Lq) * Lk_pad > kFaScoreBudget) {
        rows = static_cast<int>(kFaScoreBudget / Lk_pad) / 128 * 128;
        if (rows < 128) rows = 128;
        if (rows > Lq) rows = Lq;
    }

    Tensor Qh  = Tensor::empty_on(Device::CUDA, Lq, hdp, dt);
    Tensor Kh  = Tensor::empty_on(Device::CUDA, Lk_pad, hdp, dt);
    Tensor Vth = Tensor::empty_on(Device::CUDA, hdp, Lk_pad, dt);
    Tensor Oh  = Tensor::empty_on(Device::CUDA, Lq, hdp, dt);
    Tensor S   = Tensor::empty_on(Device::CUDA, rows, Lk_pad, Dtype::FP32);
    Tensor P   = Tensor::empty_on(Device::CUDA, rows, Lk_pad, dt);
    T* qh = static_cast<T*>(Qh.data);
    T* kh = static_cast<T*>(Kh.data);
    T* vth = static_cast<T*>(Vth.data);
    T* oh = static_cast<T*>(Oh.data);
    float* s = static_cast<float*>(S.data);
    T* p = static_cast<T*>(P.data);

    if (hdp != hd) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(qh, 0, sizeof(T) * size_t(Lq) * hdp, stream));
    }
    if (hdp != hd || Lk_pad != Lk) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(kh, 0, sizeof(T) * size_t(Lk_pad) * hdp, stream));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(vth, 0, sizeof(T) * size_t(hdp) * Lk_pad, stream));
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(hd));
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    constexpr int CP = 256;

    for (int h = 0; h < num_heads; ++h) {
        const int off = h * hd;
        fa_gather_head_kernel<T><<<grid_for(Lq * hd, CP), CP, 0, stream>>>(Q, qh, Lq, D, off, hd, hdp);
        fa_gather_head_kernel<T><<<grid_for(Lk * hd, CP), CP, 0, stream>>>(K, kh, Lk, D, off, hd, hdp);
        fa_gather_head_T_kernel<T><<<grid_for(Lk * hd, CP), CP, 0, stream>>>(V, vth, Lk, D, off, hd, Lk_pad);
        for (int q0 = 0; q0 < Lq; q0 += rows) {
            const int m = Lq - q0 < rows ? Lq - q0 : rows;
            if (!detail::cuda::mma_gemm::launch_f32out(qh + size_t(q0) * hdp, kh, s, m, Lk_pad, hdp,
                                                       stream)) {
                return false;   // same answer for every call: nothing written to O yet
            }
            fa_softmax_rows_f32_kernel<T><<<m, sm_block, 0, stream>>>(s, p, Lk, Lk_pad, scale, mask);
            fp16_internal::launch_matmul_ABT(p, vth, oh + size_t(q0) * hdp, m, hdp, Lk_pad);
        }
        fa_scatter_head_kernel<T><<<grid_for(Lq * hd, CP), CP, 0, stream>>>(oh, O, Lq, D, off, hd, hdp);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

void flash_attention_forward(const Tensor& Q,
                             const Tensor& K,
                             const Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             Tensor& O) {
    // Dtype-dispatched: FP16 or BF16. All Q/K/V (and O) must share one dtype.
    // BF16 exists so brodiffusion can run bf16 attention; FP16 behaviour is
    // kept byte-identical.
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_forward: Q, K, V must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("flash_attention_forward: Q, K, V dtype must match");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_forward: causal requires Lq == Lk");
    }
    const int head_dim = D / num_heads;
    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    // ── Fused FlashAttention-2 path ───────────────────────────────────────
    // Tiled online-softmax WMMA kernel reading the interleaved (L, D) layout
    // directly: no per-head extraction, no (Lq, Lk) score materialisation.
    // Covers the instantiated head_dims (see flash_fused::supported); masked
    // and unmasked, causal and not, FP16 and BF16. Everything else falls
    // through to the paths below.
    if (flash_fused::supported(head_dim)) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        if (bf16) {
            flash_fused::launch(
                reinterpret_cast<const __nv_bfloat16*>(Q.data),
                reinterpret_cast<const __nv_bfloat16*>(K.data),
                reinterpret_cast<const __nv_bfloat16*>(V.data),
                d_mask,
                reinterpret_cast<__nv_bfloat16*>(O.data),
                Lq, Lk, D, num_heads, head_dim, causal, stream);
        } else {
            flash_fused::launch(
                reinterpret_cast<const __half*>(Q.data),
                reinterpret_cast<const __half*>(K.data),
                reinterpret_cast<const __half*>(V.data),
                d_mask,
                reinterpret_cast<__half*>(O.data),
                Lq, Lk, D, num_heads, head_dim, causal, stream);
        }
        return;
    }

    // Non-causal, any other head_dim: the per-head tensor-core pipeline, with
    // the scores and the softmax in FP32 (see fa_per_head_forward).
    if (!causal) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        const bool done = bf16
            ? fa_per_head_forward(reinterpret_cast<const __nv_bfloat16*>(Q.data),
                                  reinterpret_cast<const __nv_bfloat16*>(K.data),
                                  reinterpret_cast<const __nv_bfloat16*>(V.data), d_mask,
                                  reinterpret_cast<__nv_bfloat16*>(O.data), dt,
                                  Lq, Lk, D, num_heads, head_dim, stream)
            : fa_per_head_forward(reinterpret_cast<const __half*>(Q.data),
                                  reinterpret_cast<const __half*>(K.data),
                                  reinterpret_cast<const __half*>(V.data), d_mask,
                                  reinterpret_cast<__half*>(O.data), dt,
                                  Lq, Lk, D, num_heads, head_dim, stream);
        if (done) return;
    }

    // Causal masking outside the fused kernel's instantiated head_dims, and
    // any problem the FP32-score GEMM cannot take (pre-sm_80), run the
    // online-softmax flash kernel: FP32 scores, no (Lq, Lk) buffer.
    {
        const size_t shmem = (static_cast<size_t>(FA_KTILE) + FA_BLOCK) * sizeof(float);
        // head_dim parallelisation in the kernel uses up to 8 d-slots/thread.
        if ((head_dim + FA_BLOCK - 1) / FA_BLOCK > 8) {
            throw std::runtime_error("flash_attention_forward: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
        }
        dim3 grid(Lq, num_heads, 1);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        if (bf16) {
            flash_attention_bf16_kernel<<<grid, FA_BLOCK, shmem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Q.data),
                reinterpret_cast<const __nv_bfloat16*>(K.data),
                reinterpret_cast<const __nv_bfloat16*>(V.data),
                d_mask,
                reinterpret_cast<__nv_bfloat16*>(O.data),
                Lq, Lk, D, head_dim,
                causal ? 1 : 0);
        } else {
            flash_attention_kernel<<<grid, FA_BLOCK, shmem, stream>>>(
                reinterpret_cast<const __half*>(Q.data),
                reinterpret_cast<const __half*>(K.data),
                reinterpret_cast<const __half*>(V.data),
                d_mask,
                reinterpret_cast<__half*>(O.data),
                Lq, Lk, D, head_dim,
                causal ? 1 : 0);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// Project (ctx → K, ctx → V) using the same linear_forward_batched_fp16
// call as flash_attention_qkvo_forward, producing (Lk, D) FP16 buffers
// in the exact layout flash_attention_forward consumes.
void flash_attention_project_kv(const Tensor& ctx,
                                    const Tensor& Wk, const Tensor* bk,
                                    const Tensor& Wv, const Tensor* bv,
                                    Tensor& K_out,
                                    Tensor& V_out) {
    const Dtype dt = ctx.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_project_kv: tensors must be FP16 or BF16");
    }
    if (Wk.dtype != dt || Wv.dtype != dt ||
        (bk && bk->dtype != dt) || (bv && bv->dtype != dt)) {
        throw std::runtime_error("flash_attention_project_kv: dtype mismatch");
    }
    const int Lk = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D = Wk.rows;
    if (Wk.cols != D_ctx || Wv.rows != D || Wv.cols != D_ctx) {
        throw std::runtime_error("flash_attention_project_kv: Wk/Wv shape mismatch");
    }
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != dt) {
        K_out.resize(Lk, D, dt);
    }
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != dt) {
        V_out.resize(Lk, D, dt);
    }
    if (Lk == 0 || D == 0) return;
    linear_forward_batched_fp16(Wk, bk, ctx, K_out);
    linear_forward_batched_fp16(Wv, bv, ctx, V_out);
}

// Core attention with caller-supplied K/V (pre-projected). Projects X → Q
// with Wq/bq, runs the tiled attention core, then applies Wo/bo. This is
// the same composition flash_attention_qkvo_forward uses on its cached
// path; both entry points delegate here so numerics are bitwise-identical.
void flash_attention_q_with_kv_cached_forward(const Tensor& X,
                                                  const Tensor& K,
                                                  const Tensor& V,
                                                  const Tensor& Wq, const Tensor* bq,
                                                  const Tensor& Wo, const Tensor* bo,
                                                  const float* d_mask,
                                                  int num_heads,
                                                  bool causal,
                                                  Tensor& O) {
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: tensors must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt || Wq.dtype != dt || Wo.dtype != dt ||
        (bq && bq->dtype != dt) || (bo && bo->dtype != dt)) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: dtype mismatch");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: K/V shape mismatch");
    }
    if (Wq.rows != D || Wq.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: Wq/Wo shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Qp = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor Op = Tensor::empty_on(Device::CUDA, Lq, D, dt);

    linear_forward_batched_fp16(Wq, bq, X, Qp);
    flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
    linear_forward_batched_fp16(Wo, bo, Op, O);
}

// Variant that fuses Q/K/V/O projections at the boundary. Delegates each
// projection to linear_forward_batched_fp16 so optional biases are
// folded in. Ctx==nullptr means self-attention (Ctx = X).
void flash_attention_qkvo_forward(const Tensor& X,
                                      const Tensor* Ctx,
                                      const Tensor& Wq, const Tensor* bq,
                                      const Tensor& Wk, const Tensor* bk,
                                      const Tensor& Wv, const Tensor* bv,
                                      const Tensor& Wo, const Tensor* bo,
                                      const float* d_mask,
                                      int num_heads,
                                      bool causal,
                                      Tensor& O) {
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_qkvo_forward: tensors must be FP16 or BF16");
    }
    if (Wq.dtype != dt || Wk.dtype != dt || Wv.dtype != dt || Wo.dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_forward: weight dtype must match X");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    if (Ctx && Ctx->dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_forward: Ctx dtype must match X");
    }
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_forward: num_heads must divide D");
    }

    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    // Compose via the two new helpers — keeps the order of CUDA ops identical
    // to the pre-refactor code (linear_Wk, linear_Wv, linear_Wq, attention,
    // linear_Wo would actually be reordered; we mirror that by inlining Q+attn
    // into the cached helper and projecting K/V here first). Both helpers are
    // dtype-dispatched, so BF16 flows through transparently.
    Tensor Kp = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    Tensor Vp = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    flash_attention_project_kv(kv_src, Wk, bk, Wv, bv, Kp, Vp);
    flash_attention_q_with_kv_cached_forward(
        X, Kp, Vp, Wq, bq, Wo, bo, d_mask, num_heads, causal, O);
}

// ─── W8A16 variants of the three fused flash-attention ops ─────────────────
//
// Identical composition to the FP16 versions, but every linear projection
// goes through linear_forward_batched_int8w_fp16 instead of
// linear_forward_batched_fp16. Attention core stays FP16 (activations
// are never quantised). Each quantised weight needs its own per-output-row
// FP32 scale tensor (shape (out, 1)). Biases remain FP16.

void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                               const Tensor& Wk_int8,
                                               const Tensor& sk,
                                               const Tensor* bk,
                                               const Tensor& Wv_int8,
                                               const Tensor& sv,
                                               const Tensor* bv,
                                               Tensor& K_out,
                                               Tensor& V_out) {
    if (ctx.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: ctx must be FP16");
    }
    if (Wk_int8.dtype != Dtype::INT8 || Wv_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: Wk/Wv must be INT8");
    }
    const int Lk = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D = Wk_int8.rows;
    if (Wk_int8.cols != D_ctx || Wv_int8.rows != D || Wv_int8.cols != D_ctx) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: Wk/Wv shape mismatch");
    }
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != Dtype::FP16) {
        K_out.resize(Lk, D, Dtype::FP16);
    }
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != Dtype::FP16) {
        V_out.resize(Lk, D, Dtype::FP16);
    }
    if (Lk == 0 || D == 0) return;
    linear_forward_batched_int8w_fp16(Wk_int8, sk, bk, ctx, K_out);
    linear_forward_batched_int8w_fp16(Wv_int8, sv, bv, ctx, V_out);
}

void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X,
                                                     const Tensor& K,
                                                     const Tensor& V,
                                                     const Tensor& Wq_int8,
                                                     const Tensor& sq,
                                                     const Tensor* bq,
                                                     const Tensor& Wo_int8,
                                                     const Tensor& so,
                                                     const Tensor* bo,
                                                     const float* d_mask,
                                                     int num_heads,
                                                     bool causal,
                                                     Tensor& O) {
    if (X.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: X/K/V must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: Wq/Wo must be INT8");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: K/V shape mismatch");
    }
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: Wq/Wo shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Qp = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    Tensor Op = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);

    linear_forward_batched_int8w_fp16(Wq_int8, sq, bq, X, Qp);
    flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
    linear_forward_batched_int8w_fp16(Wo_int8, so, bo, Op, O);
}

void flash_attention_qkvo_int8w_fp16(const Tensor& X,
                                         const Tensor* Ctx,
                                         const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                         const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                         const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                         const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                         const float* d_mask,
                                         int num_heads,
                                         bool causal,
                                         Tensor& O) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: X must be FP16");
    }
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: Ctx must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: all weights must be INT8");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wk_int8.rows != D || Wk_int8.cols != D_ctx ||
        Wv_int8.rows != D || Wv_int8.cols != D_ctx ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Kp = Tensor::empty_on(Device::CUDA, Lk, D, Dtype::FP16);
    Tensor Vp = Tensor::empty_on(Device::CUDA, Lk, D, Dtype::FP16);
    flash_attention_project_kv_int8w_fp16(kv_src,
                                              Wk_int8, sk, bk,
                                              Wv_int8, sv, bv,
                                              Kp, Vp);
    flash_attention_q_with_kv_cached_int8w_fp16(
        X, Kp, Vp,
        Wq_int8, sq, bq,
        Wo_int8, so, bo,
        d_mask, num_heads, causal, O);
}

// ─── Recompute-style FP16 / BF16 backward ──────────────────────────────────
//
// Strategy: re-run the forward to reconstruct O_attn (post-attention, pre-Wo),
// then reverse each stage.
//   1. Re-project X→Q, Ctx→K, Ctx→V via linear_forward_batched_fp16
//      (bit-identical to the forward).
//   2. O_attn (Lq, D) = flash_attention_forward(Q, K, V).
//   3. Wo+bo backward: dO_attn = dO·Wo, dWo += dO^T·O_attn,
//      dbo += colsum(dO). linear_backward_batched handles all three
//      with FP16 storage + FP32 scratch.
//   4. dQ, dK, dV = flash_attention_backward(Q, K, V, dO_attn): FP32 scores,
//      softmax, dP and D_q; P and dS narrowed only as GEMM operands.
//   5. Q,K,V-projection backward: linear_backward_batched for each.
//      Self-attn (Ctx=null) accumulates Q/K/V dX contributions; cross-attn
//      sends Q→dX, K+V→dCtx.
void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask,
    int num_heads,
    bool causal,
    const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo) {

    // ── Argument validation (matches forward; backward adds dO/grads). ────
    // Dtype-dispatched: FP16 or BF16; every participating tensor shares it.
    const Dtype dt = X.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_qkvo_backward: tensors must be FP16 or BF16");
    }
    if (dO.dtype != dt || Wq.dtype != dt || Wk.dtype != dt ||
        Wv.dtype != dt || Wo.dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_backward: all tensors must share dtype");
    }
    if (Ctx && Ctx->dtype != dt) {
        throw std::runtime_error("flash_attention_qkvo_backward: Ctx dtype must match X");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const bool self_attn = (Ctx == nullptr);
    if (self_attn) {
        if (dCtx != nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be null when Ctx is null");
        }
    } else {
        if (dCtx == nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be non-null when Ctx is non-null");
        }
    }
    // bias / grad-bias symmetry: callers must match the forward's bias presence
    // exactly on the grad side. Anything else would mean we're either silently
    // discarding a gradient or writing into a buffer the caller didn't provide.
    auto bias_pair_ok = [](const Tensor* b, const Tensor* db) {
        return static_cast<bool>(b) == static_cast<bool>(db);
    };
    if (!bias_pair_ok(bq, dbq) || !bias_pair_ok(bk, dbk) ||
        !bias_pair_ok(bv, dbv) || !bias_pair_ok(bo, dbo)) {
        throw std::runtime_error("flash_attention_qkvo_backward: bias/grad-bias presence mismatch");
    }

    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward: shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_qkvo_backward: causal requires Lq == Lk");
    }

    if (dX.rows != Lq || dX.cols != D || dX.dtype != dt) {
        dX.resize(Lq, D, dt);
    }
    if (!self_attn) {
        if (dCtx->rows != Lk || dCtx->cols != D_ctx || dCtx->dtype != dt) {
            dCtx->resize(Lk, D_ctx, dt);
        }
        dCtx->zero();
    }
    dX.zero();

    if (Lq == 0 || Lk == 0 || D == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    // ── 1. Recompute forward projections (same call as forward). ──────────
    Tensor Q = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor K = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    Tensor V = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    linear_forward_batched_fp16(Wq, bq, X,      Q);
    linear_forward_batched_fp16(Wk, bk, kv_src, K);
    linear_forward_batched_fp16(Wv, bv, kv_src, V);

    // ── 2. Recompute O_attn (Lq, D) = attention(Q, K, V) for Wo's backward. ─
    // The same attention the forward ran (flash_attention_forward: FP32
    // scores and softmax on every path).
    Tensor O_attn = Tensor::empty_on(Device::CUDA, Lq, D, dt);  // post-attn, pre-Wo
    flash_attention_forward(Q, K, V, d_mask, num_heads, causal, O_attn);

    // ── 3. Wo + bo backward via linear_backward_batched. ──────────────
    //   forward: O(Lq, D) = O_attn(Lq, D) @ Wo(D, D)^T + bo
    //   bwd outputs:
    //     dO_attn(Lq, D) = dO @ Wo  (linear_backward_batched's dX_BD)
    //     dWo(D, D)     += dO^T @ O_attn
    //     dbo           += colsum(dO)
    Tensor dO_attn = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    {
        // linear_backward_batched needs a non-null dB even when bias isn't
        // present in the forward. We always have dWo; for bo absent we feed a
        // scratch dB tensor and discard.
        Tensor scratch_db = Tensor::empty_on(Device::CUDA, 0, 0, dt);
        const bool has_bo = (bo != nullptr);
        if (!has_bo) {
            scratch_db.resize(D, 1, dt);
            scratch_db.zero();
        }
        if (bf16) {
            fa_linear_backward_batched_bf16(Wo, O_attn, dO, dO_attn, dWo,
                                            has_bo ? *dbo : scratch_db);
        } else {
            linear_backward_batched(Wo, O_attn, dO, dO_attn, dWo,
                                        has_bo ? *dbo : scratch_db);
        }
    }

    // ── 4. Attention-core backward: dQ, dK, dV from dO_attn. ──────────────
    // flash_attention_backward's pipeline (FP32 scores, softmax, dP and D_q;
    // see flash_attention_backward.cu).
    Tensor dQ = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor dK = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    Tensor dV = Tensor::empty_on(Device::CUDA, Lk, D, dt);
    flash_attention_backward(Q, K, V, O_attn, dO_attn, d_mask, num_heads, causal, dQ, dK, dV);

    // ── 5. Q / K / V projection backward. ─────────────────────────────────
    // forward: Q = X·Wq^T + bq  →  dX_q = dQ·Wq, dWq += dQ^T·X, dbq += colsum(dQ).
    // Same for K, V but with kv_src as the input. We use linear_backward_batched
    // which handles all three with FP16 storage + FP32 accumulator scratch.
    auto run_proj_back = [&](const Tensor& W, const Tensor& In,
                             const Tensor& dOut, Tensor& dIn_out,
                             Tensor& dW_acc, const Tensor* b_fwd,
                             Tensor* db_acc) {
        Tensor scratch_db = Tensor::empty_on(Device::CUDA, 0, 0, dt);
        bool has_b = (b_fwd != nullptr);
        if (!has_b) {
            scratch_db.resize(W.rows, 1, dt);
            scratch_db.zero();
        }
        if (bf16) {
            fa_linear_backward_batched_bf16(W, In, dOut, dIn_out, dW_acc,
                                            has_b ? *db_acc : scratch_db);
        } else {
            linear_backward_batched(W, In, dOut, dIn_out, dW_acc,
                                        has_b ? *db_acc : scratch_db);
        }
    };

    Tensor dX_from_Q = Tensor::empty_on(Device::CUDA, Lq, D, dt);
    Tensor dX_from_K = Tensor::empty_on(Device::CUDA, Lk, D_ctx, dt);
    Tensor dX_from_V = Tensor::empty_on(Device::CUDA, Lk, D_ctx, dt);

    run_proj_back(Wq, X,      dQ, dX_from_Q, dWq, bq, dbq);
    run_proj_back(Wk, kv_src, dK, dX_from_K, dWk, bk, dbk);
    run_proj_back(Wv, kv_src, dV, dX_from_V, dWv, bv, dbv);

    // ── 6. Accumulate dX / dCtx. ───────────────────────────────────────────
    // Self-attn: dX = dQ-path + dK-path + dV-path (all share input X).
    // Cross-attn: dX = dQ-path; dCtx = dK-path + dV-path (kv_src side).
    auto add_into = [&](Tensor& dst, const Tensor& src, int n) {
        const int blocks = (n + 255) / 256;
        if (bf16) {
            fa_bf16_add_inplace_kernel<<<blocks, 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(dst.data),
                reinterpret_cast<const __nv_bfloat16*>(src.data), n);
        } else {
            fa_fp16_add_inplace_kernel<<<blocks, 256, 0, stream>>>(
                reinterpret_cast<__half*>(dst.data),
                reinterpret_cast<const __half*>(src.data), n);
        }
    };
    add_into(dX, dX_from_Q, Lq * D);
    if (self_attn) {
        add_into(dX, dX_from_K, Lq * D);
        add_into(dX, dX_from_V, Lq * D);
    } else {
        add_into(*dCtx, dX_from_K, Lk * D_ctx);
        add_into(*dCtx, dX_from_V, Lk * D_ctx);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── Vtable contribution ───────────────────────────────────────────────────
//
// Wires this cluster's ops into the CUDA OpsVTable. flash_attention_decode
// is owned by src/cuda/kv_cache.cu (different cluster), so it's NOT assigned
// here. flash_attention_backward lives in flash_attention_backward.cu and is
// wired here for cluster locality.

void flash_attention_varlen_backward(const ::brotensor::Tensor& Q,
                                     const ::brotensor::Tensor& K,
                                     const ::brotensor::Tensor& V,
                                     const ::brotensor::Tensor& O,
                                     const ::brotensor::Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size,
                                     int max_seqlen_q,
                                     int max_seqlen_k,
                                     int num_heads,
                                     int head_dim,
                                     bool causal,
                                     ::brotensor::Tensor& dQ,
                                     ::brotensor::Tensor& dK,
                                     ::brotensor::Tensor& dV);

// ─── flash_attention_varlen_forward ────────────────────────────────────────
//
// Packed variable-length flash attention (Qwen3-VL window attention). Q/K/V
// are one big (total_tokens, num_heads*head_dim) tensor each. cu_seqlens_q/k
// are length B+1 INT32 prefix sums on the same device as Q/K/V (no host
// pointer; same convention as `const float* d_mask`).
//
// One CUDA block per (q_global, head). Each block locates its sequence with
// a linear scan over cu_seqlens (B is small for visual workloads — a few
// dozen at most) and bounds its K-tile loop to the sequence's K range. The
// online-softmax math is byte-identical to flash_attention_kernel; only the
// K-row bounds and the causal diagonal (relative to the per-sequence Q
// origin) change.

template <typename T>
__global__ void flash_attention_varlen_kernel(
        const T* __restrict__ Q,             // (total_q, D)
        const T* __restrict__ K,             // (total_k, D)
        const T* __restrict__ V,             // (total_k, D)
        const int* __restrict__ cu_q,        // (B+1)
        const int* __restrict__ cu_k,        // (B+1)
        T* __restrict__ Out,                 // (total_q, D)
        int B, int D, int head_dim,
        int causal) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FA_KTILE;

    const int q_global = blockIdx.x;
    const int h        = blockIdx.y;
    const int tid      = threadIdx.x;
    const int head_off = h * head_dim;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    // Locate sequence b such that cu_q[b] <= q_global < cu_q[b+1].
    int b = 0;
    while (b < B && cu_q[b + 1] <= q_global) ++b;
    if (b >= B) return;
    const int q_beg = cu_q[b];
    const int k_beg = cu_k[b];
    const int k_end = cu_k[b + 1];
    const int Lk    = k_end - k_beg;
    if (Lk <= 0) {
        // No keys — write zeros for this query row.
        for (int d = tid; d < head_dim; d += blockDim.x) {
            Out[q_global * D + head_off + d] = T(0.0f);
        }
        return;
    }
    const int q_local = q_global - q_beg;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (int k0 = 0; k0 < Lk; k0 += FA_KTILE) {
        if (causal && k0 > q_local) break;
        int klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;
        if (causal && k0 + klen - 1 > q_local) klen = q_local - k0 + 1;

        // 1. Scores.
        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k_beg + k0 + t;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<float>(Q[q_global * D + head_off + d]) *
                       static_cast<float>(K[kg * D + head_off + d]);
            }
            scores[t] = dot * inv_sqrt;
        }
        __syncthreads();

        // 2. Tile max.
        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            __syncthreads();
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;

        // 3. Exponentiate, sum.
        const bool tile_empty = (m_new <= -1e29f);
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) local_sum += scores[t];
        red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const float tile_sum = red[0];

        // 4. Rescale.
        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = __expf(run_max - m_new);
        }

        // 5. Update partial output.
        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] *
                       static_cast<float>(V[(k_beg + k0 + t) * D + head_off + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q_global * D + head_off + d] = T(partial[slot] * inv);
    }
}

void flash_attention_varlen_forward(const Tensor& Q,
                                    const Tensor& K,
                                    const Tensor& V,
                                    const int32_t* cu_seqlens_q,
                                    const int32_t* cu_seqlens_k,
                                    int batch_size,
                                    int max_seqlen_q,
                                    int max_seqlen_k,
                                    int num_heads,
                                    int head_dim,
                                    bool causal,
                                    Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("flash_attention_varlen_forward: Q, K, V must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("flash_attention_varlen_forward: Q, K, V dtype must match");
    }
    if (num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_varlen_forward: num_heads/head_dim must be positive");
    }
    const int D = num_heads * head_dim;
    const int total_q = Q.rows;
    const int total_k = K.rows;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k) {
        throw std::runtime_error("flash_attention_varlen_forward: shape mismatch");
    }
    if (batch_size < 0) {
        throw std::runtime_error("flash_attention_varlen_forward: batch_size must be non-negative");
    }
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) {
        throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens_q/k required when batch_size > 0");
    }
    if (max_seqlen_q < 0 || max_seqlen_k < 0) {
        throw std::runtime_error("flash_attention_varlen_forward: max_seqlen_q/k must be non-negative");
    }
    if ((head_dim + FA_BLOCK - 1) / FA_BLOCK > 8) {
        throw std::runtime_error("flash_attention_varlen_forward: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
    }
    if (O.rows != total_q || O.cols != D || O.dtype != dt) {
        O.resize(total_q, D, dt);
    }
    if (total_q == 0 || D == 0 || batch_size == 0) return;
    (void)max_seqlen_q; (void)max_seqlen_k;

    // Single-sequence non-causal FP16/BF16 is exactly flash_attention_forward
    // over the same packed (L, D) layout (the packing invariant fixes
    // cu = [0, total]), and that path runs the WMMA tensor-core GEMMs —
    // ~16x the throughput of the scalar online-softmax kernel below at
    // transformer-encoder shapes (e.g. DINOv3 ViT-H, TripoSplat flow DiT).
    //
    // Every path keeps the scores in FP32, so this is purely a speed choice.
    // The fused FlashAttention-2 kernel wins everywhere it is instantiated.
    // flash_attention_forward's per-head fallback (any other head_dim) pays
    // ~7 launches per head, a ~0.4 ms floor, so it only overtakes the scalar
    // kernel below from about a 1024 x 1024 score matrix up (1.2x at hd 20,
    // 5-23x at L 4096).
    if (batch_size == 1 && !causal && (dt == Dtype::FP16 || dt == Dtype::BF16) &&
        (flash_fused::supported(head_dim) ||
         static_cast<size_t>(Q.rows) * static_cast<size_t>(K.rows) >= (size_t(1) << 20))) {
        flash_attention_forward(Q, K, V, /*d_mask=*/nullptr, num_heads,
                                /*causal=*/false, O);
        return;
    }

    const size_t shmem = (static_cast<size_t>(FA_KTILE) + FA_BLOCK) * sizeof(float);
    dim3 grid(total_q, num_heads, 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    if (dt == Dtype::BF16) {
        flash_attention_varlen_kernel<__nv_bfloat16><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data),
            reinterpret_cast<const __nv_bfloat16*>(K.data),
            reinterpret_cast<const __nv_bfloat16*>(V.data),
            reinterpret_cast<const int*>(cu_seqlens_q),
            reinterpret_cast<const int*>(cu_seqlens_k),
            reinterpret_cast<__nv_bfloat16*>(O.data),
            batch_size, D, head_dim, causal ? 1 : 0);
    } else if (dt == Dtype::FP32) {
        flash_attention_varlen_kernel<float><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const float*>(Q.data),
            reinterpret_cast<const float*>(K.data),
            reinterpret_cast<const float*>(V.data),
            reinterpret_cast<const int*>(cu_seqlens_q),
            reinterpret_cast<const int*>(cu_seqlens_k),
            reinterpret_cast<float*>(O.data),
            batch_size, D, head_dim, causal ? 1 : 0);
    } else {
        flash_attention_varlen_kernel<__half><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __half*>(Q.data),
            reinterpret_cast<const __half*>(K.data),
            reinterpret_cast<const __half*>(V.data),
            reinterpret_cast<const int*>(cu_seqlens_q),
            reinterpret_cast<const int*>(cu_seqlens_k),
            reinterpret_cast<__half*>(O.data),
            batch_size, D, head_dim, causal ? 1 : 0);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── flash_attention_windowed_forward ──────────────────────────────────────
//
// Sliding-window causal self-attention. The Lq queries occupy the last Lq
// positions of a length-Lk causal sequence (q_offset = Lk - Lq): query row r is
// at absolute position aq = r + q_offset and attends keys [lo, aq] with
// lo = max(0, aq-window+1) (window <= 0 => lo = 0, plain causal). Lq == Lk
// (q_offset 0) is self-attention; Lq < Lk is an incremental decode block over a
// K/V cache (Lq == 1 attends the whole cache). One CUDA block per (query, head).
// The online-softmax math is byte-identical to flash_attention_varlen_kernel.
// d_mask is an optional length-Lk device key mask (1 valid / 0 invalid).
template <typename T>
__global__ void flash_attention_windowed_kernel(
        const T* __restrict__ Q,             // (Lq, Dq)   Dq = num_heads*head_dim
        const T* __restrict__ K,             // (Lk, Dkv)  Dkv = n_kv*head_dim
        const T* __restrict__ V,             // (Lk, Dkv)
        const float* __restrict__ mask,      // (Lk) or null
        T* __restrict__ Out,                 // (Lq, Dq)
        int Lk, int Dq, int Dkv, int head_dim, int window, int q_offset,
        int group, int causal) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FA_KTILE;

    const int q          = blockIdx.x;        // query row in [0, Lq)
    const int h          = blockIdx.y;        // query head
    const int tid        = threadIdx.x;
    const int head_off   = h * head_dim;          // Q/Out (Dq-wide)
    const int head_off_kv = (h / group) * head_dim; // K/V (Dkv-wide), GQA group
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    const int aq = q + q_offset;              // absolute causal position
    int lo = 0;
    int k_hi = Lk - 1;
    if (causal) {
        lo = (window > 0) ? (aq - window + 1) : 0;
        if (lo < 0) lo = 0;
        k_hi = aq;
    } else if (window > 0) {
        lo = aq - window / 2;
        if (lo < 0) lo = 0;
        k_hi = aq + window / 2;
        if (k_hi >= Lk) k_hi = Lk - 1;
    }

    float run_max = -1e30f;
    float run_sum = 0.0f;
    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    const int k0_start = (lo / FA_KTILE) * FA_KTILE;   // align down to a tile
    for (int k0 = k0_start; k0 <= k_hi; k0 += FA_KTILE) {
        int klen = FA_KTILE;
        if (k0 + klen - 1 > k_hi) klen = k_hi - k0 + 1;   // causal trim

        // 1. Scores (keys below the window or masked out -> -inf).
        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            float s;
            if (kg < lo || (mask && mask[kg] <= 0.5f)) {
                s = -1e30f;
            } else {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += static_cast<float>(Q[q * Dq + head_off + d]) *
                           static_cast<float>(K[kg * Dkv + head_off_kv + d]);
                }
                s = dot * inv_sqrt;
            }
            scores[t] = s;
        }
        __syncthreads();

        // 2. Tile max.
        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = red[tid + stride];
                if (other > red[tid]) red[tid] = other;
            }
            __syncthreads();
        }
        const float tile_max = red[0];
        const float m_new = (tile_max > run_max) ? tile_max : run_max;

        // 3. Exponentiate, sum.
        const bool tile_empty = (m_new <= -1e29f);
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) local_sum += scores[t];
        red[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        const float tile_sum = red[0];

        // Fully-masked tile (every key below the window or masked off): the
        // running max is unchanged, every score is an exact 0, and step 5
        // would multiply V by it — a provable no-op. Skip it so a fixed-
        // capacity masked cache (rows >= valid length masked off) costs
        // O(valid), not O(capacity), per query.
        if (tile_sum == 0.0f && tile_max <= -1e29f) {
            __syncthreads();
            continue;
        }

        // 4. Rescale.
        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = __expf(run_max - m_new);
        }

        // 5. Update partial output.
        int slot = 0;
        for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                acc += scores[t] *
                       static_cast<float>(V[(k0 + t) * Dkv + head_off_kv + d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++slot) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * Dq + head_off + d] = T(partial[slot] * inv);
    }
}

void flash_attention_windowed_forward(const Tensor& Q,
                                      const Tensor& K,
                                      const Tensor& V,
                                      const float* d_mask,
                                      int num_heads,
                                      int window,
                                      Tensor& O,
                                      bool causal) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32)
        throw std::runtime_error("flash_attention_windowed_forward: Q, K, V must be FP16, BF16, or FP32");
    if (K.dtype != dt || V.dtype != dt)
        throw std::runtime_error("flash_attention_windowed_forward: Q, K, V dtype must match");
    if (num_heads <= 0)
        throw std::runtime_error("flash_attention_windowed_forward: num_heads must be positive");
    const int Lq  = Q.rows;
    const int Lk  = K.rows;
    const int Dq  = Q.cols;        // num_heads * head_dim
    const int Dkv = K.cols;        // n_kv * head_dim (GQA when < Dq)
    if (Dq % num_heads != 0)
        throw std::runtime_error("flash_attention_windowed_forward: num_heads must divide D");
    const int head_dim = Dq / num_heads;
    if (V.cols != Dkv || V.rows != Lk)
        throw std::runtime_error("flash_attention_windowed_forward: shape mismatch");
    if (Dkv == 0 || Dkv % head_dim != 0)
        throw std::runtime_error("flash_attention_windowed_forward: K/V width must be a head_dim multiple");
    const int n_kv = Dkv / head_dim;
    if (num_heads % n_kv != 0)
        throw std::runtime_error("flash_attention_windowed_forward: num_heads must be a multiple of n_kv");
    if (Lk < Lq)
        throw std::runtime_error("flash_attention_windowed_forward: requires Lk >= Lq");
    if ((head_dim + FA_BLOCK - 1) / FA_BLOCK > 8)
        throw std::runtime_error("flash_attention_windowed_forward: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt)
        O.resize(Lq, Dq, dt);
    if (Lq == 0 || Lk == 0 || Dq == 0) return;

    const int q_offset = Lk - Lq;
    const int group    = num_heads / n_kv;
    const int causal_i = causal ? 1 : 0;
    const size_t shmem = (static_cast<size_t>(FA_KTILE) + FA_BLOCK) * sizeof(float);
    dim3 grid(Lq, num_heads, 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    if (dt == Dtype::BF16) {
        flash_attention_windowed_kernel<__nv_bfloat16><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data),
            reinterpret_cast<const __nv_bfloat16*>(K.data),
            reinterpret_cast<const __nv_bfloat16*>(V.data),
            d_mask, reinterpret_cast<__nv_bfloat16*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group, causal_i);
    } else if (dt == Dtype::FP32) {
        flash_attention_windowed_kernel<float><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const float*>(Q.data),
            reinterpret_cast<const float*>(K.data),
            reinterpret_cast<const float*>(V.data),
            d_mask, reinterpret_cast<float*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group, causal_i);
    } else {
        flash_attention_windowed_kernel<__half><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __half*>(Q.data),
            reinterpret_cast<const __half*>(K.data),
            reinterpret_cast<const __half*>(V.data),
            d_mask, reinterpret_cast<__half*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group, causal_i);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── flash_attention_gqa_forward ───────────────────────────────────────────
//
// GQA generalisation of flash_attention_forward, causal OR bidirectional, built
// on the same online-softmax windowed kernel (window disabled). Q is
// num_q_heads-wide, K/V num_kv_heads-wide; causal == false attends every key —
// the bidirectional encoder prefill. FP16 / BF16 / FP32.
void flash_attention_gqa_forward(const Tensor& Q,
                                 const Tensor& K,
                                 const Tensor& V,
                                 const float* d_mask,
                                 int num_q_heads,
                                 int num_kv_heads,
                                 bool causal,
                                 Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32)
        throw std::runtime_error("flash_attention_gqa_forward: Q, K, V must be FP16, BF16, or FP32");
    if (K.dtype != dt || V.dtype != dt)
        throw std::runtime_error("flash_attention_gqa_forward: Q, K, V dtype must match");
    if (num_q_heads <= 0 || num_kv_heads <= 0)
        throw std::runtime_error("flash_attention_gqa_forward: head counts must be positive");
    const int Lq  = Q.rows;
    const int Lk  = K.rows;
    const int Dq  = Q.cols;
    const int Dkv = K.cols;
    if (Dq % num_q_heads != 0)
        throw std::runtime_error("flash_attention_gqa_forward: num_q_heads must divide Q.cols");
    const int head_dim = Dq / num_q_heads;
    if (V.cols != Dkv || V.rows != Lk)
        throw std::runtime_error("flash_attention_gqa_forward: shape mismatch");
    if (Dkv != num_kv_heads * head_dim)
        throw std::runtime_error("flash_attention_gqa_forward: K/V width must be num_kv_heads*head_dim");
    if (num_q_heads % num_kv_heads != 0)
        throw std::runtime_error("flash_attention_gqa_forward: num_kv_heads must divide num_q_heads");
    if (Lk < Lq)
        throw std::runtime_error("flash_attention_gqa_forward: requires Lk >= Lq");
    if (causal && Lq != Lk)
        throw std::runtime_error("flash_attention_gqa_forward: causal requires Lq == Lk");
    if ((head_dim + FA_BLOCK - 1) / FA_BLOCK > 8)
        throw std::runtime_error("flash_attention_gqa_forward: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt)
        O.resize(Lq, Dq, dt);
    if (Lq == 0 || Lk == 0 || Dq == 0) return;

    const int q_offset  = Lk - Lq;
    const int group     = num_q_heads / num_kv_heads;
    const int causal_i  = causal ? 1 : 0;
    const size_t shmem  = (static_cast<size_t>(FA_KTILE) + FA_BLOCK) * sizeof(float);
    dim3 grid(Lq, num_q_heads, 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    if (dt == Dtype::BF16) {
        flash_attention_windowed_kernel<__nv_bfloat16><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data),
            reinterpret_cast<const __nv_bfloat16*>(K.data),
            reinterpret_cast<const __nv_bfloat16*>(V.data),
            d_mask, reinterpret_cast<__nv_bfloat16*>(O.data),
            Lk, Dq, Dkv, head_dim, /*window=*/0, q_offset, group, causal_i);
    } else if (dt == Dtype::FP32) {
        flash_attention_windowed_kernel<float><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const float*>(Q.data),
            reinterpret_cast<const float*>(K.data),
            reinterpret_cast<const float*>(V.data),
            d_mask, reinterpret_cast<float*>(O.data),
            Lk, Dq, Dkv, head_dim, /*window=*/0, q_offset, group, causal_i);
    } else {
        flash_attention_windowed_kernel<__half><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __half*>(Q.data),
            reinterpret_cast<const __half*>(K.data),
            reinterpret_cast<const __half*>(V.data),
            d_mask, reinterpret_cast<__half*>(O.data),
            Lk, Dq, Dkv, head_dim, /*window=*/0, q_offset, group, causal_i);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_flash_attention(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_forward                       = &flash_attention_forward;
    v.flash_attention_gqa_forward                   = &flash_attention_gqa_forward;
    v.flash_attention_windowed_forward              = &flash_attention_windowed_forward;
    v.flash_attention_varlen_forward                = &flash_attention_varlen_forward;
    v.flash_attention_qkvo_forward                  = &flash_attention_qkvo_forward;
    v.flash_attention_qkvo_backward                 = &flash_attention_qkvo_backward;
    v.flash_attention_backward                      = &flash_attention_backward;
    v.flash_attention_varlen_backward               = &flash_attention_varlen_backward;
    v.flash_attention_project_kv                    = &flash_attention_project_kv;
    v.flash_attention_q_with_kv_cached_forward      = &flash_attention_q_with_kv_cached_forward;
    v.flash_attention_qkvo_int8w_fp16               = &flash_attention_qkvo_int8w_fp16;
    v.flash_attention_q_with_kv_cached_int8w_fp16   = &flash_attention_q_with_kv_cached_int8w_fp16;
    v.flash_attention_project_kv_int8w_fp16         = &flash_attention_project_kv_int8w_fp16;
}

} // namespace detail::cuda

} // namespace brotensor
