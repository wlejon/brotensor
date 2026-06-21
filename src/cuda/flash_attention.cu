#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include "fp16_internal.cuh"
#include "flash_fused_internal.cuh"
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
} // namespace detail::cuda

void* cuda_current_stream();

namespace {

constexpr int FA_BLOCK = 128;
constexpr int FA_KTILE = 64;

// ─── Per-head extract / pack-back kernels ──────────────────────────────────
//
// Source X laid out as (L, D) with D = num_heads * head_dim and the head
// dimension contiguous within each row: X[l, h*head_dim + d]. The WMMA
// matmul path wants a contiguous (L, head_dim) view per head.

__global__ void extract_head_LD_kernel(const __half* __restrict__ X,
                                       __half* __restrict__ Y,
                                       int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[l * head_dim + d] = X[l * D + head_off + d];
}

// Extract a single head and TRANSPOSE on the way in: Y has layout
// (head_dim, ldY) so element (d, l) = X[l, head_off + d]. Used to produce a
// (head_dim, L) "B"-style operand for the second GEMM via launch_matmul_ABT.
// ldY >= L is the row stride of Y — the caller may pad it to an 8-element
// multiple so the GEMM keeps its vectorised (int4) loads; pad columns are
// untouched (the caller zeroes them once).
__global__ void extract_head_DL_kernel(const __half* __restrict__ X,
                                       __half* __restrict__ Y,
                                       int L, int D, int head_off, int head_dim,
                                       int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    // Cooperatively write Y[d * ldY + l] = X[l * D + head_off + d]. Choose
    // mapping that gives coalesced loads of X (d innermost in source) and
    // strided writes to Y — strided writes are fine for fp16 throughput here.
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[d * ldY + l] = X[l * D + head_off + d];
}

// Inverse of extract_head_LD: write a per-head (Lq, head_dim) block back
// into the (Lq, D) output at column slot [head_off, head_off+head_dim).
__global__ void pack_head_LD_kernel(const __half* __restrict__ Y,
                                    __half* __restrict__ Out,
                                    int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Out[l * D + head_off + d] = Y[l * head_dim + d];
}

// Row-wise softmax over S(Lq, Lk) with a scalar scale (1/sqrt(head_dim))
// and optional Lk-shaped float mask: positions with mask[k] <= 0.5 are
// dropped (score forced to -inf). One block per query row, blockDim
// chosen by the launcher. ldS >= Lk is the row stride; pad columns
// [Lk, ldS) are written as exact zeros so a downstream GEMM over the
// padded width adds nothing.
__global__ void scale_mask_softmax_rows_kernel(__half* __restrict__ S,
                                               int Lq, int Lk,
                                               float scale,
                                               const float* __restrict__ mask,
                                               int ldS) {
    extern __shared__ float ssm[];  // size = blockDim.x
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __half* row = S + static_cast<size_t>(q) * static_cast<size_t>(ldS);
    for (int k = Lk + tid; k < ldS; k += blockDim.x) row[k] = __float2half(0.0f);

    // 1. find row max (with scale and mask applied).
    float local_max = -1e30f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __half2float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
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
    // All threads must finish reading ssm[0] before any thread writes ssm[tid]
    // below; otherwise warps that race ahead clobber ssm[0] with local_sum
    // before slower warps load rmax (caught by compute-sanitizer racecheck,
    // manifested as cross-process non-determinism in cross-attn outputs).
    __syncthreads();

    // 2. exponentiate, accumulate sum.
    float local_sum = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __half2float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
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

    // 3. normalise.
    for (int k = tid; k < Lk; k += blockDim.x) {
        const float e = __half2float(row[k]);
        row[k] = __float2half(e * inv);
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

// ─── Backward-helper kernels ───────────────────────────────────────────────
//
// All of these run on per-head buffers (Q_h, K_h, V_h of shape (L, hd)) and
// the (Lq, Lk) probability matrix `P` re-derived from the forward math by a
// scaled, optionally masked / causal row-softmax. The forward already
// materialises an (Lq, Lk) buffer per head (`S` in flash_attention_forward's
// WMMA path); the backward sweeps that same buffer twice — once to compute
// the per-row `D_q = Σ_k P[q,k] · dP[q,k]`, once to scatter dQ/dK/dV — so we
// don't add a fundamentally new memory cost. Causal/mask are applied during
// the softmax recompute, so downstream kernels see P[q,k]=0 at invalid (q,k)
// pairs and naturally produce zero contributions to dV/dK/dQ at those
// positions.

// Row-wise softmax with optional Lk-shaped mask AND optional causal masking.
// One block per query row. Same algorithm as scale_mask_softmax_rows_kernel
// but with an extra "k > q ⇒ -inf" clause when `causal != 0`. Used by the
// backward recompute regardless of the forward path the user invoked; for
// correctness this only needs to match the forward's P to FP16 tolerance.
__global__ void fa_scale_mask_causal_softmax_rows_kernel(__half* __restrict__ S,
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

// dP[q, k] = sum_d dO_attn_h[q, d] * V_h[k, d]   (per-head)
//   dO_attn_h: (Lq, hd) FP16
//   V_h:       (Lk, hd) FP16
//   dP:        (Lq, Lk) FP16 (overwritten)
// One thread per (q, k). FP32 accumulation.
__global__ void fa_dP_kernel(const __half* __restrict__ dOh,
                             const __half* __restrict__ Vh,
                             __half* __restrict__ dP,
                             int Lq, int Lk, int hd) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || k >= Lk) return;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) {
        acc += __half2float(dOh[q * hd + d]) *
               __half2float(Vh[k * hd + d]);
    }
    dP[q * Lk + k] = __float2half(acc);
}

// In-place dS overwrite of P: dS[q, k] = P[q, k] * (dP[q, k] - D_q) * scale
// where D_q = sum_k P[q, k] * dP[q, k]. One block per query row.
// Reads P and dP, writes dS over P (we don't need P after this kernel for
// the dV path — dV is computed BEFORE this transform — so reusing P's
// memory is safe). `scale` folds in the 1/sqrt(hd) factor that was applied
// when forming S.
__global__ void fa_dS_from_P_dP_kernel(__half* __restrict__ P_dS,
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

// dV_h[k, d] += sum_q P[q, k] * dO_attn_h[q, d]   (overwrite, not accumulate;
// we own the buffer for this head's pass).
__global__ void fa_dVh_kernel(const __half* __restrict__ P,
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
__global__ void fa_dQh_kernel(const __half* __restrict__ dS,
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
__global__ void fa_dKh_kernel(const __half* __restrict__ dS,
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

// FP16 in-place add: dst[i] += src[i] (FP32 sum, written back as FP16).
__global__ void fa_fp16_add_inplace_kernel(__half* __restrict__ dst,
                                           const __half* __restrict__ src,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

// ─── BF16 kernels (verbatim copies of the FP16 kernels above, with ──────────
//     __half→__nv_bfloat16 / __half2float→__bfloat162float /
//     __float2half→__float2bfloat16). All real math stays in float. ──────────
//
// The non-causal forward/backward WMMA path uses fp16_internal::launch_matmul_ABT,
// which is FP16-only (tensor cores). BF16 cannot use that kernel, so this file
// carries its own self-contained naive BF16 matmul (matmul_ABT_bf16_kernel)
// with FP32 accumulation — numerically equivalent to the FP16 naive fallback.

__global__ void extract_head_LD_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                            __nv_bfloat16* __restrict__ Y,
                                            int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[l * head_dim + d] = X[l * D + head_off + d];
}

__global__ void extract_head_DL_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                            __nv_bfloat16* __restrict__ Y,
                                            int L, int D, int head_off, int head_dim,
                                            int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[d * ldY + l] = X[l * D + head_off + d];
}

__global__ void pack_head_LD_bf16_kernel(const __nv_bfloat16* __restrict__ Y,
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
// Same contract as fp16_internal::launch_matmul_ABT; one thread per output.
__global__ void matmul_ABT_bf16_kernel(const __nv_bfloat16* __restrict__ A,
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
                                   __nv_bfloat16* C, int M, int N, int K) {
    if (M == 0 || N == 0) return;
    const int total = M * N;
    const int block = 128;
    const int grid  = (total + block - 1) / block;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    matmul_ABT_bf16_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

__global__ void scale_mask_softmax_rows_bf16_kernel(__nv_bfloat16* __restrict__ S,
                                                    int Lq, int Lk,
                                                    float scale,
                                                    const float* __restrict__ mask,
                                                    int ldS) {
    extern __shared__ float ssm[];  // size = blockDim.x
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __nv_bfloat16* row = S + static_cast<size_t>(q) * static_cast<size_t>(ldS);
    for (int k = Lk + tid; k < ldS; k += blockDim.x) row[k] = __float2bfloat16(0.0f);

    float local_max = -1e30f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __bfloat162float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
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
    __syncthreads();

    float local_sum = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __bfloat162float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
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

__global__ void fa_scale_mask_causal_softmax_rows_bf16_kernel(
        __nv_bfloat16* __restrict__ S,
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

__global__ void fa_dP_bf16_kernel(const __nv_bfloat16* __restrict__ dOh,
                                  const __nv_bfloat16* __restrict__ Vh,
                                  __nv_bfloat16* __restrict__ dP,
                                  int Lq, int Lk, int hd) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || k >= Lk) return;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) {
        acc += __bfloat162float(dOh[q * hd + d]) *
               __bfloat162float(Vh[k * hd + d]);
    }
    dP[q * Lk + k] = __float2bfloat16(acc);
}

__global__ void fa_dS_from_P_dP_bf16_kernel(__nv_bfloat16* __restrict__ P_dS,
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

__global__ void fa_dVh_bf16_kernel(const __nv_bfloat16* __restrict__ P,
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

__global__ void fa_dQh_bf16_kernel(const __nv_bfloat16* __restrict__ dS,
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

__global__ void fa_dKh_bf16_kernel(const __nv_bfloat16* __restrict__ dS,
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

// BF16 in-place add: dst[i] += src[i] (FP32 sum, written back as BF16).
__global__ void fa_bf16_add_inplace_kernel(__nv_bfloat16* __restrict__ dst,
                                           const __nv_bfloat16* __restrict__ src,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

// ─── BF16 batched-linear helpers ────────────────────────────────────────────
//
// linear_forward_batched_fp16 / linear_backward_batched (gemm.cu / batched_ops.cu)
// are fixed-FP16 / FP16-or-FP32 and out of this chunk's scope. The flash qkvo
// ops need a BF16 projection path, so flash_attention.cu carries its own
// self-contained BF16 batched-linear forward + backward — same contracts:
//   forward:  Y_BD(B, out) = X_BD(B, in) @ W(out, in)^T + bias
//   backward: dX_BD = dY·W ; dW += dY^T·X (FP32 scratch) ; dB += colsum(dY).

__global__ void fa_bf16_bias_add_kernel(__nv_bfloat16* __restrict__ Y,
                                        const __nv_bfloat16* __restrict__ bias,
                                        int B, int out_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * out_dim;
    if (idx >= total) return;
    const int j = idx % out_dim;
    const float yv = __bfloat162float(Y[idx]);
    const float bv = __bfloat162float(bias[j]);
    Y[idx] = __float2bfloat16(yv + bv);
}

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

// ─── BF16 batched-linear host wrappers (file-local) ─────────────────────────
//
// BF16 twins of linear_forward_batched_fp16 / linear_backward_batched, built
// on the file-local naive BF16 matmul + FP32-scratch fold kernels above. Used
// by the BF16 path of the qkvo flash-attention ops.

namespace {

void fa_linear_forward_batched_bf16(const Tensor& W, const Tensor* bias,
                                    const Tensor& X_BD, Tensor& Y_BD) {
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("fa_linear_forward_batched_bf16: shape mismatch");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != Dtype::BF16) {
        Y_BD.resize(B, out_dim, Dtype::BF16);
    }
    if (B == 0 || out_dim == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    launch_matmul_ABT_bf16(
        static_cast<const __nv_bfloat16*>(X_BD.data),
        static_cast<const __nv_bfloat16*>(W.data),
        static_cast<__nv_bfloat16*>(Y_BD.data),
        B, out_dim, in_dim);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    if (bias && bias->size() > 0) {
        const int total = B * out_dim;
        const int blocks = (total + 255) / 256;
        fa_bf16_bias_add_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<__nv_bfloat16*>(Y_BD.data),
            static_cast<const __nv_bfloat16*>(bias->data),
            B, out_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

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

    // Causal masking is not yet supported by the WMMA path; fall through to
    // the original online-softmax flash kernel for that case. SD1.5 does
    // not use causal here, so the fast path covers our production workload.
    if (causal) {
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
                1);
        } else {
            flash_attention_kernel<<<grid, FA_BLOCK, shmem, stream>>>(
                reinterpret_cast<const __half*>(Q.data),
                reinterpret_cast<const __half*>(K.data),
                reinterpret_cast<const __half*>(V.data),
                d_mask,
                reinterpret_cast<__half*>(O.data),
                Lq, Lk, D, head_dim,
                1);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // ── Fused FlashAttention-2 path ───────────────────────────────────────
    // Tiled online-softmax WMMA kernel reading the interleaved (L, D) layout
    // directly: no per-head extraction, no (Lq, Lk) score materialisation.
    // Covers the instantiated head_dims (see flash_fused::supported); masked
    // and unmasked, FP16 and BF16. Everything else falls through to the
    // per-head GEMM path below.
    if (!causal && flash_fused::supported(head_dim)) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        if (bf16) {
            flash_fused::launch(
                reinterpret_cast<const __nv_bfloat16*>(Q.data),
                reinterpret_cast<const __nv_bfloat16*>(K.data),
                reinterpret_cast<const __nv_bfloat16*>(V.data),
                d_mask,
                reinterpret_cast<__nv_bfloat16*>(O.data),
                Lq, Lk, D, num_heads, head_dim, stream);
        } else {
            flash_fused::launch(
                reinterpret_cast<const __half*>(Q.data),
                reinterpret_cast<const __half*>(K.data),
                reinterpret_cast<const __half*>(V.data),
                d_mask,
                reinterpret_cast<__half*>(O.data),
                Lq, Lk, D, num_heads, head_dim, stream);
        }
        return;
    }

    // ── WMMA per-head path ────────────────────────────────────────────────
    // For each head h:
    //   1. Extract Qh(Lq, head_dim), Kh(Lk, head_dim), Vth(head_dim, Lk).
    //   2. S(Lq, Lk) = Qh @ Kh^T               via launch_matmul_ABT.
    //   3. S = softmax_row(S * inv_sqrt_hd, mask).
    //   4. Oh(Lq, head_dim) = S @ Vth^T        via launch_matmul_ABT.
    //   5. Pack Oh back into O at slot [h*hd .. (h+1)*hd).
    //
    // BF16 cannot use the FP16 WMMA matmul (tensor cores are FP16/TF32 here);
    // it routes through the file-local naive BF16 matmul instead. The per-head
    // pipeline is otherwise identical.
    //
    // Scratch tensors are scoped local; for SD1.5 worst-case (Lq=Lk=4096,
    // head_dim=40) the S buffer is 32 MB and the per-head buffers a few
    // hundred KB. Allocator reuse makes subsequent calls effectively free.
    //
    // Both FP16 and BF16 pad the Lk axis of Kh / Vth / S to a multiple of 8:
    // the WMMA GEMM's vectorised int4 loads need 8-element row strides along K
    // and N, and an unaligned Lk (e.g. TripoSplat's 12294-token joint sequence)
    // would demote both GEMMs to the naive fallback — a ~40x cliff. Pad rows of
    // Kh are zeroed once (giving pad scores of exactly 0 pre-softmax), the
    // softmax writes exact zeros into the pad columns of S, and Vth's pad
    // columns are zeroed once, so the padded second GEMM adds exactly nothing —
    // the result is bit-identical to the unpadded path. (sm_80+ runs BF16 WMMA
    // fragments, so BF16 takes the same tensor-core matmul as FP16.)
    const int Lk_pad = (Lk + 7) & ~7;
    Tensor Qh = Tensor::empty_on(Device::CUDA, Lq, head_dim, dt);
    Tensor Kh = Tensor::empty_on(Device::CUDA, Lk_pad, head_dim, dt);
    Tensor Vth = Tensor::empty_on(Device::CUDA, head_dim, Lk_pad, dt);
    Tensor S = Tensor::empty_on(Device::CUDA, Lq, Lk_pad, dt);
    Tensor Oh = Tensor::empty_on(Device::CUDA, Lq, head_dim, dt);

    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(head_dim));

    constexpr int CP_BLOCK = 256;
    // softmax block: scale with Lk but cap to keep shared/reduction sane.
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    if (sm_block > 1024) sm_block = 1024;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    // Zero the Lk pads once per call: Kh's pad ROWS (extract writes only the
    // valid rows, and rewrites the same region every head, so the pad stays
    // zero across the head loop) and Vth's whole buffer (its pad is the tail
    // COLUMNS of every row — interleaved, so blanket-zero the 16-bit buffer).
    if (Lk_pad != Lk) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<__half*>(Kh.data) + static_cast<size_t>(Lk) * head_dim, 0,
            static_cast<size_t>(Lk_pad - Lk) * head_dim * sizeof(__half), stream));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
            Vth.data, 0,
            static_cast<size_t>(head_dim) * Lk_pad * sizeof(__half), stream));
    }

    for (int h = 0; h < num_heads; ++h) {
        const int head_off = h * head_dim;
        const int total_q = Lq * head_dim;
        const int total_k = Lk * head_dim;
        const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);

        if (bf16) {
            extract_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Q.data),
                reinterpret_cast<__nv_bfloat16*>(Qh.data),
                Lq, D, head_off, head_dim);
            extract_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(K.data),
                reinterpret_cast<__nv_bfloat16*>(Kh.data),
                Lk, D, head_off, head_dim);
            extract_head_DL_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(V.data),
                reinterpret_cast<__nv_bfloat16*>(Vth.data),
                Lk, D, head_off, head_dim, Lk_pad);
            // S(Lq, Lk_pad) = Qh @ Kh^T — pad rows of Kh are zero, so pad cols
            // of S come out 0 (overwritten by the softmax). BF16 WMMA.
            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                reinterpret_cast<__nv_bfloat16*>(S.data),
                Lq, Lk_pad, head_dim);
            scale_mask_softmax_rows_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(S.data),
                Lq, Lk, inv_sqrt, d_mask, Lk_pad);
            // Oh(Lq, hd) = S(Lq, Lk_pad) @ Vth(hd, Lk_pad)^T — pad columns of
            // both operands are zero, contributing exactly nothing. BF16 WMMA.
            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __nv_bfloat16*>(S.data),
                reinterpret_cast<const __nv_bfloat16*>(Vth.data),
                reinterpret_cast<__nv_bfloat16*>(Oh.data),
                Lq, head_dim, Lk_pad);
            pack_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Oh.data),
                reinterpret_cast<__nv_bfloat16*>(O.data),
                Lq, D, head_off, head_dim);
            continue;
        }

        // 1. Extract per-head buffers.
        extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(Q.data),
            reinterpret_cast<__half*>(Qh.data),
            Lq, D, head_off, head_dim);
        extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(K.data),
            reinterpret_cast<__half*>(Kh.data),
            Lk, D, head_off, head_dim);
        extract_head_DL_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(V.data),
            reinterpret_cast<__half*>(Vth.data),
            Lk, D, head_off, head_dim, Lk_pad);

        // 2. S(Lq, Lk_pad) = Qh(Lq, hd) @ Kh(Lk_pad, hd)^T. Pad rows of Kh are
        //    zero, so pad columns of S come out 0 (overwritten by the softmax).
        fp16_internal::launch_matmul_ABT(
            reinterpret_cast<const __half*>(Qh.data),
            reinterpret_cast<const __half*>(Kh.data),
            reinterpret_cast<__half*>(S.data),
            Lq, Lk_pad, head_dim);

        // 3. Row-wise softmax (scaled, optionally masked) over the valid Lk;
        //    writes exact zeros into the pad columns.
        scale_mask_softmax_rows_kernel<<<Lq, sm_block, shmem, stream>>>(
            reinterpret_cast<__half*>(S.data),
            Lq, Lk, inv_sqrt, d_mask, Lk_pad);

        // 4. Oh(Lq, hd) = S(Lq, Lk_pad) @ Vth(hd, Lk_pad)^T — the pad columns
        //    of both operands are zero, contributing exactly nothing.
        fp16_internal::launch_matmul_ABT(
            reinterpret_cast<const __half*>(S.data),
            reinterpret_cast<const __half*>(Vth.data),
            reinterpret_cast<__half*>(Oh.data),
            Lq, head_dim, Lk_pad);

        // 5. Pack back into the per-head slot of O.
        pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(Oh.data),
            reinterpret_cast<__half*>(O.data),
            Lq, D, head_off, head_dim);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
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
    const bool bf16 = (dt == Dtype::BF16);
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
    if (bf16) {
        fa_linear_forward_batched_bf16(Wk, bk, ctx, K_out);
        fa_linear_forward_batched_bf16(Wv, bv, ctx, V_out);
    } else {
        linear_forward_batched_fp16(Wk, bk, ctx, K_out);
        linear_forward_batched_fp16(Wv, bv, ctx, V_out);
    }
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
    const bool bf16 = (dt == Dtype::BF16);
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

    if (bf16) {
        fa_linear_forward_batched_bf16(Wq, bq, X, Qp);
        flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
        fa_linear_forward_batched_bf16(Wo, bo, Op, O);
    } else {
        linear_forward_batched_fp16(Wq, bq, X, Qp);
        flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
        linear_forward_batched_fp16(Wo, bo, Op, O);
    }
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

// ─── Recompute-style FP16 backward ─────────────────────────────────────────
//
// Strategy: re-run the forward up to the per-head softmax + V matmul to
// reconstruct O_attn (post-attention, pre-Wo) AND each head's P matrix.
//   1. Re-project X→Q, Ctx→K, Ctx→V via linear_forward_batched_fp16
//      (bit-identical to the forward).
//   2. Run the per-head extract/matmul/softmax pipeline (mirroring the
//      forward's WMMA path) to land O_attn (Lq, D).
//   3. Wo+bo backward: dO_attn = dO·Wo, dWo += dO^T·O_attn,
//      dbo += colsum(dO). linear_backward_batched handles all three
//      with FP16 storage + FP32 scratch.
//   4. Per head: re-extract Q_h, K_h, V_h, dO_attn_h. Re-derive P. Compute
//      dV_h, dP, dS = P*(dP-D_q)*inv_sqrt, dQ_h, dK_h. Pack dQ/dK/dV back.
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
    const int hd = D / num_heads;

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
    if (bf16) {
        fa_linear_forward_batched_bf16(Wq, bq, X,      Q);
        fa_linear_forward_batched_bf16(Wk, bk, kv_src, K);
        fa_linear_forward_batched_bf16(Wv, bv, kv_src, V);
    } else {
        linear_forward_batched_fp16(Wq, bq, X,      Q);
        linear_forward_batched_fp16(Wk, bk, kv_src, K);
        linear_forward_batched_fp16(Wv, bv, kv_src, V);
    }

    // ── 2. Per-head recompute of P and O_attn (Lq, D). ────────────────────
    Tensor Qh = Tensor::empty_on(Device::CUDA, Lq, hd, dt);
    Tensor Kh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);
    Tensor Vh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);
    Tensor Vth = Tensor::empty_on(Device::CUDA, hd, Lk, dt);   // V transposed for matmul_ABT in O_attn step
    Tensor P_main = Tensor::empty_on(Device::CUDA, Lq, Lk, dt); // P during main bwd sweep (per head)
    Tensor O_attn = Tensor::empty_on(Device::CUDA, Lq, D, dt);  // full (Lq, D) post-attn pre-Wo

    constexpr int CP_BLOCK = 256;
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    if (sm_block > 1024) sm_block = 1024;
    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(hd));

    // Recompute pass 1: fill O_attn so we can run Wo backward against it.
    // We use launch_matmul_ABT for both matmuls — mirror forward's WMMA path
    // (the file-local naive BF16 matmul for the BF16 dtype).
    {
        Tensor S = Tensor::empty_on(Device::CUDA, Lq, Lk, dt);
        Tensor Oh = Tensor::empty_on(Device::CUDA, Lq, hd, dt);
        for (int h = 0; h < num_heads; ++h) {
            const int head_off = h * hd;
            const int total_q = Lq * hd;
            const int total_k = Lk * hd;
            const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);
            if (bf16) {
                extract_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Q.data),
                    reinterpret_cast<__nv_bfloat16*>(Qh.data),
                    Lq, D, head_off, hd);
                extract_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(K.data),
                    reinterpret_cast<__nv_bfloat16*>(Kh.data),
                    Lk, D, head_off, hd);
                extract_head_DL_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(V.data),
                    reinterpret_cast<__nv_bfloat16*>(Vth.data),
                    Lk, D, head_off, hd, /*ldY=*/Lk);
                launch_matmul_ABT_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                    reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                    reinterpret_cast<__nv_bfloat16*>(S.data),
                    Lq, Lk, hd);
                fa_scale_mask_causal_softmax_rows_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(S.data),
                    Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);
                launch_matmul_ABT_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(S.data),
                    reinterpret_cast<const __nv_bfloat16*>(Vth.data),
                    reinterpret_cast<__nv_bfloat16*>(Oh.data),
                    Lq, hd, Lk);
                pack_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Oh.data),
                    reinterpret_cast<__nv_bfloat16*>(O_attn.data),
                    Lq, D, head_off, hd);
                continue;
            }
            extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(Q.data),
                reinterpret_cast<__half*>(Qh.data),
                Lq, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(K.data),
                reinterpret_cast<__half*>(Kh.data),
                Lk, D, head_off, hd);
            extract_head_DL_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(V.data),
                reinterpret_cast<__half*>(Vth.data),
                Lk, D, head_off, hd, /*ldY=*/Lk);

            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __half*>(Qh.data),
                reinterpret_cast<const __half*>(Kh.data),
                reinterpret_cast<__half*>(S.data),
                Lq, Lk, hd);

            fa_scale_mask_causal_softmax_rows_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(S.data),
                Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);

            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __half*>(S.data),
                reinterpret_cast<const __half*>(Vth.data),
                reinterpret_cast<__half*>(Oh.data),
                Lq, hd, Lk);

            pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(Oh.data),
                reinterpret_cast<__half*>(O_attn.data),
                Lq, D, head_off, hd);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

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

    // ── 4. Per-head backward sweep. ───────────────────────────────────────
    Tensor dQ = Tensor::empty_on(Device::CUDA, Lq, D, dt); dQ.zero();
    Tensor dK = Tensor::empty_on(Device::CUDA, Lk, D, dt); dK.zero();
    Tensor dV = Tensor::empty_on(Device::CUDA, Lk, D, dt); dV.zero();
    {
        Tensor dOh = Tensor::empty_on(Device::CUDA, Lq, hd, dt);
        Tensor dP = Tensor::empty_on(Device::CUDA, Lq, Lk, dt);
        Tensor dQh = Tensor::empty_on(Device::CUDA, Lq, hd, dt);
        Tensor dKh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);
        Tensor dVh = Tensor::empty_on(Device::CUDA, Lk, hd, dt);
        for (int h = 0; h < num_heads; ++h) {
            const int head_off = h * hd;
            const int total_q = Lq * hd;
            const int total_k = Lk * hd;
            const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);

            if (bf16) {
                extract_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Q.data),
                    reinterpret_cast<__nv_bfloat16*>(Qh.data),
                    Lq, D, head_off, hd);
                extract_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(K.data),
                    reinterpret_cast<__nv_bfloat16*>(Kh.data),
                    Lk, D, head_off, hd);
                extract_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(V.data),
                    reinterpret_cast<__nv_bfloat16*>(Vh.data),
                    Lk, D, head_off, hd);
                extract_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dO_attn.data),
                    reinterpret_cast<__nv_bfloat16*>(dOh.data),
                    Lq, D, head_off, hd);

                launch_matmul_ABT_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                    reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                    reinterpret_cast<__nv_bfloat16*>(P_main.data),
                    Lq, Lk, hd);
                fa_scale_mask_causal_softmax_rows_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(P_main.data),
                    Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);

                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                    fa_dVh_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(P_main.data),
                        reinterpret_cast<const __nv_bfloat16*>(dOh.data),
                        reinterpret_cast<__nv_bfloat16*>(dVh.data),
                        Lq, Lk, hd);
                }
                {
                    dim3 block(16, 16);
                    dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
                    fa_dP_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(dOh.data),
                        reinterpret_cast<const __nv_bfloat16*>(Vh.data),
                        reinterpret_cast<__nv_bfloat16*>(dP.data),
                        Lq, Lk, hd);
                }
                {
                    fa_dS_from_P_dP_bf16_kernel<<<Lq, sm_block, shmem, stream>>>(
                        reinterpret_cast<__nv_bfloat16*>(P_main.data),
                        reinterpret_cast<const __nv_bfloat16*>(dP.data),
                        Lq, Lk, inv_sqrt);
                }
                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
                    fa_dQh_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(P_main.data),
                        reinterpret_cast<const __nv_bfloat16*>(Kh.data),
                        reinterpret_cast<__nv_bfloat16*>(dQh.data),
                        Lq, Lk, hd);
                }
                {
                    dim3 block(16, 16);
                    dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                    fa_dKh_bf16_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(P_main.data),
                        reinterpret_cast<const __nv_bfloat16*>(Qh.data),
                        reinterpret_cast<__nv_bfloat16*>(dKh.data),
                        Lq, Lk, hd);
                }
                pack_head_LD_bf16_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dQh.data),
                    reinterpret_cast<__nv_bfloat16*>(dQ.data),
                    Lq, D, head_off, hd);
                pack_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dKh.data),
                    reinterpret_cast<__nv_bfloat16*>(dK.data),
                    Lk, D, head_off, hd);
                pack_head_LD_bf16_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(dVh.data),
                    reinterpret_cast<__nv_bfloat16*>(dV.data),
                    Lk, D, head_off, hd);
                continue;
            }

            extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(Q.data),
                reinterpret_cast<__half*>(Qh.data),
                Lq, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(K.data),
                reinterpret_cast<__half*>(Kh.data),
                Lk, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(V.data),
                reinterpret_cast<__half*>(Vh.data),
                Lk, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dO_attn.data),
                reinterpret_cast<__half*>(dOh.data),
                Lq, D, head_off, hd);

            // Recompute P for this head — same code path as pass 1.
            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __half*>(Qh.data),
                reinterpret_cast<const __half*>(Kh.data),
                reinterpret_cast<__half*>(P_main.data),
                Lq, Lk, hd);
            fa_scale_mask_causal_softmax_rows_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(P_main.data),
                Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);

            // dV_h = P^T · dO_attn_h  (Lk, hd)
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fa_dVh_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(P_main.data),
                    reinterpret_cast<const __half*>(dOh.data),
                    reinterpret_cast<__half*>(dVh.data),
                    Lq, Lk, hd);
            }

            // dP = dO_attn_h · V_h^T   (Lq, Lk)
            {
                dim3 block(16, 16);
                dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
                fa_dP_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(dOh.data),
                    reinterpret_cast<const __half*>(Vh.data),
                    reinterpret_cast<__half*>(dP.data),
                    Lq, Lk, hd);
            }

            // dS = P * (dP - D_q) * inv_sqrt — written in-place over P_main.
            {
                fa_dS_from_P_dP_kernel<<<Lq, sm_block, shmem, stream>>>(
                    reinterpret_cast<__half*>(P_main.data),
                    reinterpret_cast<const __half*>(dP.data),
                    Lq, Lk, inv_sqrt);
            }

            // dQ_h = dS · K_h
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
                fa_dQh_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(P_main.data),
                    reinterpret_cast<const __half*>(Kh.data),
                    reinterpret_cast<__half*>(dQh.data),
                    Lq, Lk, hd);
            }
            // dK_h = dS^T · Q_h
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fa_dKh_kernel<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(P_main.data),
                    reinterpret_cast<const __half*>(Qh.data),
                    reinterpret_cast<__half*>(dKh.data),
                    Lq, Lk, hd);
            }

            // Pack dQ_h / dK_h / dV_h back into (Lq, D) / (Lk, D).
            pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dQh.data),
                reinterpret_cast<__half*>(dQ.data),
                Lq, D, head_off, hd);
            pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dKh.data),
                reinterpret_cast<__half*>(dK.data),
                Lk, D, head_off, hd);
            pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(dVh.data),
                reinterpret_cast<__half*>(dV.data),
                Lk, D, head_off, hd);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

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

void flash_attention_backward(const ::brotensor::Tensor& Q,
                              const ::brotensor::Tensor& K,
                              const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& O,
                              const ::brotensor::Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              ::brotensor::Tensor& dQ,
                              ::brotensor::Tensor& dK,
                              ::brotensor::Tensor& dV);

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
    if (batch_size == 1 && !causal && (dt == Dtype::FP16 || dt == Dtype::BF16)) {
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
        int group) {
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
    int lo = (window > 0) ? (aq - window + 1) : 0;
    if (lo < 0) lo = 0;
    const int k_hi = aq;                       // inclusive causal upper bound

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
                                      Tensor& O) {
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
    const size_t shmem = (static_cast<size_t>(FA_KTILE) + FA_BLOCK) * sizeof(float);
    dim3 grid(Lq, num_heads, 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    if (dt == Dtype::BF16) {
        flash_attention_windowed_kernel<__nv_bfloat16><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data),
            reinterpret_cast<const __nv_bfloat16*>(K.data),
            reinterpret_cast<const __nv_bfloat16*>(V.data),
            d_mask, reinterpret_cast<__nv_bfloat16*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group);
    } else if (dt == Dtype::FP32) {
        flash_attention_windowed_kernel<float><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const float*>(Q.data),
            reinterpret_cast<const float*>(K.data),
            reinterpret_cast<const float*>(V.data),
            d_mask, reinterpret_cast<float*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group);
    } else {
        flash_attention_windowed_kernel<__half><<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __half*>(Q.data),
            reinterpret_cast<const __half*>(K.data),
            reinterpret_cast<const __half*>(V.data),
            d_mask, reinterpret_cast<__half*>(O.data),
            Lk, Dq, Dkv, head_dim, window, q_offset, group);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_flash_attention(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_forward                       = &flash_attention_forward;
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
