#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include "fp16_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

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
// (head_dim, L) so element (d, l) = X[l, head_off + d]. Used to produce a
// (head_dim, L) "B"-style operand for the second GEMM via launch_matmul_ABT.
__global__ void extract_head_DL_kernel(const __half* __restrict__ X,
                                       __half* __restrict__ Y,
                                       int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    // Cooperatively write Y[d * L + l] = X[l * D + head_off + d]. Choose
    // mapping that gives coalesced loads of X (d innermost in source) and
    // strided writes to Y — strided writes are fine for fp16 throughput here.
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[d * L + l] = X[l * D + head_off + d];
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
// chosen by the launcher.
__global__ void scale_mask_softmax_rows_kernel(__half* __restrict__ S,
                                               int Lq, int Lk,
                                               float scale,
                                               const float* __restrict__ mask) {
    extern __shared__ float ssm[];  // size = blockDim.x
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __half* row = S + static_cast<size_t>(q) * static_cast<size_t>(Lk);

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
// materialises an (Lq, Lk) buffer per head (`S` in flash_attention_forward_gpu's
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

} // namespace

void flash_attention_forward_gpu(const GpuTensor& Q,
                                 const GpuTensor& K,
                                 const GpuTensor& V,
                                 const float* d_mask,
                                 int num_heads,
                                 bool causal,
                                 GpuTensor& O) {
    if (Q.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_forward_gpu: Q, K, V must be FP16");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward_gpu: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward_gpu: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_forward_gpu: causal requires Lq == Lk");
    }
    const int head_dim = D / num_heads;
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    // Causal masking is not yet supported by the WMMA path; fall through to
    // the original online-softmax flash kernel for that case. SD1.5 does
    // not use causal here, so the fast path covers our production workload.
    if (causal) {
        const size_t shmem = (static_cast<size_t>(FA_KTILE) + FA_BLOCK) * sizeof(float);
        // head_dim parallelisation in the kernel uses up to 8 d-slots/thread.
        if ((head_dim + FA_BLOCK - 1) / FA_BLOCK > 8) {
            throw std::runtime_error("flash_attention_forward_gpu: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
        }
        dim3 grid(Lq, num_heads, 1);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        flash_attention_kernel<<<grid, FA_BLOCK, shmem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_fp16()),
            reinterpret_cast<const __half*>(K.data_fp16()),
            reinterpret_cast<const __half*>(V.data_fp16()),
            d_mask,
            reinterpret_cast<__half*>(O.data_fp16()),
            Lq, Lk, D, head_dim,
            1);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
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
    // Scratch tensors are scoped local; for SD1.5 worst-case (Lq=Lk=4096,
    // head_dim=40) the S buffer is 32 MB and the per-head buffers a few
    // hundred KB. Allocator reuse makes subsequent calls effectively free.
    GpuTensor Qh(Lq, head_dim, Dtype::FP16);
    GpuTensor Kh(Lk, head_dim, Dtype::FP16);
    GpuTensor Vth(head_dim, Lk, Dtype::FP16);
    GpuTensor S(Lq, Lk, Dtype::FP16);
    GpuTensor Oh(Lq, head_dim, Dtype::FP16);

    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(head_dim));

    constexpr int CP_BLOCK = 256;
    // softmax block: scale with Lk but cap to keep shared/reduction sane.
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    if (sm_block > 1024) sm_block = 1024;

    for (int h = 0; h < num_heads; ++h) {
        const int head_off = h * head_dim;

        // 1. Extract per-head buffers.
        const int total_q = Lq * head_dim;
        const int total_k = Lk * head_dim;
        extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK>>>(
            reinterpret_cast<const __half*>(Q.data_fp16()),
            reinterpret_cast<__half*>(Qh.data_fp16()),
            Lq, D, head_off, head_dim);
        extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
            reinterpret_cast<const __half*>(K.data_fp16()),
            reinterpret_cast<__half*>(Kh.data_fp16()),
            Lk, D, head_off, head_dim);
        extract_head_DL_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
            reinterpret_cast<const __half*>(V.data_fp16()),
            reinterpret_cast<__half*>(Vth.data_fp16()),
            Lk, D, head_off, head_dim);

        // 2. S(Lq, Lk) = Qh(Lq, hd) @ Kh(Lk, hd)^T.
        fp16_internal::launch_matmul_ABT(
            reinterpret_cast<const __half*>(Qh.data_fp16()),
            reinterpret_cast<const __half*>(Kh.data_fp16()),
            reinterpret_cast<__half*>(S.data_fp16()),
            Lq, Lk, head_dim);

        // 3. Row-wise softmax (scaled, optionally masked).
        const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);
        scale_mask_softmax_rows_kernel<<<Lq, sm_block, shmem>>>(
            reinterpret_cast<__half*>(S.data_fp16()),
            Lq, Lk, inv_sqrt, d_mask);

        // 4. Oh(Lq, hd) = S(Lq, Lk) @ Vth(hd, Lk)^T  — Vth is already (hd, Lk).
        fp16_internal::launch_matmul_ABT(
            reinterpret_cast<const __half*>(S.data_fp16()),
            reinterpret_cast<const __half*>(Vth.data_fp16()),
            reinterpret_cast<__half*>(Oh.data_fp16()),
            Lq, head_dim, Lk);

        // 5. Pack back into the per-head slot of O.
        pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK>>>(
            reinterpret_cast<const __half*>(Oh.data_fp16()),
            reinterpret_cast<__half*>(O.data_fp16()),
            Lq, D, head_off, head_dim);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// Project (ctx → K, ctx → V) using the same linear_forward_batched_fp16_gpu
// call as flash_attention_qkvo_forward_gpu, producing (Lk, D) FP16 buffers
// in the exact layout flash_attention_forward_gpu consumes.
void flash_attention_project_kv_gpu(const GpuTensor& ctx,
                                    const GpuTensor& Wk, const GpuTensor* bk,
                                    const GpuTensor& Wv, const GpuTensor* bv,
                                    GpuTensor& K_out,
                                    GpuTensor& V_out) {
    if (ctx.dtype != Dtype::FP16 || Wk.dtype != Dtype::FP16 ||
        Wv.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_project_kv_gpu: all tensors must be FP16");
    }
    const int Lk = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D = Wk.rows;
    if (Wk.cols != D_ctx || Wv.rows != D || Wv.cols != D_ctx) {
        throw std::runtime_error("flash_attention_project_kv_gpu: Wk/Wv shape mismatch");
    }
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != Dtype::FP16) {
        K_out.resize(Lk, D, Dtype::FP16);
    }
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != Dtype::FP16) {
        V_out.resize(Lk, D, Dtype::FP16);
    }
    if (Lk == 0 || D == 0) return;
    linear_forward_batched_fp16_gpu(Wk, bk, ctx, K_out);
    linear_forward_batched_fp16_gpu(Wv, bv, ctx, V_out);
}

// Core attention with caller-supplied K/V (pre-projected). Projects X → Q
// with Wq/bq, runs the tiled attention core, then applies Wo/bo. This is
// the same composition flash_attention_qkvo_forward_gpu uses on its cached
// path; both entry points delegate here so numerics are bitwise-identical.
void flash_attention_q_with_kv_cached_forward_gpu(const GpuTensor& X,
                                                  const GpuTensor& K,
                                                  const GpuTensor& V,
                                                  const GpuTensor& Wq, const GpuTensor* bq,
                                                  const GpuTensor& Wo, const GpuTensor* bo,
                                                  const float* d_mask,
                                                  int num_heads,
                                                  bool causal,
                                                  GpuTensor& O) {
    if (X.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16 ||
        Wq.dtype != Dtype::FP16 || Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward_gpu: all tensors must be FP16");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward_gpu: K/V shape mismatch");
    }
    if (Wq.rows != D || Wq.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward_gpu: Wq/Wo shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward_gpu: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    GpuTensor Qp(Lq, D, Dtype::FP16);
    GpuTensor Op(Lq, D, Dtype::FP16);

    linear_forward_batched_fp16_gpu(Wq, bq, X, Qp);
    flash_attention_forward_gpu(Qp, K, V, d_mask, num_heads, causal, Op);
    linear_forward_batched_fp16_gpu(Wo, bo, Op, O);
}

// Variant that fuses Q/K/V/O projections at the boundary. Delegates each
// projection to linear_forward_batched_fp16_gpu so optional biases are
// folded in. Ctx==nullptr means self-attention (Ctx = X).
void flash_attention_qkvo_forward_gpu(const GpuTensor& X,
                                      const GpuTensor* Ctx,
                                      const GpuTensor& Wq, const GpuTensor* bq,
                                      const GpuTensor& Wk, const GpuTensor* bk,
                                      const GpuTensor& Wv, const GpuTensor* bv,
                                      const GpuTensor& Wo, const GpuTensor* bo,
                                      const float* d_mask,
                                      int num_heads,
                                      bool causal,
                                      GpuTensor& O) {
    if (X.dtype != Dtype::FP16 || Wq.dtype != Dtype::FP16 ||
        Wk.dtype != Dtype::FP16 || Wv.dtype != Dtype::FP16 ||
        Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: all tensors must be FP16");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const GpuTensor& kv_src = Ctx ? *Ctx : X;
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: Ctx must be FP16");
    }
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: num_heads must divide D");
    }

    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    // Compose via the two new helpers — keeps the order of CUDA ops identical
    // to the pre-refactor code (linear_Wk, linear_Wv, linear_Wq, attention,
    // linear_Wo would actually be reordered; we mirror that by inlining Q+attn
    // into the cached helper and projecting K/V here first).
    GpuTensor Kp(Lk, D, Dtype::FP16);
    GpuTensor Vp(Lk, D, Dtype::FP16);
    flash_attention_project_kv_gpu(kv_src, Wk, bk, Wv, bv, Kp, Vp);
    flash_attention_q_with_kv_cached_forward_gpu(
        X, Kp, Vp, Wq, bq, Wo, bo, d_mask, num_heads, causal, O);
}

// ─── Recompute-style FP16 backward ─────────────────────────────────────────
//
// Strategy: re-run the forward up to the per-head softmax + V matmul to
// reconstruct O_attn (post-attention, pre-Wo) AND each head's P matrix.
//   1. Re-project X→Q, Ctx→K, Ctx→V via linear_forward_batched_fp16_gpu
//      (bit-identical to the forward).
//   2. Run the per-head extract/matmul/softmax pipeline (mirroring the
//      forward's WMMA path) to land O_attn (Lq, D).
//   3. Wo+bo backward: dO_attn = dO·Wo, dWo += dO^T·O_attn,
//      dbo += colsum(dO). linear_backward_batched_gpu handles all three
//      with FP16 storage + FP32 scratch.
//   4. Per head: re-extract Q_h, K_h, V_h, dO_attn_h. Re-derive P. Compute
//      dV_h, dP, dS = P*(dP-D_q)*inv_sqrt, dQ_h, dK_h. Pack dQ/dK/dV back.
//   5. Q,K,V-projection backward: linear_backward_batched_gpu for each.
//      Self-attn (Ctx=null) accumulates Q/K/V dX contributions; cross-attn
//      sends Q→dX, K+V→dCtx.
void flash_attention_qkvo_backward_gpu(
    const GpuTensor& X, const GpuTensor* Ctx,
    const GpuTensor& Wq, const GpuTensor* bq,
    const GpuTensor& Wk, const GpuTensor* bk,
    const GpuTensor& Wv, const GpuTensor* bv,
    const GpuTensor& Wo, const GpuTensor* bo,
    const float* d_mask,
    int num_heads,
    bool causal,
    const GpuTensor& dO,
    GpuTensor& dX, GpuTensor* dCtx,
    GpuTensor& dWq, GpuTensor* dbq,
    GpuTensor& dWk, GpuTensor* dbk,
    GpuTensor& dWv, GpuTensor* dbv,
    GpuTensor& dWo, GpuTensor* dbo) {

    // ── Argument validation (matches forward; backward adds dO/grads). ────
    if (X.dtype != Dtype::FP16 || dO.dtype != Dtype::FP16 ||
        Wq.dtype != Dtype::FP16 || Wk.dtype != Dtype::FP16 ||
        Wv.dtype != Dtype::FP16 || Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_backward_gpu: all tensors must be FP16");
    }
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_backward_gpu: Ctx must be FP16");
    }
    const bool self_attn = (Ctx == nullptr);
    if (self_attn) {
        if (dCtx != nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward_gpu: dCtx must be null when Ctx is null");
        }
    } else {
        if (dCtx == nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward_gpu: dCtx must be non-null when Ctx is non-null");
        }
    }
    // bias / grad-bias symmetry: callers must match the forward's bias presence
    // exactly on the grad side. Anything else would mean we're either silently
    // discarding a gradient or writing into a buffer the caller didn't provide.
    auto bias_pair_ok = [](const GpuTensor* b, const GpuTensor* db) {
        return static_cast<bool>(b) == static_cast<bool>(db);
    };
    if (!bias_pair_ok(bq, dbq) || !bias_pair_ok(bk, dbk) ||
        !bias_pair_ok(bv, dbv) || !bias_pair_ok(bo, dbo)) {
        throw std::runtime_error("flash_attention_qkvo_backward_gpu: bias/grad-bias presence mismatch");
    }

    const int Lq = X.rows;
    const int D  = X.cols;
    const GpuTensor& kv_src = Ctx ? *Ctx : X;
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward_gpu: shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward_gpu: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_backward_gpu: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_qkvo_backward_gpu: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dX.rows != Lq || dX.cols != D || dX.dtype != Dtype::FP16) {
        dX.resize(Lq, D, Dtype::FP16);
    }
    if (!self_attn) {
        if (dCtx->rows != Lk || dCtx->cols != D_ctx || dCtx->dtype != Dtype::FP16) {
            dCtx->resize(Lk, D_ctx, Dtype::FP16);
        }
        dCtx->zero();
    }
    dX.zero();

    if (Lq == 0 || Lk == 0 || D == 0) return;

    // ── 1. Recompute forward projections (same call as forward). ──────────
    GpuTensor Q(Lq, D, Dtype::FP16);
    GpuTensor K(Lk, D, Dtype::FP16);
    GpuTensor V(Lk, D, Dtype::FP16);
    linear_forward_batched_fp16_gpu(Wq, bq, X,      Q);
    linear_forward_batched_fp16_gpu(Wk, bk, kv_src, K);
    linear_forward_batched_fp16_gpu(Wv, bv, kv_src, V);

    // ── 2. Per-head recompute of P and O_attn (Lq, D). ────────────────────
    GpuTensor Qh(Lq, hd, Dtype::FP16);
    GpuTensor Kh(Lk, hd, Dtype::FP16);
    GpuTensor Vh(Lk, hd, Dtype::FP16);
    GpuTensor Vth(hd, Lk, Dtype::FP16);   // V transposed for matmul_ABT in O_attn step
    GpuTensor P_main(Lq, Lk, Dtype::FP16); // P during main bwd sweep (per head)
    GpuTensor O_attn(Lq, D, Dtype::FP16);  // full (Lq, D) post-attn pre-Wo

    constexpr int CP_BLOCK = 256;
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    if (sm_block > 1024) sm_block = 1024;
    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(hd));

    // Recompute pass 1: fill O_attn so we can run Wo backward against it.
    // We use launch_matmul_ABT for both matmuls — mirror forward's WMMA path.
    {
        GpuTensor S(Lq, Lk, Dtype::FP16);
        GpuTensor Oh(Lq, hd, Dtype::FP16);
        for (int h = 0; h < num_heads; ++h) {
            const int head_off = h * hd;
            const int total_q = Lq * hd;
            const int total_k = Lk * hd;
            extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(Q.data_fp16()),
                reinterpret_cast<__half*>(Qh.data_fp16()),
                Lq, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(K.data_fp16()),
                reinterpret_cast<__half*>(Kh.data_fp16()),
                Lk, D, head_off, hd);
            extract_head_DL_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(V.data_fp16()),
                reinterpret_cast<__half*>(Vth.data_fp16()),
                Lk, D, head_off, hd);

            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __half*>(Qh.data_fp16()),
                reinterpret_cast<const __half*>(Kh.data_fp16()),
                reinterpret_cast<__half*>(S.data_fp16()),
                Lq, Lk, hd);

            const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);
            fa_scale_mask_causal_softmax_rows_kernel<<<Lq, sm_block, shmem>>>(
                reinterpret_cast<__half*>(S.data_fp16()),
                Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);

            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __half*>(S.data_fp16()),
                reinterpret_cast<const __half*>(Vth.data_fp16()),
                reinterpret_cast<__half*>(Oh.data_fp16()),
                Lq, hd, Lk);

            pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(Oh.data_fp16()),
                reinterpret_cast<__half*>(O_attn.data_fp16()),
                Lq, D, head_off, hd);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // ── 3. Wo + bo backward via linear_backward_batched_gpu. ──────────────
    //   forward: O(Lq, D) = O_attn(Lq, D) @ Wo(D, D)^T + bo
    //   bwd outputs:
    //     dO_attn(Lq, D) = dO @ Wo  (linear_backward_batched_gpu's dX_BD)
    //     dWo(D, D)     += dO^T @ O_attn
    //     dbo           += colsum(dO)
    GpuTensor dO_attn(Lq, D, Dtype::FP16);
    {
        // linear_backward_batched_gpu needs a non-null dB even when bias isn't
        // present in the forward. We always have dWo; for bo absent we feed a
        // scratch dB tensor and discard.
        GpuTensor scratch_db;
        const bool has_bo = (bo != nullptr);
        if (!has_bo) {
            scratch_db.resize(D, 1, Dtype::FP16);
            scratch_db.zero();
        }
        linear_backward_batched_gpu(Wo, O_attn, dO, dO_attn, dWo,
                                    has_bo ? *dbo : scratch_db);
    }

    // ── 4. Per-head backward sweep. ───────────────────────────────────────
    GpuTensor dQ(Lq, D, Dtype::FP16); dQ.zero();
    GpuTensor dK(Lk, D, Dtype::FP16); dK.zero();
    GpuTensor dV(Lk, D, Dtype::FP16); dV.zero();
    {
        GpuTensor dOh(Lq, hd, Dtype::FP16);
        GpuTensor dP(Lq, Lk, Dtype::FP16);
        GpuTensor dQh(Lq, hd, Dtype::FP16);
        GpuTensor dKh(Lk, hd, Dtype::FP16);
        GpuTensor dVh(Lk, hd, Dtype::FP16);
        for (int h = 0; h < num_heads; ++h) {
            const int head_off = h * hd;
            const int total_q = Lq * hd;
            const int total_k = Lk * hd;

            extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(Q.data_fp16()),
                reinterpret_cast<__half*>(Qh.data_fp16()),
                Lq, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(K.data_fp16()),
                reinterpret_cast<__half*>(Kh.data_fp16()),
                Lk, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(V.data_fp16()),
                reinterpret_cast<__half*>(Vh.data_fp16()),
                Lk, D, head_off, hd);
            extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(dO_attn.data_fp16()),
                reinterpret_cast<__half*>(dOh.data_fp16()),
                Lq, D, head_off, hd);

            // Recompute P for this head — same code path as pass 1.
            fp16_internal::launch_matmul_ABT(
                reinterpret_cast<const __half*>(Qh.data_fp16()),
                reinterpret_cast<const __half*>(Kh.data_fp16()),
                reinterpret_cast<__half*>(P_main.data_fp16()),
                Lq, Lk, hd);
            const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);
            fa_scale_mask_causal_softmax_rows_kernel<<<Lq, sm_block, shmem>>>(
                reinterpret_cast<__half*>(P_main.data_fp16()),
                Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);

            // dV_h = P^T · dO_attn_h  (Lk, hd)
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fa_dVh_kernel<<<grid, block>>>(
                    reinterpret_cast<const __half*>(P_main.data_fp16()),
                    reinterpret_cast<const __half*>(dOh.data_fp16()),
                    reinterpret_cast<__half*>(dVh.data_fp16()),
                    Lq, Lk, hd);
            }

            // dP = dO_attn_h · V_h^T   (Lq, Lk)
            {
                dim3 block(16, 16);
                dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
                fa_dP_kernel<<<grid, block>>>(
                    reinterpret_cast<const __half*>(dOh.data_fp16()),
                    reinterpret_cast<const __half*>(Vh.data_fp16()),
                    reinterpret_cast<__half*>(dP.data_fp16()),
                    Lq, Lk, hd);
            }

            // dS = P * (dP - D_q) * inv_sqrt — written in-place over P_main.
            {
                const size_t shmem2 = static_cast<size_t>(sm_block) * sizeof(float);
                fa_dS_from_P_dP_kernel<<<Lq, sm_block, shmem2>>>(
                    reinterpret_cast<__half*>(P_main.data_fp16()),
                    reinterpret_cast<const __half*>(dP.data_fp16()),
                    Lq, Lk, inv_sqrt);
            }

            // dQ_h = dS · K_h
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
                fa_dQh_kernel<<<grid, block>>>(
                    reinterpret_cast<const __half*>(P_main.data_fp16()),
                    reinterpret_cast<const __half*>(Kh.data_fp16()),
                    reinterpret_cast<__half*>(dQh.data_fp16()),
                    Lq, Lk, hd);
            }
            // dK_h = dS^T · Q_h
            {
                dim3 block(16, 16);
                dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
                fa_dKh_kernel<<<grid, block>>>(
                    reinterpret_cast<const __half*>(P_main.data_fp16()),
                    reinterpret_cast<const __half*>(Qh.data_fp16()),
                    reinterpret_cast<__half*>(dKh.data_fp16()),
                    Lq, Lk, hd);
            }

            // Pack dQ_h / dK_h / dV_h back into (Lq, D) / (Lk, D).
            pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(dQh.data_fp16()),
                reinterpret_cast<__half*>(dQ.data_fp16()),
                Lq, D, head_off, hd);
            pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(dKh.data_fp16()),
                reinterpret_cast<__half*>(dK.data_fp16()),
                Lk, D, head_off, hd);
            pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK>>>(
                reinterpret_cast<const __half*>(dVh.data_fp16()),
                reinterpret_cast<__half*>(dV.data_fp16()),
                Lk, D, head_off, hd);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // ── 5. Q / K / V projection backward. ─────────────────────────────────
    // forward: Q = X·Wq^T + bq  →  dX_q = dQ·Wq, dWq += dQ^T·X, dbq += colsum(dQ).
    // Same for K, V but with kv_src as the input. We use linear_backward_batched_gpu
    // which handles all three with FP16 storage + FP32 accumulator scratch.
    auto run_proj_back = [&](const GpuTensor& W, const GpuTensor& In,
                             const GpuTensor& dOut, GpuTensor& dIn_out,
                             GpuTensor& dW_acc, const GpuTensor* b_fwd,
                             GpuTensor* db_acc) {
        GpuTensor scratch_db;
        bool has_b = (b_fwd != nullptr);
        if (!has_b) {
            scratch_db.resize(W.rows, 1, Dtype::FP16);
            scratch_db.zero();
        }
        linear_backward_batched_gpu(W, In, dOut, dIn_out, dW_acc,
                                    has_b ? *db_acc : scratch_db);
    };

    GpuTensor dX_from_Q(Lq, D, Dtype::FP16);
    GpuTensor dX_from_K(Lk, D_ctx, Dtype::FP16);
    GpuTensor dX_from_V(Lk, D_ctx, Dtype::FP16);

    run_proj_back(Wq, X,      dQ, dX_from_Q, dWq, bq, dbq);
    run_proj_back(Wk, kv_src, dK, dX_from_K, dWk, bk, dbk);
    run_proj_back(Wv, kv_src, dV, dX_from_V, dWv, bv, dbv);

    // ── 6. Accumulate dX / dCtx. ───────────────────────────────────────────
    // Self-attn: dX = dQ-path + dK-path + dV-path (all share input X).
    // Cross-attn: dX = dQ-path; dCtx = dK-path + dV-path (kv_src side).
    {
        const int blocks = (Lq * D + 255) / 256;
        // dX <- dX_from_Q.
        fa_fp16_add_inplace_kernel<<<blocks, 256>>>(
            reinterpret_cast<__half*>(dX.data_fp16()),
            reinterpret_cast<const __half*>(dX_from_Q.data_fp16()),
            Lq * D);
    }
    if (self_attn) {
        const int blocks = (Lq * D + 255) / 256;
        fa_fp16_add_inplace_kernel<<<blocks, 256>>>(
            reinterpret_cast<__half*>(dX.data_fp16()),
            reinterpret_cast<const __half*>(dX_from_K.data_fp16()),
            Lq * D);
        fa_fp16_add_inplace_kernel<<<blocks, 256>>>(
            reinterpret_cast<__half*>(dX.data_fp16()),
            reinterpret_cast<const __half*>(dX_from_V.data_fp16()),
            Lq * D);
    } else {
        const int blocks = (Lk * D_ctx + 255) / 256;
        fa_fp16_add_inplace_kernel<<<blocks, 256>>>(
            reinterpret_cast<__half*>(dCtx->data_fp16()),
            reinterpret_cast<const __half*>(dX_from_K.data_fp16()),
            Lk * D_ctx);
        fa_fp16_add_inplace_kernel<<<blocks, 256>>>(
            reinterpret_cast<__half*>(dCtx->data_fp16()),
            reinterpret_cast<const __half*>(dX_from_V.data_fp16()),
            Lk * D_ctx);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
