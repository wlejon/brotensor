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
        flash_attention_kernel<<<grid, FA_BLOCK, shmem>>>(
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

    GpuTensor Qp(Lq, D, Dtype::FP16);
    GpuTensor Kp(Lk, D, Dtype::FP16);
    GpuTensor Vp(Lk, D, Dtype::FP16);
    GpuTensor Op(Lq, D, Dtype::FP16);

    linear_forward_batched_fp16_gpu(Wq, bq, X,      Qp);
    linear_forward_batched_fp16_gpu(Wk, bk, kv_src, Kp);
    linear_forward_batched_fp16_gpu(Wv, bv, kv_src, Vp);

    flash_attention_forward_gpu(Qp, Kp, Vp, d_mask, num_heads, causal, Op);

    linear_forward_batched_fp16_gpu(Wo, bo, Op, O);
}

} // namespace brotensor
