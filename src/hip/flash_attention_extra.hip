// Native AMD HIP Flash Attention Decode and Packed QKV for RDNA 3.5 (Wave32).
//
// Specialized routines for:
// - Incremental decode attention (Lq=1 per sequence, KV cache)
// - Masked incremental decode with softcapping
// - Packed QKV sequence attention with explicit sequence bounds

#include <brotensor/detail/dispatch.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/flash_internal.h"
#include "detail/hip_check.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::hip {

namespace {

using fa::FaScalar;
using fa::kFaBlock;
using fa::kFaHdChunk;
using fa::kMaxHdPerThread;
using fa::wave32_max;
using fa::wave32_sum;
using fa::cur_stream;

constexpr int kFaKtile = 64;

// ─── Incremental Decode Attention Kernel ─────────────────────────────────────

template <typename T>
__global__ void hip_flash_attention_decode_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K_cache,
        const T* __restrict__ V_cache,
        const float* __restrict__ mask,
        T* __restrict__ Out,
        int Lq, int valid_len, int Dq, int Dkv, int head_dim, int seq_offset,
        float attn_softcap, int window, int group, int pos_from_mask) {
    (void)Lq;
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + kFaKtile;

    const int q           = blockIdx.x;
    const int h           = blockIdx.y;
    const int d_base      = blockIdx.z * kFaHdChunk;
    const int d_end       = min(head_dim, d_base + kFaHdChunk);
    const int tid         = threadIdx.x;
    const int warp_id     = tid >> 5;
    const int lane        = tid & 31;
    const int num_warps   = blockDim.x >> 5;

    const int h_kv        = h / group;
    const int head_off_q  = h * head_dim;
    const int head_off_kv = h_kv * head_dim;
    const float inv_sqrt  = rsqrtf(static_cast<float>(head_dim));

    int p_q = seq_offset + q;
    if (pos_from_mask) {
        // decode_masked with a window: the query sits at the highest valid key
        // index of the fixed-capacity cache, not at its end (the CPU / CUDA
        // contract). Keys past it are masked anyway, so it bounds k_hi too.
        int local = -1;
        for (int kg = tid; kg <= p_q; kg += blockDim.x) {
            if (mask[kg] > 0.5f) local = kg;
        }
        float pmax = wave32_max(static_cast<float>(local));
        if (lane == 0) red[warp_id] = pmax;
        __syncthreads();
        pmax = -1.0f;
        if (tid < num_warps) pmax = red[tid];
        pmax = wave32_max(pmax);
        if (warp_id == 0) red[lane] = pmax;
        __syncthreads();
        pmax = red[0];
        __syncthreads();
        if (pmax >= 0.0f) p_q = static_cast<int>(pmax);
    }
    const int lo = (window > 0 && p_q - window + 1 > 0) ? (p_q - window + 1) : 0;
    const int k_hi = p_q;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[kMaxHdPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxHdPerThread; ++i) partial[i] = 0.0f;

    const T* q_vec = Q + static_cast<size_t>(q) * Dq + head_off_q;

    for (int k0 = (lo / kFaKtile) * kFaKtile; k0 <= k_hi; k0 += kFaKtile) {
        int klen = kFaKtile;
        if (k0 + klen - 1 > k_hi) klen = k_hi - k0 + 1;

        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            if (kg < lo || (mask && mask[kg] <= 0.5f)) {
                scores[t] = -1e30f;
            } else {
                const T* k_vec = K_cache + static_cast<size_t>(kg) * Dkv + head_off_kv;
                float s = FaScalar<T>::dot(q_vec, k_vec, head_dim) * inv_sqrt;
                if (attn_softcap > 0.0f) {
                    s = attn_softcap * tanhf(s / attn_softcap);
                }
                scores[t] = s;
            }
        }
        __syncthreads();

        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) local_max = fmaxf(local_max, scores[t]);
        local_max = wave32_max(local_max);
        if (lane == 0) red[warp_id] = local_max;
        __syncthreads();

        float tile_max = -1e30f;
        if (tid < num_warps) tile_max = red[tid];
        tile_max = wave32_max(tile_max);
        if (warp_id == 0) red[lane] = tile_max;
        __syncthreads();
        tile_max = red[0];
        // Every warp must have read the max before the sum reduction below
        // reuses red: a fast warp writing red[warp_id] raced a slow one here.
        __syncthreads();

        const float m_new = fmaxf(tile_max, run_max);
        const bool tile_empty = (m_new <= -1e29f);

        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
            local_sum += e;
        }
        local_sum = wave32_sum(local_sum);
        if (lane == 0) red[warp_id] = local_sum;
        __syncthreads();

        float tile_sum = 0.0f;
        if (tid < num_warps) tile_sum = red[tid];
        tile_sum = wave32_sum(tile_sum);
        if (warp_id == 0) red[lane] = tile_sum;
        __syncthreads();
        tile_sum = red[0];

        float alpha = (run_max <= -1e29f) ? 0.0f : __expf(run_max - m_new);

        int slot = 0;
        for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
            if (slot >= kMaxHdPerThread) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                const int kg = k0 + t;
                const T* v_vec = V_cache + static_cast<size_t>(kg) * Dkv + head_off_kv;
                acc += scores[t] * FaScalar<T>::to_f32(v_vec[d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
        if (slot >= kMaxHdPerThread) break;
        Out[static_cast<size_t>(q) * Dq + head_off_q + d] = FaScalar<T>::from_f32(partial[slot] * inv);
    }
}

// ─── Packed QKV In-place / Split Sequence Attention Kernel ───────────────────

template <typename T>
__global__ void hip_flash_attention_packed_qkv_kernel(
        const T* __restrict__ QKV,
        const int* __restrict__ bounds,
        int num_heads, int head_dim,
        int hw, T* __restrict__ Out) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + kFaKtile;

    const int q_global  = blockIdx.x;
    const int h         = blockIdx.y;
    const int d_base    = blockIdx.z * kFaHdChunk;
    const int d_end     = min(head_dim, d_base + kFaHdChunk);
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;
    const int num_warps = blockDim.x >> 5;

    const int D = num_heads * head_dim;
    const int stride_row = 3 * D;
    const int head_off_q = h * head_dim;
    const int head_off_k = D + head_off_q;
    const int head_off_v = 2 * D + head_off_q;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    // Bidirectional: row q attends to its own sequence [start, end), clipped
    // to the |q - k| <= hw band when windowed (hw < 0 = no window). Both
    // bounds are global row indices into the packed buffer.
    const int seq_start = bounds[2 * q_global];
    const int seq_end   = bounds[2 * q_global + 1];
    const int lo = hw >= 0 ? max(seq_start, q_global - hw) : seq_start;
    const int hi = hw >= 0 ? min(seq_end, q_global + hw + 1) : seq_end;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[kMaxHdPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxHdPerThread; ++i) partial[i] = 0.0f;

    const T* q_vec = QKV + static_cast<size_t>(q_global) * stride_row + head_off_q;

    for (int k0 = lo; k0 < hi; k0 += kFaKtile) {
        const int klen = min(kFaKtile, hi - k0);

        for (int t = tid; t < klen; t += blockDim.x) {
            const T* k_vec = QKV + static_cast<size_t>(k0 + t) * stride_row + head_off_k;
            scores[t] = FaScalar<T>::dot(q_vec, k_vec, head_dim) * inv_sqrt;
        }
        __syncthreads();

        float local_max = -1e30f;
        for (int t = tid; t < klen; t += blockDim.x) local_max = fmaxf(local_max, scores[t]);
        local_max = wave32_max(local_max);
        if (lane == 0) red[warp_id] = local_max;
        __syncthreads();

        float tile_max = -1e30f;
        if (tid < num_warps) tile_max = red[tid];
        tile_max = wave32_max(tile_max);
        if (warp_id == 0) red[lane] = tile_max;
        __syncthreads();
        tile_max = red[0];
        // Every warp must have read the max before the sum reduction below
        // reuses red: a fast warp writing red[warp_id] raced a slow one here.
        __syncthreads();

        const float m_new = fmaxf(tile_max, run_max);
        const bool tile_empty = (m_new <= -1e29f);

        float local_sum = 0.0f;
        for (int t = tid; t < klen; t += blockDim.x) {
            const float e = tile_empty ? 0.0f : __expf(scores[t] - m_new);
            scores[t] = e;
            local_sum += e;
        }
        local_sum = wave32_sum(local_sum);
        if (lane == 0) red[warp_id] = local_sum;
        __syncthreads();

        float tile_sum = 0.0f;
        if (tid < num_warps) tile_sum = red[tid];
        tile_sum = wave32_sum(tile_sum);
        if (warp_id == 0) red[lane] = tile_sum;
        __syncthreads();
        tile_sum = red[0];

        float alpha = (run_max <= -1e29f) ? 0.0f : __expf(run_max - m_new);

        int slot = 0;
        for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
            if (slot >= kMaxHdPerThread) break;
            float acc = alpha * partial[slot];
            for (int t = 0; t < klen; ++t) {
                const T* v_vec = QKV + static_cast<size_t>(k0 + t) * stride_row + head_off_v;
                acc += scores[t] * FaScalar<T>::to_f32(v_vec[d]);
            }
            partial[slot] = acc;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int slot = 0;
    for (int d = d_base + tid; d < d_end; d += blockDim.x, ++slot) {
        if (slot >= kMaxHdPerThread) break;
        Out[static_cast<size_t>(q_global) * D + head_off_q + d] = FaScalar<T>::from_f32(partial[slot] * inv);
    }
}

} // namespace

void hip_flash_attention_packed_qkv_forward(const ::brotensor::Tensor& QKV,
                                            const ::brotensor::Tensor& seq_bounds,
                                            int num_heads, int window,
                                            ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const Dtype dt = QKV.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("hip_flash_attention_packed_qkv_forward: QKV must be FP16, BF16, or FP32");
    }
    const int total_tokens = QKV.rows;
    const int three_D = QKV.cols;
    if (three_D % (3 * num_heads) != 0) {
        throw std::runtime_error("hip_flash_attention_packed_qkv_forward: QKV.cols not divisible by 3*num_heads");
    }
    const int D = three_D / 3;
    const int head_dim = D / num_heads;
    if (seq_bounds.dtype != Dtype::INT32 || seq_bounds.rows != total_tokens || seq_bounds.cols != 2) {
        throw std::runtime_error("hip_flash_attention_packed_qkv_forward: seq_bounds must be (L, 2) INT32");
    }
    // window is the full band width, matching the CPU/CUDA |q - k| <= window/2.
    const int hw = window > 0 ? window / 2 : -1;

    if (O.rows != total_tokens || O.cols != D || O.dtype != dt) {
        O.resize(total_tokens, D, dt);
    }
    if (total_tokens == 0) return;

    dim3 grid(total_tokens, num_heads, fa::fa_hd_chunks(head_dim));
    const size_t shmem = (static_cast<size_t>(kFaKtile) + kFaBlock) * sizeof(float);
    hipStream_t stream = cur_stream();

    if (dt == Dtype::FP16) {
        hip_flash_attention_packed_qkv_kernel<__half><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const __half*>(QKV.data), static_cast<const int*>(seq_bounds.data),
            num_heads, head_dim, hw, static_cast<__half*>(O.data));
    } else if (dt == Dtype::BF16) {
        hip_flash_attention_packed_qkv_kernel<hip_bfloat16><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const hip_bfloat16*>(QKV.data), static_cast<const int*>(seq_bounds.data),
            num_heads, head_dim, hw, static_cast<hip_bfloat16*>(O.data));
    } else {
        hip_flash_attention_packed_qkv_kernel<float><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const float*>(QKV.data), static_cast<const int*>(seq_bounds.data),
            num_heads, head_dim, hw, static_cast<float*>(O.data));
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_flash_attention_decode(const ::brotensor::Tensor& Q,
                                const ::brotensor::Tensor& K_cache,
                                const ::brotensor::Tensor& V_cache,
                                int valid_len, int num_q_heads, int num_kv_heads,
                                ::brotensor::Tensor& O,
                                float attn_softcap, int window) {
    using ::brotensor::Dtype;
    const Dtype dt = Q.dtype;
    const int Lq  = Q.rows;
    const int Dq  = Q.cols;
    const int Dkv = K_cache.cols;
    const int head_dim = Dq / num_q_heads;
    const int group = num_q_heads / num_kv_heads;

    if (O.rows != Lq || O.cols != Dq || O.dtype != dt) {
        O.resize(Lq, Dq, dt);
    }
    if (valid_len <= 0 || Lq <= 0) {
        O.zero();
        return;
    }

    const int seq_offset = valid_len - Lq;
    const size_t shmem = (static_cast<size_t>(kFaKtile) + kFaBlock) * sizeof(float);
    hipStream_t stream = cur_stream();
    dim3 grid(Lq, num_q_heads, fa::fa_hd_chunks(head_dim));

    if (dt == Dtype::FP16) {
        hip_flash_attention_decode_kernel<__half><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const __half*>(Q.data), static_cast<const __half*>(K_cache.data), static_cast<const __half*>(V_cache.data),
            nullptr, static_cast<__half*>(O.data),
            Lq, valid_len, Dq, Dkv, head_dim, seq_offset, attn_softcap, window, group, 0);
    } else if (dt == Dtype::BF16) {
        hip_flash_attention_decode_kernel<hip_bfloat16><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const hip_bfloat16*>(Q.data), static_cast<const hip_bfloat16*>(K_cache.data), static_cast<const hip_bfloat16*>(V_cache.data),
            nullptr, static_cast<hip_bfloat16*>(O.data),
            Lq, valid_len, Dq, Dkv, head_dim, seq_offset, attn_softcap, window, group, 0);
    } else {
        hip_flash_attention_decode_kernel<float><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const float*>(Q.data), static_cast<const float*>(K_cache.data), static_cast<const float*>(V_cache.data),
            nullptr, static_cast<float*>(O.data),
            Lq, valid_len, Dq, Dkv, head_dim, seq_offset, attn_softcap, window, group, 0);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_flash_attention_decode_masked(const ::brotensor::Tensor& Q,
                                       const ::brotensor::Tensor& K_cache,
                                       const ::brotensor::Tensor& V_cache,
                                       const float* d_mask,
                                       int num_q_heads, int num_kv_heads,
                                       ::brotensor::Tensor& O,
                                       float attn_softcap, int window) {
    using ::brotensor::Dtype;
    const Dtype dt = Q.dtype;
    const int Lq  = Q.rows;
    const int Dq  = Q.cols;
    const int Dkv = K_cache.cols;
    const int head_dim = Dq / num_q_heads;
    const int group = num_q_heads / num_kv_heads;
    const int valid_len = K_cache.rows;

    if (O.rows != 1 || O.cols != Dq || O.dtype != dt) {
        O.resize(1, Dq, dt);
    }
    if (valid_len <= 0) {
        O.zero();
        return;
    }

    const int seq_offset = valid_len - Lq;
    const size_t shmem = (static_cast<size_t>(kFaKtile) + kFaBlock) * sizeof(float);
    hipStream_t stream = cur_stream();
    dim3 grid(1, num_q_heads, fa::fa_hd_chunks(head_dim));

    if (dt == Dtype::FP16) {
        hip_flash_attention_decode_kernel<__half><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const __half*>(Q.data), static_cast<const __half*>(K_cache.data), static_cast<const __half*>(V_cache.data),
            d_mask, static_cast<__half*>(O.data),
            1, valid_len, Dq, Dkv, head_dim, seq_offset, attn_softcap, window, group,
            window > 0 ? 1 : 0);
    } else if (dt == Dtype::BF16) {
        hip_flash_attention_decode_kernel<hip_bfloat16><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const hip_bfloat16*>(Q.data), static_cast<const hip_bfloat16*>(K_cache.data), static_cast<const hip_bfloat16*>(V_cache.data),
            d_mask, static_cast<hip_bfloat16*>(O.data),
            1, valid_len, Dq, Dkv, head_dim, seq_offset, attn_softcap, window, group,
            window > 0 ? 1 : 0);
    } else {
        hip_flash_attention_decode_kernel<float><<<grid, kFaBlock, shmem, stream>>>(
            static_cast<const float*>(Q.data), static_cast<const float*>(K_cache.data), static_cast<const float*>(V_cache.data),
            d_mask, static_cast<float*>(O.data),
            1, valid_len, Dq, Dkv, head_dim, seq_offset, attn_softcap, window, group,
            window > 0 ? 1 : 0);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
