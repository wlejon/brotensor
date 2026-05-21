// KV-cache append + causal flash-attention decode. Phase 2G port — kernel
// bodies unchanged.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

void kv_cache_append(const Tensor& K_new, const Tensor& V_new,
                     int cur_len, Tensor& K_cache, Tensor& V_cache) {
    const Dtype dt = K_new.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("kv_cache_append: tensors must be FP16 or BF16");
    }
    if (V_new.dtype != dt || K_cache.dtype != dt || V_cache.dtype != dt) {
        throw std::runtime_error("kv_cache_append: all tensors must share the same dtype");
    }
    if (K_new.cols != V_new.cols || K_new.cols != K_cache.cols ||
        K_cache.cols != V_cache.cols) {
        throw std::runtime_error("kv_cache_append: column mismatch");
    }
    if (K_new.rows != V_new.rows) {
        throw std::runtime_error("kv_cache_append: K_new/V_new row mismatch");
    }
    if (K_cache.rows != V_cache.rows) {
        throw std::runtime_error("kv_cache_append: K_cache/V_cache row mismatch");
    }
    const int L_new = K_new.rows;
    const int L_max = K_cache.rows;
    const int D     = K_new.cols;
    if (cur_len < 0 || cur_len + L_new > L_max) {
        throw std::runtime_error("kv_cache_append: cur_len + L_new exceeds cache capacity");
    }
    if (L_new == 0 || D == 0) return;

    const size_t elem = 2; // FP16 and BF16 are both 2 bytes
    const size_t bytes = static_cast<size_t>(L_new) * D * elem;
    const size_t dst_off = static_cast<size_t>(cur_len) * D * elem;

    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char*>(K_cache.data) + dst_off,
        K_new.data, bytes, cudaMemcpyDeviceToDevice));
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char*>(V_cache.data) + dst_off,
        V_new.data, bytes, cudaMemcpyDeviceToDevice));
}

namespace {

constexpr int FAD_BLOCK = 128;
constexpr int FAD_KTILE = 64;

__global__ void flash_attention_decode_kernel(
        const __half* __restrict__ Q,
        const __half* __restrict__ K,
        const __half* __restrict__ V,
        __half* __restrict__ Out,
        int Lq, int valid_len, int D, int head_dim, int seq_offset) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FAD_KTILE;

    const int q   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int head_off = h * head_dim;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));
    const int p_q = seq_offset + q;

    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    float run_max = -1e30f;
    float run_sum = 0.0f;

    for (int k0 = 0; k0 < valid_len; k0 += FAD_KTILE) {
        if (k0 > p_q) break;
        int klen = (valid_len - k0) < FAD_KTILE ? (valid_len - k0) : FAD_KTILE;
        if (k0 + klen - 1 > p_q) klen = p_q - k0 + 1;

        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += __half2float(Q[q * D + head_off + d]) *
                       __half2float(K[kg * D + head_off + d]);
            }
            scores[t] = dot * inv_sqrt;
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
                       __half2float(V[(k0 + t) * D + head_off + d]);
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
        Out[q * D + head_off + d] = __float2half(partial[slot] * inv);
    }
}

// BF16 twin: verbatim copy of flash_attention_decode_kernel with
// __half -> __nv_bfloat16, __half2float -> __bfloat162float,
// __float2half -> __float2bfloat16.
__global__ void flash_attention_decode_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V,
        __nv_bfloat16* __restrict__ Out,
        int Lq, int valid_len, int D, int head_dim, int seq_offset) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FAD_KTILE;

    const int q   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int head_off = h * head_dim;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));
    const int p_q = seq_offset + q;

    constexpr int MAX_HD_PER_THREAD = 8;
    float partial[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    float run_max = -1e30f;
    float run_sum = 0.0f;

    for (int k0 = 0; k0 < valid_len; k0 += FAD_KTILE) {
        if (k0 > p_q) break;
        int klen = (valid_len - k0) < FAD_KTILE ? (valid_len - k0) : FAD_KTILE;
        if (k0 + klen - 1 > p_q) klen = p_q - k0 + 1;

        for (int t = tid; t < klen; t += blockDim.x) {
            const int kg = k0 + t;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += __bfloat162float(Q[q * D + head_off + d]) *
                       __bfloat162float(K[kg * D + head_off + d]);
            }
            scores[t] = dot * inv_sqrt;
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

} // namespace

void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_heads, Tensor& O) {
    if (Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16) {
        throw std::runtime_error("flash_attention_decode: tensors must be FP16 or BF16");
    }
    if (K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) {
        throw std::runtime_error("flash_attention_decode: all tensors must share the same dtype");
    }
    const int Lq = Q.rows;
    const int D  = Q.cols;
    if (K_cache.cols != D || V_cache.cols != D) {
        throw std::runtime_error("flash_attention_decode: K/V cache cols must match Q.cols");
    }
    if (valid_len < 0 || valid_len > K_cache.rows || valid_len > V_cache.rows) {
        throw std::runtime_error("flash_attention_decode: invalid valid_len");
    }
    if (valid_len < Lq) {
        throw std::runtime_error("flash_attention_decode: valid_len must be >= Lq");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_decode: num_heads must divide D");
    }
    const int head_dim = D / num_heads;
    if ((head_dim + FAD_BLOCK - 1) / FAD_BLOCK > 8) {
        throw std::runtime_error("flash_attention_decode: head_dim too large (max 8 * FAD_BLOCK = 1024)");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Q.dtype) {
        O.resize(Lq, D, Q.dtype);
    }
    if (Lq == 0 || D == 0 || valid_len == 0) return;

    const int seq_offset = valid_len - Lq;
    const size_t shmem = (static_cast<size_t>(FAD_KTILE) + FAD_BLOCK) * sizeof(float);
    dim3 grid(Lq, num_heads, 1);
    if (Q.dtype == Dtype::FP16) {
        flash_attention_decode_kernel<<<grid, FAD_BLOCK, shmem>>>(
            static_cast<const __half*>(Q.data),
            static_cast<const __half*>(K_cache.data),
            static_cast<const __half*>(V_cache.data),
            static_cast<__half*>(O.data),
            Lq, valid_len, D, head_dim, seq_offset);
    } else {
        flash_attention_decode_bf16_kernel<<<grid, FAD_BLOCK, shmem>>>(
            static_cast<const __nv_bfloat16*>(Q.data),
            static_cast<const __nv_bfloat16*>(K_cache.data),
            static_cast<const __nv_bfloat16*>(V_cache.data),
            static_cast<__nv_bfloat16*>(O.data),
            Lq, valid_len, D, head_dim, seq_offset);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
