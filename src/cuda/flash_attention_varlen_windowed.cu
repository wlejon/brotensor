#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include "flash_fused_internal.cuh"
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace {

constexpr int FA_BLOCK = 128;
constexpr int FA_KTILE = 64;

template <typename T>
__global__ void flash_attention_varlen_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const int* __restrict__ cu_q,
        const int* __restrict__ cu_k,
        T* __restrict__ Out,
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

    int b = 0;
    while (b < B && cu_q[b + 1] <= q_global) ++b;
    if (b >= B) return;
    const int q_beg = cu_q[b];
    const int k_beg = cu_k[b];
    const int k_end = cu_k[b + 1];
    const int Lk    = k_end - k_beg;
    if (Lk <= 0) {
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
        const float alpha = (run_max <= -1e29f) ? 0.0f : __expf(run_max - m_new);

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

template <typename T>
__global__ void flash_attention_windowed_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const float* __restrict__ mask,
        T* __restrict__ Out,
        int Lk, int Dq, int Dkv, int head_dim, int window, int q_offset,
        int group, int causal) {
    extern __shared__ float s_smem[];
    float* scores = s_smem;
    float* red    = s_smem + FA_KTILE;

    const int q           = blockIdx.x;
    const int h           = blockIdx.y;
    const int tid         = threadIdx.x;
    const int head_off    = h * head_dim;
    const int head_off_kv = (h / group) * head_dim;
    const float inv_sqrt  = rsqrtf(static_cast<float>(head_dim));

    const int aq = q + q_offset;
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

    const int k0_start = (lo / FA_KTILE) * FA_KTILE;
    for (int k0 = k0_start; k0 <= k_hi; k0 += FA_KTILE) {
        int klen = FA_KTILE;
        if (k0 + klen - 1 > k_hi) klen = k_hi - k0 + 1;

        const bool is_interior = (k0 >= lo) && (klen == FA_KTILE) && (mask == nullptr);
        if (is_interior) {
            for (int t = tid; t < FA_KTILE; t += blockDim.x) {
                const int kg = k0 + t;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += static_cast<float>(Q[q * Dq + head_off + d]) *
                           static_cast<float>(K[kg * Dkv + head_off_kv + d]);
                }
                scores[t] = dot * inv_sqrt;
            }
        } else {
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

        if (tile_sum == 0.0f && tile_max <= -1e29f) {
            __syncthreads();
            continue;
        }

        const float alpha = (run_max <= -1e29f) ? 0.0f : __expf(run_max - m_new);

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

} // namespace

namespace detail::cuda {

void flash_attention_forward(const Tensor& Q,
                             const Tensor& K,
                             const Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             Tensor& O);

bool flash_attention_varlen_f32_hd64(const float* Q, const float* K, const float* V,
                                     const int* cu_q, const int* cu_k, float* O,
                                     int total_q, int batch, int num_heads, int head_dim,
                                     bool causal, cudaStream_t stream);

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

    if (batch_size == 1 && !causal && (dt == Dtype::FP16 || dt == Dtype::BF16) &&
        (flash_fused::supported(head_dim) ||
         static_cast<size_t>(Q.rows) * static_cast<size_t>(K.rows) >= (size_t(1) << 20))) {
        flash_attention_forward(Q, K, V, /*d_mask=*/nullptr, num_heads,
                                /*causal=*/false, O);
        return;
    }

    if (batch_size > 1 && (dt == Dtype::FP16 || dt == Dtype::BF16) &&
        flash_fused::supported(head_dim) && max_seqlen_q > 0 && max_seqlen_k > 0 &&
        static_cast<long long>(total_q) == static_cast<long long>(batch_size) * max_seqlen_q &&
        static_cast<long long>(total_k) == static_cast<long long>(batch_size) * max_seqlen_k &&
        batch_size <= 65535 && (!causal || max_seqlen_q == max_seqlen_k)) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        if (dt == Dtype::BF16) {
            flash_fused::launch(reinterpret_cast<const __nv_bfloat16*>(Q.data),
                                reinterpret_cast<const __nv_bfloat16*>(K.data),
                                reinterpret_cast<const __nv_bfloat16*>(V.data), nullptr,
                                reinterpret_cast<__nv_bfloat16*>(O.data), max_seqlen_q,
                                max_seqlen_k, D, num_heads, head_dim, causal, stream, batch_size);
        } else {
            flash_fused::launch(reinterpret_cast<const __half*>(Q.data),
                                reinterpret_cast<const __half*>(K.data),
                                reinterpret_cast<const __half*>(V.data), nullptr,
                                reinterpret_cast<__half*>(O.data), max_seqlen_q, max_seqlen_k, D,
                                num_heads, head_dim, causal, stream, batch_size);
        }
        return;
    }

    if (dt == Dtype::FP32 &&
        flash_attention_varlen_f32_hd64(
            static_cast<const float*>(Q.data), static_cast<const float*>(K.data),
            static_cast<const float*>(V.data), reinterpret_cast<const int*>(cu_seqlens_q),
            reinterpret_cast<const int*>(cu_seqlens_k), static_cast<float*>(O.data), total_q,
            batch_size, num_heads, head_dim, causal,
            reinterpret_cast<cudaStream_t>(cuda_current_stream()))) {
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
    const int Dq  = Q.cols;
    const int Dkv = K.cols;
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
    if (dt != Dtype::FP32 && flash_fused::supported(head_dim)) {
        if (dt == Dtype::BF16)
            flash_fused::launch(reinterpret_cast<const __nv_bfloat16*>(Q.data),
                                reinterpret_cast<const __nv_bfloat16*>(K.data),
                                reinterpret_cast<const __nv_bfloat16*>(V.data), d_mask,
                                reinterpret_cast<__nv_bfloat16*>(O.data), Lq, Lk, Dq, num_q_heads, head_dim,
                                causal, stream, /*batch=*/1, num_kv_heads);
        else
            flash_fused::launch(reinterpret_cast<const __half*>(Q.data), reinterpret_cast<const __half*>(K.data),
                                reinterpret_cast<const __half*>(V.data), d_mask,
                                reinterpret_cast<__half*>(O.data), Lq, Lk, Dq, num_q_heads, head_dim, causal,
                                stream, /*batch=*/1, num_kv_heads);
        return;
    }
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

} // namespace detail::cuda

} // namespace brotensor
