#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include "flash_fused_internal.cuh"
#include "ops_fa_mir.h"
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

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

    detail::cuda::dispatch_fa_rows(
        detail::cuda::kFaModeVarlen, dt, Q.device,
        Q.data, K.data, V.data, O.data,
        nullptr, cu_seqlens_q, cu_seqlens_k,
        total_q, total_k, D, D, D,
        num_heads, num_heads, head_dim,
        0, causal, false, batch_size, 0, 0.0f);
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
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt)
        O.resize(Lq, Dq, dt);
    if (Lq == 0 || Lk == 0 || Dq == 0) return;

    detail::cuda::dispatch_fa_rows(
        detail::cuda::kFaModeRows, dt, Q.device,
        Q.data, K.data, V.data, O.data,
        d_mask, nullptr, nullptr,
        Lq, Lk, Dq, Dkv, Dq,
        num_heads, n_kv, head_dim,
        window, causal, false, 0, Lk - Lq, 0.0f);
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
    if (O.rows != Lq || O.cols != Dq || O.dtype != dt)
        O.resize(Lq, Dq, dt);
    if (Lq == 0 || Lk == 0 || Dq == 0) return;

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
    detail::cuda::dispatch_fa_rows(
        detail::cuda::kFaModeRows, dt, Q.device,
        Q.data, K.data, V.data, O.data,
        d_mask, nullptr, nullptr,
        Lq, Lk, Dq, Dkv, Dq,
        num_q_heads, num_kv_heads, head_dim,
        0, causal, false, 0, Lk - Lq, 0.0f);
}

} // namespace detail::cuda

} // namespace brotensor
