// KV-cache append + causal flash-attention decode.

#include "detail/cuda_check.h"
#include "decode_gqa.cuh"
#include "ops_fa_mir.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

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
        K_new.data, bytes, cudaMemcpyDeviceToDevice, cur_stream()));
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char*>(V_cache.data) + dst_off,
        V_new.data, bytes, cudaMemcpyDeviceToDevice, cur_stream()));
}

void flash_attention_decode(const Tensor& Q,
                            const Tensor& K_cache, const Tensor& V_cache,
                            int valid_len, int num_q_heads, int num_kv_heads,
                            Tensor& O, float attn_softcap, int window) {
    if (Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16) {
        throw std::runtime_error("flash_attention_decode: tensors must be FP16 or BF16");
    }
    if (K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) {
        throw std::runtime_error("flash_attention_decode: all tensors must share the same dtype");
    }
    const int Lq  = Q.rows;
    const int Dq  = Q.cols;            // num_q_heads  * head_dim
    const int Dkv = K_cache.cols;      // num_kv_heads * head_dim
    if (V_cache.cols != Dkv) {
        throw std::runtime_error("flash_attention_decode: K_cache.cols != V_cache.cols");
    }
    if (valid_len < 0 || valid_len > K_cache.rows || valid_len > V_cache.rows) {
        throw std::runtime_error("flash_attention_decode: invalid valid_len");
    }
    if (valid_len < Lq) {
        throw std::runtime_error("flash_attention_decode: valid_len must be >= Lq");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error("flash_attention_decode: num_q_heads / num_kv_heads must be positive");
    }
    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode: num_kv_heads must divide num_q_heads");
    }
    if (Dq % num_q_heads != 0 || Dkv % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode: head_dim does not divide cols cleanly");
    }
    const int head_dim = Dq / num_q_heads;
    if (Dkv / num_kv_heads != head_dim) {
        throw std::runtime_error("flash_attention_decode: head_dim mismatch between Q and K/V");
    }
    if (O.rows != Lq || O.cols != Dq || O.dtype != Q.dtype) {
        O.resize(Lq, Dq, Q.dtype);
    }
    if (Lq == 0 || Dq == 0 || valid_len == 0) return;

    dispatch_fa_rows(kFaModeRows, Q.dtype, Q.device,
                     Q.data, K_cache.data, V_cache.data, O.data,
                     nullptr, nullptr, nullptr,
                     Lq, valid_len, Dq, Dkv, Dq,
                     num_q_heads, num_kv_heads, head_dim,
                     window, true, false, 0, valid_len - Lq, attn_softcap);
}

void flash_attention_decode_masked(const Tensor& Q,
                                   const Tensor& K_cache,
                                   const Tensor& V_cache,
                                   const float* d_mask,
                                   int num_q_heads, int num_kv_heads,
                                   Tensor& O, float attn_softcap, int window) {
    if (Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16) {
        throw std::runtime_error("flash_attention_decode_masked: tensors must be FP16 or BF16");
    }
    if (K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) {
        throw std::runtime_error("flash_attention_decode_masked: all tensors must share the same dtype");
    }
    if (d_mask == nullptr) {
        throw std::runtime_error("flash_attention_decode_masked: d_mask must not be null");
    }
    if (Q.rows != 1) {
        throw std::runtime_error("flash_attention_decode_masked: Q must be a single row (L_q == 1)");
    }
    const int Dq  = Q.cols;
    const int Dkv = K_cache.cols;
    const int cap = K_cache.rows;
    if (V_cache.cols != Dkv) {
        throw std::runtime_error("flash_attention_decode_masked: K_cache.cols != V_cache.cols");
    }
    if (V_cache.rows != cap) {
        throw std::runtime_error("flash_attention_decode_masked: K_cache/V_cache row mismatch");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error("flash_attention_decode_masked: num_q_heads / num_kv_heads must be positive");
    }
    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode_masked: num_kv_heads must divide num_q_heads");
    }
    if (Dq % num_q_heads != 0 || Dkv % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention_decode_masked: head_dim does not divide cols cleanly");
    }
    const int head_dim = Dq / num_q_heads;
    if (Dkv / num_kv_heads != head_dim) {
        throw std::runtime_error("flash_attention_decode_masked: head_dim mismatch between Q and K/V");
    }
    if (O.rows != 1 || O.cols != Dq || O.dtype != Q.dtype) {
        O.resize(1, Dq, Q.dtype);
    }
    if (Dq == 0 || cap == 0) return;

    dispatch_fa_rows(kFaModeRows, Q.dtype, Q.device,
                     Q.data, K_cache.data, V_cache.data, O.data,
                     d_mask, nullptr, nullptr,
                     1, cap, Dq, Dkv, Dq,
                     num_q_heads, num_kv_heads, head_dim,
                     window, false, window > 0, 0, 0, attn_softcap);
}

void flash_attention_decode_len(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
                                const int32_t* d_len, int num_q_heads, int num_kv_heads, Tensor& O) {
    constexpr const char* op = "flash_attention_decode_len";
    auto fail = [&](const char* m) { throw std::runtime_error(std::string(op) + ": " + m); };
    if (Q.dtype != Dtype::FP16 && Q.dtype != Dtype::BF16) fail("tensors must be FP16 or BF16");
    if (K_cache.dtype != Q.dtype || V_cache.dtype != Q.dtype) fail("all tensors must share the same dtype");
    if (d_len == nullptr) fail("d_len must not be null");
    if (Q.rows < 1) fail("Q needs at least one row");
    const int R = Q.rows;
    const int Dq = Q.cols, Dkv = K_cache.cols, cap = K_cache.rows;
    if (V_cache.cols != Dkv || V_cache.rows != cap) fail("K_cache / V_cache shape mismatch");
    if (num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0)
        fail("num_kv_heads must be positive and divide num_q_heads");
    if (Dq % num_q_heads != 0 || Dkv != (Dq / num_q_heads) * num_kv_heads) fail("head layout mismatch");
    const int head_dim = Dq / num_q_heads;
    if (O.rows != R || O.cols != Dq || O.dtype != Q.dtype) O.resize(R, Dq, Q.dtype);
    if (Dq == 0 || cap == 0) return;
    const int group = num_q_heads / num_kv_heads;
    if (decode_gqa::supported(head_dim, group)) {
        decode_gqa::launch(Q.data, K_cache.data, V_cache.data, d_len, O.data, cap, num_q_heads, num_kv_heads,
                           head_dim, R, Q.dtype == Dtype::BF16, cur_stream());
        return;
    }
    dispatch_fa_rows(kFaModeLen, Q.dtype, Q.device,
                     Q.data, K_cache.data, V_cache.data, O.data,
                     nullptr, d_len, nullptr,
                     R, cap, Dq, Dkv, Dq,
                     num_q_heads, num_kv_heads, head_dim,
                     0, true, false, 0, 0, 0.0f);
}

} // namespace brotensor::detail::cuda
