// ─── CPU KV-cache append + flash-attention decode (CHUNK 4) ────────────────
//
// FP32 scalar host implementations. Ports src/cuda/kv_cache.cu.
//
// IMPORTANT — dtype: the GPU kv-cache ops run FP16 internally (their tensors
// must be FP16). The CPU backend is FP32-only, so the CPU impls require FP32
// tensors. CPU↔GPU parity feeds FP16 to the GPU and FP32 to the CPU and
// compares with a loose FP16-driven tolerance (see test_kv_cache_parity.cpp).
//
// ── kv_cache_append ──
//   Copies the L_new rows of K_new / V_new into K_cache / V_cache starting at
//   row `cur_len`. Layout is row-major (rows, D); the destination slice is
//   rows [cur_len, cur_len+L_new). Caches are pre-allocated by the caller to
//   their max length L_max; rows outside the written slice are untouched.
//   The op OVERWRITES exactly the [cur_len, cur_len+L_new) slice.
//
// ── flash_attention_decode ──
//   Causal multi-head attention of Lq query rows against the first valid_len
//   rows of the K/V cache. Q/K/V/O are (rows, D) with D = num_heads*head_dim;
//   head h occupies columns [h*head_dim, (h+1)*head_dim). Scale = 1/sqrt(hd).
//   Query row q maps to absolute sequence position p_q = seq_offset + q where
//   seq_offset = valid_len - Lq; key kg is attended iff kg <= p_q (causal).
//   Softmax over the valid causal keys; O OVERWRITTEN. Numerically this is a
//   plain (max-subtracted) softmax — equivalent to the GPU's online/streaming
//   softmax, just without the tiling.

#include <brotensor/tensor.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

} // namespace

void kv_cache_append(const ::brotensor::Tensor& K_new,
                     const ::brotensor::Tensor& V_new,
                     int cur_len,
                     ::brotensor::Tensor& K_cache,
                     ::brotensor::Tensor& V_cache) {
    check_fp32(K_new,   "kv_cache_append", "K_new");
    check_fp32(V_new,   "kv_cache_append", "V_new");
    check_fp32(K_cache, "kv_cache_append", "K_cache");
    check_fp32(V_cache, "kv_cache_append", "V_cache");
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

    const std::size_t n = static_cast<std::size_t>(L_new) * D * sizeof(float);
    const std::size_t dst_off = static_cast<std::size_t>(cur_len) * D;
    std::memcpy(K_cache.host_f32_mut() + dst_off, K_new.host_f32(), n);
    std::memcpy(V_cache.host_f32_mut() + dst_off, V_new.host_f32(), n);
}

void flash_attention_decode(const ::brotensor::Tensor& Q,
                            const ::brotensor::Tensor& K_cache,
                            const ::brotensor::Tensor& V_cache,
                            int valid_len,
                            int num_q_heads, int num_kv_heads,
                            ::brotensor::Tensor& O) {
    check_fp32(Q,       "flash_attention_decode", "Q");
    check_fp32(K_cache, "flash_attention_decode", "K_cache");
    check_fp32(V_cache, "flash_attention_decode", "V_cache");
    const int Lq  = Q.rows;
    const int Dq  = Q.cols;
    const int Dkv = K_cache.cols;
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
    const int q_per_kv = num_q_heads / num_kv_heads;
    if (O.rows != Lq || O.cols != Dq || O.dtype != Dtype::FP32) {
        O.resize(Lq, Dq, Dtype::FP32);
    }
    if (Lq == 0 || Dq == 0 || valid_len == 0) return;

    const int seq_offset = valid_len - Lq;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const float* Qp = Q.host_f32();
    const float* Kp = K_cache.host_f32();
    const float* Vp = V_cache.host_f32();
    float* Op = O.host_f32_mut();

    std::vector<float> scores;
    for (int q = 0; q < Lq; ++q) {
        const int p_q = seq_offset + q;
        const int klen = p_q + 1;   // causal: keys 0..p_q inclusive
        for (int hq = 0; hq < num_q_heads; ++hq) {
            const int hkv = hq / q_per_kv;
            const int q_head_off  = hq  * head_dim;
            const int kv_head_off = hkv * head_dim;
            const float* qrow = Qp + q * Dq + q_head_off;

            // Scores against the valid causal keys.
            scores.assign(klen, 0.0f);
            float run_max = -1e30f;
            for (int kg = 0; kg < klen; ++kg) {
                const float* krow = Kp + kg * Dkv + kv_head_off;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += qrow[d] * krow[d];
                const float s = dot * inv_sqrt;
                scores[kg] = s;
                if (s > run_max) run_max = s;
            }
            // Stable softmax.
            float sum = 0.0f;
            for (int kg = 0; kg < klen; ++kg) {
                const float e = std::exp(scores[kg] - run_max);
                scores[kg] = e;
                sum += e;
            }
            const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            // Weighted sum of V (KV head's V).
            float* orow = Op + q * Dq + q_head_off;
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int kg = 0; kg < klen; ++kg) {
                    acc += scores[kg] * Vp[kg * Dkv + kv_head_off + d];
                }
                orow[d] = acc * inv;
            }
        }
    }
}

} // namespace brotensor::detail::cpu
