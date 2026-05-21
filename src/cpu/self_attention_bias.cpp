// ─── CPU self-attention with additive pre-softmax bias ─────────────────────
//
// FP32 scalar host implementation. Ports src/cuda/self_attention_bias.cu —
// FP32 path only (the CPU backend is FP32-only per CLAUDE.md).
//
// Multi-head self-attention with an optional per-head (L, L) additive bias:
//   S[h,q,k] = scale * (Q_h[q] . K_h[k]) + attn_bias[h*L+q, k]
//   O        = concat_h( softmax_k(S[h]) @ V_h ) @ Wo
//
// Wq/Wk/Wv/Wo are (D, D); per-head split takes contiguous weight rows
// hh*dh .. hh*dh+dh. d_mask is a length-L key-validity buffer that also gates
// padded query rows (their output row is zeroed). attn_bias is FP32 or null.
// O is fully overwritten.

#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("self_attention_bias_forward: ") +
                                 name + " must be FP32 (CPU backend is FP32-only)");
    }
}

} // namespace

void self_attention_bias_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Wq,
                                 const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv,
                                 const ::brotensor::Tensor& Wo,
                                 const float* d_mask,
                                 const ::brotensor::Tensor* attn_bias,
                                 int num_heads, float scale,
                                 ::brotensor::Tensor& O) {
    check_fp32(X, "X");
    check_fp32(Wq, "Wq"); check_fp32(Wk, "Wk");
    check_fp32(Wv, "Wv"); check_fp32(Wo, "Wo");
    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("self_attention_bias_forward: num_heads must divide D");
    }
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo must be (D, D)");
    }
    const int H  = num_heads;
    const int dh = D / H;
    const float* bias = nullptr;
    if (attn_bias && attn_bias->data) {
        check_fp32(*attn_bias, "attn_bias");
        if (attn_bias->size() != H * L * L) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be (num_heads*L, L)");
        }
        bias = attn_bias->host_f32();
    }
    if (O.rows != L || O.cols != D || O.dtype != Dtype::FP32) {
        O.resize(L, D, Dtype::FP32);
    }
    if (L == 0 || D == 0) return;

    const float* Xp  = X.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* Op = O.host_f32_mut();

    // Per-head projections: Qh / Kh / Vh laid out (H*L, dh).
    std::vector<float> Qh(static_cast<size_t>(H) * L * dh);
    std::vector<float> Kh(static_cast<size_t>(H) * L * dh);
    std::vector<float> Vh(static_cast<size_t>(H) * L * dh);
    auto project = [&](const float* W, std::vector<float>& Out) {
        for (int hh = 0; hh < H; ++hh) {
            for (int i = 0; i < L; ++i) {
                const float* xr = Xp + static_cast<size_t>(i) * D;
                for (int j = 0; j < dh; ++j) {
                    const float* wr = W + static_cast<size_t>(hh * dh + j) * D;
                    float acc = 0.0f;
                    for (int k = 0; k < D; ++k) acc += xr[k] * wr[k];
                    Out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc;
                }
            }
        }
    };
    project(Wqp, Qh);
    project(Wkp, Kh);
    project(Wvp, Vh);

    // Yconcat (L, D): per-head attention output, concatenated.
    std::vector<float> Yc(static_cast<size_t>(L) * D, 0.0f);
    std::vector<float> srow(L);

    for (int hh = 0; hh < H; ++hh) {
        for (int i = 0; i < L; ++i) {
            if (d_mask && d_mask[i] < 0.5f) continue;  // padded query → 0 row
            // scores = scale * Q.K + bias
            const float* qr = &Qh[(static_cast<size_t>(hh) * L + i) * dh];
            float row_max = -1e30f;
            for (int j = 0; j < L; ++j) {
                if (d_mask && d_mask[j] < 0.5f) { srow[j] = 0.0f; continue; }
                const float* kr = &Kh[(static_cast<size_t>(hh) * L + j) * dh];
                float s = 0.0f;
                for (int k = 0; k < dh; ++k) s += qr[k] * kr[k];
                s *= scale;
                if (bias) s += bias[(static_cast<size_t>(hh) * L + i) * L + j];
                srow[j] = s;
                if (s > row_max) row_max = s;
            }
            // softmax over valid keys
            float sum = 0.0f;
            for (int j = 0; j < L; ++j) {
                if (d_mask && d_mask[j] < 0.5f) { srow[j] = 0.0f; continue; }
                const float e = std::exp(srow[j] - row_max);
                srow[j] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            // weighted sum of V
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < L; ++j) {
                    if (d_mask && d_mask[j] < 0.5f) continue;
                    acc += srow[j] * inv *
                           Vh[(static_cast<size_t>(hh) * L + j) * dh + k];
                }
                Yc[static_cast<size_t>(i) * D + (hh * dh + k)] = acc;
            }
        }
    }

    // Output projection O = Yconcat @ Wo^T, query-mask gated.
    for (int i = 0; i < L; ++i) {
        if (d_mask && d_mask[i] < 0.5f) {
            for (int c = 0; c < D; ++c) Op[static_cast<size_t>(i) * D + c] = 0.0f;
            continue;
        }
        const float* yr = &Yc[static_cast<size_t>(i) * D];
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<size_t>(c) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            Op[static_cast<size_t>(i) * D + c] = acc;
        }
    }
}

} // namespace brotensor::detail::cpu
