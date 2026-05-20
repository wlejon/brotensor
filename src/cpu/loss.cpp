// ─── CPU loss ops (CHUNK 1) ────────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU loss kernels in
// src/cuda/loss.cu — formulas reproduced verbatim, FP32 path only.
//
//   mse_vec_forward    — mean of squared error over all elements.
//   mse_vec_backward   — dPred = (2/n) * (pred - target), overwrite.
//   softmax_xent_fused — stable softmax + cross-entropy over the flat
//                        tensor; writes probs, dLogits = p - t, returns loss.

#include <brotensor/tensor.h>

#include <cmath>

namespace brotensor::detail::cpu {

float mse_vec_forward(const ::brotensor::Tensor& pred,
                      const ::brotensor::Tensor& target) {
    const int n = pred.size();
    if (n == 0) return 0.0f;
    const float* pp = pred.host_f32();
    const float* tp = target.host_f32();
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = pp[i] - tp[i];
        sum += d * d;
    }
    return sum / static_cast<float>(n);
}

void mse_vec_backward(const ::brotensor::Tensor& pred,
                      const ::brotensor::Tensor& target,
                      ::brotensor::Tensor& dPred) {
    const int n = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols ||
        dPred.dtype != ::brotensor::Dtype::FP32) {
        dPred.resize(pred.rows, pred.cols, ::brotensor::Dtype::FP32);
    }
    if (n == 0) return;
    const float scale = 2.0f / static_cast<float>(n);
    const float* pp = pred.host_f32();
    const float* tp = target.host_f32();
    float* dp = dPred.host_f32_mut();
    for (int i = 0; i < n; ++i) dp[i] = scale * (pp[i] - tp[i]);
}

float softmax_xent_fused(const ::brotensor::Tensor& logits,
                         const ::brotensor::Tensor& target,
                         const float* d_mask,
                         ::brotensor::Tensor& probs,
                         ::brotensor::Tensor& dLogits) {
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols ||
        probs.dtype != ::brotensor::Dtype::FP32) {
        probs.resize(logits.rows, logits.cols, ::brotensor::Dtype::FP32);
    }
    if (dLogits.rows != logits.rows || dLogits.cols != logits.cols ||
        dLogits.dtype != ::brotensor::Dtype::FP32) {
        dLogits.resize(logits.rows, logits.cols, ::brotensor::Dtype::FP32);
    }
    if (n == 0) return 0.0f;

    const float* lp = logits.host_f32();
    const float* tp = target.host_f32();
    float* pp = probs.host_f32_mut();
    float* dp = dLogits.host_f32_mut();

    // Stable softmax over the flat tensor (single segment).
    float m = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (d_mask && d_mask[i] < 0.5f) continue;
        if (lp[i] > m) m = lp[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (d_mask && d_mask[i] < 0.5f) { pp[i] = 0.0f; continue; }
        const float e = std::exp(lp[i] - m);
        pp[i] = e;
        sum += e;
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    float loss = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (d_mask && d_mask[i] < 0.5f) { dp[i] = 0.0f; continue; }
        const float p = pp[i] * inv;
        pp[i] = p;
        const float t = tp[i];
        if (t > 0.0f) {
            const float pc = p > 1e-12f ? p : 1e-12f;
            loss -= t * std::log(pc);
        }
        dp[i] = p - t;
    }
    return loss;
}

} // namespace brotensor::detail::cpu
