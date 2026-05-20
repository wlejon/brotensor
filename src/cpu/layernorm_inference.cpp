// ─── CPU inference-batched layernorm + causal-mask helper (CHUNK 1) ────────
//
// FP32 scalar host implementations.
//
//   layernorm_forward_inference_batched — per-row LayerNorm over an (R, D)
//       tensor. Ports src/cuda/layernorm.cu's
//       layernorm_forward_inference_batched_kernel: mean/variance per row,
//       rstd = 1/sqrt(var + eps), y = gamma * xhat + beta. FP32 only.
//
//   build_causal_mask_row — fills an (L, 1) mask: mask[k] = (k <= q) ? 1 : 0.
//       Ports src/cuda/elementwise.cu's causal_mask_row_kernel. Its only
//       Tensor operand is the output; the dispatcher resolves the device
//       from `mask` itself (see src/ops.cpp::build_causal_mask_row).

#include <brotensor/tensor.h>

#include <cmath>
#include <cstddef>

namespace brotensor::detail::cpu {

void layernorm_forward_inference_batched(const ::brotensor::Tensor& X_RD,
                                         const ::brotensor::Tensor& gamma,
                                         const ::brotensor::Tensor& beta,
                                         ::brotensor::Tensor& Y_RD,
                                         float eps) {
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D ||
        Y_RD.dtype != ::brotensor::Dtype::FP32) {
        Y_RD.resize(R, D, ::brotensor::Dtype::FP32);
    }
    if (R == 0 || D == 0) return;

    const float* xp = X_RD.host_f32();
    const float* gp = gamma.host_f32();
    const float* bp = beta.host_f32();
    float* yp = Y_RD.host_f32_mut();

    const float invD = 1.0f / static_cast<float>(D);
    for (int row = 0; row < R; ++row) {
        const float* xr = xp + static_cast<std::size_t>(row) * D;
        float* yr = yp + static_cast<std::size_t>(row) * D;

        float mean = 0.0f;
        for (int i = 0; i < D; ++i) mean += xr[i];
        mean *= invD;

        float var = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float d = xr[i] - mean;
            var += d * d;
        }
        var *= invD;
        const float rstd = 1.0f / std::sqrt(var + eps);

        for (int i = 0; i < D; ++i) {
            const float xh = (xr[i] - mean) * rstd;
            yr[i] = gp[i] * xh + bp[i];
        }
    }
}

void build_causal_mask_row(int L, int q, ::brotensor::Tensor& mask) {
    if (mask.rows != L || mask.cols != 1 ||
        mask.dtype != ::brotensor::Dtype::FP32) {
        mask.resize(L, 1, ::brotensor::Dtype::FP32);
    }
    if (L <= 0) return;
    float* mp = mask.host_f32_mut();
    for (int k = 0; k < L; ++k) mp[k] = (k <= q) ? 1.0f : 0.0f;
}

} // namespace brotensor::detail::cpu
