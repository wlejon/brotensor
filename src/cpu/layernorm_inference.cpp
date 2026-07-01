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

        float sum = 0.0f, sumsq = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float v = xr[i];
            sum   += v;
            sumsq += v * v;
        }
        const float mean = sum * invD;
        const float var  = sumsq * invD - mean * mean;
        const float rstd = 1.0f / std::sqrt(var + eps);

        for (int i = 0; i < D; ++i) {
            const float xh = (xr[i] - mean) * rstd;
            yr[i] = gp[i] * xh + bp[i];
        }
    }
}

void layernorm_forward_batched_with_caches(const ::brotensor::Tensor& X_RD,
                                           const ::brotensor::Tensor& gamma,
                                           const ::brotensor::Tensor& beta,
                                           ::brotensor::Tensor& Y_RD,
                                           ::brotensor::Tensor& Xhat_RD,
                                           ::brotensor::Tensor& Mean_R,
                                           ::brotensor::Tensor& Rstd_R,
                                           float eps) {
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D ||
        Y_RD.dtype != ::brotensor::Dtype::FP32) {
        Y_RD.resize(R, D, ::brotensor::Dtype::FP32);
    }
    if (Xhat_RD.rows != R || Xhat_RD.cols != D ||
        Xhat_RD.dtype != ::brotensor::Dtype::FP32) {
        Xhat_RD.resize(R, D, ::brotensor::Dtype::FP32);
    }
    if (Mean_R.rows != R || Mean_R.cols != 1 ||
        Mean_R.dtype != ::brotensor::Dtype::FP32) {
        Mean_R.resize(R, 1, ::brotensor::Dtype::FP32);
    }
    if (Rstd_R.rows != R || Rstd_R.cols != 1 ||
        Rstd_R.dtype != ::brotensor::Dtype::FP32) {
        Rstd_R.resize(R, 1, ::brotensor::Dtype::FP32);
    }
    if (R == 0 || D == 0) return;

    const float* xp = X_RD.host_f32();
    const float* gp = gamma.host_f32();
    const float* bp = beta.host_f32();
    float* yp = Y_RD.host_f32_mut();
    float* hp = Xhat_RD.host_f32_mut();
    float* mp = Mean_R.host_f32_mut();
    float* sp = Rstd_R.host_f32_mut();

    const float invD = 1.0f / static_cast<float>(D);
    for (int row = 0; row < R; ++row) {
        const float* xr = xp + static_cast<std::size_t>(row) * D;
        float* yr = yp + static_cast<std::size_t>(row) * D;
        float* hr = hp + static_cast<std::size_t>(row) * D;

        float sum = 0.0f, sumsq = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float v = xr[i];
            sum   += v;
            sumsq += v * v;
        }
        const float mean = sum * invD;
        const float var  = sumsq * invD - mean * mean;
        const float rstd = 1.0f / std::sqrt(var + eps);

        mp[row] = mean;
        sp[row] = rstd;

        for (int i = 0; i < D; ++i) {
            const float xh = (xr[i] - mean) * rstd;
            hr[i] = xh;
            yr[i] = gp[i] * xh + bp[i];
        }
    }
}

void layernorm_backward_batched_with_caches(const ::brotensor::Tensor& dY_RD,
                                            const ::brotensor::Tensor& Xhat_RD,
                                            const ::brotensor::Tensor& gamma,
                                            const ::brotensor::Tensor& Rstd_R,
                                            ::brotensor::Tensor& dX_RD,
                                            ::brotensor::Tensor& dGamma,
                                            ::brotensor::Tensor& dBeta) {
    const int R = dY_RD.rows;
    const int D = dY_RD.cols;
    if (dX_RD.rows != R || dX_RD.cols != D ||
        dX_RD.dtype != ::brotensor::Dtype::FP32) {
        dX_RD.resize(R, D, ::brotensor::Dtype::FP32);
    }
    if (R == 0 || D == 0) return;

    const float* dyp = dY_RD.host_f32();
    const float* hp  = Xhat_RD.host_f32();
    const float* gp  = gamma.host_f32();
    const float* sp  = Rstd_R.host_f32();
    float* dxp = dX_RD.host_f32_mut();
    float* dgp = dGamma.host_f32_mut();
    float* dbp = dBeta.host_f32_mut();

    const float nf = static_cast<float>(D);
    for (int row = 0; row < R; ++row) {
        const float* dyr = dyp + static_cast<std::size_t>(row) * D;
        const float* hr  = hp  + static_cast<std::size_t>(row) * D;
        float* dxr = dxp + static_cast<std::size_t>(row) * D;
        const float rstd = sp[row];

        float sum_dxh = 0.0f;
        float sum_dxh_xhat = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float g = dyr[i];
            dgp[i] += g * hr[i];
            dbp[i] += g;
            const float dxh = g * gp[i];
            sum_dxh += dxh;
            sum_dxh_xhat += dxh * hr[i];
        }
        const float scale = rstd / nf;
        for (int i = 0; i < D; ++i) {
            const float dxh = dyr[i] * gp[i];
            dxr[i] = scale * (nf * dxh - sum_dxh - hr[i] * sum_dxh_xhat);
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
