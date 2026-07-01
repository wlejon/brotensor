// ─── CPU BatchNorm ops ─────────────────────────────────────────────────────
//
// Standard NCHW BatchNorm, FP32-only (CPU backend is FP32-only). Statistics
// are reduced over (N, H, W) for each channel — i.e. one mean / var per C.
// This is the variant pretrained ResNet / DETR-ResNet50 / classic
// Mask2Former backbones use; differs from GroupNorm (which reduces over
// a (channels_per_group, H, W) tile within a single sample).
//
// Three slots:
//
//   batch_norm_forward    — training. Computes per-channel batch mean/var
//                           over (N, H, W); writes Y; updates running_mean /
//                           running_var in place via momentum; saves
//                           batch mean and rstd for the backward pass.
//                           Convention:
//                             running = (1 - momentum) * running
//                                       + momentum     * batch_stat
//                           (PyTorch's nn.BatchNorm2d convention; the "batch"
//                           variance fed into running_var is the *unbiased*
//                           estimator, sumsq/(M-1) - mean*mean*M/(M-1).)
//
//   batch_norm_inference  — uses running_mean / running_var; pure forward;
//                           no state mutation. This is what loaded pretrained
//                           checkpoints want during inference.
//
//   batch_norm_backward   — given X + saved batch mean/rstd from forward,
//                           computes dX (overwritten) plus dGamma / dBeta
//                           (accumulated; caller zeros).
//
// Reduction width M = N * H * W. Running-var uses unbiased denom (M-1) when
// M > 1; for M==1 it stays equal to the biased value (matches PyTorch which
// just uses biased when bessel correction is undefined). Forward Y and
// backward dX use the *biased* var (= 1/M sum (x-mean)^2) — same as PyTorch.

#include <brotensor/tensor.h>

#include <cmath>
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

inline void check_per_channel(const ::brotensor::Tensor& t,
                              int C, const char* op, const char* name) {
    if (t.size() != C) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must have C elements");
    }
}

} // namespace

void batch_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        ::brotensor::Tensor& running_mean,
                        ::brotensor::Tensor& running_var,
                        int N, int C, int H, int W,
                        float eps, float momentum,
                        ::brotensor::Tensor& Y,
                        ::brotensor::Tensor& saved_mean,
                        ::brotensor::Tensor& saved_rstd) {
    check_fp32(X,            "batch_norm_forward", "X");
    check_fp32(gamma,        "batch_norm_forward", "gamma");
    check_fp32(beta,         "batch_norm_forward", "beta");
    check_fp32(running_mean, "batch_norm_forward", "running_mean");
    check_fp32(running_var,  "batch_norm_forward", "running_var");
    check_per_channel(gamma,        C, "batch_norm_forward", "gamma");
    check_per_channel(beta,         C, "batch_norm_forward", "beta");
    check_per_channel(running_mean, C, "batch_norm_forward", "running_mean");
    check_per_channel(running_var,  C, "batch_norm_forward", "running_var");

    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (saved_mean.rows != C || saved_mean.cols != 1 ||
        saved_mean.dtype != Dtype::FP32) {
        saved_mean.resize(C, 1, Dtype::FP32);
    }
    if (saved_rstd.rows != C || saved_rstd.cols != 1 ||
        saved_rstd.dtype != Dtype::FP32) {
        saved_rstd.resize(C, 1, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    const float* gp = gamma.host_f32();
    const float* bp = beta.host_f32();
    float* rmp = running_mean.host_f32_mut();
    float* rvp = running_var.host_f32_mut();
    float* Yp = Y.host_f32_mut();
    float* smp = saved_mean.host_f32_mut();
    float* srp = saved_rstd.host_f32_mut();

    const int M = N * spatial;
    const float inv_M = 1.0f / static_cast<float>(M);
    // Bessel correction factor for the running-var update. For M==1 we leave
    // the biased estimate (matches PyTorch's behaviour).
    const float bessel = (M > 1) ? static_cast<float>(M) /
                                   static_cast<float>(M - 1)
                                 : 1.0f;

    for (int c = 0; c < C; ++c) {
        // Pass 1: sum + sum-of-squares for this channel across (N, H, W).
        float sum = 0.0f, sumsq = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* x_chan = Xp + (n * C + c) * spatial;
            for (int s = 0; s < spatial; ++s) {
                const float v = x_chan[s];
                sum   += v;
                sumsq += v * v;
            }
        }
        const float mean    = sum * inv_M;
        const float var_b   = sumsq * inv_M - mean * mean;  // biased
        const float rstd    = 1.0f / std::sqrt(var_b + eps);
        const float var_unb = var_b * bessel;                // unbiased

        // Save for backward.
        smp[c] = mean;
        srp[c] = rstd;

        // Update running stats (PyTorch convention).
        rmp[c] = (1.0f - momentum) * rmp[c] + momentum * mean;
        rvp[c] = (1.0f - momentum) * rvp[c] + momentum * var_unb;

        // Pass 2: write Y = (x - mean) * rstd * gamma + beta.
        const float gv = gp[c];
        const float bv = bp[c];
        for (int n = 0; n < N; ++n) {
            const float* x_chan = Xp + (n * C + c) * spatial;
            float*       y_chan = Yp + (n * C + c) * spatial;
            for (int s = 0; s < spatial; ++s) {
                y_chan[s] = (x_chan[s] - mean) * rstd * gv + bv;
            }
        }
    }
}

void batch_norm_inference(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& gamma,
                          const ::brotensor::Tensor& beta,
                          const ::brotensor::Tensor& running_mean,
                          const ::brotensor::Tensor& running_var,
                          int N, int C, int H, int W,
                          float eps,
                          ::brotensor::Tensor& Y) {
    check_fp32(X,            "batch_norm_inference", "X");
    check_fp32(gamma,        "batch_norm_inference", "gamma");
    check_fp32(beta,         "batch_norm_inference", "beta");
    check_fp32(running_mean, "batch_norm_inference", "running_mean");
    check_fp32(running_var,  "batch_norm_inference", "running_var");
    check_per_channel(gamma,        C, "batch_norm_inference", "gamma");
    check_per_channel(beta,         C, "batch_norm_inference", "beta");
    check_per_channel(running_mean, C, "batch_norm_inference", "running_mean");
    check_per_channel(running_var,  C, "batch_norm_inference", "running_var");

    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp  = X.host_f32();
    const float* gp  = gamma.host_f32();
    const float* bp  = beta.host_f32();
    const float* rmp = running_mean.host_f32();
    const float* rvp = running_var.host_f32();
    float* Yp = Y.host_f32_mut();

    // Precompute per-channel affine y = x * a_c + b_c.
    // (x - mu) / sqrt(var + eps) * gamma + beta
    //   = x * (gamma / sqrt(var+eps)) + (beta - mu * gamma / sqrt(var+eps))
    for (int c = 0; c < C; ++c) {
        const float inv = 1.0f / std::sqrt(rvp[c] + eps);
        const float a = gp[c] * inv;
        const float b = bp[c] - rmp[c] * a;
        for (int n = 0; n < N; ++n) {
            const float* x_chan = Xp + (n * C + c) * spatial;
            float*       y_chan = Yp + (n * C + c) * spatial;
            for (int s = 0; s < spatial; ++s) {
                y_chan[s] = x_chan[s] * a + b;
            }
        }
    }
}

void batch_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& saved_mean,
                         const ::brotensor::Tensor& saved_rstd,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W,
                         ::brotensor::Tensor& dX,
                         ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta) {
    check_fp32(X,           "batch_norm_backward", "X");
    check_fp32(gamma,       "batch_norm_backward", "gamma");
    check_fp32(saved_mean,  "batch_norm_backward", "saved_mean");
    check_fp32(saved_rstd,  "batch_norm_backward", "saved_rstd");
    check_fp32(dY,          "batch_norm_backward", "dY");
    check_fp32(dGamma,      "batch_norm_backward", "dGamma");
    check_fp32(dBeta,       "batch_norm_backward", "dBeta");
    check_per_channel(gamma,       C, "batch_norm_backward", "gamma");
    check_per_channel(saved_mean,  C, "batch_norm_backward", "saved_mean");
    check_per_channel(saved_rstd,  C, "batch_norm_backward", "saved_rstd");
    if (dGamma.rows != C || dGamma.cols != 1 ||
        dBeta.rows  != C || dBeta.cols  != 1) {
        throw std::runtime_error("batch_norm_backward: dGamma/dBeta must be (C,1)");
    }

    const int spatial = H * W;
    const int cols = C * spatial;
    if (dY.rows != N || dY.cols != cols) {
        throw std::runtime_error("batch_norm_backward: dY shape mismatch");
    }
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("batch_norm_backward: X shape mismatch");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp  = X.host_f32();
    const float* gp  = gamma.host_f32();
    const float* mp  = saved_mean.host_f32();
    const float* rp  = saved_rstd.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    float* dGp = dGamma.host_f32_mut();
    float* dBp = dBeta.host_f32_mut();

    const int M = N * spatial;
    const float inv_M = 1.0f / static_cast<float>(M);

    // Scratch buffer for x̂, cached during the first pass and reused in the
    // dX pass so it doesn't need to be recomputed from X a second time.
    std::vector<float> xhat_buf(static_cast<std::size_t>(M));

    // Per-channel: derive dGamma, dBeta, and the two reduction sums used
    // by the dX formula. Then a second pass over (N, H, W) writes dX.
    //
    //   xhat = (x - mean) * rstd
    //   dxhat = dY * gamma
    //   dGamma_c += sum (dY * xhat)
    //   dBeta_c  += sum dY
    //   dX = rstd * (dxhat - (sum dxhat + xhat * sum(dxhat*xhat)) / M)
    for (int c = 0; c < C; ++c) {
        const float mean = mp[c];
        const float rstd = rp[c];
        const float gv   = gp[c];

        float sum_dY     = 0.0f;
        float sum_dY_xh  = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* x_chan  = Xp  + (n * C + c) * spatial;
            const float* dy_chan = dYp + (n * C + c) * spatial;
            float*       xh_chan = xhat_buf.data() + n * spatial;
            for (int s = 0; s < spatial; ++s) {
                const float xh = (x_chan[s] - mean) * rstd;
                xh_chan[s] = xh;
                sum_dY     += dy_chan[s];
                sum_dY_xh  += dy_chan[s] * xh;
            }
        }

        dGp[c] += sum_dY_xh;   // accumulate
        dBp[c] += sum_dY;      // accumulate

        // For dX we need sum1 = sum dxhat = gv * sum_dY,
        // and sum2 = sum (dxhat * xhat) = gv * sum_dY_xh.
        const float sum1 = gv * sum_dY;
        const float sum2 = gv * sum_dY_xh;

        for (int n = 0; n < N; ++n) {
            const float* dy_chan = dYp + (n * C + c) * spatial;
            float*       dx_chan = dXp + (n * C + c) * spatial;
            const float* xh_chan = xhat_buf.data() + n * spatial;
            for (int s = 0; s < spatial; ++s) {
                const float xh  = xh_chan[s];
                const float dxh = dy_chan[s] * gv;
                dx_chan[s] = rstd * (dxh - (sum1 + xh * sum2) * inv_M);
            }
        }
    }
}

} // namespace brotensor::detail::cpu
