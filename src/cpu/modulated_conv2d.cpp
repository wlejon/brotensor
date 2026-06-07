// ─── CPU modulated_conv2d (StyleGAN3) ───────────────────────────────────────
//
// The StyleGAN synthesis-layer core: per-sample style modulation of the conv
// weights, optional demodulation, then a standard stride-1 conv per sample.
// FP32 reference mirroring NVlabs `modulated_conv2d` (the pure-PyTorch path).
//
// Realized by looping the batch and reusing the validated CPU conv2d kernels
// (groups=1) on a per-sample weight — only the weight construction + demod is
// new. Layouts: X (N,C_in*H*W) NCHW; W (C_out, C_in*kH*kW) OIHW; s (N,C_in).
//
//   w'[o,i,kh,kw] = W[o,i,kh,kw] * s[n,i]
//   dcoef[n,o]    = demodulate ? rsqrt(Σ_{i,kh,kw} w'^2 + eps) : 1
//   w''           = w' * dcoef[n,o]
//   Y[n]          = conv2d(X[n], w'', pad, stride=1)
//
// Backward (per n; dw'' = conv2d_backward_weight(X[n],dY[n])):
//   g[o]    = Σ dw''[o,..] * w'[o,..]
//   dw'[o]  = demodulate ? dw''[o]*dcoef - g[o]*dcoef^3*w'[o] : dw''[o]
//   dW[o]  += Σ_n dw'[n,o] * s[n,i]      (accumulate — caller zeros dW)
//   ds[n,i] = Σ_{o,kh,kw} dw'[o,i,kh,kw] * W[o,i,kh,kw]   (overwrite)
//   dX[n]   = conv2d_backward_input(w''[n], dY[n])         (overwrite)

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

// Reused CPU conv2d kernels (defined in conv2d.cpp, same namespace).
void conv2d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, ::brotensor::Tensor& Y);
void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups,
                           ::brotensor::Tensor& dX);
void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int dil_h, int dil_w, int groups,
                            ::brotensor::Tensor& dWt);

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " +
                                 name + " must be FP32 (CPU backend is "
                                 "FP32-only)");
    }
}

// Non-owning CPU FP32 view over one row of a (N, cols) tensor.
inline ::brotensor::Tensor row_view(const ::brotensor::Tensor& T, int n, int cols) {
    float* base = const_cast<float*>(T.host_f32()) + static_cast<size_t>(n) * cols;
    return ::brotensor::Tensor::view(Device::CPU, base, 1, cols, Dtype::FP32);
}
inline ::brotensor::Tensor row_view_mut(::brotensor::Tensor& T, int n, int cols) {
    float* base = T.host_f32_mut() + static_cast<size_t>(n) * cols;
    return ::brotensor::Tensor::view(Device::CPU, base, 1, cols, Dtype::FP32);
}

} // namespace

void modulated_conv2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& W,
                              const ::brotensor::Tensor& s,
                              int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW,
                              int pad_h, int pad_w,
                              bool demodulate, float eps,
                              ::brotensor::Tensor& dcoef,
                              ::brotensor::Tensor& Y) {
    check_fp32(X, "modulated_conv2d_forward", "X");
    check_fp32(W, "modulated_conv2d_forward", "W");
    check_fp32(s, "modulated_conv2d_forward", "s");
    const int wk = C_in * kH * kW;
    if (X.rows != N || X.cols != C_in * H * Wd)
        throw std::runtime_error("modulated_conv2d_forward: X shape mismatch");
    if (W.rows != C_out || W.cols != wk)
        throw std::runtime_error("modulated_conv2d_forward: W shape mismatch");
    if (s.rows != N || s.cols != C_in)
        throw std::runtime_error("modulated_conv2d_forward: s shape mismatch");
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    if (H_out <= 0 || W_out <= 0)
        throw std::runtime_error("modulated_conv2d_forward: non-positive output shape");
    if (dcoef.rows != N || dcoef.cols != C_out || dcoef.dtype != Dtype::FP32)
        dcoef.resize(N, C_out, Dtype::FP32);
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP32)
        Y.resize(N, out_cols, Dtype::FP32);
    if (N == 0 || out_cols == 0) return;

    const float* Wp = W.host_f32();
    const float* sp = s.host_f32();
    float* dcp = dcoef.host_f32_mut();

    ::brotensor::Tensor Wn = ::brotensor::Tensor::zeros_on(Device::CPU, C_out, wk);
    float* Wnp = Wn.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        const float* sn = sp + static_cast<size_t>(n) * C_in;
        for (int o = 0; o < C_out; ++o) {
            const float* Wo = Wp + static_cast<size_t>(o) * wk;
            float* Wno = Wnp + static_cast<size_t>(o) * wk;
            // w'[o,..] = W[o,..]*s[n,i]; demod coefficient from its norm.
            double ss = 0.0;
            for (int i = 0; i < C_in; ++i) {
                const float si = sn[i];
                for (int t = 0; t < kH * kW; ++t) {
                    const float wp = Wo[i * kH * kW + t] * si;
                    Wno[i * kH * kW + t] = wp;
                    ss += static_cast<double>(wp) * wp;
                }
            }
            const float d = demodulate
                ? 1.0f / std::sqrt(static_cast<float>(ss) + eps) : 1.0f;
            dcp[static_cast<size_t>(n) * C_out + o] = d;
            if (demodulate) for (int t = 0; t < wk; ++t) Wno[t] *= d;
        }
        ::brotensor::Tensor Xn = row_view(X, n, C_in * H * Wd);
        ::brotensor::Tensor Yn = row_view_mut(Y, n, out_cols);
        conv2d_forward(Xn, Wn, nullptr, 1, C_in, H, Wd, C_out, kH, kW,
                       1, 1, pad_h, pad_w, 1, 1, 1, Yn);
    }
}

void modulated_conv2d_backward(const ::brotensor::Tensor& X,
                               const ::brotensor::Tensor& W,
                               const ::brotensor::Tensor& s,
                               const ::brotensor::Tensor& dcoef,
                               const ::brotensor::Tensor& dY,
                               int N, int C_in, int H, int Wd,
                               int C_out, int kH, int kW,
                               int pad_h, int pad_w, bool demodulate, float eps,
                               ::brotensor::Tensor& dX,
                               ::brotensor::Tensor& dW,
                               ::brotensor::Tensor& ds) {
    check_fp32(X, "modulated_conv2d_backward", "X");
    check_fp32(W, "modulated_conv2d_backward", "W");
    check_fp32(s, "modulated_conv2d_backward", "s");
    check_fp32(dcoef, "modulated_conv2d_backward", "dcoef");
    check_fp32(dY, "modulated_conv2d_backward", "dY");
    (void)eps;  // demod coefficient is precomputed (passed in as dcoef)
    const int wk = C_in * kH * kW;
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    if (W.rows != C_out || W.cols != wk)
        throw std::runtime_error("modulated_conv2d_backward: W shape mismatch");
    if (dW.rows != C_out || dW.cols != wk)
        throw std::runtime_error("modulated_conv2d_backward: dW shape mismatch");
    if (dX.rows != N || dX.cols != C_in * H * Wd || dX.dtype != Dtype::FP32)
        dX.resize(N, C_in * H * Wd, Dtype::FP32);
    if (ds.rows != N || ds.cols != C_in || ds.dtype != Dtype::FP32)
        ds.resize(N, C_in, Dtype::FP32);
    if (N == 0) return;

    const float* Wp = W.host_f32();
    const float* sp = s.host_f32();
    const float* dcp = dcoef.host_f32();
    float* dWp = dW.host_f32_mut();   // accumulate
    float* dsp = ds.host_f32_mut();   // overwrite

    const int out_cols = C_out * H_out * W_out;
    ::brotensor::Tensor Wpp = ::brotensor::Tensor::zeros_on(Device::CPU, C_out, wk); // w''
    ::brotensor::Tensor Wpr = ::brotensor::Tensor::zeros_on(Device::CPU, C_out, wk); // w'
    ::brotensor::Tensor dWpp = ::brotensor::Tensor::zeros_on(Device::CPU, C_out, wk);
    std::vector<float> dWpr(static_cast<size_t>(C_out) * wk);  // dw'
    float* Wppp = Wpp.host_f32_mut();
    float* Wprp = Wpr.host_f32_mut();
    float* dWppp = dWpp.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        const float* sn = sp + static_cast<size_t>(n) * C_in;
        const float* dcn = dcp + static_cast<size_t>(n) * C_out;
        // Rebuild w' and w''.
        for (int o = 0; o < C_out; ++o) {
            const float* Wo = Wp + static_cast<size_t>(o) * wk;
            const float d = dcn[o];
            for (int i = 0; i < C_in; ++i) {
                const float si = sn[i];
                for (int t = 0; t < kH * kW; ++t) {
                    const int col = i * kH * kW + t;
                    const float wp = Wo[col] * si;
                    Wprp[static_cast<size_t>(o) * wk + col] = wp;
                    Wppp[static_cast<size_t>(o) * wk + col] = wp * d;
                }
            }
        }
        ::brotensor::Tensor Xn = row_view(X, n, C_in * H * Wd);
        ::brotensor::Tensor dYn = row_view(dY, n, out_cols);
        // dw'' (conv2d_backward_weight accumulates → zero first), and dX[n].
        dWpp.zero();
        conv2d_backward_weight(Xn, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                               1, 1, pad_h, pad_w, 1, 1, 1, dWpp);
        ::brotensor::Tensor dXn = row_view_mut(dX, n, C_in * H * Wd);
        conv2d_backward_input(Wpp, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                              1, 1, pad_h, pad_w, 1, 1, 1, dXn);
        // Through demod → dw'.
        for (int o = 0; o < C_out; ++o) {
            const size_t ob = static_cast<size_t>(o) * wk;
            if (demodulate) {
                const float d = dcn[o];
                double g = 0.0;
                for (int t = 0; t < wk; ++t)
                    g += static_cast<double>(dWppp[ob + t]) * Wprp[ob + t];
                const float gd3 = static_cast<float>(g) * d * d * d;
                for (int t = 0; t < wk; ++t)
                    dWpr[ob + t] = dWppp[ob + t] * d - gd3 * Wprp[ob + t];
            } else {
                for (int t = 0; t < wk; ++t) dWpr[ob + t] = dWppp[ob + t];
            }
        }
        // Accumulate dW and write ds[n].
        float* dsn = dsp + static_cast<size_t>(n) * C_in;
        for (int i = 0; i < C_in; ++i) dsn[i] = 0.0f;
        for (int o = 0; o < C_out; ++o) {
            const float* Wo = Wp + static_cast<size_t>(o) * wk;
            const size_t ob = static_cast<size_t>(o) * wk;
            for (int i = 0; i < C_in; ++i) {
                const float si = sn[i];
                double ds_acc = 0.0;
                for (int t = 0; t < kH * kW; ++t) {
                    const int col = i * kH * kW + t;
                    const float dwp = dWpr[ob + col];
                    dWp[ob + col] += dwp * si;             // accumulate dW
                    ds_acc += static_cast<double>(dwp) * Wo[col];
                }
                dsn[i] += static_cast<float>(ds_acc);      // overwrite ds[n,i]
            }
        }
    }
}

} // namespace brotensor::detail::cpu
