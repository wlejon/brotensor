// brotensor public op wrappers — 2D/3D spatial, conv, norm, pool, StyleGAN.

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <stdexcept>

namespace brotensor {

// ─── Conv2d ────────────────────────────────────────────────────────────────

void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("conv2d_forward", X, N, C_in, H, W);
    const auto& v = detail::dispatch_with_opts(X, Wt, {bias, &Y});
    if (!v.conv2d_forward) detail::throw_not_implemented("conv2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv2d_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                     stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Y);
}

void conv2d_backward_input(const Tensor& Wt, const Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups, Tensor& dX) {
    if (N <= 0 || C_in <= 0 || H <= 0 || W <= 0 || C_out <= 0) {
        detail::throw_spatial_2d_error("conv2d_backward_input", "dY",
                                       N, C_in, H, W, dY.rows, dY.cols);
    }
    if (kH > 0 && kW > 0 && stride_h > 0 && stride_w > 0 && dil_h > 0 && dil_w > 0) {
        const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
        const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
        if (H_out > 0 && W_out > 0) {
            BROTENSOR_CHECK_SPATIAL_2D("conv2d_backward_input", dY, N, C_out, H_out, W_out);
        }
    }
    const auto& v = detail::dispatch(Wt, dY, dX);
    if (!v.conv2d_backward_input) detail::throw_not_implemented("conv2d_backward_input", Wt.device);
    detail::adopt_output(dX, Wt.device);
    v.conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dX);
}

void conv2d_backward_weight(const Tensor& X, const Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups, Tensor& dWt) {
    BROTENSOR_CHECK_SPATIAL_2D("conv2d_backward_weight", X, N, C_in, H, W);
    const auto& v = detail::dispatch(X, dY, dWt);
    if (!v.conv2d_backward_weight) detail::throw_not_implemented("conv2d_backward_weight", X.device);
    detail::adopt_output(dWt, X.device);
    v.conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                             stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
}

void conv2d_backward_bias(const Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          Tensor& dB) {
    BROTENSOR_CHECK_SPATIAL_2D("conv2d_backward_bias", dY, N, C_out, H_out, W_out);
    const auto& v = detail::dispatch(dY, dB);
    if (!v.conv2d_backward_bias) detail::throw_not_implemented("conv2d_backward_bias", dY.device);
    detail::adopt_output(dB, dY.device);
    v.conv2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

void conv2d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8, const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups,
                               Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("conv2d_int8w_fp16_forward", X, N, C_in, H, W);
    const auto& v = detail::dispatch_with_opts(X, W_int8, {&scales, bias, &Y});
    if (!v.conv2d_int8w_fp16_forward)
        detail::throw_not_implemented("conv2d_int8w_fp16_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv2d_int8w_fp16_forward(X, W_int8, scales, bias,
                                N, C_in, H, W, C_out, kH, kW,
                                stride_h, stride_w, pad_h, pad_w,
                                dil_h, dil_w, groups, Y);
}

void deform_conv2d_forward(const Tensor& X, const Tensor& offset,
                           const Tensor* mask, const Tensor& Wt,
                           const Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups, int deform_groups, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("deform_conv2d_forward", X, N, C_in, H, W);
    const auto& v = detail::dispatch_with_opts(X, Wt, {&offset, mask, bias, &Y});
    if (!v.deform_conv2d_forward) detail::throw_not_implemented("deform_conv2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.deform_conv2d_forward(X, offset, mask, Wt, bias, N, C_in, H, W,
                            C_out, kH, kW, stride_h, stride_w, pad_h, pad_w,
                            dil_h, dil_w, groups, deform_groups, Y);
}

// ─── Conv3d ────────────────────────────────────────────────────────────────

void conv3d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int T, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w,
                    int groups, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_3D("conv3d_forward", X, N, C_in, T, H, W);
    const auto& v = detail::dispatch_with_opts(X, Wt, {bias, &Y});
    if (!v.conv3d_forward) detail::throw_not_implemented("conv3d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv3d_forward(X, Wt, bias, N, C_in, T, H, W, C_out, kT, kH, kW,
                     stride_t, stride_h, stride_w,
                     pad_t, pad_h, pad_w,
                     dil_t, dil_h, dil_w, groups, Y);
}

void conv3d_int8w_fp16_forward(const Tensor& X, const Tensor& W_int8,
                               const Tensor& scales, const Tensor* bias,
                               int N, int C_in, int T, int H, int W,
                               int C_out, int kT, int kH, int kW,
                               int stride_t, int stride_h, int stride_w,
                               int pad_t, int pad_h, int pad_w,
                               int dil_t, int dil_h, int dil_w,
                               int groups, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_3D("conv3d_int8w_fp16_forward", X, N, C_in, T, H, W);
    const auto& v = detail::dispatch_with_opts(X, W_int8, {&scales, bias, &Y});
    if (!v.conv3d_int8w_fp16_forward)
        detail::throw_not_implemented("conv3d_int8w_fp16_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv3d_int8w_fp16_forward(X, W_int8, scales, bias,
                                N, C_in, T, H, W, C_out, kT, kH, kW,
                                stride_t, stride_h, stride_w,
                                pad_t, pad_h, pad_w,
                                dil_t, dil_h, dil_w, groups, Y);
}

// ─── GroupNorm ─────────────────────────────────────────────────────────────

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        int N, int C, int H, int W, int num_groups,
                        float eps, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("group_norm_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, gamma, beta, Y);
    if (!v.group_norm_forward) detail::throw_not_implemented("group_norm_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.group_norm_forward(X, gamma, beta, N, C, H, W, num_groups, eps, Y);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    BROTENSOR_CHECK_SPATIAL_2D("group_norm_backward", X, N, C, H, W);
    BROTENSOR_CHECK_SPATIAL_2D("group_norm_backward", dY, N, C, H, W);
    const auto& v = detail::dispatch(X, gamma, dY, dX, dGamma, dBeta);
    if (!v.group_norm_backward) detail::throw_not_implemented("group_norm_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dGamma, X.device);
    detail::adopt_output(dBeta, X.device);
    v.group_norm_backward(X, gamma, dY, N, C, H, W, num_groups, eps,
                          dX, dGamma, dBeta);
}

// ─── Activations ───────────────────────────────────────────────────────────

void silu_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.silu_forward) detail::throw_not_implemented("silu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.silu_forward(x, y);
}
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.silu_backward) detail::throw_not_implemented("silu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.silu_backward(x, dY, dX);
}
void gelu_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.gelu_forward) detail::throw_not_implemented("gelu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.gelu_forward(x, y);
}
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.gelu_backward) detail::throw_not_implemented("gelu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.gelu_backward(x, dY, dX);
}
void gelu_exact_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.gelu_exact_forward) detail::throw_not_implemented("gelu_exact_forward", x.device);
    detail::adopt_output(y, x.device);
    v.gelu_exact_forward(x, y);
}
void gelu_exact_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.gelu_exact_backward) detail::throw_not_implemented("gelu_exact_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.gelu_exact_backward(x, dY, dX);
}
void quick_gelu_forward(const Tensor& x, Tensor& y) {
    const auto& v = detail::dispatch(x, y);
    if (!v.quick_gelu_forward) detail::throw_not_implemented("quick_gelu_forward", x.device);
    detail::adopt_output(y, x.device);
    v.quick_gelu_forward(x, y);
}
void quick_gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    const auto& v = detail::dispatch(x, dY, dX);
    if (!v.quick_gelu_backward) detail::throw_not_implemented("quick_gelu_backward", x.device);
    detail::adopt_output(dX, x.device);
    v.quick_gelu_backward(x, dY, dX);
}

// ─── Resample ──────────────────────────────────────────────────────────────

void upsample_nearest_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("upsample_nearest_2x", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.upsample_nearest_2x) detail::throw_not_implemented("upsample_nearest_2x", X.device);
    detail::adopt_output(Y, X.device);
    v.upsample_nearest_2x(X, N, C, H, W, Y);
}
void upsample_bilinear_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("upsample_bilinear_2x", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.upsample_bilinear_2x) detail::throw_not_implemented("upsample_bilinear_2x", X.device);
    detail::adopt_output(Y, X.device);
    v.upsample_bilinear_2x(X, N, C, H, W, Y);
}
void downsample_avg_2x(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("downsample_avg_2x", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.downsample_avg_2x) detail::throw_not_implemented("downsample_avg_2x", X.device);
    detail::adopt_output(Y, X.device);
    v.downsample_avg_2x(X, N, C, H, W, Y);
}
void upsample_nearest_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("upsample_nearest_2x_backward", "dY",
                                       N, C, H, W, dY.rows, dY.cols);
    }
    BROTENSOR_CHECK_SPATIAL_2D("upsample_nearest_2x_backward", dY, N, C, 2 * H, 2 * W);
    const auto& v = detail::dispatch(dY, dX);
    if (!v.upsample_nearest_2x_backward) detail::throw_not_implemented("upsample_nearest_2x_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.upsample_nearest_2x_backward(dY, N, C, H, W, dX);
}
void upsample_bilinear_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("upsample_bilinear_2x_backward", "dY",
                                       N, C, H, W, dY.rows, dY.cols);
    }
    BROTENSOR_CHECK_SPATIAL_2D("upsample_bilinear_2x_backward", dY, N, C, 2 * H, 2 * W);
    const auto& v = detail::dispatch(dY, dX);
    if (!v.upsample_bilinear_2x_backward) detail::throw_not_implemented("upsample_bilinear_2x_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.upsample_bilinear_2x_backward(dY, N, C, H, W, dX);
}
void downsample_avg_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("downsample_avg_2x_backward", "dY",
                                       N, C, H, W, dY.rows, dY.cols);
    }
    if (H % 2 == 0 && W % 2 == 0) {
        BROTENSOR_CHECK_SPATIAL_2D("downsample_avg_2x_backward", dY, N, C, H / 2, W / 2);
    }
    const auto& v = detail::dispatch(dY, dX);
    if (!v.downsample_avg_2x_backward) detail::throw_not_implemented("downsample_avg_2x_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.downsample_avg_2x_backward(dY, N, C, H, W, dX);
}

void interp2d_forward(const Tensor& X,
                      int N, int C, int H_in, int W_in, int H_out, int W_out,
                      int mode, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("interp2d_forward", X, N, C, H_in, W_in);
    const auto& v = detail::dispatch(X, Y);
    if (!v.interp2d_forward) detail::throw_not_implemented("interp2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.interp2d_forward(X, N, C, H_in, W_in, H_out, W_out, mode, Y);
}
void interp2d_backward(const Tensor& dY,
                       int N, int C, int H_in, int W_in, int H_out, int W_out,
                       int mode, Tensor& dX) {
    if (N <= 0 || C <= 0 || H_in <= 0 || W_in <= 0) {
        detail::throw_spatial_2d_error("interp2d_backward", "dY",
                                       N, C, H_in, W_in, dY.rows, dY.cols);
    }
    BROTENSOR_CHECK_SPATIAL_2D("interp2d_backward", dY, N, C, H_out, W_out);
    const auto& v = detail::dispatch(dY, dX);
    if (!v.interp2d_backward) detail::throw_not_implemented("interp2d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.interp2d_backward(dY, N, C, H_in, W_in, H_out, W_out, mode, dX);
}
void interp2d_align_corners_forward(const Tensor& X,
                                    int N, int C, int H_in, int W_in,
                                    int H_out, int W_out, int mode, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("interp2d_align_corners_forward", X, N, C, H_in, W_in);
    const auto& v = detail::dispatch(X, Y);
    if (!v.interp2d_align_corners_forward)
        detail::throw_not_implemented("interp2d_align_corners_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.interp2d_align_corners_forward(X, N, C, H_in, W_in, H_out, W_out, mode, Y);
}

void pad2d_forward(const Tensor& X, int N, int C, int H, int W,
                   int pad_top, int pad_bottom, int pad_left, int pad_right,
                   int mode, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("pad2d_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.pad2d_forward) detail::throw_not_implemented("pad2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.pad2d_forward(X, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode, Y);
}
void pad2d_backward(const Tensor& dY, int N, int C, int H, int W,
                    int pad_top, int pad_bottom, int pad_left, int pad_right,
                    int mode, Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("pad2d_backward", "dY",
                                       N, C, H, W, dY.rows, dY.cols);
    }
    const int H_out = H + pad_top + pad_bottom;
    const int W_out = W + pad_left + pad_right;
    if (H_out > 0 && W_out > 0) {
        BROTENSOR_CHECK_SPATIAL_2D("pad2d_backward", dY, N, C, H_out, W_out);
    }
    const auto& v = detail::dispatch(dY, dX);
    if (!v.pad2d_backward) detail::throw_not_implemented("pad2d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.pad2d_backward(dY, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode, dX);
}

void slice2d_forward(const Tensor& X, int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("slice2d_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.slice2d_forward) detail::throw_not_implemented("slice2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.slice2d_forward(X, N, C, H, W, h0, w0, H_out, W_out, Y);
}
void slice2d_backward(const Tensor& dY, int N, int C, int H, int W,
                      int h0, int w0, int H_out, int W_out, Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("slice2d_backward", "dY",
                                       N, C, H, W, dY.rows, dY.cols);
    }
    if (H_out > 0 && W_out > 0) {
        BROTENSOR_CHECK_SPATIAL_2D("slice2d_backward", dY, N, C, H_out, W_out);
    }
    const auto& v = detail::dispatch(dY, dX);
    if (!v.slice2d_backward) detail::throw_not_implemented("slice2d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.slice2d_backward(dY, N, C, H, W, h0, w0, H_out, W_out, dX);
}

void unfold2d_forward(const Tensor& X, int N, int C, int H, int W,
                      int kH, int kW, int stride_h, int stride_w,
                      int pad_top, int pad_bottom, int pad_left, int pad_right,
                      int mode, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("unfold2d_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.unfold2d_forward) detail::throw_not_implemented("unfold2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.unfold2d_forward(X, N, C, H, W, kH, kW, stride_h, stride_w,
                       pad_top, pad_bottom, pad_left, pad_right, mode, Y);
}

void l2_normalize_nchw_forward(const Tensor& X, int N, int C, int H, int W,
                               float eps, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("l2_normalize_nchw_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.l2_normalize_nchw_forward)
        detail::throw_not_implemented("l2_normalize_nchw_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.l2_normalize_nchw_forward(X, N, C, H, W, eps, Y);
}

void convex_upsample_forward(const Tensor& X, const Tensor& Mask,
                             int N, int C, int H, int W, int scale, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("convex_upsample_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Mask, Y);
    if (!v.convex_upsample_forward)
        detail::throw_not_implemented("convex_upsample_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.convex_upsample_forward(X, Mask, N, C, H, W, scale, Y);
}

void top_k_rows(const Tensor& X, int k, Tensor& Vals, Tensor& Idx) {
    const auto& v = detail::dispatch(X, Vals, Idx);
    if (!v.top_k_rows) detail::throw_not_implemented("top_k_rows", X.device);
    detail::adopt_output(Vals, X.device);
    detail::adopt_output(Idx, X.device);
    v.top_k_rows(X, k, Vals, Idx);
}

void adaptive_avg_pool2d_forward(const Tensor& X, int N, int C, int H, int W,
                                 int H_out, int W_out, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("adaptive_avg_pool2d_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.adaptive_avg_pool2d_forward)
        detail::throw_not_implemented("adaptive_avg_pool2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.adaptive_avg_pool2d_forward(X, N, C, H, W, H_out, W_out, Y);
}
void adaptive_avg_pool2d_backward(const Tensor& dY, int N, int C, int H, int W,
                                  int H_out, int W_out, Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("adaptive_avg_pool2d_backward", "dY",
                                       N, C, H, W, dY.rows, dY.cols);
    }
    BROTENSOR_CHECK_SPATIAL_2D("adaptive_avg_pool2d_backward", dY, N, C, H_out, W_out);
    const auto& v = detail::dispatch(dY, dX);
    if (!v.adaptive_avg_pool2d_backward)
        detail::throw_not_implemented("adaptive_avg_pool2d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.adaptive_avg_pool2d_backward(dY, N, C, H, W, H_out, W_out, dX);
}

void max_pool2d_forward(const Tensor& X, int N, int C, int H, int W,
                        int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w, Tensor& Y, Tensor& Idx) {
    BROTENSOR_CHECK_SPATIAL_2D("max_pool2d_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y, Idx);
    if (!v.max_pool2d_forward)
        detail::throw_not_implemented("max_pool2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    detail::adopt_output(Idx, X.device);
    v.max_pool2d_forward(X, N, C, H, W, kH, kW, stride_h, stride_w,
                         pad_h, pad_w, Y, Idx);
}
void max_pool2d_backward(const Tensor& dY, const Tensor& Idx,
                         int N, int C, int H, int W, int H_out, int W_out,
                         Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("max_pool2d_backward", "dY",
                                       N, C, H, W, dY.rows, dY.cols);
    }
    BROTENSOR_CHECK_SPATIAL_2D("max_pool2d_backward", dY, N, C, H_out, W_out);
    const auto& v = detail::dispatch(dY, Idx, dX);
    if (!v.max_pool2d_backward)
        detail::throw_not_implemented("max_pool2d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.max_pool2d_backward(dY, Idx, N, C, H, W, H_out, W_out, dX);
}

void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y) {
    const auto& v = detail::dispatch(X, Idx, Y);
    if (!v.gather_rows) detail::throw_not_implemented("gather_rows", X.device);
    detail::adopt_output(Y, X.device);
    v.gather_rows(X, Idx, Y);
}
void scatter_rows_add(const Tensor& dY, const Tensor& Idx, int R, Tensor& dX) {
    const auto& v = detail::dispatch(dY, Idx, dX);
    if (!v.scatter_rows_add)
        detail::throw_not_implemented("scatter_rows_add", dY.device);
    detail::adopt_output(dX, dY.device);
    v.scatter_rows_add(dY, Idx, R, dX);
}
void scatter_rows(const Tensor& Y, const Tensor& Idx, Tensor& X) {
    const auto& v = detail::dispatch(Y, Idx, X);
    if (!v.scatter_rows) detail::throw_not_implemented("scatter_rows", Y.device);
    v.scatter_rows(Y, Idx, X);
}

void conv_transpose2d_forward(const Tensor& X, const Tensor& Wt,
                              const Tensor* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("conv_transpose2d_forward", X, N, C_in, H, W);
    const auto& v = detail::dispatch(X, Wt, Y);
    if (!v.conv_transpose2d_forward)
        detail::throw_not_implemented("conv_transpose2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.conv_transpose2d_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                               stride_h, stride_w, pad_h, pad_w,
                               output_padding_h, output_padding_w,
                               dil_h, dil_w, groups, Y);
}
void conv_transpose2d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int H, int W,
                                     int C_out, int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     Tensor& dX) {
    if (N <= 0 || C_in <= 0 || H <= 0 || W <= 0 || C_out <= 0) {
        detail::throw_spatial_2d_error("conv_transpose2d_backward_input", "dY",
                                       N, C_in, H, W, dY.rows, dY.cols);
    }
    const auto& v = detail::dispatch(Wt, dY, dX);
    if (!v.conv_transpose2d_backward_input)
        detail::throw_not_implemented("conv_transpose2d_backward_input", Wt.device);
    detail::adopt_output(dX, Wt.device);
    v.conv_transpose2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                                      stride_h, stride_w, pad_h, pad_w,
                                      output_padding_h, output_padding_w,
                                      dil_h, dil_w, groups, dX);
}
void conv_transpose2d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int H, int W,
                                      int C_out, int kH, int kW,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      Tensor& dWt) {
    BROTENSOR_CHECK_SPATIAL_2D("conv_transpose2d_backward_weight", X, N, C_in, H, W);
    const auto& v = detail::dispatch(X, dY, dWt);
    if (!v.conv_transpose2d_backward_weight)
        detail::throw_not_implemented("conv_transpose2d_backward_weight", X.device);
    detail::adopt_output(dWt, X.device);
    v.conv_transpose2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                                       stride_h, stride_w, pad_h, pad_w,
                                       output_padding_h, output_padding_w,
                                       dil_h, dil_w, groups, dWt);
}
void conv_transpose2d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int H_out, int W_out, Tensor& dB) {
    BROTENSOR_CHECK_SPATIAL_2D("conv_transpose2d_backward_bias", dY, N, C_out, H_out, W_out);
    const auto& v = detail::dispatch(dY, dB);
    if (!v.conv_transpose2d_backward_bias)
        detail::throw_not_implemented("conv_transpose2d_backward_bias", dY.device);
    detail::adopt_output(dB, dY.device);
    v.conv_transpose2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

void window_partition_forward(const Tensor& X, int N, int C, int H, int W,
                              int window, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("window_partition_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.window_partition_forward)
        detail::throw_not_implemented("window_partition_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.window_partition_forward(X, N, C, H, W, window, Y);
}
void window_reverse_forward(const Tensor& X, int N, int C, int H, int W,
                            int window, Tensor& Y) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("window_reverse_forward", "X",
                                       N, C, H, W, X.rows, X.cols);
    }
    if (window > 0 && H % window == 0 && W % window == 0) {
        BROTENSOR_CHECK_SPATIAL_2D("window_reverse_forward", X,
                                   N * (H / window) * (W / window), C, window, window);
    }
    const auto& v = detail::dispatch(X, Y);
    if (!v.window_reverse_forward)
        detail::throw_not_implemented("window_reverse_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.window_reverse_forward(X, N, C, H, W, window, Y);
}

// ─── NCHW <-> sequence ─────────────────────────────────────────────────────

void nchw_to_sequence(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("nchw_to_sequence", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.nchw_to_sequence) detail::throw_not_implemented("nchw_to_sequence", X.device);
    detail::adopt_output(Y, X.device);
    v.nchw_to_sequence(X, N, C, H, W, Y);
}

void sequence_to_nchw(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        detail::throw_spatial_2d_error("sequence_to_nchw", "X",
                                       N, C, H, W, X.rows, X.cols);
    }
    BROTENSOR_CHECK_SPATIAL_1D("sequence_to_nchw", X, N * H * W, C, 1);
    const auto& v = detail::dispatch(X, Y);
    if (!v.sequence_to_nchw) detail::throw_not_implemented("sequence_to_nchw", X.device);
    detail::adopt_output(Y, X.device);
    v.sequence_to_nchw(X, N, C, H, W, Y);
}

void spatial_merge_2x2_forward(const Tensor& X, int N, int C, int H, int W,
                               bool channel_major, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("spatial_merge_2x2_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, Y);
    if (!v.spatial_merge_2x2_forward)
        detail::throw_not_implemented("spatial_merge_2x2_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.spatial_merge_2x2_forward(X, N, C, H, W, channel_major, Y);
}

void pixel_shuffle_upsample_2x_forward(const Tensor& X, int N, int C_in,
                                       int H, int W, int C_out, Tensor& Y) {
    if (N > 0 && C_in > 0 && H > 0 && W > 0) {
        BROTENSOR_CHECK_SPATIAL_2D("pixel_shuffle_upsample_2x_forward", X, N, C_in, H, W);
    }
    const auto& v = detail::dispatch(X, Y);
    if (!v.pixel_shuffle_upsample_2x_forward)
        detail::throw_not_implemented("pixel_shuffle_upsample_2x_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, Y);
}

void patch_unpack_forward(const Tensor& tokens, int hp, int wp, int P,
                          int C_total, int C_keep, bool channel_major,
                          Tensor& Y) {
    const auto& v = detail::dispatch(tokens, Y);
    if (!v.patch_unpack_forward)
        detail::throw_not_implemented("patch_unpack_forward", tokens.device);
    detail::adopt_output(Y, tokens.device);
    v.patch_unpack_forward(tokens, hp, wp, P, C_total, C_keep, channel_major, Y);
}

// ─── ResBlock ──────────────────────────────────────────────────────────────

void resblock_forward(const Tensor& X,
                      const Tensor& gamma1, const Tensor& beta1,
                      const Tensor& W1, const Tensor* b1,
                      const Tensor* t_emb_shift,
                      const Tensor& gamma2, const Tensor& beta2,
                      const Tensor& W2, const Tensor* b2,
                      const Tensor* Wskip, const Tensor* bskip,
                      int N, int C_in, int C_out, int H, int W,
                      int num_groups, float eps, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("resblock_forward", X, N, C_in, H, W);
    const auto& v = detail::dispatch_with_opts(
        X, gamma1, {&beta1, &W1, b1, t_emb_shift, &gamma2, &beta2, &W2, b2,
                    Wskip, bskip, &Y});
    if (!v.resblock_forward) detail::throw_not_implemented("resblock_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.resblock_forward(X, gamma1, beta1, W1, b1, t_emb_shift,
                       gamma2, beta2, W2, b2, Wskip, bskip,
                       N, C_in, C_out, H, W, num_groups, eps, Y);
}

void resblock_forward_int8w_fp16(const Tensor& X,
                                 const Tensor& gamma1, const Tensor& beta1,
                                 const Tensor& W1_int8, const Tensor& s1,
                                 const Tensor* b1,
                                 const Tensor* t_emb_shift,
                                 const Tensor& gamma2, const Tensor& beta2,
                                 const Tensor& W2_int8, const Tensor& s2,
                                 const Tensor* b2,
                                 const Tensor* Wskip_int8, const Tensor* sskip,
                                 const Tensor* bskip,
                                 int N, int C_in, int C_out, int H, int W,
                                 int num_groups, float eps, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("resblock_forward_int8w_fp16", X, N, C_in, H, W);
    const auto& v = detail::dispatch_with_opts(
        X, gamma1, {&beta1, &W1_int8, &s1, b1, t_emb_shift,
                    &gamma2, &beta2, &W2_int8, &s2, b2,
                    Wskip_int8, sskip, bskip, &Y});
    if (!v.resblock_forward_int8w_fp16)
        detail::throw_not_implemented("resblock_forward_int8w_fp16", X.device);
    detail::adopt_output(Y, X.device);
    v.resblock_forward_int8w_fp16(X, gamma1, beta1, W1_int8, s1, b1, t_emb_shift,
                                  gamma2, beta2, W2_int8, s2, b2,
                                  Wskip_int8, sskip, bskip,
                                  N, C_in, C_out, H, W, num_groups, eps, Y);
}

void resblock_backward(const Tensor& X,
                       const Tensor& gamma1, const Tensor& beta1,
                       const Tensor& W1, const Tensor* b1,
                       const Tensor* t_emb_shift,
                       const Tensor& gamma2, const Tensor& beta2,
                       const Tensor& W2, const Tensor* b2,
                       const Tensor* Wskip, const Tensor* bskip,
                       int N, int C_in, int C_out, int H, int W,
                       int num_groups, float eps,
                       const Tensor& dY,
                       Tensor& dX,
                       Tensor& dGamma1, Tensor& dBeta1,
                       Tensor& dW1, Tensor* db1,
                       Tensor* dt_emb_shift,
                       Tensor& dGamma2, Tensor& dBeta2,
                       Tensor& dW2, Tensor* db2,
                       Tensor* dWskip, Tensor* dbskip) {
    BROTENSOR_CHECK_SPATIAL_2D("resblock_backward", X, N, C_in, H, W);
    BROTENSOR_CHECK_SPATIAL_2D("resblock_backward", dY, N, C_out, H, W);
    const auto& v = detail::dispatch_with_opts(
        X, gamma1, {&beta1, &W1, b1, t_emb_shift, &gamma2, &beta2, &W2, b2,
                    Wskip, bskip, &dY, &dX, &dGamma1, &dBeta1, &dW1, db1,
                    dt_emb_shift, &dGamma2, &dBeta2, &dW2, db2, dWskip, dbskip});
    if (!v.resblock_backward) detail::throw_not_implemented("resblock_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dGamma1, X.device);
    detail::adopt_output(dBeta1, X.device);
    detail::adopt_output(dW1, X.device);
    if (db1) detail::adopt_output(*db1, X.device);
    if (dt_emb_shift) detail::adopt_output(*dt_emb_shift, X.device);
    detail::adopt_output(dGamma2, X.device);
    detail::adopt_output(dBeta2, X.device);
    detail::adopt_output(dW2, X.device);
    if (db2) detail::adopt_output(*db2, X.device);
    if (dWskip) detail::adopt_output(*dWskip, X.device);
    if (dbskip) detail::adopt_output(*dbskip, X.device);
    v.resblock_backward(X, gamma1, beta1, W1, b1, t_emb_shift,
                        gamma2, beta2, W2, b2, Wskip, bskip,
                        N, C_in, C_out, H, W, num_groups, eps,
                        dY, dX, dGamma1, dBeta1, dW1, db1, dt_emb_shift,
                        dGamma2, dBeta2, dW2, db2, dWskip, dbskip);
}

// ─── BatchNorm ─────────────────────────────────────────────────────────────

void batch_norm_forward(const Tensor& X,
                        const Tensor& gamma, const Tensor& beta,
                        Tensor& running_mean, Tensor& running_var,
                        int N, int C, int H, int W,
                        float eps, float momentum,
                        Tensor& Y,
                        Tensor& saved_mean, Tensor& saved_rstd) {
    BROTENSOR_CHECK_SPATIAL_2D("batch_norm_forward", X, N, C, H, W);
    const auto& v = detail::dispatch(X, gamma, beta, running_mean, running_var,
                                     Y, saved_mean, saved_rstd);
    if (!v.batch_norm_forward)
        detail::throw_not_implemented("batch_norm_forward", X.device);
    detail::adopt_output(running_mean, X.device);
    detail::adopt_output(running_var,  X.device);
    detail::adopt_output(Y,            X.device);
    detail::adopt_output(saved_mean,   X.device);
    detail::adopt_output(saved_rstd,   X.device);
    v.batch_norm_forward(X, gamma, beta, running_mean, running_var,
                         N, C, H, W, eps, momentum,
                         Y, saved_mean, saved_rstd);
}

void batch_norm_inference(const Tensor& X,
                          const Tensor& gamma, const Tensor& beta,
                          const Tensor& running_mean,
                          const Tensor& running_var,
                          int N, int C, int H, int W,
                          float eps,
                          Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("batch_norm_inference", X, N, C, H, W);
    const auto& v = detail::dispatch(X, gamma, beta, running_mean, running_var, Y);
    if (!v.batch_norm_inference)
        detail::throw_not_implemented("batch_norm_inference", X.device);
    detail::adopt_output(Y, X.device);
    v.batch_norm_inference(X, gamma, beta, running_mean, running_var,
                           N, C, H, W, eps, Y);
}

void batch_norm_backward(const Tensor& X,
                         const Tensor& gamma,
                         const Tensor& saved_mean,
                         const Tensor& saved_rstd,
                         const Tensor& dY,
                         int N, int C, int H, int W,
                         Tensor& dX,
                         Tensor& dGamma, Tensor& dBeta) {
    BROTENSOR_CHECK_SPATIAL_2D("batch_norm_backward", X, N, C, H, W);
    BROTENSOR_CHECK_SPATIAL_2D("batch_norm_backward", dY, N, C, H, W);
    const auto& v = detail::dispatch(X, gamma, saved_mean, saved_rstd, dY,
                                     dX, dGamma, dBeta);
    if (!v.batch_norm_backward)
        detail::throw_not_implemented("batch_norm_backward", X.device);
    detail::adopt_output(dX,     X.device);
    detail::adopt_output(dGamma, X.device);
    detail::adopt_output(dBeta,  X.device);
    v.batch_norm_backward(X, gamma, saved_mean, saved_rstd, dY,
                          N, C, H, W, dX, dGamma, dBeta);
}

// ─── Image preprocessing helpers ───────────────────────────────────────────

void image_normalize(const Tensor& X,
                     const Tensor& mean, const Tensor& std_,
                     int N, int C, int H, int W,
                     Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("image_normalize", X, N, C, H, W);
    const auto& v = detail::dispatch(X, mean, std_, Y);
    if (!v.image_normalize)
        detail::throw_not_implemented("image_normalize", X.device);
    detail::adopt_output(Y, X.device);
    v.image_normalize(X, mean, std_, N, C, H, W, Y);
}

void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  Tensor& Y) {
    const auto& v = detail::dispatch(Y);
    if (!v.image_u8_to_f32_nhwc_to_nchw)
        detail::throw_not_implemented("image_u8_to_f32_nhwc_to_nchw", Y.device);
    detail::adopt_output(Y, Y.device);
    v.image_u8_to_f32_nhwc_to_nchw(src, N, H, W, C, scale, bias, Y);
}

// ─── StyleGAN3 bias_act / upfirdn2d / modulated_conv2d ─────────────────────

void bias_act_forward(const Tensor& X, const Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_1D("bias_act_forward", X, N, C, HW);
    const auto& v = detail::dispatch_with_opts(X, {b, &Y});
    if (!v.bias_act_forward) detail::throw_not_implemented("bias_act_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.bias_act_forward(X, b, N, C, HW, act, alpha, gain, clamp, Y);
}

void bias_act_backward(const Tensor& dY, const Tensor& X, const Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp, Tensor& dX, Tensor* dB) {
    BROTENSOR_CHECK_SPATIAL_1D("bias_act_backward", dY, N, C, HW);
    BROTENSOR_CHECK_SPATIAL_1D("bias_act_backward", X, N, C, HW);
    const auto& v = detail::dispatch_with_opts(dY, X, {b, dB, &dX});
    if (!v.bias_act_backward) detail::throw_not_implemented("bias_act_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    if (dB) detail::adopt_output(*dB, dY.device);
    v.bias_act_backward(dY, X, b, N, C, HW, act, alpha, gain, clamp, dX, dB);
}

void upfirdn2d_forward(const Tensor& X, const Tensor& f,
                       int N, int C, int H, int Wd, int fH, int fW,
                       int up_x, int up_y, int down_x, int down_y,
                       int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("upfirdn2d_forward", X, N, C, H, Wd);
    const auto& v = detail::dispatch(X, f, Y);
    if (!v.upfirdn2d_forward) detail::throw_not_implemented("upfirdn2d_forward", X.device);
    detail::adopt_output(Y, X.device);
    v.upfirdn2d_forward(X, f, N, C, H, Wd, fH, fW, up_x, up_y, down_x, down_y,
                        pad_x0, pad_x1, pad_y0, pad_y1, flip_filter, gain, Y);
}

void upfirdn2d_backward(const Tensor& dY, const Tensor& f,
                        int N, int C, int H, int Wd, int fH, int fW,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, Tensor& dX) {
    if (N <= 0 || C <= 0 || H <= 0 || Wd <= 0) {
        detail::throw_spatial_2d_error("upfirdn2d_backward", "dY",
                                       N, C, H, Wd, dY.rows, dY.cols);
    }
    const auto& v = detail::dispatch(dY, f, dX);
    if (!v.upfirdn2d_backward) detail::throw_not_implemented("upfirdn2d_backward", dY.device);
    detail::adopt_output(dX, dY.device);
    v.upfirdn2d_backward(dY, f, N, C, H, Wd, fH, fW, up_x, up_y, down_x, down_y,
                         pad_x0, pad_x1, pad_y0, pad_y1, flip_filter, gain, dX);
}

void modulated_conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& s,
                              int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW,
                              int pad_h, int pad_w,
                              bool demodulate, float eps,
                              Tensor& dcoef, Tensor& Y) {
    BROTENSOR_CHECK_SPATIAL_2D("modulated_conv2d_forward", X, N, C_in, H, Wd);
    const auto& v = detail::dispatch(X, W, s, dcoef, Y);
    if (!v.modulated_conv2d_forward)
        detail::throw_not_implemented("modulated_conv2d_forward", X.device);
    detail::adopt_output(dcoef, X.device);
    detail::adopt_output(Y, X.device);
    v.modulated_conv2d_forward(X, W, s, N, C_in, H, Wd, C_out, kH, kW,
                               pad_h, pad_w, demodulate, eps, dcoef, Y);
}

void modulated_conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& s,
                               const Tensor& dcoef, const Tensor& dY,
                               int N, int C_in, int H, int Wd,
                               int C_out, int kH, int kW,
                               int pad_h, int pad_w, bool demodulate, float eps,
                               Tensor& dX, Tensor& dW, Tensor& ds) {
    BROTENSOR_CHECK_SPATIAL_2D("modulated_conv2d_backward", X, N, C_in, H, Wd);
    if (kH > 0 && kW > 0) {
        const int H_out = H + 2 * pad_h - (kH - 1);
        const int W_out = Wd + 2 * pad_w - (kW - 1);
        if (H_out > 0 && W_out > 0) {
            BROTENSOR_CHECK_SPATIAL_2D("modulated_conv2d_backward", dY, N, C_out, H_out, W_out);
        }
    }
    const auto& v = detail::dispatch(X, W, s, dcoef, dY, dX, dW, ds);
    if (!v.modulated_conv2d_backward)
        detail::throw_not_implemented("modulated_conv2d_backward", X.device);
    detail::adopt_output(dX, X.device);
    detail::adopt_output(dW, X.device);
    detail::adopt_output(ds, X.device);
    v.modulated_conv2d_backward(X, W, s, dcoef, dY, N, C_in, H, Wd, C_out, kH, kW,
                                pad_h, pad_w, demodulate, eps, dX, dW, ds);
}

} // namespace brotensor
