// CUDA 2D transposed convolution forward + backward ops backed by
// direct/implicit-GEMM Brass MIR conv2d kernels.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Forward declarations of intra-backend helpers.
bool conv_transpose2d_f32_gemm(const float* X, const float* Wt, const float* bias, float* Y,
                               int N, int Cin, int H, int W, int Cout, int kH, int kW, int sh,
                               int sw, int ph, int pw, int dh, int dw, int Ho, int Wo,
                               cudaStream_t stream);

void dispatch_conv_direct(std::uint32_t mode, ::brotensor::Dtype dt,
                          const void* x, const void* w, void* y, const void* bias,
                          int N, int Cin, int T, int H, int Wd,
                          int Cout, int To, int Ho, int Wo,
                          int kT, int kH, int kW,
                          int st, int sh, int sw,
                          int pt, int ph, int pw,
                          int dt_, int dh, int dw, int groups);

void conv2d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y);

void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            ::brotensor::Tensor& dWt);

void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB);

void pad2d_forward(const ::brotensor::Tensor& X, int N, int C, int H, int W,
                   int pad_top, int pad_bottom, int pad_left, int pad_right,
                   int mode, ::brotensor::Tensor& Y);

void slice2d_forward(const ::brotensor::Tensor& X, int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out, ::brotensor::Tensor& Y);

namespace {

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

inline void require_fp32(const char* op, const ::brotensor::Tensor& t,
                         const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (CUDA conv_transpose2d is FP32-only)");
    }
}

inline int convt2d_out(int L, int stride, int padding, int output_padding,
                       int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

inline void check_geometry(const char* op, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int output_padding_h, int output_padding_w,
                           int dil_h, int dil_w) {
    if (kH < 1 || kW < 1 || stride_h < 1 || stride_w < 1
        || dil_h < 1 || dil_w < 1 || pad_h < 0 || pad_w < 0
        || output_padding_h < 0 || output_padding_w < 0) {
        fail(op, "kH/kW/stride/dilation >=1 and pad/output_padding >=0");
    }
    if (output_padding_h >= stride_h && output_padding_h >= dil_h) {
        fail(op, "output_padding_h must be < stride_h or < dil_h");
    }
    if (output_padding_w >= stride_w && output_padding_w >= dil_w) {
        fail(op, "output_padding_w must be < stride_w or < dil_w");
    }
}

}  // namespace

void conv_transpose2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              ::brotensor::Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        fail(op, "X must be FP32, FP16 or BF16");
    }
    if (Wt.dtype != X.dtype) fail(op, "Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) fail(op, "bias dtype must match X");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");

    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (X.rows != N || X.cols != C_in * H * W) {
        fail(op, "X shape must be (N, C_in*H*W)");
    }
    if (bias && (bias->rows != C_out || bias->cols != 1)) {
        fail(op, "bias shape must be (C_out, 1)");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    if (X.dtype == Dtype::FP32 && groups == 1 &&
        conv_transpose2d_f32_gemm(static_cast<const float*>(X.data),
                                  static_cast<const float*>(Wt.data),
                                  bias ? static_cast<const float*>(bias->data) : nullptr,
                                  static_cast<float*>(Y.data), N, C_in, H, W, C_out,
                                  kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h,
                                  dil_w, H_out, W_out, cur_stream())) {
        return;
    }
    dispatch_conv_direct(
        1u, X.dtype, X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, H_out, W_out, 1, kH, kW,
        1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv_transpose2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int H, int W,
                                     int C_out, int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     ::brotensor::Tensor& dX) {
    const char* op = "conv_transpose2d_backward_input";
    require_fp32(op, Wt, "Wt");
    require_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;

    const int Hc = (H_out + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int Wc = (W_out + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (Hc == H && Wc == W) {
        conv2d_forward(dY, Wt, nullptr, N, C_out, H_out, W_out, C_in, kH, kW,
                       stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dX);
        return;
    }
    ::brotensor::Tensor full = ::brotensor::Tensor::empty_on(dY.device, N, C_in * Hc * Wc, dY.dtype);
    conv2d_forward(dY, Wt, nullptr, N, C_out, H_out, W_out, C_in, kH, kW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, full);
    slice2d_forward(full, N, C_in, Hc, Wc, 0, 0, H, W, dX);
}

void conv_transpose2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int H, int W,
                                      int C_out, int kH, int kW,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      ::brotensor::Tensor& dWt) {
    const char* op = "conv_transpose2d_backward_weight";
    require_fp32(op, X, "X");
    require_fp32(op, dY, "dY");
    require_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (dWt.rows != C_in || dWt.cols != Cg_out * kHW) {
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (X.rows != N || X.cols != C_in * H * W) {
        fail(op, "X shape must be (N, C_in*H*W)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    if (C_in == 0 || Cg_out == 0 || kHW == 0 || N == 0) return;

    const int Hc = (H_out + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int Wc = (W_out + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (Hc == H && Wc == W) {
        conv2d_backward_weight(dY, X, N, C_out, H_out, W_out, C_in, kH, kW,
                               stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
        return;
    }
    ::brotensor::Tensor Xp = ::brotensor::Tensor::empty_on(X.device, N, C_in * Hc * Wc, X.dtype);
    pad2d_forward(X, N, C_in, H, W, 0, Hc - H, 0, Wc - W, 0, Xp);
    conv2d_backward_weight(dY, Xp, N, C_out, H_out, W_out, C_in, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
}

void conv_transpose2d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    ::brotensor::Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    require_fp32(op, dY, "dY");
    require_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1) {
        fail(op, "dB shape must be (C_out, 1)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;
    conv2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

void fill_cuda_vtable_conv_transpose2d(::brotensor::detail::OpsVTable& v) {
    v.conv_transpose2d_forward         = &conv_transpose2d_forward;
    v.conv_transpose2d_backward_input  = &conv_transpose2d_backward_input;
    v.conv_transpose2d_backward_weight = &conv_transpose2d_backward_weight;
    v.conv_transpose2d_backward_bias   = &conv_transpose2d_backward_bias;
}

}  // namespace brotensor::detail::cuda
