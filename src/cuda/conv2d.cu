#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int CONV_BLOCK = 256;

// One thread per output element. Naive direct-conv reduction over
// (C_in, kH, kW). FP32 accumulator; FP16 load/store.
__global__ void conv2d_forward_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ Wt,
        const __half* __restrict__ bias,   // may be null
        __half* __restrict__ Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Unflatten idx → (n, oc, oh, ow).
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int oc = t % C_out;
    const int n  = t / C_out;

    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    float acc = 0.0f;
    // Weight base for this output channel: (oc, 0, 0, 0) in OIHW.
    const int w_oc_base = oc * C_in * kH * kW;
    // Input base for this sample.
    const int x_n_base = n * C_in * H * W;

    for (int ic = 0; ic < C_in; ++ic) {
        const int w_ic_base = w_oc_base + ic * kH * kW;
        const int x_ic_base = x_n_base + ic * H * W;
        for (int kh = 0; kh < kH; ++kh) {
            const int in_h = in_h_origin + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int in_w = in_w_origin + kw * dil_w;
                if (in_w < 0 || in_w >= W) continue;
                const float x_v = __half2float(X[x_ic_base + in_h * W + in_w]);
                const float w_v = __half2float(Wt[w_ic_base + kh * kW + kw]);
                acc += x_v * w_v;
            }
        }
    }
    if (bias) {
        acc += __half2float(bias[oc]);
    }
    Y[idx] = __float2half(acc);
}

} // namespace

void conv2d_forward_gpu(const GpuTensor& X,
                        const GpuTensor& Wt,
                        const GpuTensor* bias,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        int dil_h, int dil_w,
                        GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || Wt.dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_forward_gpu: X and Wt must be FP16");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_forward_gpu: bias must be FP16");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward_gpu: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    const __half* x_p  = reinterpret_cast<const __half*>(X.data_fp16());
    const __half* w_p  = reinterpret_cast<const __half*>(Wt.data_fp16());
    const __half* b_p  = bias ? reinterpret_cast<const __half*>(bias->data_fp16())
                              : nullptr;
    __half* y_p        = reinterpret_cast<__half*>(Y.data_fp16());

    conv2d_forward_kernel<<<blocks, CONV_BLOCK>>>(
        x_p, w_p, b_p, y_p,
        N, C_in, H, W,
        C_out, kH, kW,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
