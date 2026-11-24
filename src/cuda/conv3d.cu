// CUDA conv3d_forward + conv3d_int8w_fp16_forward (Qwen3-VL patch embed).
//
// Mirrors src/cuda/conv2d.cu's naive direct-conv kernels with an added T
// (depth) axis. No WMMA fast path — the patch embedder is small and a single
// direct kernel is fine per Chunk A scope.
//
// Layouts (match the CPU/Metal ports):
//   X       : NCTHW
//   Wt      : OICTHW (grouped — I-dim sized as Cg_in for grouped conv)
//   bias    : (C_out, 1), optional
//   Y       : NCTHW
//
// FP32 accumulator; storage dtype T is __half / __nv_bfloat16 / float for the
// generic forward, and FP16 / INT8 / FP32 (X / W / scales) for the W8A16 path.
// Y is OVERWRITTEN.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

#include "detail/cuda_check.h"

namespace brotensor {

void* cuda_current_stream();

namespace {

constexpr int CONV_BLOCK = 256;

// W8A16 variant: X / bias FP16, W INT8 with per-c_out FP32 dequant scale.
__global__ void conv3d_int8w_fp16_forward_kernel(
        const __half* __restrict__ X,
        const int8_t* __restrict__ W,
        const float*  __restrict__ scales,
        const __half* __restrict__ bias,
        __half* __restrict__ Y,
        int N, int C_in, int T_in, int H, int W_in_,
        int C_out, int kT, int kH, int kW,
        int T_out, int H_out, int W_out,
        int stride_t, int stride_h, int stride_w,
        int pad_t, int pad_h, int pad_w,
        int dil_t, int dil_h, int dil_w,
        int /*groups*/, int Cg_in, int Cg_out,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int ot = t % T_out;
    t /= T_out;
    const int oc = t % C_out;
    const int n  = t / C_out;

    const int in_t_origin = ot * stride_t - pad_t;
    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    const float scale = scales[oc];

    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const int w_oc_base = oc * Cg_in * kT * kH * kW;
    const int x_n_base = n * C_in * T_in * H * W_in_;

    float acc = 0.0f;
    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const int w_ic_base = w_oc_base + ic_local * kT * kH * kW;
        const int x_ic_base = x_n_base + ic * T_in * H * W_in_;
        for (int kt = 0; kt < kT; ++kt) {
            const int in_t = in_t_origin + kt * dil_t;
            if (in_t < 0 || in_t >= T_in) continue;
            for (int kh = 0; kh < kH; ++kh) {
                const int in_h = in_h_origin + kh * dil_h;
                if (in_h < 0 || in_h >= H) continue;
                for (int kw = 0; kw < kW; ++kw) {
                    const int in_w = in_w_origin + kw * dil_w;
                    if (in_w < 0 || in_w >= W_in_) continue;
                    const float xv = __half2float(
                        X[x_ic_base + (in_t * H + in_h) * W_in_ + in_w]);
                    const float wv = static_cast<float>(
                        W[w_ic_base + (kt * kH + kh) * kW + kw]) * scale;
                    acc += xv * wv;
                }
            }
        }
    }
    if (bias) acc += __half2float(bias[oc]);
    Y[idx] = __float2half(acc);
}

} // namespace

namespace detail::cuda {

void dispatch_conv_direct(std::uint32_t mode, Dtype dt,
                          const void* x, const void* w, void* y, const void* bias,
                          int N, int Cin, int T, int H, int Wd,
                          int Cout, int To, int Ho, int Wo,
                          int kT, int kH, int kW,
                          int st, int sh, int sw,
                          int pt, int ph, int pw,
                          int dt_, int dh, int dw, int groups);

void conv3d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int T_in, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv3d_forward: X must be FP16, BF16 or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: bias dtype must match X");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv3d_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_forward: non-positive output shape");
    }
    const int out_cols = C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    dispatch_conv_direct(
        0u, X.dtype, X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, T_in, H, W, C_out, T_out, H_out, W_out, kT, kH, kW,
        stride_t, stride_h, stride_w, pad_t, pad_h, pad_w, dil_t, dil_h, dil_w, groups);
}

void conv3d_int8w_fp16_forward(const ::brotensor::Tensor& X,
                               const ::brotensor::Tensor& W_int8,
                               const ::brotensor::Tensor& scales,
                               const ::brotensor::Tensor* bias,
                               int N, int C_in, int T_in, int H, int W,
                               int C_out, int kT, int kH, int kW,
                               int stride_t, int stride_h, int stride_w,
                               int pad_t, int pad_h, int pad_w,
                               int dil_t, int dil_h, int dil_w, int groups,
                               ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: X must be FP16");
    }
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: scales must be FP32");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv3d_int8w_fp16_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    if (W_int8.rows != C_out || W_int8.cols != Cg_in * kT * kH * kW) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: W shape mismatch");
    }
    if (scales.rows != C_out || scales.cols != 1) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: scales shape mismatch");
    }
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: non-positive output shape");
    }
    const int out_cols = C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = bias ? static_cast<const __half*>(bias->data) : nullptr;

    const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    conv3d_int8w_fp16_forward_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
        static_cast<const __half*>(X.data),
        static_cast<const int8_t*>(W_int8.data),
        static_cast<const float*>(scales.data),
        b_p,
        static_cast<__half*>(Y.data),
        N, C_in, T_in, H, W, C_out, kT, kH, kW,
        T_out, H_out, W_out,
        stride_t, stride_h, stride_w,
        pad_t, pad_h, pad_w,
        dil_t, dil_h, dil_w,
        groups, Cg_in, Cg_out, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// Per-cluster vtable contribution for conv3d (forward + INT8w variant).
void fill_cuda_vtable_conv3d(::brotensor::detail::OpsVTable& v) {
    v.conv3d_forward             = &conv3d_forward;
    v.conv3d_int8w_fp16_forward  = &conv3d_int8w_fp16_forward;
}

} // namespace detail::cuda

} // namespace brotensor
