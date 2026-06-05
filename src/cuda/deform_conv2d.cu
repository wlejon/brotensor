// ─── CUDA modulated deformable conv2d (torchvision deform_conv2d v2, fwd) ───
//
// One thread per output element; FP32 accumulator, storage dtype T (FP16/FP32).
// Direct fused form of torchvision's deformable_im2col + GEMM: each kH×kW tap
// is bilinearly sampled from X at a per-tap, per-pixel offset location with ZERO
// padding (torchvision convention), optionally reweighted by the mask
// modulator, then reduced against the OIHW weight. Mirrors the CPU impl in
// src/cpu/deform_conv2d.cpp exactly.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

#include "detail/cuda_check.h"

namespace brotensor {

void* cuda_current_stream();

namespace {

constexpr int DCONV_BLOCK = 256;

template <typename T>
__device__ inline float dload(const T* p);
template <> __device__ inline float dload<float>(const float* p)   { return *p; }
template <> __device__ inline float dload<__half>(const __half* p) { return __half2float(*p); }

template <typename T>
__device__ inline void dstore(T* p, float v);
template <> __device__ inline void dstore<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void dstore<__half>(__half* p, float v) { *p = __float2half(v); }

// torchvision bilinear_interpolate: zero outside [0,H)×[0,W), per-corner guard.
template <typename T>
__device__ inline float dbilinear(const T* in, int H, int W, float h, float w) {
    if (h <= -1.0f || static_cast<float>(H) <= h ||
        w <= -1.0f || static_cast<float>(W) <= w) {
        return 0.0f;
    }
    int h_low = static_cast<int>(floorf(h));
    int w_low = static_cast<int>(floorf(w));
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    float lh = h - h_low, lw = w - w_low;
    float hh = 1.0f - lh, hw = 1.0f - lw;
    float v1 = (h_low >= 0 && w_low >= 0) ? dload<T>(&in[h_low * W + w_low]) : 0.0f;
    float v2 = (h_low >= 0 && w_high <= W - 1) ? dload<T>(&in[h_low * W + w_high]) : 0.0f;
    float v3 = (h_high <= H - 1 && w_low >= 0) ? dload<T>(&in[h_high * W + w_low]) : 0.0f;
    float v4 = (h_high <= H - 1 && w_high <= W - 1) ? dload<T>(&in[h_high * W + w_high]) : 0.0f;
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
__global__ void deform_conv2d_forward_kernel(
        const T* __restrict__ X,
        const T* __restrict__ offset,
        const T* __restrict__ mask,    // may be null
        const T* __restrict__ Wt,
        const T* __restrict__ bias,    // may be null
        T* __restrict__ Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int Cg_in, int Cg_out, int c_per_off_grp,
        int deform_groups, int total) {
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
    const int ksz = kH * kW;

    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const int w_oc_base = oc * Cg_in * ksz;

    const T* off_n  = offset + (size_t)n * deform_groups * 2 * ksz * H_out * W_out;
    const T* mask_n = mask ? mask + (size_t)n * deform_groups * ksz * H_out * W_out : nullptr;

    float acc = bias ? dload<T>(&bias[oc]) : 0.0f;
    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const int off_grp = ic / c_per_off_grp;
        const T* in_ch = X + ((size_t)n * C_in + ic) * H * W;
        const T* off_grp_base = off_n + (size_t)off_grp * 2 * ksz * H_out * W_out;
        const T* mask_grp_base = mask_n ? mask_n + (size_t)off_grp * ksz * H_out * W_out : nullptr;
        const int w_ic_base = w_oc_base + ic_local * ksz;
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int tap = kh * kW + kw;
                const float off_y = dload<T>(&off_grp_base[((2 * tap) * H_out + oh) * W_out + ow]);
                const float off_x = dload<T>(&off_grp_base[((2 * tap + 1) * H_out + oh) * W_out + ow]);
                const float m = mask_grp_base
                    ? dload<T>(&mask_grp_base[(tap * H_out + oh) * W_out + ow]) : 1.0f;
                const float y = in_h_origin + kh * dil_h + off_y;
                const float x = in_w_origin + kw * dil_w + off_x;
                const float val = dbilinear<T>(in_ch, H, W, y, x);
                acc += dload<T>(&Wt[w_ic_base + tap]) * (m * val);
            }
        }
    }
    dstore<T>(&Y[idx], acc);
}

} // namespace

namespace detail::cuda {

void deform_conv2d_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& offset,
                           const ::brotensor::Tensor* mask,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups, int deform_groups,
                           ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("deform_conv2d_forward: X must be FP16 or FP32");
    }
    if (Wt.dtype != X.dtype || offset.dtype != X.dtype ||
        (mask && mask->dtype != X.dtype) || (bias && bias->dtype != X.dtype)) {
        throw std::runtime_error(
            "deform_conv2d_forward: X, offset, mask, Wt, bias dtype must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "deform_conv2d_forward: groups must divide C_in and C_out");
    }
    if (deform_groups < 1 || C_in % deform_groups != 0) {
        throw std::runtime_error(
            "deform_conv2d_forward: deform_groups must divide C_in");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int c_per_off_grp = C_in / deform_groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("deform_conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    const int blocks = (total + DCONV_BLOCK - 1) / DCONV_BLOCK;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    if (X.dtype == Dtype::FP16) {
        deform_conv2d_forward_kernel<__half><<<blocks, DCONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(offset.data),
            mask ? static_cast<const __half*>(mask->data) : nullptr,
            static_cast<const __half*>(Wt.data),
            bias ? static_cast<const __half*>(bias->data) : nullptr,
            static_cast<__half*>(Y.data),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, c_per_off_grp, deform_groups, total);
    } else {
        deform_conv2d_forward_kernel<float><<<blocks, DCONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(offset.data),
            mask ? static_cast<const float*>(mask->data) : nullptr,
            static_cast<const float*>(Wt.data),
            bias ? static_cast<const float*>(bias->data) : nullptr,
            static_cast<float*>(Y.data),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            Cg_in, Cg_out, c_per_off_grp, deform_groups, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_deform_conv(::brotensor::detail::OpsVTable& v) {
    v.deform_conv2d_forward = &deform_conv2d_forward;
}

} // namespace detail::cuda

} // namespace brotensor
