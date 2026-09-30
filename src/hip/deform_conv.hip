// Native AMD HIP modulated deformable conv2d (torchvision deform_conv2d v2, fwd).
//
// Optimized for Wave32 wavefronts on RDNA 3.5.

#include "detail/conv_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int DCONV_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int dconv_grid(long long n) {
    long long blocks = (n + DCONV_BLOCK - 1) / DCONV_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

template <typename T>
__device__ inline float dload(const T* p);
template <> __device__ inline float dload<float>(const float* p)   { return *p; }
template <> __device__ inline float dload<__half>(const __half* p) { return __half2float(*p); }

template <typename T>
__device__ inline void dstore(T* p, float v);
template <> __device__ inline void dstore<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void dstore<__half>(__half* p, float v) { *p = __float2half(v); }

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
        const T* __restrict__ mask,
        const T* __restrict__ Wt,
        const T* __restrict__ bias,
        T* __restrict__ Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int groups, int deform_groups,
        int Cg_in, int Cg_out, int Cdg_in,
        int H_out, int W_out,
        long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = static_cast<int>(idx % W_out);
    long long t = idx / W_out;
    const int oh = static_cast<int>(t % H_out);
    t /= H_out;
    const int oc = static_cast<int>(t % C_out);
    const int n  = static_cast<int>(t / C_out);

    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    const long long w_oc_base = (long long)oc * Cg_in * kH * kW;
    const long long x_n_base  = (long long)n * C_in * H * W;
    const long long off_n_base = (long long)n * deform_groups * 2 * kH * kW * H_out * W_out;
    const long long mask_n_base = (long long)n * deform_groups * kH * kW * H_out * W_out;

    float acc = 0.0f;

    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const int dg = ic / Cdg_in;
        const long long w_ic_base = w_oc_base + (long long)ic_local * kH * kW;
        const long long x_ic_base = x_n_base  + (long long)ic * H * W;
        const long long off_dg_base = off_n_base + (long long)dg * 2 * kH * kW * H_out * W_out;
        const long long mask_dg_base = mask_n_base + (long long)dg * kH * kW * H_out * W_out;

        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int tap = kh * kW + kw;
                const long long off_h_idx = off_dg_base + ((2 * tap)     * H_out + oh) * W_out + ow;
                const long long off_w_idx = off_dg_base + ((2 * tap + 1) * H_out + oh) * W_out + ow;
                const float d_h = dload<T>(&offset[off_h_idx]);
                const float d_w = dload<T>(&offset[off_w_idx]);
                const float h_im = in_h_origin + kh * dil_h + d_h;
                const float w_im = in_w_origin + kw * dil_w + d_w;

                float x_val = dbilinear<T>(&X[x_ic_base], H, W, h_im, w_im);
                if (mask) {
                    const long long m_idx = mask_dg_base + (tap * H_out + oh) * W_out + ow;
                    x_val *= dload<T>(&mask[m_idx]);
                }
                const long long w_idx = w_ic_base + tap;
                acc += x_val * dload<T>(&Wt[w_idx]);
            }
        }
    }

    if (bias) acc += dload<T>(&bias[oc]);
    dstore<T>(&Y[idx], acc);
}

} // namespace

void deform_conv2d_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& offset,
                           const ::brotensor::Tensor* mask,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W, int C_out,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dil_h, int dil_w,
                           int groups, int deform_groups,
                           ::brotensor::Tensor& Y) {
    if (X.dtype != ::brotensor::Dtype::FP16 && X.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error("deform_conv2d_forward: X must be FP16 or FP32");
    }
    if (offset.dtype != X.dtype || Wt.dtype != X.dtype ||
        (mask && mask->dtype != X.dtype) ||
        (bias && bias->dtype != X.dtype)) {
        throw std::runtime_error("deform_conv2d_forward: operand dtypes must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("deform_conv2d_forward: groups must divide C_in and C_out");
    }
    if (deform_groups < 1 || C_in % deform_groups != 0) {
        throw std::runtime_error("deform_conv2d_forward: deform_groups must divide C_in");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int Cdg_in = C_in  / deform_groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("deform_conv2d_forward: non-positive output shape");
    }
    const long long out_cols = (long long)C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const long long total = (long long)N * out_cols;
    if (total == 0) return;

    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        const __half* m_ptr = mask ? static_cast<const __half*>(mask->data) : nullptr;
        const __half* b_ptr = bias ? static_cast<const __half*>(bias->data) : nullptr;
        deform_conv2d_forward_kernel<__half><<<dconv_grid(total), DCONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(offset.data),
            m_ptr,
            static_cast<const __half*>(Wt.data),
            b_ptr,
            static_cast<__half*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, deform_groups, Cg_in, Cg_out, Cdg_in,
            H_out, W_out, total);
    } else {
        const float* m_ptr = mask ? static_cast<const float*>(mask->data) : nullptr;
        const float* b_ptr = bias ? static_cast<const float*>(bias->data) : nullptr;
        deform_conv2d_forward_kernel<float><<<dconv_grid(total), DCONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(offset.data),
            m_ptr,
            static_cast<const float*>(Wt.data),
            b_ptr,
            static_cast<float*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, deform_groups, Cg_in, Cg_out, Cdg_in,
            H_out, W_out, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
