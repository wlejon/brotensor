// Native AMD HIP 3D convolution (conv3d_forward and conv3d_int8w_fp16_forward).
//
// Optimized for Wave32 wavefronts on RDNA 3.5.

#include "detail/conv_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int CONV3D_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int conv3d_grid(long long n) {
    long long blocks = (n + CONV3D_BLOCK - 1) / CONV3D_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

template <typename T>
__device__ inline float c3d_load_f32(const T* p);
template <> __device__ inline float c3d_load_f32<float>(const float* p)   { return *p; }
template <> __device__ inline float c3d_load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float c3d_load_f32<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T>
__device__ inline void c3d_store_f32(T* p, float v);
template <> __device__ inline void c3d_store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void c3d_store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void c3d_store_f32<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

template <typename T>
__global__ void conv3d_forward_kernel(
        const T* __restrict__ X,
        const T* __restrict__ Wt,
        const T* __restrict__ bias,
        T* __restrict__ Y,
        int N, int C_in, int T_in, int H, int W,
        int C_out, int kT, int kH, int kW,
        int T_out, int H_out, int W_out,
        int stride_t, int stride_h, int stride_w,
        int pad_t, int pad_h, int pad_w,
        int dil_t, int dil_h, int dil_w,
        int groups, int Cg_in, int Cg_out,
        long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = static_cast<int>(idx % W_out);
    long long t = idx / W_out;
    const int oh = static_cast<int>(t % H_out);
    t /= H_out;
    const int ot = static_cast<int>(t % T_out);
    t /= T_out;
    const int oc = static_cast<int>(t % C_out);
    const int n  = static_cast<int>(t / C_out);

    const int in_t_origin = ot * stride_t - pad_t;
    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    float acc = 0.0f;
    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const long long w_oc_base = (long long)oc * Cg_in * kT * kH * kW;
    const long long x_n_base  = (long long)n * C_in * T_in * H * W;

    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const long long w_ic_base = w_oc_base + (long long)ic_local * kT * kH * kW;
        const long long x_ic_base = x_n_base  + (long long)ic * T_in * H * W;

        for (int kt_idx = 0; kt_idx < kT; ++kt_idx) {
            const int in_t = in_t_origin + kt_idx * dil_t;
            if (in_t < 0 || in_t >= T_in) continue;

            for (int kh = 0; kh < kH; ++kh) {
                const int in_h = in_h_origin + kh * dil_h;
                if (in_h < 0 || in_h >= H) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    const int in_w = in_w_origin + kw * dil_w;
                    if (in_w < 0 || in_w >= W) continue;

                    const long long x_offset =
                        x_ic_base + (((long long)in_t * H + in_h) * W + in_w);
                    const long long w_offset =
                        w_ic_base + (((long long)kt_idx * kH + kh) * kW + kw);
                    acc += c3d_load_f32<T>(&X[x_offset]) * c3d_load_f32<T>(&Wt[w_offset]);
                }
            }
        }
    }

    if (bias) acc += c3d_load_f32<T>(&bias[oc]);
    c3d_store_f32<T>(&Y[idx], acc);
}

__global__ void conv3d_int8w_fp16_forward_kernel(
        const __half* __restrict__ X,
        const int8_t* __restrict__ W_int8,
        const float*  __restrict__ scales,
        const __half* __restrict__ bias,
        __half*       __restrict__ Y,
        int N, int C_in, int T_in, int H, int W,
        int C_out, int kT, int kH, int kW,
        int T_out, int H_out, int W_out,
        int stride_t, int stride_h, int stride_w,
        int pad_t, int pad_h, int pad_w,
        int dil_t, int dil_h, int dil_w,
        int groups, int Cg_in, int Cg_out,
        long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = static_cast<int>(idx % W_out);
    long long t = idx / W_out;
    const int oh = static_cast<int>(t % H_out);
    t /= H_out;
    const int ot = static_cast<int>(t % T_out);
    t /= T_out;
    const int oc = static_cast<int>(t % C_out);
    const int n  = static_cast<int>(t / C_out);

    const int in_t_origin = ot * stride_t - pad_t;
    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    float acc = 0.0f;
    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const long long w_oc_base = (long long)oc * Cg_in * kT * kH * kW;
    const long long x_n_base  = (long long)n * C_in * T_in * H * W;

    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const long long w_ic_base = w_oc_base + (long long)ic_local * kT * kH * kW;
        const long long x_ic_base = x_n_base  + (long long)ic * T_in * H * W;

        for (int kt_idx = 0; kt_idx < kT; ++kt_idx) {
            const int in_t = in_t_origin + kt_idx * dil_t;
            if (in_t < 0 || in_t >= T_in) continue;

            for (int kh = 0; kh < kH; ++kh) {
                const int in_h = in_h_origin + kh * dil_h;
                if (in_h < 0 || in_h >= H) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    const int in_w = in_w_origin + kw * dil_w;
                    if (in_w < 0 || in_w >= W) continue;

                    const long long x_offset =
                        x_ic_base + (((long long)in_t * H + in_h) * W + in_w);
                    const long long w_offset =
                        w_ic_base + (((long long)kt_idx * kH + kh) * kW + kw);
                    const float x_val = __half2float(X[x_offset]);
                    const float w_val = static_cast<float>(W_int8[w_offset]);
                    acc += x_val * w_val;
                }
            }
        }
    }

    acc *= scales[oc];
    if (bias) acc += __half2float(bias[oc]);
    Y[idx] = __float2half(acc);
}

} // namespace

void conv3d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int T_in, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w, int groups,
                    ::brotensor::Tensor& Y) {
    if (X.dtype != ::brotensor::Dtype::FP16 &&
        X.dtype != ::brotensor::Dtype::FP32 &&
        X.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error("conv3d_forward: X must be FP16, BF16 or FP32");
    }
    if (Wt.dtype != X.dtype) throw std::runtime_error("conv3d_forward: Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) throw std::runtime_error("conv3d_forward: bias dtype must match X");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv3d_forward: groups must divide C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_forward: non-positive output shape");
    }
    const long long out_cols = (long long)C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const long long total = (long long)N * out_cols;
    if (total == 0) return;

    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        const __half* b_ptr = bias ? static_cast<const __half*>(bias->data) : nullptr;
        conv3d_forward_kernel<__half><<<conv3d_grid(total), CONV3D_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(Wt.data),
            b_ptr,
            static_cast<__half*>(Y.data),
            N, C_in, T_in, H, W, C_out, kT, kH, kW,
            T_out, H_out, W_out,
            stride_t, stride_h, stride_w,
            pad_t, pad_h, pad_w, dil_t, dil_h, dil_w,
            groups, Cg_in, Cg_out, total);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        const hip_bfloat16* b_ptr = bias ? static_cast<const hip_bfloat16*>(bias->data) : nullptr;
        conv3d_forward_kernel<hip_bfloat16><<<conv3d_grid(total), CONV3D_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data),
            static_cast<const hip_bfloat16*>(Wt.data),
            b_ptr,
            static_cast<hip_bfloat16*>(Y.data),
            N, C_in, T_in, H, W, C_out, kT, kH, kW,
            T_out, H_out, W_out,
            stride_t, stride_h, stride_w,
            pad_t, pad_h, pad_w, dil_t, dil_h, dil_w,
            groups, Cg_in, Cg_out, total);
    } else {
        const float* b_ptr = bias ? static_cast<const float*>(bias->data) : nullptr;
        conv3d_forward_kernel<float><<<conv3d_grid(total), CONV3D_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(Wt.data),
            b_ptr,
            static_cast<float*>(Y.data),
            N, C_in, T_in, H, W, C_out, kT, kH, kW,
            T_out, H_out, W_out,
            stride_t, stride_h, stride_w,
            pad_t, pad_h, pad_w, dil_t, dil_h, dil_w,
            groups, Cg_in, Cg_out, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
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
    if (X.dtype != ::brotensor::Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: X must be FP16");
    }
    if (W_int8.dtype != ::brotensor::Dtype::INT8) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: W_int8 must be INT8");
    }
    if (scales.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: scales must be FP32");
    }
    if (bias && bias->dtype != ::brotensor::Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: groups must divide C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: non-positive output shape");
    }
    const long long out_cols = (long long)C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != ::brotensor::Dtype::FP16) {
        Y.resize(N, out_cols, ::brotensor::Dtype::FP16);
    }
    const long long total = (long long)N * out_cols;
    if (total == 0) return;

    hipStream_t stream = cur_stream();
    const __half* b_ptr = bias ? static_cast<const __half*>(bias->data) : nullptr;
    conv3d_int8w_fp16_forward_kernel<<<conv3d_grid(total), CONV3D_BLOCK, 0, stream>>>(
        static_cast<const __half*>(X.data),
        static_cast<const int8_t*>(W_int8.data),
        static_cast<const float*>(scales.data),
        b_ptr,
        static_cast<__half*>(Y.data),
        N, C_in, T_in, H, W, C_out, kT, kH, kW,
        T_out, H_out, W_out,
        stride_t, stride_h, stride_w,
        pad_t, pad_h, pad_w, dil_t, dil_h, dil_w,
        groups, Cg_in, Cg_out, total);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
