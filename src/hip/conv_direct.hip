// Native AMD HIP direct and depthwise 2D convolution kernels for RDNA 3.5 (gfx1151, Wave32).

#include "detail/conv_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int CONV_BLOCK = 256;
constexpr int kWave32 = 32;

inline hipStream_t cur_stream(int dev = -1) {
    return reinterpret_cast<hipStream_t>(hip_current_stream(dev));
}

template <typename T>
__device__ __forceinline__ float load_f32(const T* p);
template <> __device__ __forceinline__ float load_f32<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ __forceinline__ float load_f32<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T>
__device__ __forceinline__ void store_f32(T* p, float v);
template <> __device__ __forceinline__ void store_f32<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ __forceinline__ void store_f32<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

// ─── Direct Fallback Kernels ────────────────────────────────────────────────
template <typename T>
__global__ void conv2d_forward_kernel(
        const T* __restrict__ X,
        const T* __restrict__ Wt,
        const T* __restrict__ bias,
        T* __restrict__ Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int groups, int Cg_in, int Cg_out,
        long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = static_cast<int>(idx % W_out);
    long long t = idx / W_out;
    const int oh = static_cast<int>(t % H_out);
    t /= H_out;
    const int oc = static_cast<int>(t % C_out);
    const int n  = static_cast<int>(t / C_out);

    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    float acc = 0.0f;
    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const long long w_oc_base = (long long)oc * Cg_in * kH * kW;
    const long long x_n_base  = (long long)n * C_in * H * W;

    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const long long w_ic_base = w_oc_base + (long long)ic_local * kH * kW;
        const long long x_ic_base = x_n_base  + (long long)ic * H * W;
        for (int kh = 0; kh < kH; ++kh) {
            const int in_h = in_h_origin + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int in_w = in_w_origin + kw * dil_w;
                if (in_w < 0 || in_w >= W) continue;
                acc += load_f32<T>(&X[x_ic_base + in_h * W + in_w]) *
                       load_f32<T>(&Wt[w_ic_base + kh * kW + kw]);
            }
        }
    }
    if (bias) acc += load_f32<T>(&bias[oc]);
    store_f32<T>(&Y[idx], acc);
}

template <typename T>
__global__ void conv2d_backward_input_kernel(
        const T* __restrict__ Wt,
        const T* __restrict__ dY,
        T* __restrict__ dX,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int groups, int Cg_in, int Cg_out,
        long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int j = static_cast<int>(idx % W);
    long long t = idx / W;
    const int i = static_cast<int>(t % H);
    t /= H;
    const int c_in = static_cast<int>(t % C_in);
    const int n    = static_cast<int>(t / C_in);

    const int g = c_in / Cg_in;
    const int c_in_local = c_in - g * Cg_in;
    const int oc_lo = g * Cg_out;
    const int oc_hi = oc_lo + Cg_out;

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        const int num_h = i + pad_h - dil_h * kh;
        if (num_h < 0 || num_h % stride_h != 0) continue;
        const int i_out = num_h / stride_h;
        if (i_out < 0 || i_out >= H_out) continue;

        for (int kw = 0; kw < kW; ++kw) {
            const int num_w = j + pad_w - dil_w * kw;
            if (num_w < 0 || num_w % stride_w != 0) continue;
            const int j_out = num_w / stride_w;
            if (j_out < 0 || j_out >= W_out) continue;

            for (int c_out = oc_lo; c_out < oc_hi; ++c_out) {
                const long long dy_idx = (((long long)n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                const long long w_idx  = (((long long)c_out * Cg_in + c_in_local) * kH + kh) * kW + kw;
                acc += load_f32<T>(&dY[dy_idx]) * load_f32<T>(&Wt[w_idx]);
            }
        }
    }
    store_f32<T>(&dX[idx], acc);
}

template <typename T>
__global__ void conv2d_backward_weight_kernel(
        const T* __restrict__ X,
        const T* __restrict__ dY,
        float* __restrict__ dWt_scratch,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int groups, int Cg_in, int Cg_out,
        long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int kw = static_cast<int>(idx % kW);
    long long t = idx / kW;
    const int kh = static_cast<int>(t % kH);
    t /= kH;
    const int c_in_local = static_cast<int>(t % Cg_in);
    const int c_out      = static_cast<int>(t / Cg_in);

    const int g = c_out / Cg_out;
    const int c_in = g * Cg_in + c_in_local;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int i_out = 0; i_out < H_out; ++i_out) {
            const int in_h = i_out * stride_h - pad_h + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int j_out = 0; j_out < W_out; ++j_out) {
                const int in_w = j_out * stride_w - pad_w + kw * dil_w;
                if (in_w < 0 || in_w >= W) continue;
                const long long x_idx  = (((long long)n * C_in  + c_in)  * H     + in_h)  * W     + in_w;
                const long long dy_idx = (((long long)n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                acc += load_f32<T>(&dY[dy_idx]) * load_f32<T>(&X[x_idx]);
            }
        }
    }
    dWt_scratch[idx] = acc;
}

// ─── Depthwise Wave32 Kernels ───────────────────────────────────────────────
template <typename T>
__global__ void depthwise_conv2d_forward_wave32_kernel(
        const T* __restrict__ X,
        const T* __restrict__ Wt,
        const T* __restrict__ bias,
        T* __restrict__ Y,
        int N, int C, int H, int W,
        int kH, int kW, int H_out, int W_out,
        int stride_h, int stride_w, int pad_h, int pad_w,
        int dil_h, int dil_w, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ow = idx % W_out;
    long long t = idx / W_out;
    int oh = t % H_out;
    t /= H_out;
    int c = t % C;
    int n = t / C;

    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;
    const long long x_base = ((long long)n * C + c) * H * W;
    const int w_base = c * kH * kW;

    float acc = bias ? load_f32<T>(&bias[c]) : 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        int ih = in_h_origin + kh * dil_h;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = in_w_origin + kw * dil_w;
            if (iw < 0 || iw >= W) continue;
            acc += load_f32<T>(&X[x_base + ih * W + iw]) * load_f32<T>(&Wt[w_base + kh * kW + kw]);
        }
    }
    store_f32<T>(&Y[idx], acc);
}

template <typename T>
__global__ void depthwise_conv2d_backward_input_wave32_kernel(
        const T* __restrict__ Wt,
        const T* __restrict__ dY,
        T* __restrict__ dX,
        int N, int C, int H, int W,
        int kH, int kW, int H_out, int W_out,
        int stride_h, int stride_w, int pad_h, int pad_w,
        int dil_h, int dil_w, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int iw = idx % W;
    long long t = idx / W;
    int ih = t % H;
    t /= H;
    int c = t % C;
    int n = t / C;

    const long long dy_base = ((long long)n * C + c) * H_out * W_out;
    const int w_base = c * kH * kW;

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        int num_h = ih + pad_h - kh * dil_h;
        if (num_h < 0 || num_h % stride_h != 0) continue;
        int oh = num_h / stride_h;
        if (oh >= H_out) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int num_w = iw + pad_w - kw * dil_w;
            if (num_w < 0 || num_w % stride_w != 0) continue;
            int ow = num_w / stride_w;
            if (ow >= W_out) continue;
            acc += load_f32<T>(&dY[dy_base + oh * W_out + ow]) * load_f32<T>(&Wt[w_base + kh * kW + kw]);
        }
    }
    store_f32<T>(&dX[idx], acc);
}

template <typename T>
__global__ void depthwise_conv2d_backward_weight_wave32_kernel(
        const T* __restrict__ X,
        const T* __restrict__ dY,
        float* __restrict__ dWt_scratch,
        int N, int C, int H, int W,
        int kH, int kW, int H_out, int W_out,
        int stride_h, int stride_w, int pad_h, int pad_w,
        int dil_h, int dil_w) {
    const int idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int kw = idx % kW;
    int t = idx / kW;
    const int kh = t % kH;
    const int c  = t / kH;

    const int spatial = H_out * W_out;
    const long long M = (long long)N * spatial;

    float acc = 0.0f;
    for (long long m = tid; m < M; m += blockDim.x) {
        const int n  = static_cast<int>(m / spatial);
        const int sp = static_cast<int>(m % spatial);
        const int i_out = sp / W_out;
        const int j_out = sp % W_out;
        const int in_h = i_out * stride_h - pad_h + kh * dil_h;
        if (in_h < 0 || in_h >= H) continue;
        const int in_w = j_out * stride_w - pad_w + kw * dil_w;
        if (in_w < 0 || in_w >= W) continue;
        const long long x_idx  = (((long long)n * C + c) * H     + in_h)  * W     + in_w;
        const long long dy_idx = (((long long)n * C + c) * H_out + i_out) * W_out + j_out;
        acc += load_f32<T>(&dY[dy_idx]) * load_f32<T>(&X[x_idx]);
    }

    __shared__ float s_acc[CONV_BLOCK];
    s_acc[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 16; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        __syncthreads();
    }
    if (tid < 16) {
        float v = s_acc[tid] + s_acc[tid + 16];
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset, kWave32);
        }
        if (tid == 0) dWt_scratch[idx] = v;
    }
}

} // namespace

void depthwise_conv2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C, int H, int W,
                              int kH, int kW, int H_out, int W_out,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int dil_h, int dil_w, long long total,
                              ::brotensor::Tensor& Y) {
    hipStream_t stream = cur_stream(X.device.index);
    const int blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        const __half* b_ptr = bias ? static_cast<const __half*>(bias->data) : nullptr;
        depthwise_conv2d_forward_wave32_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(Wt.data), b_ptr,
            static_cast<__half*>(Y.data), N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        const hip_bfloat16* b_ptr = bias ? static_cast<const hip_bfloat16*>(bias->data) : nullptr;
        depthwise_conv2d_forward_wave32_kernel<hip_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(Wt.data), b_ptr,
            static_cast<hip_bfloat16*>(Y.data), N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    } else {
        const float* b_ptr = bias ? static_cast<const float*>(bias->data) : nullptr;
        depthwise_conv2d_forward_wave32_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(Wt.data), b_ptr,
            static_cast<float*>(Y.data), N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void depthwise_conv2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C, int H, int W,
                                     int kH, int kW, int H_out, int W_out,
                                     int stride_h, int stride_w, int pad_h, int pad_w,
                                     int dil_h, int dil_w, long long total,
                                     ::brotensor::Tensor& dX) {
    hipStream_t stream = cur_stream(Wt.device.index);
    const int blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
    if (Wt.dtype == ::brotensor::Dtype::FP16) {
        depthwise_conv2d_backward_input_wave32_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(Wt.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    } else if (Wt.dtype == ::brotensor::Dtype::BF16) {
        depthwise_conv2d_backward_input_wave32_kernel<hip_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(Wt.data), static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data), N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    } else {
        depthwise_conv2d_backward_input_wave32_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(Wt.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void depthwise_conv2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      float* dWt_scratch,
                                      int N, int C, int H, int W,
                                      int kH, int kW, int H_out, int W_out,
                                      int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dil_h, int dil_w, int total) {
    hipStream_t stream = cur_stream(X.device.index);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        depthwise_conv2d_backward_weight_wave32_kernel<__half><<<total, CONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data), dWt_scratch,
            N, C, H, W, kH, kW, H_out, W_out, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        depthwise_conv2d_backward_weight_wave32_kernel<hip_bfloat16><<<total, CONV_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(dY.data), dWt_scratch,
            N, C, H, W, kH, kW, H_out, W_out, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);
    } else {
        depthwise_conv2d_backward_weight_wave32_kernel<float><<<total, CONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data), dWt_scratch,
            N, C, H, W, kH, kW, H_out, W_out, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void conv2d_direct_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W, int C_out,
                           int kH, int kW, int H_out, int W_out,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups, int Cg_in, int Cg_out,
                           long long total, ::brotensor::Tensor& Y) {
    hipStream_t stream = cur_stream(X.device.index);
    const int blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        const __half* b_ptr = bias ? static_cast<const __half*>(bias->data) : nullptr;
        conv2d_forward_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(Wt.data), b_ptr,
            static_cast<__half*>(Y.data), N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        const hip_bfloat16* b_ptr = bias ? static_cast<const hip_bfloat16*>(bias->data) : nullptr;
        conv2d_forward_kernel<hip_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(Wt.data), b_ptr,
            static_cast<hip_bfloat16*>(Y.data), N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    } else {
        const float* b_ptr = bias ? static_cast<const float*>(bias->data) : nullptr;
        conv2d_forward_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(Wt.data), b_ptr,
            static_cast<float*>(Y.data), N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void conv2d_direct_backward_input(const ::brotensor::Tensor& Wt,
                                  const ::brotensor::Tensor& dY,
                                  int N, int C_in, int H, int W, int C_out,
                                  int kH, int kW, int H_out, int W_out,
                                  int stride_h, int stride_w, int pad_h, int pad_w,
                                  int dil_h, int dil_w, int groups, int Cg_in, int Cg_out,
                                  long long total, ::brotensor::Tensor& dX) {
    hipStream_t stream = cur_stream(Wt.device.index);
    const int blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
    if (Wt.dtype == ::brotensor::Dtype::FP16) {
        conv2d_backward_input_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(Wt.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    } else if (Wt.dtype == ::brotensor::Dtype::BF16) {
        conv2d_backward_input_kernel<hip_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(Wt.data), static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data), N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    } else {
        conv2d_backward_input_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(Wt.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void conv2d_direct_backward_weight(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor& dY,
                                   float* dWt_scratch,
                                   int N, int C_in, int H, int W, int C_out,
                                   int kH, int kW, int H_out, int W_out,
                                   int stride_h, int stride_w, int pad_h, int pad_w,
                                   int dil_h, int dil_w, int groups, int Cg_in, int Cg_out,
                                   long long total) {
    hipStream_t stream = cur_stream(X.device.index);
    const int blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        conv2d_backward_weight_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data), dWt_scratch,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        conv2d_backward_weight_kernel<hip_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(dY.data), dWt_scratch,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    } else {
        conv2d_backward_weight_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data), dWt_scratch,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
