// Native AMD HIP SwiGLU, GEGLU, and Bias-Act activations for RDNA 3.5 (gfx1151).
//
// Supports FP32, FP16 (__half), and BF16 (hip_bfloat16).

#include "detail/elementwise_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);
void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);

namespace {

constexpr int EW_BLOCK = 256;
constexpr float kSqrt2OverPi = 0.7978845608028654f;   // sqrt(2/pi)
constexpr float kGeluCoeff   = 0.044715f;
constexpr float kInvSqrt2    = 0.7071067811865475f;
constexpr float kInvSqrt2Pi  = 0.3989422804014327f;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int grid_for(long long n) {
    long long b = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return static_cast<int>(b);
}

__device__ __forceinline__ float to_f32(float v)           { return v; }
__device__ __forceinline__ float to_f32(__half v)          { return __half2float(v); }
__device__ __forceinline__ float to_f32(hip_bfloat16 v)    { return float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ hip_bfloat16 from_f32<hip_bfloat16>(float v) { return hip_bfloat16(v); }

__device__ __forceinline__ float silu_scalar(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float silu_grad_scalar(float v) {
    const float s = 1.0f / (1.0f + __expf(-v));
    return s * (1.0f + v * (1.0f - s));
}

__device__ __forceinline__ float gelu_tanh_scalar(float x) {
    const float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_tanh_grad_scalar(float x) {
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float inner = kSqrt2OverPi * (x + kGeluCoeff * x3);
    const float tanh_v = tanhf(inner);
    const float sech2 = 1.0f - tanh_v * tanh_v;
    const float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x2);
    return 0.5f * (1.0f + tanh_v) + 0.5f * x * sech2 * d_inner;
}

__device__ __forceinline__ float gelu_exact_scalar(float x) {
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float gelu_exact_grad_scalar(float x) {
    return 0.5f * (1.0f + erff(x * kInvSqrt2)) + x * kInvSqrt2Pi * __expf(-0.5f * x * x);
}

// ─── SwiGLU Kernels ────────────────────────────────────────────────────────

template <typename T>
__global__ void swiglu_forward_kernel(const T* __restrict__ X, T* __restrict__ Y, int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a  = to_f32(X[b * two_d + d]);
        const float bh = to_f32(X[b * two_d + D + d]);
        Y[idx] = from_f32<T>(silu_scalar(a) * bh);
    }
}

template <typename T>
__global__ void swiglu_backward_kernel(const T* __restrict__ X, const T* __restrict__ dY,
                                       T* __restrict__ dX, int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a      = to_f32(X[b * two_d + d]);
        const float bh     = to_f32(X[b * two_d + D + d]);
        const float dy     = to_f32(dY[idx]);
        const float s_a    = silu_scalar(a);
        const float s_a_p  = silu_grad_scalar(a);
        dX[b * two_d + d]     = from_f32<T>(dy * bh * s_a_p);
        dX[b * two_d + D + d] = from_f32<T>(dy * s_a);
    }
}

// ─── GEGLU Kernels ─────────────────────────────────────────────────────────

template <typename T, bool Exact>
__global__ void geglu_forward_kernel(const T* __restrict__ X, T* __restrict__ Y, int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a   = to_f32(X[b * two_d + d]);
        const float g_v = to_f32(X[b * two_d + D + d]);
        const float act = Exact ? gelu_exact_scalar(g_v) : gelu_tanh_scalar(g_v);
        Y[idx] = from_f32<T>(a * act);
    }
}

template <typename T, bool Exact>
__global__ void geglu_backward_kernel(const T* __restrict__ X, const T* __restrict__ dY,
                                      T* __restrict__ dX, int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a      = to_f32(X[b * two_d + d]);
        const float bh     = to_f32(X[b * two_d + D + d]);
        const float dy     = to_f32(dY[idx]);
        const float g      = Exact ? gelu_exact_scalar(bh)      : gelu_tanh_scalar(bh);
        const float gprime = Exact ? gelu_exact_grad_scalar(bh) : gelu_tanh_grad_scalar(bh);
        dX[b * two_d + d]     = from_f32<T>(dy * g);
        dX[b * two_d + D + d] = from_f32<T>(dy * a * gprime);
    }
}

// ─── Bias-Act Kernels ──────────────────────────────────────────────────────

constexpr int ACT_LINEAR = 0;
constexpr int ACT_LRELU  = 1;

__device__ __forceinline__ float apply_act(float t, int act, float alpha) {
    if (act == ACT_LRELU) return t > 0.0f ? t : alpha * t;
    return t;
}
__device__ __forceinline__ float act_grad(float t, int act, float alpha) {
    if (act == ACT_LRELU) return t > 0.0f ? 1.0f : alpha;
    return 1.0f;
}

template <typename T>
__global__ void bias_act_forward_kernel(const T* __restrict__ X, const T* __restrict__ b,
                                        long long total, int C, int HW,
                                        int act, float alpha, float gain, float clamp,
                                        T* __restrict__ Y) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int c = static_cast<int>((i / HW) % C);
        const float bias_v = b ? to_f32(b[c]) : 0.0f;
        float t = to_f32(X[i]) + bias_v;
        float y = apply_act(t, act, alpha) * gain;
        if (clamp >= 0.0f) {
            if (y > clamp) y = clamp;
            else if (y < -clamp) y = -clamp;
        }
        Y[i] = from_f32<T>(y);
    }
}

template <typename T>
__global__ void bias_act_backward_kernel(const T* __restrict__ dY, const T* __restrict__ X,
                                         const T* __restrict__ b, long long total, int C, int HW,
                                         int act, float alpha, float gain, float clamp,
                                         T* __restrict__ dX, float* __restrict__ dB_scratch) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int c = static_cast<int>((i / HW) % C);
        const float bias_v = b ? to_f32(b[c]) : 0.0f;
        const float t = to_f32(X[i]) + bias_v;
        const float y_pre = apply_act(t, act, alpha) * gain;
        const bool clamped = (clamp >= 0.0f) && (y_pre > clamp || y_pre < -clamp);
        float dt = clamped ? 0.0f : (to_f32(dY[i]) * gain * act_grad(t, act, alpha));
        dX[i] = from_f32<T>(dt);
        if (dB_scratch) {
            atomicAdd(&dB_scratch[c], dt);
        }
    }
}

template <typename T>
__global__ void fold_fp32_into_T(const float* __restrict__ src, T* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = from_f32<T>(to_f32(dst[i]) + src[i]);
}

} // namespace

void swiglu_forward(const Tensor& X, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.cols % 2 != 0) throw std::runtime_error("swiglu_forward: X.cols must be even");
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) Y.resize(B, D, X.dtype);
    const int total = B * D;
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        swiglu_forward_kernel<float><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), B, D);
    } else if (X.dtype == Dtype::FP16) {
        swiglu_forward_kernel<__half><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), B, D);
    } else {
        swiglu_forward_kernel<hip_bfloat16><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), B, D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    using ::brotensor::Dtype;
    if (X.cols % 2 != 0) throw std::runtime_error("swiglu_backward: X.cols must be even");
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) dX.resize(B, 2 * D, X.dtype);
    const int total = B * D;
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        swiglu_backward_kernel<float><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D);
    } else if (X.dtype == Dtype::FP16) {
        swiglu_backward_kernel<__half><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), B, D);
    } else {
        swiglu_backward_kernel<hip_bfloat16><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data), B, D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void geglu_forward(const Tensor& X, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.cols % 2 != 0) throw std::runtime_error("geglu_forward: X.cols must be even");
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) Y.resize(B, D, X.dtype);
    const int total = B * D;
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        geglu_forward_kernel<float, false><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), B, D);
    } else if (X.dtype == Dtype::FP16) {
        geglu_forward_kernel<__half, false><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), B, D);
    } else {
        geglu_forward_kernel<hip_bfloat16, false><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), B, D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    using ::brotensor::Dtype;
    if (X.cols % 2 != 0) throw std::runtime_error("geglu_backward: X.cols must be even");
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) dX.resize(B, 2 * D, X.dtype);
    const int total = B * D;
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        geglu_backward_kernel<float, false><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D);
    } else if (X.dtype == Dtype::FP16) {
        geglu_backward_kernel<__half, false><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), B, D);
    } else {
        geglu_backward_kernel<hip_bfloat16, false><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data), B, D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void geglu_exact_forward(const Tensor& X, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.cols % 2 != 0) throw std::runtime_error("geglu_exact_forward: X.cols must be even");
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) Y.resize(B, D, X.dtype);
    const int total = B * D;
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        geglu_forward_kernel<float, true><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), B, D);
    } else if (X.dtype == Dtype::FP16) {
        geglu_forward_kernel<__half, true><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), B, D);
    } else {
        geglu_forward_kernel<hip_bfloat16, true><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), B, D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    using ::brotensor::Dtype;
    if (X.cols % 2 != 0) throw std::runtime_error("geglu_exact_backward: X.cols must be even");
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) dX.resize(B, 2 * D, X.dtype);
    const int total = B * D;
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        geglu_backward_kernel<float, true><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D);
    } else if (X.dtype == Dtype::FP16) {
        geglu_backward_kernel<__half, true><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), B, D);
    } else {
        geglu_backward_kernel<hip_bfloat16, true><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data), B, D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void bias_act_forward(const Tensor& X, const Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, Tensor& Y) {
    using ::brotensor::Dtype;
    const long long total = static_cast<long long>(N) * C * HW;
    if (Y.size() != total || Y.dtype != X.dtype) Y.resize(N, C * HW, X.dtype);
    if (total == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        bias_act_forward_kernel<float><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), b ? static_cast<const float*>(b->data) : nullptr,
            total, C, HW, act, alpha, gain, clamp, static_cast<float*>(Y.data));
    } else if (X.dtype == Dtype::FP16) {
        bias_act_forward_kernel<__half><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), b ? static_cast<const __half*>(b->data) : nullptr,
            total, C, HW, act, alpha, gain, clamp, static_cast<__half*>(Y.data));
    } else {
        bias_act_forward_kernel<hip_bfloat16><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), b ? static_cast<const hip_bfloat16*>(b->data) : nullptr,
            total, C, HW, act, alpha, gain, clamp, static_cast<hip_bfloat16*>(Y.data));
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void bias_act_backward(const Tensor& dY, const Tensor& X, const Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp, Tensor& dX, Tensor* dB) {
    using ::brotensor::Dtype;
    const long long total = static_cast<long long>(N) * C * HW;
    if (dX.size() != total || dX.dtype != X.dtype) dX.resize(N, C * HW, X.dtype);
    if (total == 0) return;

    hipStream_t stream = cur_stream();
    float* dB_scratch = nullptr;
    if (dB) {
        dB_scratch = static_cast<float*>(hip_alloc(C * sizeof(float)));
        BROTENSOR_HIP_CHECK(hipMemsetAsync(dB_scratch, 0, C * sizeof(float), stream));
    }

    if (X.dtype == Dtype::FP32) {
        bias_act_backward_kernel<float><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data), static_cast<const float*>(X.data),
            b ? static_cast<const float*>(b->data) : nullptr,
            total, C, HW, act, alpha, gain, clamp, static_cast<float*>(dX.data), dB_scratch);
    } else if (X.dtype == Dtype::FP16) {
        bias_act_backward_kernel<__half><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(dY.data), static_cast<const __half*>(X.data),
            b ? static_cast<const __half*>(b->data) : nullptr,
            total, C, HW, act, alpha, gain, clamp, static_cast<__half*>(dX.data), dB_scratch);
    } else {
        bias_act_backward_kernel<hip_bfloat16><<<grid_for(total), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(dY.data), static_cast<const hip_bfloat16*>(X.data),
            b ? static_cast<const hip_bfloat16*>(b->data) : nullptr,
            total, C, HW, act, alpha, gain, clamp, static_cast<hip_bfloat16*>(dX.data), dB_scratch);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    if (dB && dB_scratch) {
        const int fold_blocks = (C + EW_BLOCK - 1) / EW_BLOCK;
        if (dB->dtype == Dtype::FP32) {
            fold_fp32_into_T<float><<<fold_blocks, EW_BLOCK, 0, stream>>>(dB_scratch, static_cast<float*>(dB->data), C);
        } else if (dB->dtype == Dtype::FP16) {
            fold_fp32_into_T<__half><<<fold_blocks, EW_BLOCK, 0, stream>>>(dB_scratch, static_cast<__half*>(dB->data), C);
        } else {
            fold_fp32_into_T<hip_bfloat16><<<fold_blocks, EW_BLOCK, 0, stream>>>(dB_scratch, static_cast<hip_bfloat16*>(dB->data), C);
        }
        BROTENSOR_HIP_CHECK(hipGetLastError());
        hip_free(dB_scratch);
    }
}

} // namespace brotensor::detail::hip
