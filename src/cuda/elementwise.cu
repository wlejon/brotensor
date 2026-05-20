#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor::detail::cuda {

namespace {

constexpr int EW_BLOCK = 256;

__device__ inline float silu_scalar(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ inline float gelu_tanh_scalar(float v) {
    // GELU with tanh approximation (matches PyTorch's approximate="tanh").
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(u));
}

__device__ inline float silu_grad_scalar(float v) {
    // d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
    const float s = 1.0f / (1.0f + __expf(-v));
    return s * (1.0f + v * (1.0f - s));
}

__device__ inline float gelu_tanh_grad_scalar(float v) {
    // Derivative of gelu_tanh_scalar w.r.t. v.
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    const float t = tanhf(u);
    const float dudx = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx;
}

__device__ inline float gelu_exact_scalar(float v) {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))). Matches PyTorch's
    // default `torch.nn.functional.gelu` (approximate="none").
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

__device__ inline float gelu_exact_grad_scalar(float v) {
    // d/dx [0.5*x*(1+erf(x/√2))] = 0.5*(1+erf(x/√2)) + x*φ(x)
    // where φ(x) = (1/√(2π)) * exp(-x²/2).
    constexpr float kInvSqrt2  = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f; // 1/sqrt(2π)
    const float cdf_term = 0.5f * (1.0f + erff(v * kInvSqrt2));
    const float pdf      = kInvSqrt2Pi * __expf(-0.5f * v * v);
    return cdf_term + v * pdf;
}

__device__ inline float quick_gelu_grad_scalar(float v) {
    // d/dx [x * sigmoid(1.702*x)] = s + x * 1.702 * s * (1 - s).
    const float s = 1.0f / (1.0f + __expf(-1.702f * v));
    return s + v * 1.702f * s * (1.0f - s);
}

__global__ void relu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* __restrict__ x,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
    }
}

__global__ void tanh_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = tanhf(x[i]);
    }
}

__global__ void tanh_backward_kernel(const float* __restrict__ y,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv = y[i];
        dX[i] = dY[i] * (1.0f - yv * yv);
    }
}

__global__ void sigmoid_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

__global__ void sigmoid_backward_kernel(const float* __restrict__ y,
                                        const float* __restrict__ dY,
                                        float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv = y[i];
        dX[i] = dY[i] * yv * (1.0f - yv);
    }
}

__global__ void add_inplace_kernel(float* __restrict__ y,
                                   const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += x[i];
    }
}

__global__ void add_scalar_inplace_kernel(float* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += s;
    }
}

__global__ void scale_inplace_kernel(float* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] *= s;
    }
}

__global__ void build_slot_mask_kernel(const float* __restrict__ x,
                                       float* __restrict__ mask,
                                       int offset, int K, int stride) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const float v = x[offset + k * stride];
    mask[k] = v > 0.5f ? 1.0f : 0.0f;
}

__global__ void silu_forward_fp32_kernel(const float* __restrict__ x,
                                         float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = silu_scalar(x[i]);
    }
}

__global__ void silu_forward_fp16_kernel(const __half* __restrict__ x,
                                         __half* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(silu_scalar(__half2float(x[i])));
    }
}

__global__ void gelu_forward_fp32_kernel(const float* __restrict__ x,
                                         float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = gelu_tanh_scalar(x[i]);
    }
}

__global__ void gelu_forward_fp16_kernel(const __half* __restrict__ x,
                                         __half* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(gelu_tanh_scalar(__half2float(x[i])));
    }
}

__global__ void add_inplace_fp16_kernel(__half* __restrict__ y,
                                        const __half* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float a = __half2float(y[i]);
        const float b = __half2float(x[i]);
        y[i] = __float2half(a + b);
    }
}

__global__ void scale_inplace_fp16_kernel(__half* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(__half2float(y[i]) * s);
    }
}

__global__ void add_scalar_inplace_fp16_kernel(__half* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(__half2float(y[i]) + s);
    }
}

__global__ void clamp_fp32_kernel(float* __restrict__ y, float lo, float hi, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = y[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        y[i] = v;
    }
}

__global__ void clamp_fp16_kernel(__half* __restrict__ y, float lo, float hi, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = __half2float(y[i]);
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        y[i] = __float2half(v);
    }
}

__device__ inline float quick_gelu_scalar(float v) {
    // OpenAI CLIP's QuickGELU: x * sigmoid(1.702 * x).
    return v / (1.0f + __expf(-1.702f * v));
}

__global__ void quick_gelu_forward_fp32_kernel(const float* __restrict__ x,
                                               float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = quick_gelu_scalar(x[i]);
    }
}

__global__ void quick_gelu_forward_fp16_kernel(const __half* __restrict__ x,
                                               __half* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(quick_gelu_scalar(__half2float(x[i])));
    }
}

__global__ void mul_inplace_fp32_kernel(float* __restrict__ y,
                                        const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] *= x[i];
    }
}

__global__ void mul_inplace_fp16_kernel(__half* __restrict__ y,
                                        const __half* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float a = __half2float(y[i]);
        const float b = __half2float(x[i]);
        y[i] = __float2half(a * b);
    }
}

// Y(B, D) = X_a(B, D) * gelu(X_b(B, D)) where X is (B, 2D) — A is the first
// half along the last dim, B is the second half.
__global__ void geglu_forward_fp16_kernel(const __half* __restrict__ X,
                                          __half* __restrict__ Y,
                                          int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a = __half2float(X[b * two_d + d]);
        const float gv_raw = __half2float(X[b * two_d + D + d]);
        Y[idx] = __float2half(a * gelu_tanh_scalar(gv_raw));
    }
}

__global__ void silu_backward_fp32_kernel(const float* __restrict__ x,
                                          const float* __restrict__ dY,
                                          float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = dY[i] * silu_grad_scalar(x[i]);
    }
}

__global__ void silu_backward_fp16_kernel(const __half* __restrict__ x,
                                          const __half* __restrict__ dY,
                                          __half* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __half2float(x[i]);
        const float dyv = __half2float(dY[i]);
        dX[i] = __float2half(dyv * silu_grad_scalar(xv));
    }
}

__global__ void gelu_backward_fp32_kernel(const float* __restrict__ x,
                                          const float* __restrict__ dY,
                                          float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = dY[i] * gelu_tanh_grad_scalar(x[i]);
    }
}

__global__ void gelu_backward_fp16_kernel(const __half* __restrict__ x,
                                          const __half* __restrict__ dY,
                                          __half* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __half2float(x[i]);
        const float dyv = __half2float(dY[i]);
        dX[i] = __float2half(dyv * gelu_tanh_grad_scalar(xv));
    }
}

__global__ void quick_gelu_backward_fp32_kernel(const float* __restrict__ x,
                                                const float* __restrict__ dY,
                                                float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = dY[i] * quick_gelu_grad_scalar(x[i]);
    }
}

__global__ void quick_gelu_backward_fp16_kernel(const __half* __restrict__ x,
                                                const __half* __restrict__ dY,
                                                __half* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __half2float(x[i]);
        const float dyv = __half2float(dY[i]);
        dX[i] = __float2half(dyv * quick_gelu_grad_scalar(xv));
    }
}

// Y(B, D) = X_a(B, D) * gelu(X_b(B, D)), FP32 variant.
__global__ void geglu_forward_fp32_kernel(const float* __restrict__ X,
                                          float* __restrict__ Y,
                                          int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a = X[b * two_d + d];
        const float gv_raw = X[b * two_d + D + d];
        Y[idx] = a * gelu_tanh_scalar(gv_raw);
    }
}

// dX(B, 2D): dA = dY * g; dB_half = dY * A * gelu'(B_half).
__global__ void geglu_backward_fp32_kernel(const float* __restrict__ X,
                                           const float* __restrict__ dY,
                                           float* __restrict__ dX,
                                           int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a       = X[b * two_d + d];
        const float bh      = X[b * two_d + D + d];
        const float dy      = dY[idx];
        const float g       = gelu_tanh_scalar(bh);
        const float gprime  = gelu_tanh_grad_scalar(bh);
        dX[b * two_d + d]     = dy * g;
        dX[b * two_d + D + d] = dy * a * gprime;
    }
}

__global__ void geglu_backward_fp16_kernel(const __half* __restrict__ X,
                                           const __half* __restrict__ dY,
                                           __half* __restrict__ dX,
                                           int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a       = __half2float(X[b * two_d + d]);
        const float bh      = __half2float(X[b * two_d + D + d]);
        const float dy      = __half2float(dY[idx]);
        const float g       = gelu_tanh_scalar(bh);
        const float gprime  = gelu_tanh_grad_scalar(bh);
        dX[b * two_d + d]     = __float2half(dy * g);
        dX[b * two_d + D + d] = __float2half(dy * a * gprime);
    }
}

__global__ void gelu_exact_forward_fp32_kernel(const float* __restrict__ x,
                                               float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = gelu_exact_scalar(x[i]);
    }
}

__global__ void gelu_exact_forward_fp16_kernel(const __half* __restrict__ x,
                                               __half* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(gelu_exact_scalar(__half2float(x[i])));
    }
}

__global__ void gelu_exact_backward_fp32_kernel(const float* __restrict__ x,
                                                const float* __restrict__ dY,
                                                float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = dY[i] * gelu_exact_grad_scalar(x[i]);
    }
}

__global__ void gelu_exact_backward_fp16_kernel(const __half* __restrict__ x,
                                                const __half* __restrict__ dY,
                                                __half* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __half2float(x[i]);
        const float dyv = __half2float(dY[i]);
        dX[i] = __float2half(dyv * gelu_exact_grad_scalar(xv));
    }
}

__global__ void geglu_exact_forward_fp32_kernel(const float* __restrict__ X,
                                                float* __restrict__ Y,
                                                int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a = X[b * two_d + d];
        const float gv_raw = X[b * two_d + D + d];
        Y[idx] = a * gelu_exact_scalar(gv_raw);
    }
}

__global__ void geglu_exact_forward_fp16_kernel(const __half* __restrict__ X,
                                                __half* __restrict__ Y,
                                                int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a = __half2float(X[b * two_d + d]);
        const float gv_raw = __half2float(X[b * two_d + D + d]);
        Y[idx] = __float2half(a * gelu_exact_scalar(gv_raw));
    }
}

__global__ void geglu_exact_backward_fp32_kernel(const float* __restrict__ X,
                                                 const float* __restrict__ dY,
                                                 float* __restrict__ dX,
                                                 int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a       = X[b * two_d + d];
        const float bh      = X[b * two_d + D + d];
        const float dy      = dY[idx];
        const float g       = gelu_exact_scalar(bh);
        const float gprime  = gelu_exact_grad_scalar(bh);
        dX[b * two_d + d]     = dy * g;
        dX[b * two_d + D + d] = dy * a * gprime;
    }
}

__global__ void geglu_exact_backward_fp16_kernel(const __half* __restrict__ X,
                                                 const __half* __restrict__ dY,
                                                 __half* __restrict__ dX,
                                                 int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a       = __half2float(X[b * two_d + d]);
        const float bh      = __half2float(X[b * two_d + D + d]);
        const float dy      = __half2float(dY[idx]);
        const float g       = gelu_exact_scalar(bh);
        const float gprime  = gelu_exact_grad_scalar(bh);
        dX[b * two_d + d]     = __float2half(dy * g);
        dX[b * two_d + D + d] = __float2half(dy * a * gprime);
    }
}

__global__ void causal_mask_row_kernel(float* __restrict__ mask, int L, int q) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= L) return;
    mask[k] = (k <= q) ? 1.0f : 0.0f;
}

inline int grid_for(int n) {
    int blocks = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

} // anonymous namespace

using ::brotensor::Tensor;
using ::brotensor::Dtype;

void relu_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    relu_forward_kernel<<<grid_for(n), EW_BLOCK>>>(
        static_cast<const float*>(x.data),
        static_cast<float*>(y.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols) dX.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    relu_backward_kernel<<<grid_for(n), EW_BLOCK>>>(
        static_cast<const float*>(x.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dX.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    tanh_forward_kernel<<<grid_for(n), EW_BLOCK>>>(
        static_cast<const float*>(x.data),
        static_cast<float*>(y.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols) dX.resize(y.rows, y.cols);
    const int n = y.size();
    if (n == 0) return;
    tanh_backward_kernel<<<grid_for(n), EW_BLOCK>>>(
        static_cast<const float*>(y.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dX.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    sigmoid_forward_kernel<<<grid_for(n), EW_BLOCK>>>(
        static_cast<const float*>(x.data),
        static_cast<float*>(y.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols) dX.resize(y.rows, y.cols);
    const int n = y.size();
    if (n == 0) return;
    sigmoid_backward_kernel<<<grid_for(n), EW_BLOCK>>>(
        static_cast<const float*>(y.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dX.data), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_inplace(Tensor& y, const Tensor& x) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        if (x.dtype != Dtype::FP16) {
            throw std::runtime_error("add_inplace: dtype mismatch");
        }
        add_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<__half*>(y.data),
            static_cast<const __half*>(x.data), n);
    } else {
        add_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<float*>(y.data),
            static_cast<const float*>(x.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_scalar_inplace(Tensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        add_scalar_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<__half*>(y.data), s, n);
    } else {
        add_scalar_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<float*>(y.data), s, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void clamp(Tensor& y, float lo, float hi) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        clamp_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<__half*>(y.data), lo, hi, n);
    } else {
        clamp_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<float*>(y.data), lo, hi, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void scale_inplace(Tensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        scale_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<__half*>(y.data), s, n);
    } else {
        scale_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<float*>(y.data), s, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void mul_inplace(Tensor& y, const Tensor& x) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("mul_inplace: shape/dtype mismatch");
    }
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        mul_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<__half*>(y.data),
            static_cast<const __half*>(x.data), n);
    } else {
        mul_inplace_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<float*>(y.data),
            static_cast<const float*>(x.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void silu_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        silu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<__half*>(y.data), n);
    } else {
        silu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<float*>(y.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void gelu_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        gelu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<__half*>(y.data), n);
    } else {
        gelu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<float*>(y.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void quick_gelu_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        quick_gelu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<__half*>(y.data), n);
    } else {
        quick_gelu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<float*>(y.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        silu_backward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else {
        silu_backward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        gelu_backward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else {
        gelu_backward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void quick_gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        quick_gelu_backward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else {
        quick_gelu_backward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void geglu_forward(const Tensor& X, Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_forward: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        geglu_forward_fp16_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            B, D);
    } else {
        geglu_forward_fp32_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_backward: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        geglu_backward_fp16_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            B, D);
    } else {
        geglu_backward_fp32_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void gelu_exact_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        gelu_exact_forward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<__half*>(y.data), n);
    } else {
        gelu_exact_forward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<float*>(y.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void gelu_exact_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        gelu_exact_backward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else {
        gelu_exact_backward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(
            static_cast<const float*>(x.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void geglu_exact_forward(const Tensor& X, Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_exact_forward: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        geglu_exact_forward_fp16_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            B, D);
    } else {
        geglu_exact_forward_fp32_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_exact_backward: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        geglu_exact_backward_fp16_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            B, D);
    } else {
        geglu_exact_backward_fp32_kernel<<<grid_for(total), EW_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
    if (mask.rows != L || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(L, 1, Dtype::FP32);
    }
    if (L <= 0) return;
    const int blocks = (L + EW_BLOCK - 1) / EW_BLOCK;
    causal_mask_row_kernel<<<blocks, EW_BLOCK>>>(
        static_cast<float*>(mask.data), L, q);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride,
                     Tensor& mask) {
    if (mask.rows != K || mask.cols != 1) mask.resize(K, 1);
    if (K <= 0) return;
    const int blocks = (K + EW_BLOCK - 1) / EW_BLOCK;
    build_slot_mask_kernel<<<blocks, EW_BLOCK>>>(
        static_cast<const float*>(x.data),
        static_cast<float*>(mask.data),
        offset, K, stride);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── Vtable contribution ───────────────────────────────────────────────────
//
// Forward decls for ops registered here whose implementations live in sibling
// .cu files in this cluster (rms_norm.cu, swiglu.cu).
void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y);
void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma);
void swiglu_forward(const Tensor& X, Tensor& Y);
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX);

void fill_cuda_vtable_elementwise(::brotensor::detail::OpsVTable& v) {
    v.relu_forward            = &relu_forward;
    v.relu_backward           = &relu_backward;
    v.tanh_forward            = &tanh_forward;
    v.tanh_backward           = &tanh_backward;
    v.sigmoid_forward         = &sigmoid_forward;
    v.sigmoid_backward        = &sigmoid_backward;
    v.add_inplace             = &add_inplace;
    v.add_scalar_inplace      = &add_scalar_inplace;
    v.scale_inplace           = &scale_inplace;
    v.mul_inplace             = &mul_inplace;
    v.clamp                   = &clamp;
    v.silu_forward            = &silu_forward;
    v.silu_backward           = &silu_backward;
    v.gelu_forward            = &gelu_forward;
    v.gelu_backward           = &gelu_backward;
    v.gelu_exact_forward      = &gelu_exact_forward;
    v.gelu_exact_backward     = &gelu_exact_backward;
    v.quick_gelu_forward      = &quick_gelu_forward;
    v.quick_gelu_backward     = &quick_gelu_backward;
    v.geglu_forward           = &geglu_forward;
    v.geglu_backward          = &geglu_backward;
    v.geglu_exact_forward     = &geglu_exact_forward;
    v.geglu_exact_backward    = &geglu_exact_backward;
    v.build_slot_mask         = &build_slot_mask;
    v.build_causal_mask_row   = &build_causal_mask_row;
    v.rms_norm_forward        = &rms_norm_forward;
    v.rms_norm_backward       = &rms_norm_backward;
    v.swiglu_forward          = &swiglu_forward;
    v.swiglu_backward         = &swiglu_backward;
}

} // namespace brotensor::detail::cuda
