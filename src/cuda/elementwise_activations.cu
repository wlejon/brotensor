#include "elementwise_common.cuh"

namespace brotensor::detail::cuda {

namespace {

__device__ inline float silu_scalar(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(u));
}

__device__ inline float silu_grad_scalar(float v) {
    const float s = 1.0f / (1.0f + __expf(-v));
    return s * (1.0f + v * (1.0f - s));
}

__device__ inline float gelu_tanh_grad_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    const float t = tanhf(u);
    const float dudx = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx;
}

__device__ inline float gelu_exact_scalar(float v) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

__device__ inline float gelu_exact_grad_scalar(float v) {
    constexpr float kInvSqrt2   = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
    const float cdf_term = 0.5f * (1.0f + erff(v * kInvSqrt2));
    const float pdf      = kInvSqrt2Pi * __expf(-0.5f * v * v);
    return cdf_term + v * pdf;
}

__device__ inline float quick_gelu_scalar(float v) {
    return v / (1.0f + __expf(-1.702f * v));
}

__device__ inline float quick_gelu_grad_scalar(float v) {
    const float s = 1.0f / (1.0f + __expf(-1.702f * v));
    return s + v * 1.702f * s * (1.0f - s);
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

__global__ void silu_forward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                         __nv_bfloat16* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2bfloat16(silu_scalar(__bfloat162float(x[i])));
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

__global__ void silu_backward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ dY,
                                          __nv_bfloat16* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __bfloat162float(x[i]);
        const float dyv = __bfloat162float(dY[i]);
        dX[i] = __float2bfloat16(dyv * silu_grad_scalar(xv));
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

__global__ void gelu_forward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                         __nv_bfloat16* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2bfloat16(gelu_tanh_scalar(__bfloat162float(x[i])));
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

__global__ void gelu_backward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ dY,
                                          __nv_bfloat16* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __bfloat162float(x[i]);
        const float dyv = __bfloat162float(dY[i]);
        dX[i] = __float2bfloat16(dyv * gelu_tanh_grad_scalar(xv));
    }
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

__global__ void quick_gelu_forward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                               __nv_bfloat16* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2bfloat16(quick_gelu_scalar(__bfloat162float(x[i])));
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

__global__ void quick_gelu_backward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ dY,
                                                __nv_bfloat16* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __bfloat162float(x[i]);
        const float dyv = __bfloat162float(dY[i]);
        dX[i] = __float2bfloat16(dyv * quick_gelu_grad_scalar(xv));
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

__global__ void gelu_exact_forward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                               __nv_bfloat16* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2bfloat16(gelu_exact_scalar(__bfloat162float(x[i])));
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

__global__ void gelu_exact_backward_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ dY,
                                                __nv_bfloat16* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = __bfloat162float(x[i]);
        const float dyv = __bfloat162float(dY[i]);
        dX[i] = __float2bfloat16(dyv * gelu_exact_grad_scalar(xv));
    }
}

// ── GeGLU / GeGLU-Exact with 32-bit FastDivmodU32 index decomposition ───────

__global__ void geglu_forward_fp32_kernel(const float* __restrict__ X,
                                          float* __restrict__ Y,
                                          int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a = X[b * two_d + d];
        const float gv_raw = X[b * two_d + static_cast<std::uint32_t>(D) + d];
        Y[idx] = a * gelu_tanh_scalar(gv_raw);
    }
}

__global__ void geglu_forward_fp16_kernel(const __half* __restrict__ X,
                                          __half* __restrict__ Y,
                                          int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a = __half2float(X[b * two_d + d]);
        const float gv_raw = __half2float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        Y[idx] = __float2half(a * gelu_tanh_scalar(gv_raw));
    }
}

__global__ void geglu_forward_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                          __nv_bfloat16* __restrict__ Y,
                                          int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a = __bfloat162float(X[b * two_d + d]);
        const float gv_raw = __bfloat162float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        Y[idx] = __float2bfloat16(a * gelu_tanh_scalar(gv_raw));
    }
}

__global__ void geglu_backward_fp32_kernel(const float* __restrict__ X,
                                           const float* __restrict__ dY,
                                           float* __restrict__ dX,
                                           int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a      = X[b * two_d + d];
        const float bh     = X[b * two_d + static_cast<std::uint32_t>(D) + d];
        const float dy     = dY[idx];
        const float g      = gelu_tanh_scalar(bh);
        const float gprime = gelu_tanh_grad_scalar(bh);
        dX[b * two_d + d]                                 = dy * g;
        dX[b * two_d + static_cast<std::uint32_t>(D) + d] = dy * a * gprime;
    }
}

__global__ void geglu_backward_fp16_kernel(const __half* __restrict__ X,
                                           const __half* __restrict__ dY,
                                           __half* __restrict__ dX,
                                           int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a      = __half2float(X[b * two_d + d]);
        const float bh     = __half2float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        const float dy     = __half2float(dY[idx]);
        const float g      = gelu_tanh_scalar(bh);
        const float gprime = gelu_tanh_grad_scalar(bh);
        dX[b * two_d + d]                                 = __float2half(dy * g);
        dX[b * two_d + static_cast<std::uint32_t>(D) + d] = __float2half(dy * a * gprime);
    }
}

__global__ void geglu_backward_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                           const __nv_bfloat16* __restrict__ dY,
                                           __nv_bfloat16* __restrict__ dX,
                                           int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a      = __bfloat162float(X[b * two_d + d]);
        const float bh     = __bfloat162float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        const float dy     = __bfloat162float(dY[idx]);
        const float g      = gelu_tanh_scalar(bh);
        const float gprime = gelu_tanh_grad_scalar(bh);
        dX[b * two_d + d]                                 = __float2bfloat16(dy * g);
        dX[b * two_d + static_cast<std::uint32_t>(D) + d] = __float2bfloat16(dy * a * gprime);
    }
}

__global__ void geglu_exact_forward_fp32_kernel(const float* __restrict__ X,
                                                float* __restrict__ Y,
                                                int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a = X[b * two_d + d];
        const float gv_raw = X[b * two_d + static_cast<std::uint32_t>(D) + d];
        Y[idx] = a * gelu_exact_scalar(gv_raw);
    }
}

__global__ void geglu_exact_forward_fp16_kernel(const __half* __restrict__ X,
                                                __half* __restrict__ Y,
                                                int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a = __half2float(X[b * two_d + d]);
        const float gv_raw = __half2float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        Y[idx] = __float2half(a * gelu_exact_scalar(gv_raw));
    }
}

__global__ void geglu_exact_forward_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                                __nv_bfloat16* __restrict__ Y,
                                                int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a = __bfloat162float(X[b * two_d + d]);
        const float gv_raw = __bfloat162float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        Y[idx] = __float2bfloat16(a * gelu_exact_scalar(gv_raw));
    }
}

__global__ void geglu_exact_backward_fp32_kernel(const float* __restrict__ X,
                                                 const float* __restrict__ dY,
                                                 float* __restrict__ dX,
                                                 int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a      = X[b * two_d + d];
        const float bh     = X[b * two_d + static_cast<std::uint32_t>(D) + d];
        const float dy     = dY[idx];
        const float g      = gelu_exact_scalar(bh);
        const float gprime = gelu_exact_grad_scalar(bh);
        dX[b * two_d + d]                                 = dy * g;
        dX[b * two_d + static_cast<std::uint32_t>(D) + d] = dy * a * gprime;
    }
}

__global__ void geglu_exact_backward_fp16_kernel(const __half* __restrict__ X,
                                                 const __half* __restrict__ dY,
                                                 __half* __restrict__ dX,
                                                 int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a      = __half2float(X[b * two_d + d]);
        const float bh     = __half2float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        const float dy     = __half2float(dY[idx]);
        const float g      = gelu_exact_scalar(bh);
        const float gprime = gelu_exact_grad_scalar(bh);
        dX[b * two_d + d]                                 = __float2half(dy * g);
        dX[b * two_d + static_cast<std::uint32_t>(D) + d] = __float2half(dy * a * gprime);
    }
}

__global__ void geglu_exact_backward_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                                 const __nv_bfloat16* __restrict__ dY,
                                                 __nv_bfloat16* __restrict__ dX,
                                                 int B, int D, FastDivmodU32 div_d) {
    const std::uint32_t total = static_cast<std::uint32_t>(B) * static_cast<std::uint32_t>(D);
    const std::uint32_t two_d = static_cast<std::uint32_t>(2 * D);
    for (std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        std::uint32_t b = 0u, d = 0u;
        div_d.divmod(idx, b, d);
        const float a      = __bfloat162float(X[b * two_d + d]);
        const float bh     = __bfloat162float(X[b * two_d + static_cast<std::uint32_t>(D) + d]);
        const float dy     = __bfloat162float(dY[idx]);
        const float g      = gelu_exact_scalar(bh);
        const float gprime = gelu_exact_grad_scalar(bh);
        dX[b * two_d + d]                                 = __float2bfloat16(dy * g);
        dX[b * two_d + static_cast<std::uint32_t>(D) + d] = __float2bfloat16(dy * a * gprime);
    }
}

} // anonymous namespace

using ::brotensor::Tensor;
using ::brotensor::Dtype;

void silu_forward(const Tensor& x, Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        silu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n);
    } else if (x.dtype == Dtype::BF16) {
        silu_forward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(y.data), n);
    } else {
        silu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<float*>(y.data), n);
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
        gelu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n);
    } else if (x.dtype == Dtype::BF16) {
        gelu_forward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(y.data), n);
    } else {
        gelu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<float*>(y.data), n);
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
        quick_gelu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n);
    } else if (x.dtype == Dtype::BF16) {
        quick_gelu_forward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(y.data), n);
    } else {
        quick_gelu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<float*>(y.data), n);
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
        silu_backward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else if (x.dtype == Dtype::BF16) {
        silu_backward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data), n);
    } else {
        silu_backward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(dY.data),
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
        gelu_backward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else if (x.dtype == Dtype::BF16) {
        gelu_backward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data), n);
    } else {
        gelu_backward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(dY.data),
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
        quick_gelu_backward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else if (x.dtype == Dtype::BF16) {
        quick_gelu_backward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data), n);
    } else {
        quick_gelu_backward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(dY.data),
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
    const FastDivmodU32 div_d(D);
    if (X.dtype == Dtype::FP16) {
        geglu_forward_fp16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), B, D, div_d);
    } else if (X.dtype == Dtype::BF16) {
        geglu_forward_bf16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<__nv_bfloat16*>(Y.data), B, D, div_d);
    } else {
        geglu_forward_fp32_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), B, D, div_d);
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
    const FastDivmodU32 div_d(D);
    if (X.dtype == Dtype::FP16) {
        geglu_backward_fp16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), B, D, div_d);
    } else if (X.dtype == Dtype::BF16) {
        geglu_backward_bf16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data), B, D, div_d);
    } else {
        geglu_backward_fp32_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D, div_d);
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
        gelu_exact_forward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n);
    } else if (x.dtype == Dtype::BF16) {
        gelu_exact_forward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(y.data), n);
    } else {
        gelu_exact_forward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<float*>(y.data), n);
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
        gelu_exact_backward_fp16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n);
    } else if (x.dtype == Dtype::BF16) {
        gelu_exact_backward_bf16_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data), n);
    } else {
        gelu_exact_backward_fp32_kernel<<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(dY.data),
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
    const FastDivmodU32 div_d(D);
    if (X.dtype == Dtype::FP16) {
        geglu_exact_forward_fp16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), B, D, div_d);
    } else if (X.dtype == Dtype::BF16) {
        geglu_exact_forward_bf16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<__nv_bfloat16*>(Y.data), B, D, div_d);
    } else {
        geglu_exact_forward_fp32_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), B, D, div_d);
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
    const FastDivmodU32 div_d(D);
    if (X.dtype == Dtype::FP16) {
        geglu_exact_backward_fp16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), B, D, div_d);
    } else if (X.dtype == Dtype::BF16) {
        geglu_exact_backward_bf16_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data), B, D, div_d);
    } else {
        geglu_exact_backward_fp32_kernel<<<grid_for(total), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D, div_d);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
