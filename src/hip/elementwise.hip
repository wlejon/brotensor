// Native AMD HIP Elementwise arithmetic, activations, and registration for RDNA 3.5 (gfx1151).
//
// Features 128-bit vector loads (float4 for FP32 = 4 elements; 16-byte int4 for FP16/BF16 = 8 elements)
// with head/tail scalar peel for unaligned addresses.

#include "detail/elementwise_internal.h"
#include "detail/hip_check.h"

#include <brotensor/detail/dispatch.h>
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

namespace {

constexpr int EW_BLOCK = 256;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
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

// ─── Scalar activation functions ───────────────────────────────────────────

__device__ __forceinline__ float relu_fn(float v)         { return v > 0.0f ? v : 0.0f; }
__device__ __forceinline__ float relu_grad_fn(float v)    { return v > 0.0f ? 1.0f : 0.0f; }
__device__ __forceinline__ float tanh_grad_fn(float y)    { return 1.0f - y * y; }
__device__ __forceinline__ float sigmoid_fn(float v)      { return 1.0f / (1.0f + __expf(-v)); }
__device__ __forceinline__ float sigmoid_grad_fn(float y) { return y * (1.0f - y); }
__device__ __forceinline__ float silu_fn(float v)         { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float silu_grad_fn(float v) {
    const float s = 1.0f / (1.0f + __expf(-v));
    return s * (1.0f + v * (1.0f - s));
}

__device__ __forceinline__ float gelu_tanh_fn(float x) {
    const float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_tanh_grad_fn(float x) {
    const float x2 = x * x;
    const float inner = kSqrt2OverPi * (x + kGeluCoeff * x2 * x);
    const float tanh_v = tanhf(inner);
    const float sech2 = 1.0f - tanh_v * tanh_v;
    const float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x2);
    return 0.5f * (1.0f + tanh_v) + 0.5f * x * sech2 * d_inner;
}

__device__ __forceinline__ float gelu_exact_fn(float x) {
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float gelu_exact_grad_fn(float x) {
    return 0.5f * (1.0f + erff(x * kInvSqrt2)) + x * kInvSqrt2Pi * __expf(-0.5f * x * x);
}

__device__ __forceinline__ float quick_gelu_fn(float x) {
    return x / (1.0f + __expf(-1.702f * x));
}

__device__ __forceinline__ float quick_gelu_grad_fn(float x) {
    const float s = 1.0f / (1.0f + __expf(-1.702f * x));
    return s + 1.702f * x * s * (1.0f - s);
}

// ─── 128-bit Vectorized In-place Unary / Binary Kernels ─────────────────────

template <typename T, typename Op>
__global__ void unary_inplace_kernel(T* __restrict__ y, long long n, Op op) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

    // Vectorized 128-bit path if aligned to 16 bytes
    if constexpr (sizeof(T) == 4) {
        const long long nvec = n / 4;
        float4* y4 = reinterpret_cast<float4*>(y);
        for (long long i = tid; i < nvec; i += stride) {
            float4 v = y4[i];
            v.x = op(v.x);
            v.y = op(v.y);
            v.z = op(v.z);
            v.w = op(v.w);
            y4[i] = v;
        }
        for (long long i = nvec * 4 + tid; i < n; i += stride) {
            y[i] = from_f32<T>(op(to_f32(y[i])));
        }
    } else {
        const long long nvec = n / 8;
        int4* y4 = reinterpret_cast<int4*>(y);
        for (long long i = tid; i < nvec; i += stride) {
            int4 raw = y4[i];
            T* sub = reinterpret_cast<T*>(&raw);
            #pragma unroll
            for (int k = 0; k < 8; ++k) sub[k] = from_f32<T>(op(to_f32(sub[k])));
            y4[i] = raw;
        }
        for (long long i = nvec * 8 + tid; i < n; i += stride) {
            y[i] = from_f32<T>(op(to_f32(y[i])));
        }
    }
}

template <typename T, typename Op>
__global__ void unary_forward_kernel(const T* __restrict__ x, T* __restrict__ y, long long n, Op op) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

    if constexpr (sizeof(T) == 4) {
        const long long nvec = n / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* y4 = reinterpret_cast<float4*>(y);
        for (long long i = tid; i < nvec; i += stride) {
            float4 v = x4[i];
            v.x = op(v.x); v.y = op(v.y); v.z = op(v.z); v.w = op(v.w);
            y4[i] = v;
        }
        for (long long i = nvec * 4 + tid; i < n; i += stride) {
            y[i] = from_f32<T>(op(to_f32(x[i])));
        }
    } else {
        const long long nvec = n / 8;
        const int4* x4 = reinterpret_cast<const int4*>(x);
        int4* y4 = reinterpret_cast<int4*>(y);
        for (long long i = tid; i < nvec; i += stride) {
            int4 raw = x4[i];
            const T* sub_in = reinterpret_cast<const T*>(&raw);
            int4 out_raw;
            T* sub_out = reinterpret_cast<T*>(&out_raw);
            #pragma unroll
            for (int k = 0; k < 8; ++k) sub_out[k] = from_f32<T>(op(to_f32(sub_in[k])));
            y4[i] = out_raw;
        }
        for (long long i = nvec * 8 + tid; i < n; i += stride) {
            y[i] = from_f32<T>(op(to_f32(x[i])));
        }
    }
}

template <typename T, typename Op>
__global__ void binary_inplace_kernel(T* __restrict__ y, const T* __restrict__ x, long long n, Op op) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

    if constexpr (sizeof(T) == 4) {
        const long long nvec = n / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* y4 = reinterpret_cast<float4*>(y);
        for (long long i = tid; i < nvec; i += stride) {
            float4 vx = x4[i];
            float4 vy = y4[i];
            vy.x = op(vy.x, vx.x); vy.y = op(vy.y, vx.y);
            vy.z = op(vy.z, vx.z); vy.w = op(vy.w, vx.w);
            y4[i] = vy;
        }
        for (long long i = nvec * 4 + tid; i < n; i += stride) {
            y[i] = from_f32<T>(op(to_f32(y[i]), to_f32(x[i])));
        }
    } else {
        const long long nvec = n / 8;
        const int4* x4 = reinterpret_cast<const int4*>(x);
        int4* y4 = reinterpret_cast<int4*>(y);
        for (long long i = tid; i < nvec; i += stride) {
            int4 rx = x4[i];
            int4 ry = y4[i];
            const T* sx = reinterpret_cast<const T*>(&rx);
            T* sy = reinterpret_cast<T*>(&ry);
            #pragma unroll
            for (int k = 0; k < 8; ++k) sy[k] = from_f32<T>(op(to_f32(sy[k]), to_f32(sx[k])));
            y4[i] = ry;
        }
        for (long long i = nvec * 8 + tid; i < n; i += stride) {
            y[i] = from_f32<T>(op(to_f32(y[i]), to_f32(x[i])));
        }
    }
}

template <typename T, typename GradOp>
__global__ void act_backward_kernel(const T* __restrict__ p, const T* __restrict__ dY,
                                    T* __restrict__ dX, long long n, GradOp gop) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

    for (long long i = tid; i < n; i += stride) {
        const float pv  = to_f32(p[i]);
        const float dyv = to_f32(dY[i]);
        dX[i] = from_f32<T>(dyv * gop(pv));
    }
}

// ─── Channel Bias & Row Bias ────────────────────────────────────────────────

template <typename T>
__global__ void add_channel_bias_kernel(T* __restrict__ y, const T* __restrict__ bias,
                                        int C, int L, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = from_f32<T>(to_f32(y[i]) + to_f32(bias[i / L]));
    }
}

template <typename T>
__global__ void add_row_bias_kernel(T* __restrict__ y, const T* __restrict__ bias,
                                    int D, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = from_f32<T>(to_f32(y[i]) + to_f32(bias[i % D]));
    }
}

// ─── Dispatch Helpers ───────────────────────────────────────────────────────

template <typename Op>
void launch_unary_inplace(Tensor& y, Op op) {
    using ::brotensor::Dtype;
    const long long n = y.size();
    if (n == 0) return;
    const int grid = grid_for(n);
    hipStream_t stream = cur_stream();
    if (y.dtype == Dtype::FP32) {
        unary_inplace_kernel<float><<<grid, EW_BLOCK, 0, stream>>>(static_cast<float*>(y.data), n, op);
    } else if (y.dtype == Dtype::FP16) {
        unary_inplace_kernel<__half><<<grid, EW_BLOCK, 0, stream>>>(static_cast<__half*>(y.data), n, op);
    } else if (y.dtype == Dtype::BF16) {
        unary_inplace_kernel<hip_bfloat16><<<grid, EW_BLOCK, 0, stream>>>(static_cast<hip_bfloat16*>(y.data), n, op);
    } else {
        throw std::runtime_error("elementwise op: unsupported dtype");
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

template <typename Op>
void launch_unary_forward(const Tensor& x, Tensor& y, Op op) {
    using ::brotensor::Dtype;
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    const int grid = grid_for(n);
    hipStream_t stream = cur_stream();
    if (x.dtype == Dtype::FP32) {
        unary_forward_kernel<float><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(x.data), static_cast<float*>(y.data), n, op);
    } else if (x.dtype == Dtype::FP16) {
        unary_forward_kernel<__half><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(x.data), static_cast<__half*>(y.data), n, op);
    } else if (x.dtype == Dtype::BF16) {
        unary_forward_kernel<hip_bfloat16><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(x.data), static_cast<hip_bfloat16*>(y.data), n, op);
    } else {
        throw std::runtime_error("elementwise forward: unsupported dtype");
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

template <typename Op>
void launch_binary_inplace(Tensor& y, const Tensor& x, Op op) {
    using ::brotensor::Dtype;
    if (y.dtype != x.dtype) throw std::runtime_error("binary elementwise: dtype mismatch");
    if (y.size() != x.size()) throw std::runtime_error("binary elementwise: size mismatch");
    const long long n = y.size();
    if (n == 0) return;
    const int grid = grid_for(n);
    hipStream_t stream = cur_stream();
    if (y.dtype == Dtype::FP32) {
        binary_inplace_kernel<float><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<float*>(y.data), static_cast<const float*>(x.data), n, op);
    } else if (y.dtype == Dtype::FP16) {
        binary_inplace_kernel<__half><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<__half*>(y.data), static_cast<const __half*>(x.data), n, op);
    } else if (y.dtype == Dtype::BF16) {
        binary_inplace_kernel<hip_bfloat16><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<hip_bfloat16*>(y.data), static_cast<const hip_bfloat16*>(x.data), n, op);
    } else {
        throw std::runtime_error("binary elementwise: unsupported dtype");
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

template <typename GradOp>
void launch_act_backward(const Tensor& p, const Tensor& dY, Tensor& dX, GradOp gop) {
    using ::brotensor::Dtype;
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    const long long n = dY.size();
    if (n == 0) return;
    const int grid = grid_for(n);
    hipStream_t stream = cur_stream();
    if (dY.dtype == Dtype::FP32) {
        act_backward_kernel<float><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(p.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), n, gop);
    } else if (dY.dtype == Dtype::FP16) {
        act_backward_kernel<__half><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(p.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), n, gop);
    } else if (dY.dtype == Dtype::BF16) {
        act_backward_kernel<hip_bfloat16><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(p.data), static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data), n, gop);
    } else {
        throw std::runtime_error("activation backward: unsupported dtype");
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

template <typename Ty, typename Tx>
__global__ void binary_inplace_mixed_kernel(Ty* __restrict__ y, const Tx* __restrict__ x, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = from_f32<Ty>(to_f32(y[i]) + to_f32(x[i]));
    }
}

} // namespace

// ─── Public Op Wrappers ────────────────────────────────────────────────────

void add_inplace(Tensor& y, const Tensor& x) {
    using ::brotensor::Dtype;
    if (y.dtype == Dtype::FP32 && x.dtype == Dtype::FP16) {
        if (y.size() != x.size()) throw std::runtime_error("add_inplace: size mismatch");
        const long long n = y.size();
        if (n == 0) return;
        const int grid = grid_for(n);
        hipStream_t stream = cur_stream();
        binary_inplace_mixed_kernel<float, __half><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<float*>(y.data), static_cast<const __half*>(x.data), n);
        BROTENSOR_HIP_CHECK(hipGetLastError());
        return;
    }
    launch_binary_inplace(y, x, [] __device__ (float a, float b) { return a + b; });
}
void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD) {
    add_inplace(Y_BD, X_BD);
}
void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) {
    launch_binary_inplace(y, x, [a, b] __device__ (float yv, float xv) { return a * xv + b * yv; });
}
void add_scalar_inplace(Tensor& y, float s) {
    launch_unary_inplace(y, [s] __device__ (float v) { return v + s; });
}
void scale_inplace(Tensor& y, float s) {
    launch_unary_inplace(y, [s] __device__ (float v) { return v * s; });
}
void mul_inplace(Tensor& y, const Tensor& x) {
    launch_binary_inplace(y, x, [] __device__ (float a, float b) { return a * b; });
}
void clamp(Tensor& y, float lo, float hi) {
    launch_unary_inplace(y, [lo, hi] __device__ (float v) { return fminf(fmaxf(v, lo), hi); });
}

void relu_forward(const Tensor& x, Tensor& y) {
    launch_unary_forward(x, y, [] __device__ (float v) { return relu_fn(v); });
}
void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    relu_forward(X_BD, Y_BD);
}
void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_act_backward(x, dY, dX, [] __device__ (float xv) { return relu_grad_fn(xv); });
}
void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD, Tensor& dX_BD) {
    relu_backward(X_BD, dY_BD, dX_BD);
}

void tanh_forward(const Tensor& x, Tensor& y) {
    launch_unary_forward(x, y, [] __device__ (float v) { return tanhf(v); });
}
void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    tanh_forward(X_BD, Y_BD);
}
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_act_backward(y, dY, dX, [] __device__ (float yv) { return tanh_grad_fn(yv); });
}
void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD, Tensor& dX_BD) {
    tanh_backward(Y_BD, dY_BD, dX_BD);
}

void sigmoid_forward(const Tensor& x, Tensor& y) {
    launch_unary_forward(x, y, [] __device__ (float v) { return sigmoid_fn(v); });
}
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_act_backward(y, dY, dX, [] __device__ (float yv) { return sigmoid_grad_fn(yv); });
}

void silu_forward(const Tensor& x, Tensor& y) {
    launch_unary_forward(x, y, [] __device__ (float v) { return silu_fn(v); });
}
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_act_backward(x, dY, dX, [] __device__ (float xv) { return silu_grad_fn(xv); });
}

void gelu_forward(const Tensor& x, Tensor& y) {
    launch_unary_forward(x, y, [] __device__ (float v) { return gelu_tanh_fn(v); });
}
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_act_backward(x, dY, dX, [] __device__ (float xv) { return gelu_tanh_grad_fn(xv); });
}

void gelu_exact_forward(const Tensor& x, Tensor& y) {
    launch_unary_forward(x, y, [] __device__ (float v) { return gelu_exact_fn(v); });
}
void gelu_exact_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_act_backward(x, dY, dX, [] __device__ (float xv) { return gelu_exact_grad_fn(xv); });
}

void quick_gelu_forward(const Tensor& x, Tensor& y) {
    launch_unary_forward(x, y, [] __device__ (float v) { return quick_gelu_fn(v); });
}
void quick_gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_act_backward(x, dY, dX, [] __device__ (float xv) { return quick_gelu_grad_fn(xv); });
}

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    const int n = C * L;
    if (n == 0) return;
    if (y.dtype != bias.dtype) throw std::runtime_error("add_channel_bias_inplace: dtype mismatch");
    hipStream_t stream = cur_stream();
    using ::brotensor::Dtype;
    if (y.dtype == Dtype::FP32) {
        add_channel_bias_kernel<float><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<float*>(y.data), static_cast<const float*>(bias.data), C, L, n);
    } else if (y.dtype == Dtype::FP16) {
        add_channel_bias_kernel<__half><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<__half*>(y.data), static_cast<const __half*>(bias.data), C, L, n);
    } else {
        add_channel_bias_kernel<hip_bfloat16><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<hip_bfloat16*>(y.data), static_cast<const hip_bfloat16*>(bias.data), C, L, n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void add_row_bias_inplace(Tensor& Y, const Tensor& bias) {
    const int D = bias.size();
    const int n = Y.size();
    if (n == 0 || D == 0) return;
    if (Y.dtype != bias.dtype) throw std::runtime_error("add_row_bias_inplace: dtype mismatch");
    hipStream_t stream = cur_stream();
    using ::brotensor::Dtype;
    if (Y.dtype == Dtype::FP32) {
        add_row_bias_kernel<float><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<float*>(Y.data), static_cast<const float*>(bias.data), D, n);
    } else if (Y.dtype == Dtype::FP16) {
        add_row_bias_kernel<__half><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<__half*>(Y.data), static_cast<const __half*>(bias.data), D, n);
    } else {
        add_row_bias_kernel<hip_bfloat16><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<hip_bfloat16*>(Y.data), static_cast<const hip_bfloat16*>(bias.data), D, n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

template <typename T>
__global__ void threshold_u8_kernel(const T* __restrict__ X, float t, uint8_t* __restrict__ Y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        Y[i] = (to_f32(X[i]) > t) ? 1u : 0u;
    }
}

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    using ::brotensor::Dtype;
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::INT8) {
        Y.resize(X.rows, X.cols, Dtype::INT8);
    }
    const int n = X.size();
    if (n == 0) return;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        threshold_u8_kernel<float><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), t, static_cast<uint8_t*>(Y.data), n);
    } else if (X.dtype == Dtype::FP16) {
        threshold_u8_kernel<__half><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), t, static_cast<uint8_t*>(Y.data), n);
    } else {
        threshold_u8_kernel<hip_bfloat16><<<grid_for(n), EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), t, static_cast<uint8_t*>(Y.data), n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

template <typename InT, typename OutT>
__global__ void cast_kernel(const InT* __restrict__ src, OutT* __restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = from_f32<OutT>(to_f32(src[i]));
    }
}

void cast(const Tensor& src, Tensor& dst, ::brotensor::Dtype out_dtype) {
    using ::brotensor::Dtype;
    if (dst.rows != src.rows || dst.cols != src.cols || dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const int n = src.size();
    if (n == 0 || src.dtype == out_dtype) {
        if (src.dtype == out_dtype && src.data != dst.data) {
            BROTENSOR_HIP_CHECK(hipMemcpyAsync(dst.data, src.data, src.bytes(), hipMemcpyDeviceToDevice, cur_stream()));
        }
        return;
    }
    hipStream_t stream = cur_stream();
    const int grid = grid_for(n);

    if (src.dtype == Dtype::FP32 && out_dtype == Dtype::FP16) {
        cast_kernel<float, __half><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(src.data), static_cast<__half*>(dst.data), n);
    } else if (src.dtype == Dtype::FP16 && out_dtype == Dtype::FP32) {
        cast_kernel<__half, float><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(src.data), static_cast<float*>(dst.data), n);
    } else if (src.dtype == Dtype::FP32 && out_dtype == Dtype::BF16) {
        cast_kernel<float, hip_bfloat16><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const float*>(src.data), static_cast<hip_bfloat16*>(dst.data), n);
    } else if (src.dtype == Dtype::BF16 && out_dtype == Dtype::FP32) {
        cast_kernel<hip_bfloat16, float><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(src.data), static_cast<float*>(dst.data), n);
    } else if (src.dtype == Dtype::FP16 && out_dtype == Dtype::BF16) {
        cast_kernel<__half, hip_bfloat16><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const __half*>(src.data), static_cast<hip_bfloat16*>(dst.data), n);
    } else if (src.dtype == Dtype::BF16 && out_dtype == Dtype::FP16) {
        cast_kernel<hip_bfloat16, __half><<<grid, EW_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(src.data), static_cast<__half*>(dst.data), n);
    } else {
        throw std::runtime_error("hip cast: unsupported dtype pair");
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_elementwise(::brotensor::detail::OpsVTable& v) {
    v.add_inplace              = &add_inplace;
    v.add_inplace_batched      = &add_inplace_batched;
    v.axpby_inplace            = &axpby_inplace;
    v.add_scalar_inplace       = &add_scalar_inplace;
    v.scale_inplace            = &scale_inplace;
    v.mul_inplace              = &mul_inplace;
    v.clamp                    = &clamp;
    v.threshold_u8             = &threshold_u8;
    v.cast                     = &cast;
    v.add_channel_bias_inplace = &add_channel_bias_inplace;
    v.add_row_bias_inplace     = &add_row_bias_inplace;

    v.relu_forward             = &relu_forward;
    v.relu_backward            = &relu_backward;
    v.relu_forward_batched     = &relu_forward_batched;
    v.relu_backward_batched    = &relu_backward_batched;

    v.tanh_forward             = &tanh_forward;
    v.tanh_backward            = &tanh_backward;
    v.tanh_forward_batched     = &tanh_forward_batched;
    v.tanh_backward_batched    = &tanh_backward_batched;

    v.sigmoid_forward          = &sigmoid_forward;
    v.sigmoid_backward         = &sigmoid_backward;

    v.silu_forward             = &silu_forward;
    v.silu_backward            = &silu_backward;

    v.gelu_forward             = &gelu_forward;
    v.gelu_backward            = &gelu_backward;
    v.gelu_exact_forward       = &gelu_exact_forward;
    v.gelu_exact_backward      = &gelu_exact_backward;

    v.quick_gelu_forward       = &quick_gelu_forward;
    v.quick_gelu_backward      = &quick_gelu_backward;

    v.swiglu_forward           = &swiglu_forward;
    v.swiglu_backward          = &swiglu_backward;
    v.geglu_forward            = &geglu_forward;
    v.geglu_backward           = &geglu_backward;
    v.geglu_exact_forward      = &geglu_exact_forward;
    v.geglu_exact_backward     = &geglu_exact_backward;

    v.sum_rows                 = &sum_rows;
    v.sum_cols                 = &sum_cols;
    v.argmax_rows              = &argmax_rows;
    v.rows_count_above         = &rows_count_above;
    v.masked_mean_pool_forward = &masked_mean_pool_forward;
    v.masked_mean_pool_backward = &masked_mean_pool_backward;

    v.bias_act_forward         = &bias_act_forward;
    v.bias_act_backward        = &bias_act_backward;
}

} // namespace brotensor::detail::hip
