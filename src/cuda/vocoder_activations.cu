// ─── CUDA vocoder / codec activations (brosoundml CHUNK 4, family C) ────────
//
// CUDA port of src/cpu/vocoder_activations.cpp — FP32-only, mirroring the CPU
// contract verbatim:
//   snake_forward / snake_backward         — BigVGAN / DAC snake + snakebeta
//   elu_forward / elu_backward             — EnCodec ELU
//   leaky_relu_forward / leaky_relu_backward — HiFi-GAN leaky ReLU
//
// Layout / accumulation (identical to the CPU backend):
//   snake is per-channel over an NCL tensor stored as (N, C*L); element
//   (n, c, l) sits at flat index (n*C + c)*L + l. alpha / beta carry one
//   scalar per channel, broadcast across the (n, l) plane.
//   snake_forward            — Y  OVERWRITTEN.
//   snake_backward           — dX OVERWRITTEN; dAlpha / dBeta ACCUMULATE (+=).
//   elu / leaky_relu forward — y  OVERWRITTEN.
//   elu / leaky_relu backward— dX OVERWRITTEN (no learnable params).
//
// These audio ops are FP32-only on every backend — a non-FP32 operand throws.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int VA_BLOCK = 256;

inline int va_grid(long long n) {
    long long blocks = (n + VA_BLOCK - 1) / VA_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

inline void require_fp32(const char* op, const ::brotensor::Tensor& t,
                         const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32 (audio ops are FP32-only)");
    }
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

// Sign-preserving floor on a reciprocal denominator — keeps |d| >= 1e-9 so a
// near-zero alpha/beta degrades gracefully instead of producing NaN/Inf.
// Matches the CPU guard_denom exactly.
__device__ inline float guard_denom(float d) {
    constexpr float kMin = 1e-9f;
    if (d >= 0.0f) return d < kMin ? kMin : d;
    return d > -kMin ? -kMin : d;
}

// ─── snake ──────────────────────────────────────────────────────────────────

__global__ void snake_forward_kernel(const float* __restrict__ X,
                                     const float* __restrict__ alpha,
                                     const float* __restrict__ beta,
                                     int C, int L, long long total,
                                     float* __restrict__ Y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>((i / L) % C);
        const float a = alpha[c];
        const float r = 1.0f / guard_denom(beta ? beta[c] : a);
        const float x = X[i];
        const float s = sinf(a * x);
        Y[i] = x + r * s * s;                          // overwrite
    }
}

__global__ void snake_backward_kernel(const float* __restrict__ X,
                                      const float* __restrict__ alpha,
                                      const float* __restrict__ beta,
                                      const float* __restrict__ dY,
                                      int C, int L, long long total,
                                      float* __restrict__ dX,
                                      float* __restrict__ dAlpha,
                                      float* __restrict__ dBeta) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < total; i += (long long)blockDim.x * gridDim.x) {
        const int c = static_cast<int>((i / L) % C);
        const float a = alpha[c];
        const float r = 1.0f / guard_denom(beta ? beta[c] : a);
        const float x  = X[i];
        const float dy = dY[i];
        const float s  = sinf(a * x);
        const float co = cosf(a * x);
        const float sc = s * co;                       // sin*cos

        // dy/dx = 1 + 2*a*r*s*c
        dX[i] = dy * (1.0f + 2.0f * a * r * sc);        // overwrite

        // dy/dalpha frequency term = 2*r*x*s*c
        float dalpha = dy * (2.0f * r * x * sc);
        if (beta) {
            // snakebeta: dy/dbeta = -r^2 * s^2.
            atomicAdd(&dBeta[c], dy * (-r * r * s * s));
        } else {
            // plain snake: denom == alpha — fold the -r^2*s^2 term into dAlpha.
            dalpha += dy * (-r * r * s * s);
        }
        atomicAdd(&dAlpha[c], dalpha);
    }
}

// ─── elu ────────────────────────────────────────────────────────────────────

__global__ void elu_forward_kernel(const float* __restrict__ x, float alpha,
                                   long long n, float* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
    }
}

__global__ void elu_backward_kernel(const float* __restrict__ x,
                                    const float* __restrict__ dY, float alpha,
                                    long long n, float* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float v = x[i];
        const float g = v > 0.0f ? 1.0f : alpha * expf(v);
        dX[i] = dY[i] * g;
    }
}

// ─── leaky_relu ─────────────────────────────────────────────────────────────

__global__ void leaky_relu_forward_kernel(const float* __restrict__ x,
                                          float slope, long long n,
                                          float* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : slope * v;
    }
}

__global__ void leaky_relu_backward_kernel(const float* __restrict__ x,
                                           const float* __restrict__ dY,
                                           float slope, long long n,
                                           float* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float g = x[i] > 0.0f ? 1.0f : slope;
        dX[i] = dY[i] * g;
    }
}

} // namespace

// ─── snake wrappers ─────────────────────────────────────────────────────────

void snake_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& alpha,
                   const ::brotensor::Tensor* beta, int N, int C, int L,
                   ::brotensor::Tensor& Y) {
    require_fp32("snake_forward", X, "X");
    require_fp32("snake_forward", alpha, "alpha");
    if (beta) require_fp32("snake_forward", *beta, "beta");
    if (N < 0 || C < 0 || L < 0) fail("snake_forward", "N, C, L must be non-negative");
    const long long cols = static_cast<long long>(C) * L;
    if (X.rows != N || X.cols != cols) fail("snake_forward", "X must be shaped (N, C*L)");
    if (alpha.size() != C) fail("snake_forward", "alpha must have C elements");
    if (beta && beta->size() != C) fail("snake_forward", "beta must have C elements");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
    const long long total = static_cast<long long>(N) * cols;
    if (total == 0) return;
    snake_forward_kernel<<<va_grid(total), VA_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data),
        static_cast<const float*>(alpha.data),
        beta ? static_cast<const float*>(beta->data) : nullptr,
        C, L, total, static_cast<float*>(Y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void snake_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& alpha,
                    const ::brotensor::Tensor* beta, const ::brotensor::Tensor& dY,
                    int N, int C, int L, ::brotensor::Tensor& dX,
                    ::brotensor::Tensor& dAlpha, ::brotensor::Tensor* dBeta) {
    require_fp32("snake_backward", X, "X");
    require_fp32("snake_backward", alpha, "alpha");
    require_fp32("snake_backward", dY, "dY");
    require_fp32("snake_backward", dAlpha, "dAlpha");
    if (beta) require_fp32("snake_backward", *beta, "beta");
    if (dBeta) require_fp32("snake_backward", *dBeta, "dBeta");
    if ((beta == nullptr) != (dBeta == nullptr)) {
        fail("snake_backward", "dBeta must be non-null exactly when beta is non-null");
    }
    if (N < 0 || C < 0 || L < 0) fail("snake_backward", "N, C, L must be non-negative");
    const long long cols = static_cast<long long>(C) * L;
    if (X.rows != N || X.cols != cols) fail("snake_backward", "X must be shaped (N, C*L)");
    if (dY.rows != N || dY.cols != cols) fail("snake_backward", "dY must be shaped (N, C*L)");
    if (alpha.size() != C) fail("snake_backward", "alpha must have C elements");
    if (beta && beta->size() != C) fail("snake_backward", "beta must have C elements");
    if (dAlpha.rows != C || dAlpha.cols != 1) fail("snake_backward", "dAlpha must be (C, 1)");
    if (dBeta && (dBeta->rows != C || dBeta->cols != 1)) {
        fail("snake_backward", "dBeta must be (C, 1)");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype) {
        dX.resize(N, static_cast<int>(cols), X.dtype);
    }
    const long long total = static_cast<long long>(N) * cols;
    if (total == 0) return;
    snake_backward_kernel<<<va_grid(total), VA_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data),
        static_cast<const float*>(alpha.data),
        beta ? static_cast<const float*>(beta->data) : nullptr,
        static_cast<const float*>(dY.data),
        C, L, total,
        static_cast<float*>(dX.data),
        static_cast<float*>(dAlpha.data),
        dBeta ? static_cast<float*>(dBeta->data) : nullptr);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── elu wrappers ───────────────────────────────────────────────────────────

void elu_forward(const ::brotensor::Tensor& x, float alpha,
                 ::brotensor::Tensor& y) {
    require_fp32("elu_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    elu_forward_kernel<<<va_grid(n), VA_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(x.data), alpha, n,
        static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void elu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  float alpha, ::brotensor::Tensor& dX) {
    require_fp32("elu_backward", x, "x");
    require_fp32("elu_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    elu_backward_kernel<<<va_grid(n), VA_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(x.data),
        static_cast<const float*>(dY.data), alpha, n,
        static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── leaky_relu wrappers ────────────────────────────────────────────────────

void leaky_relu_forward(const ::brotensor::Tensor& x, float negative_slope,
                        ::brotensor::Tensor& y) {
    require_fp32("leaky_relu_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    leaky_relu_forward_kernel<<<va_grid(n), VA_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(x.data), negative_slope, n,
        static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void leaky_relu_backward(const ::brotensor::Tensor& x,
                         const ::brotensor::Tensor& dY, float negative_slope,
                         ::brotensor::Tensor& dX) {
    require_fp32("leaky_relu_backward", x, "x");
    require_fp32("leaky_relu_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    leaky_relu_backward_kernel<<<va_grid(n), VA_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(x.data),
        static_cast<const float*>(dY.data), negative_slope, n,
        static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_vocoder(::brotensor::detail::OpsVTable& v) {
    v.snake_forward       = &snake_forward;
    v.snake_backward      = &snake_backward;
    v.elu_forward         = &elu_forward;
    v.elu_backward        = &elu_backward;
    v.leaky_relu_forward  = &leaky_relu_forward;
    v.leaky_relu_backward = &leaky_relu_backward;
}

} // namespace brotensor::detail::cuda
