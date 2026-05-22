// ─── CUDA log / exp / round elementwise ops (CHUNK 6, family G) ─────────────
//
// CUDA port of src/cpu/log_exp_round.cpp — FP32-only, mirroring the CPU
// contract verbatim:
//   log_forward / log_backward   y = log(x);   dX = dY / x
//   exp_forward / exp_backward   y = exp(x);   dX = dY * exp(x)
//   round_forward                y = round-half-to-even(x)  (torch.round)
//   round_backward               straight-through estimator: dX = dY
//
// log_forward / log_backward do NOT guard the input — for x <= 0 they return
// the IEEE result so a mis-clamped pipeline fails loudly. None of these ops
// has a learnable parameter, so every backward OVERWRITES dX (input/output
// may alias). These audio ops are FP32-only — a non-FP32 operand throws.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int LER_BLOCK = 256;

inline int ler_grid(long long n) {
    long long blocks = (n + LER_BLOCK - 1) / LER_BLOCK;
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

__global__ void log_forward_kernel(const float* __restrict__ x, long long n,
                                   float* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        y[i] = logf(x[i]);
    }
}

__global__ void log_backward_kernel(const float* __restrict__ x,
                                    const float* __restrict__ dY, long long n,
                                    float* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        dX[i] = dY[i] / x[i];
    }
}

__global__ void exp_forward_kernel(const float* __restrict__ x, long long n,
                                   float* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        y[i] = expf(x[i]);
    }
}

__global__ void exp_backward_kernel(const float* __restrict__ x,
                                    const float* __restrict__ dY, long long n,
                                    float* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        // Read both inputs before writing so an in-place dX==dY alias is safe.
        dX[i] = dY[i] * expf(x[i]);
    }
}

__global__ void round_forward_kernel(const float* __restrict__ x, long long n,
                                     float* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        // nearbyintf = round-half-to-even — matches torch.round / numpy.round
        // and the CPU backend's std::nearbyint.
        y[i] = nearbyintf(x[i]);
    }
}

__global__ void round_backward_kernel(const float* __restrict__ dY, long long n,
                                      float* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        dX[i] = dY[i];                                  // straight-through
    }
}

} // namespace

// ─── log ────────────────────────────────────────────────────────────────────

void log_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32("log_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != ::brotensor::Dtype::FP32) {
        y.resize(x.rows, x.cols, ::brotensor::Dtype::FP32);
    }
    const long long n = x.size();
    if (n == 0) return;
    log_forward_kernel<<<ler_grid(n), LER_BLOCK>>>(
        static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void log_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    require_fp32("log_backward", x, "x");
    require_fp32("log_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(x.rows, x.cols, ::brotensor::Dtype::FP32);
    }
    const long long n = x.size();
    if (n == 0) return;
    log_backward_kernel<<<ler_grid(n), LER_BLOCK>>>(
        static_cast<const float*>(x.data),
        static_cast<const float*>(dY.data), n, static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── exp ────────────────────────────────────────────────────────────────────

void exp_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32("exp_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != ::brotensor::Dtype::FP32) {
        y.resize(x.rows, x.cols, ::brotensor::Dtype::FP32);
    }
    const long long n = x.size();
    if (n == 0) return;
    exp_forward_kernel<<<ler_grid(n), LER_BLOCK>>>(
        static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void exp_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    require_fp32("exp_backward", x, "x");
    require_fp32("exp_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(x.rows, x.cols, ::brotensor::Dtype::FP32);
    }
    const long long n = x.size();
    if (n == 0) return;
    exp_backward_kernel<<<ler_grid(n), LER_BLOCK>>>(
        static_cast<const float*>(x.data),
        static_cast<const float*>(dY.data), n, static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── round ──────────────────────────────────────────────────────────────────

void round_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32("round_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != ::brotensor::Dtype::FP32) {
        y.resize(x.rows, x.cols, ::brotensor::Dtype::FP32);
    }
    const long long n = x.size();
    if (n == 0) return;
    round_forward_kernel<<<ler_grid(n), LER_BLOCK>>>(
        static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void round_backward(const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX) {
    require_fp32("round_backward", dY, "dY");
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(dY.rows, dY.cols, ::brotensor::Dtype::FP32);
    }
    const long long n = dY.size();
    if (n == 0) return;
    round_backward_kernel<<<ler_grid(n), LER_BLOCK>>>(
        static_cast<const float*>(dY.data), n, static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_log_exp_round(::brotensor::detail::OpsVTable& v) {
    v.log_forward    = &log_forward;
    v.log_backward   = &log_backward;
    v.exp_forward    = &exp_forward;
    v.exp_backward   = &exp_backward;
    v.round_forward  = &round_forward;
    v.round_backward = &round_backward;
}

} // namespace brotensor::detail::cuda
