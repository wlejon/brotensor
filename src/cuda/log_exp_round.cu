// ─── CUDA log / exp / round elementwise ops (CHUNK 6, family G) ─────────────
//
// CUDA port of src/cpu/log_exp_round.cpp:
//   log_forward / log_backward   y = log(x);   dX = dY / x
//   exp_forward / exp_backward   y = exp(x);   dX = dY * exp(x)
//   round_forward                y = round-half-to-even(x)  (torch.round)
//   round_backward               straight-through estimator: dX = dY
//
// log_forward / log_backward do NOT guard the input — for x <= 0 they return
// the IEEE result so a mis-clamped pipeline fails loudly. None of these ops
// has a learnable parameter, so every backward OVERWRITES dX (input/output
// may alias).
//
// CPU is FP32-only (per CLAUDE.md); CUDA additionally supports FP16/BF16 with
// math performed in FP32, loads/stores cast at the boundary so mixed-precision
// callers don't get silently coerced to FP32.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int LER_BLOCK = 256;

inline int ler_grid(long long n) {
    long long blocks = (n + LER_BLOCK - 1) / LER_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

inline void require_fp(const char* op, const ::brotensor::Tensor& t,
                       const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

template <typename T> __device__ inline float ler_load(const T* p);
template <> __device__ inline float ler_load<float>(const float* p) { return *p; }
template <> __device__ inline float ler_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float ler_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void ler_store(T* p, float v);
template <> __device__ inline void ler_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void ler_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void ler_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__global__ void log_forward_kernel(const T* __restrict__ x, long long n,
                                   T* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        ler_store<T>(&y[i], logf(ler_load<T>(&x[i])));
    }
}

template <typename T>
__global__ void log_backward_kernel(const T* __restrict__ x,
                                    const T* __restrict__ dY, long long n,
                                    T* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        ler_store<T>(&dX[i], ler_load<T>(&dY[i]) / ler_load<T>(&x[i]));
    }
}

template <typename T>
__global__ void exp_forward_kernel(const T* __restrict__ x, long long n,
                                   T* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        ler_store<T>(&y[i], expf(ler_load<T>(&x[i])));
    }
}

template <typename T>
__global__ void exp_backward_kernel(const T* __restrict__ x,
                                    const T* __restrict__ dY, long long n,
                                    T* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        // Read both inputs before writing so an in-place dX==dY alias is safe.
        const float xv = ler_load<T>(&x[i]);
        const float gv = ler_load<T>(&dY[i]);
        ler_store<T>(&dX[i], gv * expf(xv));
    }
}

template <typename T>
__global__ void round_forward_kernel(const T* __restrict__ x, long long n,
                                     T* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        // nearbyintf = round-half-to-even — matches torch.round / numpy.round
        // and the CPU backend's std::nearbyint.
        ler_store<T>(&y[i], nearbyintf(ler_load<T>(&x[i])));
    }
}

template <typename T>
__global__ void round_backward_kernel(const T* __restrict__ dY, long long n,
                                      T* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        ler_store<T>(&dX[i], ler_load<T>(&dY[i]));      // straight-through
    }
}

template <typename FwdFn>
inline void launch_unary(const char* op, const ::brotensor::Tensor& x,
                         ::brotensor::Tensor& y, FwdFn launch) {
    require_fp(op, x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    launch(x, y, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ─── log ────────────────────────────────────────────────────────────────────

void log_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp("log_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    if (x.dtype == ::brotensor::Dtype::FP16) {
        log_forward_kernel<__half><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), n, static_cast<__half*>(y.data));
    } else if (x.dtype == ::brotensor::Dtype::BF16) {
        log_forward_kernel<__nv_bfloat16><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), n, static_cast<__nv_bfloat16*>(y.data));
    } else {
        log_forward_kernel<float><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void log_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    require_fp("log_backward", x, "x");
    require_fp("log_backward", dY, "dY");
    if (dY.dtype != x.dtype) {
        throw std::runtime_error("brotensor: log_backward: dY.dtype must match x.dtype");
    }
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    if (x.dtype == ::brotensor::Dtype::FP16) {
        log_backward_kernel<__half><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __half*>(dY.data), n, static_cast<__half*>(dX.data));
    } else if (x.dtype == ::brotensor::Dtype::BF16) {
        log_backward_kernel<__nv_bfloat16><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(dY.data), n, static_cast<__nv_bfloat16*>(dX.data));
    } else {
        log_backward_kernel<float><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data),
            static_cast<const float*>(dY.data), n, static_cast<float*>(dX.data));
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── exp ────────────────────────────────────────────────────────────────────

void exp_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp("exp_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    if (x.dtype == ::brotensor::Dtype::FP16) {
        exp_forward_kernel<__half><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), n, static_cast<__half*>(y.data));
    } else if (x.dtype == ::brotensor::Dtype::BF16) {
        exp_forward_kernel<__nv_bfloat16><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), n, static_cast<__nv_bfloat16*>(y.data));
    } else {
        exp_forward_kernel<float><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void exp_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    require_fp("exp_backward", x, "x");
    require_fp("exp_backward", dY, "dY");
    if (dY.dtype != x.dtype) {
        throw std::runtime_error("brotensor: exp_backward: dY.dtype must match x.dtype");
    }
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    if (x.dtype == ::brotensor::Dtype::FP16) {
        exp_backward_kernel<__half><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data),
            static_cast<const __half*>(dY.data), n, static_cast<__half*>(dX.data));
    } else if (x.dtype == ::brotensor::Dtype::BF16) {
        exp_backward_kernel<__nv_bfloat16><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(dY.data), n, static_cast<__nv_bfloat16*>(dX.data));
    } else {
        exp_backward_kernel<float><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data),
            static_cast<const float*>(dY.data), n, static_cast<float*>(dX.data));
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── round ──────────────────────────────────────────────────────────────────

void round_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp("round_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const long long n = x.size();
    if (n == 0) return;
    if (x.dtype == ::brotensor::Dtype::FP16) {
        round_forward_kernel<__half><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(x.data), n, static_cast<__half*>(y.data));
    } else if (x.dtype == ::brotensor::Dtype::BF16) {
        round_forward_kernel<__nv_bfloat16><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), n, static_cast<__nv_bfloat16*>(y.data));
    } else {
        round_forward_kernel<float><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void round_backward(const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX) {
    require_fp("round_backward", dY, "dY");
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    const long long n = dY.size();
    if (n == 0) return;
    if (dY.dtype == ::brotensor::Dtype::FP16) {
        round_backward_kernel<__half><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(dY.data), n, static_cast<__half*>(dX.data));
    } else if (dY.dtype == ::brotensor::Dtype::BF16) {
        round_backward_kernel<__nv_bfloat16><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(dY.data), n, static_cast<__nv_bfloat16*>(dX.data));
    } else {
        round_backward_kernel<float><<<ler_grid(n), LER_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(dY.data), n, static_cast<float*>(dX.data));
    }
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
