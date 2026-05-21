// ─── CUDA AdaLN modulation ops ─────────────────────────────────────────────
//
// DiT / SD3 / Flux broadcast-affine primitives.
//   modulate:      Y[l, d] = X[l, d] * (1 + scale[d]) + shift[d]
//   broadcast_mul: Y[l, d] = X[l, d] * v[d]
// scale / shift / v are length-D vectors broadcast across every token row.
//
// Dispatched on X.dtype (FP32 / FP16 / BF16). The scale / shift / v operands
// must share X's dtype. All arithmetic is performed in FP32; only the storage
// loads / stores change type. Both ops fully overwrite their output.

#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor {
namespace detail::cuda {

namespace {

constexpr int MOD_BLOCK = 256;

// ── typed load / store: storage type T <-> float compute ──
__device__ inline float ld(const float& x)          { return x; }
__device__ inline float ld(const __half& x)         { return __half2float(x); }
__device__ inline float ld(const __nv_bfloat16& x)  { return __bfloat162float(x); }
__device__ inline void  st(float& d, float v)         { d = v; }
__device__ inline void  st(__half& d, float v)        { d = __float2half(v); }
__device__ inline void  st(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

template <typename T>
__global__ void modulate_kernel(const T* __restrict__ X,
                                const T* __restrict__ scale,
                                const T* __restrict__ shift,
                                T* __restrict__ Y, int total, int D) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int d = idx % D;
        st(Y[idx], ld(X[idx]) * (1.0f + ld(scale[d])) + ld(shift[d]));
    }
}

template <typename T>
__global__ void broadcast_mul_kernel(const T* __restrict__ X,
                                     const T* __restrict__ v,
                                     T* __restrict__ Y, int total, int D) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int d = idx % D;
        st(Y[idx], ld(X[idx]) * ld(v[d]));
    }
}

inline int grid_for(int n) {
    int b = (n + MOD_BLOCK - 1) / MOD_BLOCK;
    if (b > 65535) b = 65535;   // grid-stride loop covers the remainder
    return b < 1 ? 1 : b;
}

inline void check_same_dtype(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& t,
                             const char* op, const char* name) {
    if (t.dtype != X.dtype) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " dtype must match X");
    }
}

} // namespace

void modulate(const ::brotensor::Tensor& X, const ::brotensor::Tensor& scale,
              const ::brotensor::Tensor& shift, ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("modulate: X must be FP32, FP16, or BF16");
    }
    check_same_dtype(X, scale, "modulate", "scale");
    check_same_dtype(X, shift, "modulate", "shift");
    const int L = X.rows;
    const int D = X.cols;
    if (scale.size() != D || shift.size() != D) {
        throw std::runtime_error("modulate: scale and shift must each have X.cols elements");
    }
    if (Y.rows != L || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(L, D, X.dtype);
    }
    const int total = L * D;
    if (total == 0) return;
    const int blocks = grid_for(total);

    switch (X.dtype) {
    case Dtype::FP32:
        modulate_kernel<float><<<blocks, MOD_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(scale.data),
            static_cast<const float*>(shift.data),
            static_cast<float*>(Y.data), total, D);
        break;
    case Dtype::FP16:
        modulate_kernel<__half><<<blocks, MOD_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(scale.data),
            static_cast<const __half*>(shift.data),
            static_cast<__half*>(Y.data), total, D);
        break;
    default:  // BF16
        modulate_kernel<__nv_bfloat16><<<blocks, MOD_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<const __nv_bfloat16*>(scale.data),
            static_cast<const __nv_bfloat16*>(shift.data),
            static_cast<__nv_bfloat16*>(Y.data), total, D);
        break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void broadcast_mul(const ::brotensor::Tensor& X, const ::brotensor::Tensor& v,
                   ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("broadcast_mul: X must be FP32, FP16, or BF16");
    }
    check_same_dtype(X, v, "broadcast_mul", "v");
    const int L = X.rows;
    const int D = X.cols;
    if (v.size() != D) {
        throw std::runtime_error("broadcast_mul: v must have X.cols elements");
    }
    if (Y.rows != L || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(L, D, X.dtype);
    }
    const int total = L * D;
    if (total == 0) return;
    const int blocks = grid_for(total);

    switch (X.dtype) {
    case Dtype::FP32:
        broadcast_mul_kernel<float><<<blocks, MOD_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(v.data),
            static_cast<float*>(Y.data), total, D);
        break;
    case Dtype::FP16:
        broadcast_mul_kernel<__half><<<blocks, MOD_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(v.data),
            static_cast<__half*>(Y.data), total, D);
        break;
    default:  // BF16
        broadcast_mul_kernel<__nv_bfloat16><<<blocks, MOD_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<__nv_bfloat16*>(Y.data), total, D);
        break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
