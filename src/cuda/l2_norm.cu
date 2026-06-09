// ─── CUDA L2-norm (per-head, last-dim) ─────────────────────────────────────
//
// Mirrors src/cpu/l2_norm.cpp. Layout: (L, num_heads * head_dim), row-major;
// head h occupies columns [h*head_dim, (h+1)*head_dim). Used to L2-normalise
// q and k per head before the gated delta-rule recurrence.
//
// Supports FP32 / FP16 / BF16 — math in FP32, loads/stores cast at the
// boundary so mixed-precision callers don't get silently coerced to FP32.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

namespace {

constexpr int L2_BLOCK = 128;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

template <typename T> __device__ inline float l2_load(const T* p);
template <> __device__ inline float l2_load<float>(const float* p)  { return *p; }
template <> __device__ inline float l2_load<__half>(const __half* p){ return __half2float(*p); }
template <> __device__ inline float l2_load<__nv_bfloat16>(const __nv_bfloat16* p){ return __bfloat162float(*p); }

template <typename T> __device__ inline void l2_store(T* p, float v);
template <> __device__ inline void l2_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void l2_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void l2_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// In-block reduction over `blockDim.x` threads using shared memory. Returns
// the sum from thread 0; other threads see the final value in sdata[0].
__device__ inline float block_sum(float v, float* sdata) {
    const int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

// One block per (row, head). blockDim.x = L2_BLOCK threads cooperate over
// head_dim. Shared memory: L2_BLOCK floats (single reduction scratch).
template <typename T>
__global__ void l2_norm_forward_kernel(const T* __restrict__ X,
                                       T* __restrict__ Y,
                                       int L, int num_heads, int head_dim,
                                       float eps) {
    extern __shared__ float sdata[];
    const int idx = blockIdx.x;
    const int r   = idx / num_heads;
    const int h   = idx - r * num_heads;
    if (r >= L) return;
    const int tid = threadIdx.x;
    const int D   = num_heads * head_dim;
    const int off = r * D + h * head_dim;
    const T* xrow = X + off;
    T*       yrow = Y + off;

    float local = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = l2_load<T>(&xrow[d]);
        local += v * v;
    }
    const float sumsq = block_sum(local, sdata);
    const float inv   = rsqrtf(sumsq + eps);

    for (int d = tid; d < head_dim; d += blockDim.x) {
        l2_store<T>(&yrow[d], l2_load<T>(&xrow[d]) * inv);
    }
}

// One block per (row, head). Shared memory: 2 * L2_BLOCK floats — one stripe
// for sumsq, one for dot(x, dY).
//
// dX_d = n * (dY_d - x_d * n^2 * dot(x, dY)), with n = 1 / sqrt(sumsq + eps).
template <typename T>
__global__ void l2_norm_backward_kernel(const T* __restrict__ X,
                                        const T* __restrict__ dY,
                                        T* __restrict__ dX,
                                        int L, int num_heads, int head_dim,
                                        float eps) {
    extern __shared__ float sdata[];
    float* s_sumsq = sdata;
    float* s_dot   = sdata + blockDim.x;

    const int idx = blockIdx.x;
    const int r   = idx / num_heads;
    const int h   = idx - r * num_heads;
    if (r >= L) return;
    const int tid = threadIdx.x;
    const int D   = num_heads * head_dim;
    const int off = r * D + h * head_dim;
    const T* xrow = X  + off;
    const T* grow = dY + off;
    T*       drow = dX + off;

    float l_sumsq = 0.0f;
    float l_dot   = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v  = l2_load<T>(&xrow[d]);
        const float gv = l2_load<T>(&grow[d]);
        l_sumsq += v * v;
        l_dot   += v * gv;
    }
    s_sumsq[tid] = l_sumsq;
    s_dot[tid]   = l_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sumsq[tid] += s_sumsq[tid + s];
            s_dot[tid]   += s_dot[tid + s];
        }
        __syncthreads();
    }
    const float sumsq = s_sumsq[0];
    const float dot   = s_dot[0];
    const float n2    = 1.0f / (sumsq + eps);
    const float n     = sqrtf(n2);
    const float c     = dot * n2;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float gv = l2_load<T>(&grow[d]);
        const float xv = l2_load<T>(&xrow[d]);
        l2_store<T>(&drow[d], n * (gv - xv * c));
    }
}

inline void check_fp(const ::brotensor::Tensor& t,
                     const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

inline void check_shape(const ::brotensor::Tensor& t,
                        int head_dim, int num_heads,
                        const char* op, const char* name) {
    if (head_dim <= 0) {
        throw std::runtime_error(std::string(op) + ": head_dim must be positive");
    }
    if (num_heads <= 0) {
        throw std::runtime_error(std::string(op) + ": num_heads must be positive");
    }
    if (t.cols != num_heads * head_dim) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 ".cols != num_heads * head_dim");
    }
}

} // namespace

void l2_norm_forward(const ::brotensor::Tensor& X,
                     int head_dim, int num_heads, float eps,
                     ::brotensor::Tensor& Y) {
    check_fp(X, "l2_norm_forward", "X");
    check_shape(X, head_dim, num_heads, "l2_norm_forward", "X");
    const int L = X.rows;
    const int D = X.cols;
    if (Y.rows != L || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    const int blocks = L * num_heads;
    const int block  = L2_BLOCK;
    const size_t shmem = block * sizeof(float);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        l2_norm_forward_kernel<__half><<<blocks, block, shmem, cur_stream()>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            L, num_heads, head_dim, eps);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        l2_norm_forward_kernel<__nv_bfloat16><<<blocks, block, shmem, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            L, num_heads, head_dim, eps);
    } else {
        l2_norm_forward_kernel<float><<<blocks, block, shmem, cur_stream()>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            L, num_heads, head_dim, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void l2_norm_backward(const ::brotensor::Tensor& X,
                      int head_dim, int num_heads, float eps,
                      const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX) {
    check_fp(X,  "l2_norm_backward", "X");
    check_fp(dY, "l2_norm_backward", "dY");
    if (dY.dtype != X.dtype) {
        throw std::runtime_error("l2_norm_backward: dY.dtype must match X.dtype");
    }
    check_shape(X,  head_dim, num_heads, "l2_norm_backward", "X");
    check_shape(dY, head_dim, num_heads, "l2_norm_backward", "dY");
    if (dY.rows != X.rows) {
        throw std::runtime_error("l2_norm_backward: dY.rows != X.rows");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (dX.rows != L || dX.cols != D || dX.dtype != X.dtype) {
        dX.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    const int blocks = L * num_heads;
    const int block  = L2_BLOCK;
    const size_t shmem = 2 * block * sizeof(float);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        l2_norm_backward_kernel<__half><<<blocks, block, shmem, cur_stream()>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            L, num_heads, head_dim, eps);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        l2_norm_backward_kernel<__nv_bfloat16><<<blocks, block, shmem, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data),
            L, num_heads, head_dim, eps);
    } else {
        l2_norm_backward_kernel<float><<<blocks, block, shmem, cur_stream()>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            L, num_heads, head_dim, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_l2_norm(::brotensor::detail::OpsVTable& v) {
    v.l2_norm_forward  = &l2_norm_forward;
    v.l2_norm_backward = &l2_norm_backward;
}

} // namespace brotensor::detail::cuda
