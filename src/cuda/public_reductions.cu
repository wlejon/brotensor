// Public reductions: sum_rows, sum_cols, argmax_rows.
// FP32 + FP16 dispatch for the sums; argmax accepts FP32/FP16 input and
// writes integer indices into an FP32 output tensor.

#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr int RED_BLOCK = 256;

template <typename T>
__device__ inline float load_f32(const T* p);
template <> __device__ inline float load_f32<float>(const float* p)   { return *p; }
template <> __device__ inline float load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float load_f32<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__device__ inline void store_f32(T* p, float v);
template <> __device__ inline void store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void store_f32<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// One block per row, threads stride-loop over N then shared-mem reduce.
template <typename T>
__global__ void sum_rows_kernel(const T* __restrict__ X, T* __restrict__ Y,
                                int M, int N) {
    __shared__ float sm[RED_BLOCK];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    if (m >= M) return;
    float acc = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        acc += load_f32<T>(&X[m * N + n]);
    }
    sm[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) store_f32<T>(&Y[m], sm[0]);
}

// One thread per output column. Straight-line accumulation over rows.
template <typename T>
__global__ void sum_cols_kernel(const T* __restrict__ X, T* __restrict__ Y,
                                int M, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) acc += load_f32<T>(&X[m * N + n]);
    store_f32<T>(&Y[n], acc);
}

template <typename T>
__global__ void argmax_rows_kernel(const T* __restrict__ X,
                                   float* __restrict__ Idx,
                                   int M, int N) {
    __shared__ float sm_val[RED_BLOCK];
    __shared__ int   sm_idx[RED_BLOCK];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    if (m >= M) return;
    float best_v = -3.4028235e38f;
    int   best_i = 0;
    for (int n = tid; n < N; n += blockDim.x) {
        const float v = load_f32<T>(&X[m * N + n]);
        if (v > best_v) { best_v = v; best_i = n; }
    }
    sm_val[tid] = best_v;
    sm_idx[tid] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sm_val[tid + s] > sm_val[tid]) {
                sm_val[tid] = sm_val[tid + s];
                sm_idx[tid] = sm_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) Idx[m] = static_cast<float>(sm_idx[0]);
}

} // namespace

void sum_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("sum_rows: X must be FP16, BF16, or FP32");
    }
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) {
        Y.resize(M, 1, X.dtype);
    }
    if (M == 0) return;
    if (N == 0) { Y.zero(); return; }
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (X.dtype == Dtype::FP16) {
        sum_rows_kernel<__half><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            M, N);
    } else if (X.dtype == Dtype::BF16) {
        sum_rows_kernel<__nv_bfloat16><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            M, N);
    } else {
        sum_rows_kernel<float><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data), M, N);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sum_cols(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("sum_cols: X must be FP16, BF16, or FP32");
    }
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) {
        Y.resize(1, N, X.dtype);
    }
    if (N == 0) return;
    if (M == 0) { Y.zero(); return; }
    const int blocks = (N + RED_BLOCK - 1) / RED_BLOCK;
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (X.dtype == Dtype::FP16) {
        sum_cols_kernel<__half><<<blocks, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            M, N);
    } else if (X.dtype == Dtype::BF16) {
        sum_cols_kernel<__nv_bfloat16><<<blocks, RED_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            M, N);
    } else {
        sum_cols_kernel<float><<<blocks, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data), M, N);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void argmax_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("argmax_rows: X must be FP16, BF16, or FP32");
    }
    const int M = X.rows;
    const int N = X.cols;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != Dtype::FP32) {
        Idx.resize(M, 1, Dtype::FP32);
    }
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (X.dtype == Dtype::FP16) {
        argmax_rows_kernel<__half><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<float*>(Idx.data), M, N);
    } else if (X.dtype == Dtype::BF16) {
        argmax_rows_kernel<__nv_bfloat16><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<float*>(Idx.data), M, N);
    } else {
        argmax_rows_kernel<float><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Idx.data), M, N);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
