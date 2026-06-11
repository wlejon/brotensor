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

// OutT selects the index storage type: float (legacy) or int32_t (so the index
// can feed a device gather with no host round-trip — the AR-decode hot path).
template <typename T, typename OutT>
__global__ void argmax_rows_kernel(const T* __restrict__ X,
                                   OutT* __restrict__ Idx,
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
    if (tid == 0) Idx[m] = static_cast<OutT>(sm_idx[0]);
}

// One block per row; threads stride over the columns counting strict-above
// hits for both thresholds in the same pass, then a shared-mem tree reduce
// folds the per-thread partial counts. Thread 0 writes counts[r] = {n_lo,
// n_hi}.
template <typename T>
__global__ void rows_count_above_kernel(const T* __restrict__ X,
                                        float t_lo, float t_hi,
                                        int32_t* __restrict__ counts,
                                        int R, int C) {
    __shared__ int sm_lo[RED_BLOCK];
    __shared__ int sm_hi[RED_BLOCK];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    if (r >= R) return;
    int n_lo = 0, n_hi = 0;
    const T* row = X + static_cast<long long>(r) * C;
    for (int c = tid; c < C; c += blockDim.x) {
        const float v = load_f32<T>(&row[c]);
        n_lo += (v > t_lo) ? 1 : 0;
        n_hi += (v > t_hi) ? 1 : 0;
    }
    sm_lo[tid] = n_lo;
    sm_hi[tid] = n_hi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm_lo[tid] += sm_lo[tid + s];
            sm_hi[tid] += sm_hi[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        counts[2 * r + 0] = sm_lo[0];
        counts[2 * r + 1] = sm_hi[0];
    }
}

// Launch argmax for input dtype T-dispatched, writing OutT indices.
template <typename OutT>
void launch_argmax(const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx,
                   int M, int N, cudaStream_t stream) {
    OutT* out = static_cast<OutT*>(Idx.data);
    if (X.dtype == Dtype::FP16) {
        argmax_rows_kernel<__half, OutT><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), out, M, N);
    } else if (X.dtype == Dtype::BF16) {
        argmax_rows_kernel<__nv_bfloat16, OutT><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(X.data), out, M, N);
    } else {
        argmax_rows_kernel<float, OutT><<<M, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), out, M, N);
    }
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

// Output dtype is opt-in: pass an INT32-typed `Idx` to get the index written as
// a device int32 (consumable as a gather index with no host round-trip);
// otherwise the index is written FP32 (the legacy default). Input X is
// FP16/BF16/FP32.
void argmax_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("argmax_rows: X must be FP16, BF16, or FP32");
    }
    const int M = X.rows;
    const int N = X.cols;
    const Dtype out_dt = (Idx.dtype == Dtype::INT32) ? Dtype::INT32 : Dtype::FP32;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != out_dt) {
        Idx.resize(M, 1, out_dt);
    }
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (out_dt == Dtype::INT32) launch_argmax<int32_t>(X, Idx, M, N, stream);
    else                        launch_argmax<float>(X, Idx, M, N, stream);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// Per-row strict-above counts at two thresholds (one pass, block per row).
// counts is (R, 2) INT32: counts[r] = { #{x > t_lo}, #{x > t_hi} }.
void rows_count_above(const ::brotensor::Tensor& X, float t_lo, float t_hi,
                      ::brotensor::Tensor& counts) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("rows_count_above: X must be FP32 or FP16");
    }
    const int R = X.rows;
    const int C = X.cols;
    if (counts.rows != R || counts.cols != 2 || counts.dtype != Dtype::INT32) {
        counts.resize(R, 2, Dtype::INT32);
    }
    if (R == 0) return;
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (C == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
            counts.data, 0, static_cast<size_t>(R) * 2 * sizeof(int32_t),
            stream));
        return;
    }
    int32_t* out = static_cast<int32_t*>(counts.data);
    if (X.dtype == Dtype::FP16) {
        rows_count_above_kernel<__half><<<R, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), t_lo, t_hi, out, R, C);
    } else {
        rows_count_above_kernel<float><<<R, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), t_lo, t_hi, out, R, C);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
