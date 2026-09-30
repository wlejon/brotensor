// Native AMD HIP Reductions for RDNA 3.5 (gfx1151, Wave32).
//
// sum_rows, sum_cols, argmax_rows, rows_count_above, masked_mean_pool.
// Supports FP32, FP16 (__half), and BF16 (hip_bfloat16).

#include "detail/elementwise_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);
void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);

namespace {

constexpr int RED_BLOCK = 256;
constexpr int kWave32 = 32;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
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

__device__ __forceinline__ float wave32_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, kWave32);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* sdata) {
    val = wave32_reduce_sum(val);
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        sdata[wave_id] = val;
    }
    __syncthreads();
    float sum = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
    sum = wave32_reduce_sum(sum);
    return sum;
}

// ─── sum_rows ──────────────────────────────────────────────────────────────

template <typename T>
__global__ void sum_rows_warp_kernel(const T* __restrict__ X, T* __restrict__ Y, int M, int N) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int m = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (m >= M) return;

    const T* row = X + static_cast<size_t>(m) * N;
    float local_sum = 0.0f;
    for (int n = lane; n < N; n += kWave32) {
        local_sum += to_f32(row[n]);
    }
    const float tot = wave32_reduce_sum(local_sum);
    if (lane == 0) {
        Y[m] = from_f32<T>(tot);
    }
}

template <typename T>
__global__ void sum_rows_block_kernel(const T* __restrict__ X, T* __restrict__ Y, int M, int N) {
    __shared__ float sdata[RED_BLOCK >> 5];
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;

    const T* row = X + static_cast<size_t>(m) * N;
    float local_sum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        local_sum += to_f32(row[n]);
    }
    const float tot = block_reduce_sum(local_sum, sdata);
    if (tid == 0) {
        Y[m] = from_f32<T>(tot);
    }
}

// ─── sum_cols ──────────────────────────────────────────────────────────────

template <typename T>
__global__ void sum_cols_kernel(const T* __restrict__ X, T* __restrict__ Y, int M, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += to_f32(X[static_cast<size_t>(m) * N + n]);
    }
    Y[n] = from_f32<T>(acc);
}

// ─── argmax_rows ───────────────────────────────────────────────────────────

template <typename T, typename OutT>
__global__ void argmax_rows_kernel(const T* __restrict__ X, OutT* __restrict__ Idx, int M, int N) {
    __shared__ float sm_val[RED_BLOCK];
    __shared__ int   sm_idx[RED_BLOCK];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    if (m >= M) return;

    float best_v = -1e38f;
    int   best_i = 0;
    const T* row = X + static_cast<size_t>(m) * N;
    for (int n = tid; n < N; n += blockDim.x) {
        const float v = to_f32(row[n]);
        if (v > best_v) {
            best_v = v;
            best_i = n;
        }
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
    if (tid == 0) {
        Idx[m] = static_cast<OutT>(sm_idx[0]);
    }
}

// ─── rows_count_above ──────────────────────────────────────────────────────

template <typename T>
__global__ void rows_count_above_kernel(const T* __restrict__ X, float t_lo, float t_hi,
                                        int32_t* __restrict__ counts, int R, int C) {
    __shared__ int sm_lo[RED_BLOCK];
    __shared__ int sm_hi[RED_BLOCK];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    if (r >= R) return;

    int n_lo = 0, n_hi = 0;
    const T* row = X + static_cast<size_t>(r) * C;
    for (int c = tid; c < C; c += blockDim.x) {
        const float v = to_f32(row[c]);
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

// ─── masked_mean_pool ──────────────────────────────────────────────────────

__global__ void count_valid_kernel(const float* __restrict__ mask, int K, int* __restrict__ out_num_valid) {
    __shared__ int sh[RED_BLOCK];
    const int tid = threadIdx.x;
    int local = 0;
    for (int k = tid; k < K; k += blockDim.x) local += (mask[k] != 0.0f);
    sh[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out_num_valid = sh[0];
}

template <typename T>
__global__ void masked_mean_pool_forward_kernel(const T* __restrict__ X,
                                                const float* __restrict__ mask,
                                                T* __restrict__ y,
                                                int K, int D,
                                                const int* __restrict__ num_valid_ptr) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    const int num_valid = mask ? *num_valid_ptr : K;
    if (num_valid == 0) {
        y[j] = from_f32<T>(0.0f);
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float m = mask ? mask[k] : 1.0f;
        if (m != 0.0f) acc += to_f32(X[k * D + j]);
    }
    y[j] = from_f32<T>(acc / static_cast<float>(num_valid));
}

template <typename T>
__global__ void masked_mean_pool_backward_kernel(const T* __restrict__ dY,
                                                 const float* __restrict__ mask,
                                                 T* __restrict__ dX,
                                                 int K, int D, int num_valid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = K * D;
    if (idx >= total) return;
    const int k = idx / D;
    const int j = idx - k * D;
    const float m = mask ? mask[k] : 1.0f;
    if (num_valid == 0 || m == 0.0f) {
        dX[idx] = from_f32<T>(0.0f);
    } else {
        dX[idx] = from_f32<T>(to_f32(dY[j]) / static_cast<float>(num_valid));
    }
}

} // namespace

void sum_rows(const Tensor& X, Tensor& Y) {
    using ::brotensor::Dtype;
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) Y.resize(M, 1, X.dtype);
    if (M == 0 || N == 0) return;

    hipStream_t stream = cur_stream();
    if (N <= 1024) {
        constexpr int warps_per_block = RED_BLOCK / kWave32;
        const int blocks = (M + warps_per_block - 1) / warps_per_block;
        if (X.dtype == Dtype::FP32) {
            sum_rows_warp_kernel<float><<<blocks, RED_BLOCK, 0, stream>>>(
                static_cast<const float*>(X.data), static_cast<float*>(Y.data), M, N);
        } else if (X.dtype == Dtype::FP16) {
            sum_rows_warp_kernel<__half><<<blocks, RED_BLOCK, 0, stream>>>(
                static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), M, N);
        } else {
            sum_rows_warp_kernel<hip_bfloat16><<<blocks, RED_BLOCK, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), M, N);
        }
    } else {
        if (X.dtype == Dtype::FP32) {
            sum_rows_block_kernel<float><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const float*>(X.data), static_cast<float*>(Y.data), M, N);
        } else if (X.dtype == Dtype::FP16) {
            sum_rows_block_kernel<__half><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), M, N);
        } else {
            sum_rows_block_kernel<hip_bfloat16><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), M, N);
        }
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void sum_cols(const Tensor& X, Tensor& Y) {
    using ::brotensor::Dtype;
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) Y.resize(1, N, X.dtype);
    if (M == 0 || N == 0) return;

    const int grid = (N + RED_BLOCK - 1) / RED_BLOCK;
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        sum_cols_kernel<float><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), M, N);
    } else if (X.dtype == Dtype::FP16) {
        sum_cols_kernel<__half><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), M, N);
    } else {
        sum_cols_kernel<hip_bfloat16><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), M, N);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void argmax_rows(const Tensor& X, Tensor& Idx) {
    using ::brotensor::Dtype;
    const int M = X.rows;
    const int N = X.cols;
    if (Idx.rows != M || Idx.cols != 1) {
        Idx.resize(M, 1, Idx.dtype == Dtype::INT32 ? Dtype::INT32 : Dtype::FP32);
    }
    if (M == 0 || N == 0) return;

    hipStream_t stream = cur_stream();
    if (Idx.dtype == Dtype::INT32) {
        if (X.dtype == Dtype::FP32) {
            argmax_rows_kernel<float, int32_t><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const float*>(X.data), static_cast<int32_t*>(Idx.data), M, N);
        } else if (X.dtype == Dtype::FP16) {
            argmax_rows_kernel<__half, int32_t><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const __half*>(X.data), static_cast<int32_t*>(Idx.data), M, N);
        } else {
            argmax_rows_kernel<hip_bfloat16, int32_t><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data), static_cast<int32_t*>(Idx.data), M, N);
        }
    } else {
        if (X.dtype == Dtype::FP32) {
            argmax_rows_kernel<float, float><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const float*>(X.data), static_cast<float*>(Idx.data), M, N);
        } else if (X.dtype == Dtype::FP16) {
            argmax_rows_kernel<__half, float><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const __half*>(X.data), static_cast<float*>(Idx.data), M, N);
        } else {
            argmax_rows_kernel<hip_bfloat16, float><<<M, RED_BLOCK, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data), static_cast<float*>(Idx.data), M, N);
        }
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts) {
    using ::brotensor::Dtype;
    const int R = X.rows;
    const int C = X.cols;
    if (counts.rows != R || counts.cols != 2 || counts.dtype != Dtype::INT32) {
        counts.resize(R, 2, Dtype::INT32);
    }
    if (R == 0 || C == 0) return;

    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        rows_count_above_kernel<float><<<R, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), t_lo, t_hi, static_cast<int32_t*>(counts.data), R, C);
    } else if (X.dtype == Dtype::FP16) {
        rows_count_above_kernel<__half><<<R, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), t_lo, t_hi, static_cast<int32_t*>(counts.data), R, C);
    } else {
        rows_count_above_kernel<hip_bfloat16><<<R, RED_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), t_lo, t_hi, static_cast<int32_t*>(counts.data), R, C);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void masked_mean_pool_forward(const Tensor& X, const float* d_mask, Tensor& y) {
    using ::brotensor::Dtype;
    const int K = X.rows;
    const int D = X.cols;
    if (y.rows != D || y.cols != 1 || y.dtype != X.dtype) y.resize(D, 1, X.dtype);
    if (K == 0 || D == 0) return;

    hipStream_t stream = cur_stream();
    int* d_num_valid = nullptr;
    if (d_mask) {
        d_num_valid = static_cast<int*>(hip_alloc(sizeof(int)));
        count_valid_kernel<<<1, RED_BLOCK, 0, stream>>>(d_mask, K, d_num_valid);
    }

    const int grid = (D + RED_BLOCK - 1) / RED_BLOCK;
    if (X.dtype == Dtype::FP32) {
        masked_mean_pool_forward_kernel<float><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), d_mask, static_cast<float*>(y.data), K, D, d_num_valid);
    } else if (X.dtype == Dtype::FP16) {
        masked_mean_pool_forward_kernel<__half><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), d_mask, static_cast<__half*>(y.data), K, D, d_num_valid);
    } else {
        masked_mean_pool_forward_kernel<hip_bfloat16><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), d_mask, static_cast<hip_bfloat16*>(y.data), K, D, d_num_valid);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    if (d_num_valid) hip_free(d_num_valid);
}

void masked_mean_pool_backward(const Tensor& dY, const float* d_mask, int K, Tensor& dX) {
    using ::brotensor::Dtype;
    const int D = dY.size();
    if (dX.rows != K || dX.cols != D || dX.dtype != dY.dtype) dX.resize(K, D, dY.dtype);
    if (K == 0 || D == 0) return;

    hipStream_t stream = cur_stream();
    int num_valid = K;
    if (d_mask) {
        int* d_nv = static_cast<int*>(hip_alloc(sizeof(int)));
        count_valid_kernel<<<1, RED_BLOCK, 0, stream>>>(d_mask, K, d_nv);
        BROTENSOR_HIP_CHECK(hipMemcpyAsync(&num_valid, d_nv, sizeof(int), hipMemcpyDeviceToHost, stream));
        BROTENSOR_HIP_CHECK(hipStreamSynchronize(stream));
        hip_free(d_nv);
    }

    const int total = K * D;
    const int grid = (total + RED_BLOCK - 1) / RED_BLOCK;
    if (dY.dtype == Dtype::FP32) {
        masked_mean_pool_backward_kernel<float><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data), d_mask, static_cast<float*>(dX.data), K, D, num_valid);
    } else if (dY.dtype == Dtype::FP16) {
        masked_mean_pool_backward_kernel<__half><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const __half*>(dY.data), d_mask, static_cast<__half*>(dX.data), K, D, num_valid);
    } else {
        masked_mean_pool_backward_kernel<hip_bfloat16><<<grid, RED_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(dY.data), d_mask, static_cast<hip_bfloat16*>(dX.data), K, D, num_valid);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
