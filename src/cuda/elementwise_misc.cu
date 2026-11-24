// Standalone small CUDA mask/threshold utilities not lowered via MIR elementwise.

#include <brotensor/tensor.h>
#include "detail/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

namespace {

constexpr int kMiscBlock = 256;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

inline int grid_for(int n) {
    int blocks = (n + kMiscBlock - 1) / kMiscBlock;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

__global__ void threshold_u8_fp32_kernel(const float* __restrict__ x, float t,
                                         int8_t* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = (x[i] > t) ? int8_t{1} : int8_t{0};
    }
}

__global__ void threshold_u8_fp16_kernel(const __half* __restrict__ x, float t,
                                         int8_t* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = (__half2float(x[i]) > t) ? int8_t{1} : int8_t{0};
    }
}

__global__ void causal_mask_row_kernel(float* __restrict__ mask, int L, int q) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= L) return;
    mask[k] = (k <= q) ? 1.0f : 0.0f;
}

__global__ void build_slot_mask_kernel(const float* __restrict__ x,
                                       float* __restrict__ mask,
                                       int offset, int K, int stride) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const float v = x[offset + k * stride];
    mask[k] = v > 0.5f ? 1.0f : 0.0f;
}

template <typename T> __device__ __forceinline__ float misc_load_f32(const T* p);
template <> __device__ __forceinline__ float misc_load_f32<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float misc_load_f32<__half>(const __half* p) { return __half2float(*p); }

template <typename T>
__global__ void rows_count_above_kernel(const T* __restrict__ X,
                                        float t_lo, float t_hi,
                                        int32_t* __restrict__ counts,
                                        int R, int C) {
    __shared__ int sm_lo[kMiscBlock];
    __shared__ int sm_hi[kMiscBlock];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    if (r >= R) return;
    int n_lo = 0, n_hi = 0;
    const T* row = X + static_cast<long long>(r) * C;
    for (int c = tid; c < C; c += blockDim.x) {
        const float v = misc_load_f32<T>(&row[c]);
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

}  // namespace

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("threshold_u8: X must be FP32 or FP16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::INT8) {
        Y.resize(X.rows, X.cols, Dtype::INT8);
    }
    const int n = X.size();
    if (n == 0) return;
    if (X.dtype == Dtype::FP16) {
        threshold_u8_fp16_kernel<<<grid_for(n), kMiscBlock, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), t,
            static_cast<int8_t*>(Y.data), n);
    } else {
        threshold_u8_fp32_kernel<<<grid_for(n), kMiscBlock, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), t,
            static_cast<int8_t*>(Y.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
    if (mask.rows != L || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(L, 1, Dtype::FP32);
    }
    if (L <= 0) return;
    const int blocks = (L + kMiscBlock - 1) / kMiscBlock;
    causal_mask_row_kernel<<<blocks, kMiscBlock, 0, cur_stream()>>>(
        static_cast<float*>(mask.data), L, q);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride, Tensor& mask) {
    if (mask.rows != K || mask.cols != 1) mask.resize(K, 1);
    if (K <= 0) return;
    const int blocks = (K + kMiscBlock - 1) / kMiscBlock;
    build_slot_mask_kernel<<<blocks, kMiscBlock, 0, cur_stream()>>>(
        static_cast<const float*>(x.data),
        static_cast<float*>(mask.data),
        offset, K, stride);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void rows_count_above(const ::brotensor::Tensor& X, float t_lo, float t_hi,
                      ::brotensor::Tensor& counts) {
    if (X.dtype != ::brotensor::Dtype::FP32 && X.dtype != ::brotensor::Dtype::FP16) {
        throw std::runtime_error("rows_count_above: X must be FP32 or FP16");
    }
    const int R = X.rows;
    const int C = X.cols;
    if (counts.rows != R || counts.cols != 2 || counts.dtype != ::brotensor::Dtype::INT32) {
        counts.resize(R, 2, ::brotensor::Dtype::INT32);
    }
    if (R == 0) return;
    if (C == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
            counts.data, 0, static_cast<size_t>(R) * 2 * sizeof(int32_t),
            cur_stream()));
        return;
    }
    int32_t* out = static_cast<int32_t*>(counts.data);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        rows_count_above_kernel<__half><<<R, kMiscBlock, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), t_lo, t_hi, out, R, C);
    } else {
        rows_count_above_kernel<float><<<R, kMiscBlock, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), t_lo, t_hi, out, R, C);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace brotensor::detail::cuda
