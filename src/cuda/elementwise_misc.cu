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

}  // namespace brotensor::detail::cuda
