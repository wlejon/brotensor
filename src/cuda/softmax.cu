#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor {
namespace detail::cuda {

namespace {

constexpr int SM_BLOCK = 256;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

// One block computes softmax over a vector of length N.
// Stable: subtract max over valid (mask==1) entries, exp, normalise.
__global__ void softmax_forward_kernel(const float* __restrict__ logits,
                                       float* __restrict__ probs,
                                       const float* __restrict__ mask,
                                       int n) {
    __shared__ float sdata[SM_BLOCK];

    const int tid = threadIdx.x;

    // Phase 1: find max over valid entries.
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) continue;
        const float v = logits[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    // Phase 2: write exp(x - m) into probs (zero on masked), accumulate sum.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) {
            probs[i] = 0.0f;
            continue;
        }
        const float e = expf(logits[i] - m);
        probs[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        probs[i] = probs[i] * inv;
    }
}

// Row-batched stable softmax: one block per row, softmax over `cols` entries.
// Replaces a per-row softmax_forward loop with a single launch over `rows`.
__global__ void softmax_rows_forward_kernel(const float* __restrict__ X,
                                            float* __restrict__ Y,
                                            int rows, int cols) {
    __shared__ float sdata[SM_BLOCK];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* logits = X + static_cast<long long>(r) * cols;
    float* probs        = Y + static_cast<long long>(r) * cols;
    const int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) {
        const float v = logits[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { const float a = sdata[tid], b = sdata[tid + s]; sdata[tid] = a > b ? a : b; }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        const float e = expf(logits[i] - m);
        probs[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float inv = sdata[0] > 0.0f ? 1.0f / sdata[0] : 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) probs[i] *= inv;
}

// FP16 forward: loads/stores in FP16, all reductions (max, sum) in FP32.
// probs is staged in an FP32 scratch buffer between phase 2 and the final
// normalise+store so we don't pay extra FP16 rounding in the intermediate.
__global__ void softmax_forward_kernel_fp16(const __half* __restrict__ logits,
                                            __half* __restrict__ probs,
                                            const float* __restrict__ mask,
                                            float* __restrict__ scratch, // length n
                                            int n) {
    __shared__ float sdata[SM_BLOCK];
    const int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) continue;
        const float v = __half2float(logits[i]);
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) {
            scratch[i] = 0.0f;
            continue;
        }
        const float e = expf(__half2float(logits[i]) - m);
        scratch[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        probs[i] = __float2half(scratch[i] * inv);
    }
}

__global__ void softmax_forward_kernel_bf16(const __nv_bfloat16* __restrict__ logits,
                                            __nv_bfloat16* __restrict__ probs,
                                            const float* __restrict__ mask,
                                            float* __restrict__ scratch, // length n
                                            int n) {
    __shared__ float sdata[SM_BLOCK];
    const int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) continue;
        const float v = __bfloat162float(logits[i]);
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) {
            scratch[i] = 0.0f;
            continue;
        }
        const float e = expf(__bfloat162float(logits[i]) - m);
        scratch[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        probs[i] = __float2bfloat16(scratch[i] * inv);
    }
}

// dL/dz_i = p_i * (dL/dp_i - sum_j dL/dp_j * p_j).
__global__ void softmax_backward_kernel(const float* __restrict__ probs,
                                        const float* __restrict__ dProbs,
                                        float* __restrict__ dLogits,
                                        int n) {
    __shared__ float sdata[SM_BLOCK];
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += dProbs[i] * probs[i];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int i = tid; i < n; i += blockDim.x) {
        dLogits[i] = probs[i] * (dProbs[i] - dot);
    }
}

__global__ void softmax_backward_kernel_fp16(const __half* __restrict__ probs,
                                             const __half* __restrict__ dProbs,
                                             __half* __restrict__ dLogits,
                                             int n) {
    __shared__ float sdata[SM_BLOCK];
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += __half2float(dProbs[i]) * __half2float(probs[i]);
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int i = tid; i < n; i += blockDim.x) {
        const float p  = __half2float(probs[i]);
        const float dp = __half2float(dProbs[i]);
        dLogits[i] = __float2half(p * (dp - dot));
    }
}

__global__ void softmax_backward_kernel_bf16(const __nv_bfloat16* __restrict__ probs,
                                             const __nv_bfloat16* __restrict__ dProbs,
                                             __nv_bfloat16* __restrict__ dLogits,
                                             int n) {
    __shared__ float sdata[SM_BLOCK];
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += __bfloat162float(dProbs[i]) * __bfloat162float(probs[i]);
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int i = tid; i < n; i += blockDim.x) {
        const float p  = __bfloat162float(probs[i]);
        const float dp = __bfloat162float(dProbs[i]);
        dLogits[i] = __float2bfloat16(p * (dp - dot));
    }
}

} // namespace

void softmax_forward(const ::brotensor::Tensor& logits,
                     ::brotensor::Tensor& probs,
                     const float* d_mask) {
    using ::brotensor::Dtype;
    if (logits.dtype != Dtype::FP16 && logits.dtype != Dtype::BF16 &&
        logits.dtype != Dtype::FP32) {
        throw std::runtime_error("softmax_forward: logits must be FP16, BF16, or FP32");
    }
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols ||
        probs.dtype != logits.dtype) {
        probs.resize(logits.rows, logits.cols, logits.dtype);
    }
    if (n == 0) return;
    if (logits.dtype == Dtype::FP16) {
        float* d_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        n * sizeof(float)));
        softmax_forward_kernel_fp16<<<1, SM_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(logits.data),
            static_cast<__half*>(probs.data),
            d_mask, d_scratch, n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_scratch);
    } else if (logits.dtype == Dtype::BF16) {
        float* d_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        n * sizeof(float)));
        softmax_forward_kernel_bf16<<<1, SM_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<__nv_bfloat16*>(probs.data),
            d_mask, d_scratch, n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_scratch);
    } else {
        softmax_forward_kernel<<<1, SM_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(logits.data),
            static_cast<float*>(probs.data), d_mask, n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

void softmax_rows_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y,
                          int rows, int cols) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32)
        throw std::runtime_error("softmax_rows_forward: X must be FP32");
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype)
        Y.resize(X.rows, X.cols, X.dtype);
    if (rows <= 0 || cols <= 0) return;
    softmax_rows_forward_kernel<<<rows, SM_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data), static_cast<float*>(Y.data), rows, cols);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void softmax_backward(const ::brotensor::Tensor& probs,
                      const ::brotensor::Tensor& dProbs,
                      ::brotensor::Tensor& dLogits) {
    using ::brotensor::Dtype;
    if (probs.dtype != Dtype::FP16 && probs.dtype != Dtype::BF16 &&
        probs.dtype != Dtype::FP32) {
        throw std::runtime_error("softmax_backward: probs must be FP16, BF16, or FP32");
    }
    if (dProbs.dtype != probs.dtype) {
        throw std::runtime_error("softmax_backward: dProbs.dtype must match probs.dtype");
    }
    const int n = probs.size();
    if (dLogits.rows != probs.rows || dLogits.cols != probs.cols ||
        dLogits.dtype != probs.dtype) {
        dLogits.resize(probs.rows, probs.cols, probs.dtype);
    }
    if (n == 0) return;
    if (probs.dtype == Dtype::FP16) {
        softmax_backward_kernel_fp16<<<1, SM_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(probs.data),
            static_cast<const __half*>(dProbs.data),
            static_cast<__half*>(dLogits.data), n);
    } else if (probs.dtype == Dtype::BF16) {
        softmax_backward_kernel_bf16<<<1, SM_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(probs.data),
            static_cast<const __nv_bfloat16*>(dProbs.data),
            static_cast<__nv_bfloat16*>(dLogits.data), n);
    } else {
        softmax_backward_kernel<<<1, SM_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(probs.data),
            static_cast<const float*>(dProbs.data),
            static_cast<float*>(dLogits.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
