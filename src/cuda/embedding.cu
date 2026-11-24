// CUDA embedding lookup forward. Backward lives in ops_scatter.cpp (MirKind::ScatterAdd).

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <cstdint>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

namespace {

constexpr int EMB_BLOCK = 256;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

__global__ void embedding_lookup_forward_kernel(const float* __restrict__ table,
                                                const int* __restrict__ idx,
                                                float* __restrict__ out,
                                                int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        out[t] = table[row * D + j];
    }
}

__global__ void embedding_lookup_forward_fp16_kernel(const __half* __restrict__ table,
                                                     const int* __restrict__ idx,
                                                     __half* __restrict__ out,
                                                     int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        out[t] = table[row * D + j];
    }
}

__global__ void embedding_lookup_forward_bf16_kernel(const __nv_bfloat16* __restrict__ table,
                                                     const int* __restrict__ idx,
                                                     __nv_bfloat16* __restrict__ out,
                                                     int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        out[t] = table[row * D + j];
    }
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    if (b > 4096) b = 4096;
    return b;
}

} // namespace

void embedding_lookup_forward(const Tensor& table,
                              const int32_t* d_idx, int B,
                              Tensor& out) {
    if (table.dtype != Dtype::FP32 && table.dtype != Dtype::FP16 &&
        table.dtype != Dtype::BF16) {
        throw std::runtime_error(
            "brotensor: embedding_lookup_forward: table dtype must be "
            "FP32, FP16, or BF16 (quantized tables must be dequantized first)");
    }
    const int D = table.cols;
    if (out.rows != B || out.cols != D || out.dtype != table.dtype) {
        out.resize(B, D, table.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (table.dtype == Dtype::FP16) {
        embedding_lookup_forward_fp16_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(table.data),
            reinterpret_cast<const int*>(d_idx),
            static_cast<__half*>(out.data),
            B, D);
    } else if (table.dtype == Dtype::BF16) {
        embedding_lookup_forward_bf16_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(table.data),
            reinterpret_cast<const int*>(d_idx),
            static_cast<__nv_bfloat16*>(out.data),
            B, D);
    } else {
        embedding_lookup_forward_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(table.data),
            reinterpret_cast<const int*>(d_idx),
            static_cast<float*>(out.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
