// CUDA embedding lookup. Phase 2G port — kernel bodies unchanged.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

namespace {

constexpr int EMB_BLOCK = 256;

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

__global__ void embedding_lookup_backward_kernel(const float* __restrict__ dOut,
                                                 const int* __restrict__ idx,
                                                 float* __restrict__ dTable,
                                                 int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        atomicAdd(&dTable[row * D + j], dOut[t]);
    }
}

__global__ void embedding_lookup_backward_kernel_fp16(
        const __half* __restrict__ dOut,
        const int* __restrict__ idx,
        float* __restrict__ dTable_scratch,
        int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        atomicAdd(&dTable_scratch[row * D + j], __half2float(dOut[t]));
    }
}

__global__ void embedding_lookup_backward_kernel_bf16(
        const __nv_bfloat16* __restrict__ dOut,
        const int* __restrict__ idx,
        float* __restrict__ dTable_scratch,
        int B, int D) {
    const int total = B * D;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += blockDim.x * gridDim.x) {
        const int b = t / D;
        const int j = t - b * D;
        const int row = idx[b];
        atomicAdd(&dTable_scratch[row * D + j], __bfloat162float(dOut[t]));
    }
}

__global__ void emb_add_fp32_into_fp16(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}

__global__ void emb_add_fp32_into_bf16(const float* __restrict__ src,
                                       __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
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
    const int D = table.cols;
    if (out.rows != B || out.cols != D || out.dtype != table.dtype) {
        out.resize(B, D, table.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (table.dtype == Dtype::FP16) {
        embedding_lookup_forward_fp16_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            static_cast<const __half*>(table.data),
            reinterpret_cast<const int*>(d_idx),
            static_cast<__half*>(out.data),
            B, D);
    } else if (table.dtype == Dtype::BF16) {
        embedding_lookup_forward_bf16_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(table.data),
            reinterpret_cast<const int*>(d_idx),
            static_cast<__nv_bfloat16*>(out.data),
            B, D);
    } else {
        embedding_lookup_forward_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            static_cast<const float*>(table.data),
            reinterpret_cast<const int*>(d_idx),
            static_cast<float*>(out.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void embedding_lookup_backward(const Tensor& dOut,
                               const int32_t* d_idx, int B,
                               Tensor& dTable) {
    if (dTable.dtype != Dtype::FP16 && dTable.dtype != Dtype::FP32 && dTable.dtype != Dtype::BF16) {
        throw std::runtime_error("embedding_lookup_backward: dTable must be FP16, BF16, or FP32");
    }
    if (dOut.dtype != dTable.dtype) {
        throw std::runtime_error("embedding_lookup_backward: dOut/dTable dtype must match");
    }
    const int D = dTable.cols;
    const int total = B * D;
    if (total == 0) return;

    if (dTable.dtype == Dtype::FP32) {
        embedding_lookup_backward_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            static_cast<const float*>(dOut.data),
            reinterpret_cast<const int*>(d_idx),
            static_cast<float*>(dTable.data), B, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else if (dTable.dtype == Dtype::FP16) {
        const int V = dTable.rows;
        const int table_n = V * D;
        float* d_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        table_n * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemset(d_scratch, 0, table_n * sizeof(float)));
        embedding_lookup_backward_kernel_fp16<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            static_cast<const __half*>(dOut.data),
            reinterpret_cast<const int*>(d_idx),
            d_scratch, B, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        emb_add_fp32_into_fp16<<<grid_for(table_n, EMB_BLOCK), EMB_BLOCK>>>(
            d_scratch, static_cast<__half*>(dTable.data), table_n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_scratch);
    } else {
        // BF16: FP32-scratch scatter-accumulate, fold back into BF16.
        const int V = dTable.rows;
        const int table_n = V * D;
        float* d_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        table_n * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemset(d_scratch, 0, table_n * sizeof(float)));
        embedding_lookup_backward_kernel_bf16<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(dOut.data),
            reinterpret_cast<const int*>(d_idx),
            d_scratch, B, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        emb_add_fp32_into_bf16<<<grid_for(table_n, EMB_BLOCK), EMB_BLOCK>>>(
            d_scratch, static_cast<__nv_bfloat16*>(dTable.data), table_n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_scratch);
    }
}

} // namespace brotensor::detail::cuda
