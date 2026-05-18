#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int EMB_BLOCK = 256;

// One thread per (b, j). Gather row d_idx[b] of `table` into out[b, :].
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

// Scatter-accumulate via atomicAdd. Multiple lookups of the same row sum.
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

// FP16 input variant: scatter into FP32 scratch via atomicAdd (FP16 atomicAdd
// is not portable across CUDA compute capabilities).
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

__global__ void emb_add_fp32_into_fp16(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    if (b > 4096) b = 4096;
    return b;
}

} // namespace

void embedding_lookup_forward_gpu(const GpuTensor& table,
                                  const int32_t* d_idx, int B,
                                  GpuTensor& out) {
    const int D = table.cols;
    if (out.rows != B || out.cols != D || out.dtype != table.dtype) {
        out.resize(B, D, table.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (table.dtype == Dtype::FP16) {
        embedding_lookup_forward_fp16_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            reinterpret_cast<const __half*>(table.data_fp16()),
            reinterpret_cast<const int*>(d_idx),
            reinterpret_cast<__half*>(out.data_fp16()),
            B, D);
    } else {
        embedding_lookup_forward_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            table.data, reinterpret_cast<const int*>(d_idx), out.data, B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void embedding_lookup_backward_gpu(const GpuTensor& dOut,
                                   const int32_t* d_idx, int B,
                                   GpuTensor& dTable) {
    if (dTable.dtype != Dtype::FP16 && dTable.dtype != Dtype::FP32) {
        throw std::runtime_error("embedding_lookup_backward_gpu: dTable must be FP16 or FP32");
    }
    if (dOut.dtype != dTable.dtype) {
        throw std::runtime_error("embedding_lookup_backward_gpu: dOut/dTable dtype must match");
    }
    const int D = dTable.cols;
    const int total = B * D;
    if (total == 0) return;

    if (dTable.dtype == Dtype::FP32) {
        embedding_lookup_backward_kernel<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            dOut.data, reinterpret_cast<const int*>(d_idx), dTable.data, B, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else {
        const int V = dTable.rows;
        const int table_n = V * D;
        float* d_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        table_n * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemset(d_scratch, 0, table_n * sizeof(float)));
        embedding_lookup_backward_kernel_fp16<<<grid_for(total, EMB_BLOCK), EMB_BLOCK>>>(
            reinterpret_cast<const __half*>(dOut.data_fp16()),
            reinterpret_cast<const int*>(d_idx),
            d_scratch, B, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        emb_add_fp32_into_fp16<<<grid_for(table_n, EMB_BLOCK), EMB_BLOCK>>>(
            d_scratch, reinterpret_cast<__half*>(dTable.data_fp16()), table_n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_scratch);
    }
}

} // namespace brotensor
