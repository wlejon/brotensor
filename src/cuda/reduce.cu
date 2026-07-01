#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor {
namespace detail::cuda {

// Defined in tensor.cu — pooled (cudaMallocAsync/cudaFreeAsync-backed)
// allocator, used here instead of raw cudaMalloc/cudaFree for the small
// per-call num_valid scratch.
void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

namespace {

constexpr int RED_BLOCK = 256;

template <typename T> __device__ inline float r_load(const T* p);
template <> __device__ inline float r_load<float>(const float* p) { return *p; }
template <> __device__ inline float r_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float r_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void r_store(T* p, float v);
template <> __device__ inline void r_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void r_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void r_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// Single-block preliminary reduction: computes num_valid (count of unmasked
// rows) ONCE and writes it to device memory, instead of every one of the D
// column-threads in the main kernel redoing this identical O(K) loop.
__global__ void count_valid_kernel(const float* __restrict__ mask, int K,
                                   int* __restrict__ out_num_valid) {
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

// One thread per output column. Each thread sums its column over valid rows
// and divides by num_valid. If num_valid == 0, writes 0. num_valid is read
// from a device scratch int precomputed once by count_valid_kernel (mask !=
// nullptr) rather than recomputed redundantly by each of the D threads.
template <typename T>
__global__ void masked_mean_pool_forward_kernel(const T* __restrict__ X,
                                                const float* __restrict__ mask,
                                                T* __restrict__ y,
                                                int K, int D,
                                                const int* __restrict__ num_valid_ptr) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    const int num_valid = mask ? *num_valid_ptr : K;

    if (num_valid == 0) { r_store<T>(&y[j], 0.0f); return; }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float m = mask ? mask[k] : 1.0f;
        if (m != 0.0f) acc += r_load<T>(&X[k * D + j]);
    }
    r_store<T>(&y[j], acc / static_cast<float>(num_valid));
}

// Distribute dY/num_valid to valid rows; zero invalid rows.
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
        r_store<T>(&dX[idx], 0.0f);
    } else {
        r_store<T>(&dX[idx], r_load<T>(&dY[j]) / static_cast<float>(num_valid));
    }
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return b;
}

inline void require_fp(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

} // namespace

void masked_mean_pool_forward(const ::brotensor::Tensor& X, const float* d_mask,
                              ::brotensor::Tensor& y) {
    require_fp(X, "masked_mean_pool_forward", "X");
    const int K = X.rows;
    const int D = X.cols;
    if (y.rows != D || y.cols != 1 || y.dtype != X.dtype) {
        y.resize(D, 1, X.dtype);
    }
    if (D == 0) return;

    // Precompute num_valid once (mask case) instead of D-fold redundantly.
    int* d_num_valid = nullptr;
    if (d_mask) {
        d_num_valid = static_cast<int*>(cuda_alloc(sizeof(int)));
        count_valid_kernel<<<1, RED_BLOCK, 0, cur_stream()>>>(d_mask, K, d_num_valid);
    }

    if (X.dtype == ::brotensor::Dtype::FP16) {
        masked_mean_pool_forward_kernel<__half><<<grid_for(D, RED_BLOCK), RED_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), d_mask,
            static_cast<__half*>(y.data), K, D, d_num_valid);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        masked_mean_pool_forward_kernel<__nv_bfloat16><<<grid_for(D, RED_BLOCK), RED_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), d_mask,
            static_cast<__nv_bfloat16*>(y.data), K, D, d_num_valid);
    } else {
        masked_mean_pool_forward_kernel<float><<<grid_for(D, RED_BLOCK), RED_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), d_mask,
            static_cast<float*>(y.data), K, D, d_num_valid);
    }
    const cudaError_t launch_err = cudaGetLastError();

    if (d_num_valid) {
        cuda_free(d_num_valid);
    }
    BROTENSOR_CUDA_CHECK(launch_err);
}

void masked_mean_pool_backward(const ::brotensor::Tensor& dY, const float* d_mask,
                               int K, ::brotensor::Tensor& dX) {
    require_fp(dY, "masked_mean_pool_backward", "dY");
    const int D = dY.size();
    if (dX.rows != K || dX.cols != D || dX.dtype != dY.dtype) {
        dX.resize(K, D, dY.dtype);
    }
    const int total = K * D;
    if (total == 0) return;

    // Need num_valid on host for the divisor. K is small — just download mask.
    int num_valid = 0;
    if (d_mask) {
        std::vector<float> hmask(K);
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(hmask.data(), d_mask, sizeof(float) * K,
                                  cudaMemcpyDeviceToHost, cur_stream()));
        BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(cur_stream()));
        for (int k = 0; k < K; ++k) num_valid += (hmask[k] != 0.0f);
    } else {
        num_valid = K;
    }

    if (dY.dtype == ::brotensor::Dtype::FP16) {
        masked_mean_pool_backward_kernel<__half><<<grid_for(total, RED_BLOCK), RED_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(dY.data), d_mask,
            static_cast<__half*>(dX.data), K, D, num_valid);
    } else if (dY.dtype == ::brotensor::Dtype::BF16) {
        masked_mean_pool_backward_kernel<__nv_bfloat16><<<grid_for(total, RED_BLOCK), RED_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(dY.data), d_mask,
            static_cast<__nv_bfloat16*>(dX.data), K, D, num_valid);
    } else {
        masked_mean_pool_backward_kernel<float><<<grid_for(total, RED_BLOCK), RED_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(dY.data), d_mask,
            static_cast<float*>(dX.data), K, D, num_valid);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
