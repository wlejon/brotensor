// Batched (inference-only) CUDA ops.

#include "detail/cuda_check.h"
#include "ops_gemm_mir.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Defined in tensor.cu (same namespace). The pooled allocator draws from a
// stream-ordered memory pool (cudaMallocAsync/cudaFreeAsync) instead of the
// synchronizing cudaMalloc/cudaFree pair.
void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

using ::brotensor::Tensor;
using ::brotensor::Dtype;

namespace {

// Current CUDA stream for hot-op launches — so kernels join a non-default
// capture/replay stream instead of silently landing on the default stream.
inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

constexpr int LBB_DX_BLOCK = 64;

template <typename T> __device__ inline float lbb_load(const T* p);
template <> __device__ inline float lbb_load<float>(const float* p)  { return *p; }
template <> __device__ inline float lbb_load<__half>(const __half* p){ return __half2float(*p); }
template <> __device__ inline float lbb_load<__nv_bfloat16>(const __nv_bfloat16* p){ return __bfloat162float(*p); }
template <typename T> __device__ inline void lbb_store(T* p, float v);
template <> __device__ inline void lbb_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void lbb_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void lbb_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__global__ void linear_backward_batched_dx_kernel(const T* __restrict__ W,
                                                  const T* __restrict__ dY,
                                                  T* __restrict__ dX,
                                                  int B, int out_dim, int in_dim) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= in_dim) return;
    const T* dY_row = dY + static_cast<size_t>(b) * out_dim;
    float acc = 0.0f;
    for (int i = 0; i < out_dim; ++i) {
        acc += lbb_load<T>(&W[static_cast<size_t>(i) * in_dim + j]) *
               lbb_load<T>(&dY_row[i]);
    }
    lbb_store<T>(&dX[static_cast<size_t>(b) * in_dim + j], acc);
}

template <typename T>
__global__ void linear_backward_batched_dw_kernel(const T* __restrict__ dY,
                                                  const T* __restrict__ X,
                                                  float* __restrict__ dW_scratch,
                                                  int B, int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += lbb_load<T>(&dY[static_cast<size_t>(b) * out_dim + i]) *
               lbb_load<T>(&X [static_cast<size_t>(b) * in_dim  + j]);
    }
    dW_scratch[static_cast<size_t>(i) * in_dim + j] = acc;
}

template <typename T>
__global__ void linear_backward_batched_db_kernel(const T* __restrict__ dY,
                                                  float* __restrict__ dB_scratch,
                                                  int B, int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += lbb_load<T>(&dY[static_cast<size_t>(b) * out_dim + i]);
    }
    dB_scratch[i] = acc;
}

__global__ void lbb_add_fp32_into_fp16(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}
__global__ void lbb_add_fp32_into_fp32(const float* __restrict__ src,
                                       float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}
__global__ void lbb_add_fp32_into_bf16(const float* __restrict__ src,
                                       __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
}

// CUDA per-launch limit for gridDim.y / gridDim.z on every supported arch.
constexpr int LBB_MAX_GRID_Y = 65535;

} // anonymous namespace

void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD) {
    const bool w16 = (W.dtype == Dtype::FP16 || W.dtype == Dtype::BF16);
    if ((W.dtype != Dtype::FP32 && !w16) || X_BD.dtype != Dtype::FP32 ||
        bias.dtype != Dtype::FP32) {
        throw std::runtime_error(
            "linear_forward_batched: X and bias must be FP32, W FP32/FP16/BF16 "
            "(use linear_forward_batched_fp16 for 16-bit activations)");
    }
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != Dtype::FP32) {
        Y_BD.resize(B, out_dim, Dtype::FP32);
    }
    if (B == 0 || out_dim == 0) return;

    const void* bias_p = bias.size() > 0 ? bias.data : nullptr;
    if (B <= 8 && in_dim > 0) {
        dispatch_mir_gemv(X_BD.data, Dtype::FP32, W.data, W.dtype, Y_BD.data, Dtype::FP32,
                          bias_p, B, out_dim, in_dim, in_dim, in_dim, out_dim, 0, 0);
        return;
    }
    dispatch_mir_gemm_simt(X_BD.data, Dtype::FP32, W.data, W.dtype, Y_BD.data, Dtype::FP32,
                           bias_p, Dtype::FP32, 1, B, out_dim, in_dim,
                           in_dim, in_dim, out_dim, 0, 0, 0,
                           false, false, 0, 0, W.device);
}

void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD,
                             Tensor& dW, Tensor& dB) {
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::FP32 && W.dtype != Dtype::BF16) {
        throw std::runtime_error("linear_backward_batched: W must be FP16, BF16, or FP32");
    }
    if (X_BD.dtype != W.dtype || dY_BD.dtype != W.dtype ||
        dW.dtype != W.dtype || dB.dtype != W.dtype) {
        throw std::runtime_error("linear_backward_batched: all tensors must share dtype");
    }
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;

    if (dX_BD.rows != B || dX_BD.cols != in_dim || dX_BD.dtype != W.dtype) {
        dX_BD.resize(B, in_dim, W.dtype);
    }
    if (B == 0) return;

    const bool is_fp16 = (W.dtype == Dtype::FP16);
    const bool is_bf16 = (W.dtype == Dtype::BF16);

    if (in_dim > 0 && out_dim > 0) {
        dim3 block(LBB_DX_BLOCK, 1);
        const int grid_x = (in_dim + LBB_DX_BLOCK - 1) / LBB_DX_BLOCK;
        for (int b0 = 0; b0 < B; b0 += LBB_MAX_GRID_Y) {
            const int b_chunk = (B - b0) < LBB_MAX_GRID_Y ? (B - b0) : LBB_MAX_GRID_Y;
            dim3 grid(grid_x, b_chunk);
            if (is_fp16) {
                linear_backward_batched_dx_kernel<__half><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const __half*>(W.data),
                    static_cast<const __half*>(dY_BD.data)
                        + static_cast<size_t>(b0) * out_dim,
                    static_cast<__half*>(dX_BD.data)
                        + static_cast<size_t>(b0) * in_dim,
                    b_chunk, out_dim, in_dim);
            } else if (is_bf16) {
                linear_backward_batched_dx_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const __nv_bfloat16*>(W.data),
                    static_cast<const __nv_bfloat16*>(dY_BD.data)
                        + static_cast<size_t>(b0) * out_dim,
                    static_cast<__nv_bfloat16*>(dX_BD.data)
                        + static_cast<size_t>(b0) * in_dim,
                    b_chunk, out_dim, in_dim);
            } else {
                linear_backward_batched_dx_kernel<float><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const float*>(W.data),
                    static_cast<const float*>(dY_BD.data)
                        + static_cast<size_t>(b0) * out_dim,
                    static_cast<float*>(dX_BD.data)
                        + static_cast<size_t>(b0) * in_dim,
                    b_chunk, out_dim, in_dim);
            }
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    if (out_dim > 0 && in_dim > 0) {
        const int dw_n = out_dim * in_dim;
        float* d_dw_scratch = static_cast<float*>(cuda_alloc(static_cast<size_t>(dw_n) * sizeof(float)));
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        if (is_fp16) {
            linear_backward_batched_dw_kernel<__half><<<grid, block, 0, cur_stream()>>>(
                static_cast<const __half*>(dY_BD.data),
                static_cast<const __half*>(X_BD.data),
                d_dw_scratch, B, out_dim, in_dim);
        } else if (is_bf16) {
            linear_backward_batched_dw_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY_BD.data),
                static_cast<const __nv_bfloat16*>(X_BD.data),
                d_dw_scratch, B, out_dim, in_dim);
        } else {
            linear_backward_batched_dw_kernel<float><<<grid, block, 0, cur_stream()>>>(
                static_cast<const float*>(dY_BD.data),
                static_cast<const float*>(X_BD.data),
                d_dw_scratch, B, out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (dw_n + 255) / 256;
        if (is_fp16) {
            lbb_add_fp32_into_fp16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_dw_scratch, static_cast<__half*>(dW.data), dw_n);
        } else if (is_bf16) {
            lbb_add_fp32_into_bf16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_dw_scratch, static_cast<__nv_bfloat16*>(dW.data), dw_n);
        } else {
            lbb_add_fp32_into_fp32<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_dw_scratch, static_cast<float*>(dW.data), dw_n);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cuda_free(d_dw_scratch);
    }

    if (out_dim > 0) {
        float* d_db_scratch = static_cast<float*>(cuda_alloc(static_cast<size_t>(out_dim) * sizeof(float)));
        const int blocks = (out_dim + 255) / 256;
        if (is_fp16) {
            linear_backward_batched_db_kernel<__half><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const __half*>(dY_BD.data),
                d_db_scratch, B, out_dim);
        } else if (is_bf16) {
            linear_backward_batched_db_kernel<__nv_bfloat16><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY_BD.data),
                d_db_scratch, B, out_dim);
        } else {
            linear_backward_batched_db_kernel<float><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const float*>(dY_BD.data),
                d_db_scratch, B, out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (out_dim + 255) / 256;
        if (is_fp16) {
            lbb_add_fp32_into_fp16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_db_scratch, static_cast<__half*>(dB.data), out_dim);
        } else if (is_bf16) {
            lbb_add_fp32_into_bf16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_db_scratch, static_cast<__nv_bfloat16*>(dB.data), out_dim);
        } else {
            lbb_add_fp32_into_fp32<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_db_scratch, static_cast<float*>(dB.data), out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cuda_free(d_db_scratch);
    }
}

} // namespace brotensor::detail::cuda
