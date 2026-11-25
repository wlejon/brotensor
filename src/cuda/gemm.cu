#include <brotensor/runtime.h>
#include "detail/cuda_check.h"
#include "detail/activations.cuh"

#include "fp16_internal.cuh"
#include "ops_gemm_mir.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor {
namespace detail::cuda {

namespace {

template <typename T> __device__ inline float g_load(const T* p);
template <> __device__ inline float g_load<float>(const float* p) { return *p; }
template <> __device__ inline float g_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float g_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void g_store(T* p, float v);
template <> __device__ inline void g_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void g_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void g_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// dX[j] = sum_i W[i, j] * dY[i]
// One thread per input column. Uses shared memory to cache tiles of dY (FP32).
constexpr int LB_DX_BLOCK = 128;
constexpr int LB_DX_TILE  = 128;

template <typename T>
__global__ void linear_backward_dx_kernel(const T* __restrict__ W,
                                          const T* __restrict__ dY,
                                          T* __restrict__ dX,
                                          int out_dim, int in_dim) {
    __shared__ float dytile[LB_DX_TILE];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int t0 = 0; t0 < out_dim; t0 += LB_DX_TILE) {
        const int t_len = (out_dim - t0) < LB_DX_TILE ? (out_dim - t0) : LB_DX_TILE;

        for (int k = tid; k < t_len; k += blockDim.x) {
            dytile[k] = g_load<T>(&dY[t0 + k]);
        }
        __syncthreads();

        if (col < in_dim) {
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += g_load<T>(&W[static_cast<size_t>(t0 + k) * in_dim + col]) * dytile[k];
            }
        }
        __syncthreads();
    }

    if (col < in_dim) {
        g_store<T>(&dX[col], acc);
    }
}

// dW[i, j] += dY[i] * x[j]. 2D grid: each thread one (i, j).
template <typename T>
__global__ void linear_backward_dw_kernel(const T* __restrict__ dY,
                                          const T* __restrict__ x,
                                          T* __restrict__ dW,
                                          int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    const size_t off = static_cast<size_t>(i) * in_dim + j;
    const float prev = g_load<T>(&dW[off]);
    const float upd  = g_load<T>(&dY[i]) * g_load<T>(&x[j]);
    g_store<T>(&dW[off], prev + upd);
}

// dB[i] += dY[i].
template <typename T>
__global__ void linear_backward_db_kernel(const T* __restrict__ dY,
                                          T* __restrict__ dB,
                                          int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    g_store<T>(&dB[i], g_load<T>(&dB[i]) + g_load<T>(&dY[i]));
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

inline void require_same_dtype(const ::brotensor::Tensor& a, const ::brotensor::Tensor& b,
                               const char* op) {
    if (a.dtype != b.dtype) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": all FP operand dtypes must match");
    }
}

} // anonymous namespace

void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp(W, "linear_forward", "W");
    require_same_dtype(W, b, "linear_forward");
    require_same_dtype(W, x, "linear_forward");
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (y.rows != out_dim || y.cols != 1 || y.dtype != W.dtype) {
        y.resize(out_dim, 1, W.dtype);
    }
    if (out_dim == 0) return;

    dispatch_mir_gemv(x.data, x.dtype, W.data, W.dtype, y.data, y.dtype, b.data,
                      1, out_dim, in_dim, in_dim, in_dim, out_dim, 0, 0);
}

// 16-bit batched linear forward: Y(B, out_dim) = X(B, in_dim) @ W(out_dim, in_dim)^T
// + optional broadcast bias, with an optional activation. Same matmul kernel as
// cross-attention's matmul_ABT — X is the (M=B, K=in_dim) side, W is the
// (N=out_dim, K=in_dim) side. Bias and activation are fused into the GEMM's
// output-store stage (no separate bias-add / activation launches, no extra HBM
// round-trips over Y). Dispatches FP16 / BF16 on the operand dtype — kept under
// the historical `_fp16` op name (ABI), but BF16 storage is fully supported.
namespace {
void linear_forward_batched_fp16_impl(const ::brotensor::Tensor& W,
                                      const ::brotensor::Tensor* bias,
                                      const ::brotensor::Tensor& X_BD,
                                      int act,
                                      ::brotensor::Tensor& Y_BD) {
    const Dtype dt = X_BD.dtype;
    if ((dt != Dtype::FP16 && dt != Dtype::BF16) || W.dtype != dt) {
        throw std::runtime_error(
            "linear_forward_batched_fp16: W and X must both be FP16 or both BF16");
    }
    if (bias && bias->dtype != dt) {
        throw std::runtime_error(
            "linear_forward_batched_fp16: bias dtype must match X/W");
    }
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("linear_forward_batched_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != dt) {
        Y_BD.resize(B, out_dim, dt);
    }
    if (B == 0 || out_dim == 0) return;

    const bool has_bias = bias && bias->size() > 0;

    // Skinny batches take the block-per-row split-K GEMV — fully coalesced
    // weight traffic, FP32 accumulation, bias + activation fused into the
    // store (the same dispatch shape as the FP32-activation batched linear).
    // The WMMA tile kernel keeps the wider-batch case.
    //
    // Threshold is B <= 4, NOT the batch's whole small end: this GEMV grid is
    // (out_dim, B) — one block per (output, row) — so it re-reads the entire
    // weight matrix B times. It runs at near-peak bandwidth, but that B× traffic
    // means its wall time grows linearly with B, while the WMMA tile kernel reads
    // the weights once (its time is ~flat in B for the short-sequence regime).
    // Measured on a 4090 (Llama-8B-shaped FP16 linears): GEMV ≈ 1.08 TB/s but B×
    // traffic; WMMA ≈ 0.24 TB/s at 1× traffic — so GEMV only wins for B ≲ 4.
    // At the old B <= 32 cutoff an L=32 prefill did 32× the weight traffic and
    // ran ~6.5× SLOWER than L=64 (101 ms vs 15 ms for an 8-layer stack) — a cliff
    // that hit every short prompt / bidirectional-encoder forward. B <= 4 keeps
    // GEMV only where its coalesced single-pass-per-row actually beats WMMA
    // (decode's B == 1, and 2–4-row micro-batches).
    if (B <= 4 && in_dim > 0) {
        dispatch_mir_gemv(X_BD.data, dt, W.data, dt, Y_BD.data, dt,
                          has_bias ? bias->data : nullptr,
                          B, out_dim, in_dim, in_dim, in_dim, out_dim, 0, act);
        return;
    }

    if (dt == Dtype::FP16) {
        fp16_internal::launch_matmul_ABT_act(
            static_cast<const __half*>(X_BD.data),
            static_cast<const __half*>(W.data),
            static_cast<__half*>(Y_BD.data),
            B, out_dim, in_dim,
            has_bias ? static_cast<const __half*>(bias->data) : nullptr, act);
    } else {  // BF16
        fp16_internal::launch_matmul_ABT_act(
            static_cast<const __nv_bfloat16*>(X_BD.data),
            static_cast<const __nv_bfloat16*>(W.data),
            static_cast<__nv_bfloat16*>(Y_BD.data),
            B, out_dim, in_dim,
            has_bias ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr, act);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}
} // namespace

void linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                 const ::brotensor::Tensor* bias,
                                 const ::brotensor::Tensor& X_BD,
                                 ::brotensor::Tensor& Y_BD) {
    linear_forward_batched_fp16_impl(W, bias, X_BD, /*act=*/0, Y_BD);
}

void linear_forward_batched_fp16_act(const ::brotensor::Tensor& W,
                                     const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD,
                                     int act,
                                     ::brotensor::Tensor& Y_BD) {
    linear_forward_batched_fp16_impl(W, bias, X_BD, act, Y_BD);
}

void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB) {
    require_fp(W, "linear_backward", "W");
    require_same_dtype(W, x,  "linear_backward");
    require_same_dtype(W, dY, "linear_backward");
    require_same_dtype(W, dW, "linear_backward");
    require_same_dtype(W, dB, "linear_backward");
    const int out_dim = W.rows;
    const int in_dim  = W.cols;

    if (dX.rows != in_dim || dX.cols != 1 || dX.dtype != W.dtype) {
        dX.resize(in_dim, 1, W.dtype);
    }

    // dX = W^T * dY (overwrite)
    if (in_dim > 0) {
        const int blocks = (in_dim + LB_DX_BLOCK - 1) / LB_DX_BLOCK;
        if (W.dtype == Dtype::FP16) {
            linear_backward_dx_kernel<__half><<<blocks, LB_DX_BLOCK, 0, cur_stream()>>>(
                static_cast<const __half*>(W.data),
                static_cast<const __half*>(dY.data),
                static_cast<__half*>(dX.data),
                out_dim, in_dim);
        } else if (W.dtype == Dtype::BF16) {
            linear_backward_dx_kernel<__nv_bfloat16><<<blocks, LB_DX_BLOCK, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(W.data),
                static_cast<const __nv_bfloat16*>(dY.data),
                static_cast<__nv_bfloat16*>(dX.data),
                out_dim, in_dim);
        } else {
            linear_backward_dx_kernel<float><<<blocks, LB_DX_BLOCK, 0, cur_stream()>>>(
                static_cast<const float*>(W.data),
                static_cast<const float*>(dY.data),
                static_cast<float*>(dX.data),
                out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    // dW += dY * x^T
    if (out_dim > 0 && in_dim > 0) {
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        if (W.dtype == Dtype::FP16) {
            linear_backward_dw_kernel<__half><<<grid, block, 0, cur_stream()>>>(
                static_cast<const __half*>(dY.data),
                static_cast<const __half*>(x.data),
                static_cast<__half*>(dW.data),
                out_dim, in_dim);
        } else if (W.dtype == Dtype::BF16) {
            linear_backward_dw_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY.data),
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<__nv_bfloat16*>(dW.data),
                out_dim, in_dim);
        } else {
            linear_backward_dw_kernel<float><<<grid, block, 0, cur_stream()>>>(
                static_cast<const float*>(dY.data),
                static_cast<const float*>(x.data),
                static_cast<float*>(dW.data),
                out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    // dB += dY
    if (out_dim > 0) {
        const int blocks = (out_dim + 255) / 256;
        if (W.dtype == Dtype::FP16) {
            linear_backward_db_kernel<__half><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const __half*>(dY.data),
                static_cast<__half*>(dB.data), out_dim);
        } else if (W.dtype == Dtype::BF16) {
            linear_backward_db_kernel<__nv_bfloat16><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY.data),
                static_cast<__nv_bfloat16*>(dB.data), out_dim);
        } else {
            linear_backward_db_kernel<float><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const float*>(dY.data),
                static_cast<float*>(dB.data), out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace detail::cuda
} // namespace brotensor
