// Public matmul: row-major C(M,N) = A(M,K) @ B(K,N), no bias.
// Dispatched on A.dtype (FP32 + FP16). FP32 accumulation throughout.
//
// The existing internal kernels (`gemm.cu`'s `linear_forward_kernel` and
// `fp16_matmul.cu`'s `launch_matmul_ABT`) compute Y = X @ W^T, which is the
// wrong layout for a plain row-major matmul: B is stored row-major as
// (K, N) but we want to read its columns as the inner dimension. We therefore
// provide direct (M,K) @ (K,N) kernels here.

#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor {

// Forward decl: thread-local current stream from runtime.cu.
void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr int MM_TILE = 16;

// 2D tiled GEMM, naive but tiled. One thread per output element; cooperative
// tile loads of A and B into shared memory.
__global__ void matmul_fp32_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    const int row = blockIdx.y * MM_TILE + threadIdx.y;
    const int col = blockIdx.x * MM_TILE + threadIdx.x;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * MM_TILE + threadIdx.x;
        const int b_row = t * MM_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

__global__ void matmul_fp16_kernel(const __half* __restrict__ A,
                                   const __half* __restrict__ B,
                                   __half* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    const int row = blockIdx.y * MM_TILE + threadIdx.y;
    const int col = blockIdx.x * MM_TILE + threadIdx.x;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * MM_TILE + threadIdx.x;
        const int b_row = t * MM_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __half2float(A[row * K + a_col]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __half2float(B[b_row * N + col]) : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

__global__ void matmul_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   __nv_bfloat16* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    const int row = blockIdx.y * MM_TILE + threadIdx.y;
    const int col = blockIdx.x * MM_TILE + threadIdx.x;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * MM_TILE + threadIdx.x;
        const int b_row = t * MM_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __bfloat162float(A[row * K + a_col]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __bfloat162float(B[b_row * N + col]) : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

} // namespace

void matmul(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
            ::brotensor::Tensor& C) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("matmul: A and B must share dtype");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (C.rows != M || C.cols != N || C.dtype != A.dtype) {
        C.resize(M, N, A.dtype);
    }
    if (M == 0 || N == 0) return;
    if (K == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(C.data, 0, C.bytes()));
        return;
    }

    dim3 block(MM_TILE, MM_TILE);
    dim3 grid((N + MM_TILE - 1) / MM_TILE, (M + MM_TILE - 1) / MM_TILE);
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (A.dtype == Dtype::FP16) {
        matmul_fp16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half*>(A.data),
            static_cast<const __half*>(B.data),
            static_cast<__half*>(C.data),
            M, N, K);
    } else if (A.dtype == Dtype::BF16) {
        matmul_bf16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(A.data),
            static_cast<const __nv_bfloat16*>(B.data),
            static_cast<__nv_bfloat16*>(C.data),
            M, N, K);
    } else {
        matmul_fp32_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(A.data),
            static_cast<const float*>(B.data),
            static_cast<float*>(C.data),
            M, N, K);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor

// ─── Vtable contribution ────────────────────────────────────────────────────
//
// Filled into the CUDA OpsVTable from the per-cluster registration entry in
// src/cuda/register.cu (Phase 2G). Includes every op in this cluster (matmul +
// utilities) as well as the slots whose implementations live in sibling files
// (linear_forward_batched, linear_backward_batched) — those are owned by
// other Phase 2 agents but share the `detail::cuda` namespace.

namespace brotensor::detail::cuda {

// Forward decls — implementations in sibling .cu files within this namespace.
void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB);
void linear_forward_batched(const ::brotensor::Tensor& W,
                            const ::brotensor::Tensor& bias,
                            const ::brotensor::Tensor& X_BD,
                            ::brotensor::Tensor& Y_BD);
void linear_backward_batched(const ::brotensor::Tensor& W,
                             const ::brotensor::Tensor& X_BD,
                             const ::brotensor::Tensor& dY_BD,
                             ::brotensor::Tensor& dX_BD,
                             ::brotensor::Tensor& dW,
                             ::brotensor::Tensor& dB);
void linear_forward_batched_fp16_act(const ::brotensor::Tensor& W,
                                     const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD,
                                     int act,
                                     ::brotensor::Tensor& Y_BD);
void linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                 const ::brotensor::Tensor* bias,
                                 const ::brotensor::Tensor& X_BD,
                                 ::brotensor::Tensor& Y_BD);
void linear_forward_batched_int8w_fp16(const ::brotensor::Tensor& W_int8,
                                       const ::brotensor::Tensor& scales,
                                       const ::brotensor::Tensor* bias,
                                       const ::brotensor::Tensor& X_BD,
                                       ::brotensor::Tensor& Y_BD);
void matmul_int8w_fp16(const ::brotensor::Tensor& W_int8,
                       const ::brotensor::Tensor& scales,
                       const ::brotensor::Tensor& X,
                       ::brotensor::Tensor& Y);
void matmul_backward(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                     const ::brotensor::Tensor& dC,
                     ::brotensor::Tensor& dA, ::brotensor::Tensor& dB);
void rope_forward(const ::brotensor::Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, ::brotensor::Tensor& Y);
void rope_backward(const ::brotensor::Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, ::brotensor::Tensor& dX);
void rope_apply(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                ::brotensor::Tensor& Y);
void rope_apply_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& cos_tbl,
                         const ::brotensor::Tensor& sin_tbl,
                         int head_dim, int num_heads, ::brotensor::Tensor& dX);
void softmax_forward(const ::brotensor::Tensor& logits,
                     ::brotensor::Tensor& probs, const float* mask);
void softmax_backward(const ::brotensor::Tensor& probs,
                      const ::brotensor::Tensor& dProbs,
                      ::brotensor::Tensor& dLogits);
void sgd_step(::brotensor::Tensor& param, ::brotensor::Tensor& grad,
              ::brotensor::Tensor& velocity, float lr, float momentum);
void adam_step(::brotensor::Tensor& param, const ::brotensor::Tensor& grad,
               ::brotensor::Tensor& m, ::brotensor::Tensor& v,
               float lr, float beta1, float beta2, float eps, int step);
void sum_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void sum_cols(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void argmax_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx);
void ddim_step(const ::brotensor::Tensor& x_t,
               const ::brotensor::Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               ::brotensor::Tensor& x_prev);
void euler_step(const ::brotensor::Tensor& x_t,
                const ::brotensor::Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                ::brotensor::Tensor& x_prev);
void dpmpp_2m_step(const ::brotensor::Tensor& x_t,
                   const ::brotensor::Tensor& eps_pred,
                   const ::brotensor::Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   ::brotensor::Tensor& x_prev, ::brotensor::Tensor& x0_out);
void timestep_embedding(const ::brotensor::Tensor& timesteps,
                        int dim, float max_period, ::brotensor::Tensor& Y);
// Defined in transpose.cu.
void nchw_to_sequence(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);
void sequence_to_nchw(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);

void fill_cuda_vtable_utils(::brotensor::detail::OpsVTable& v) {
    v.matmul                              = &matmul;
    v.matmul_backward                     = &matmul_backward;
    v.linear_forward                      = &linear_forward;
    v.linear_backward                     = &linear_backward;
    v.linear_forward_batched              = &linear_forward_batched;
    v.linear_backward_batched             = &linear_backward_batched;
    v.linear_forward_batched_fp16         = &linear_forward_batched_fp16;
    v.linear_forward_batched_fp16_act     = &linear_forward_batched_fp16_act;
    v.linear_forward_batched_int8w_fp16   = &linear_forward_batched_int8w_fp16;
    v.matmul_int8w_fp16                   = &matmul_int8w_fp16;
    v.rope_forward                        = &rope_forward;
    v.rope_backward                       = &rope_backward;
    v.rope_apply                          = &rope_apply;
    v.rope_apply_backward                 = &rope_apply_backward;
    v.softmax_forward                     = &softmax_forward;
    v.softmax_backward                    = &softmax_backward;
    v.sgd_step                            = &sgd_step;
    v.adam_step                           = &adam_step;
    v.sum_rows                            = &sum_rows;
    v.sum_cols                            = &sum_cols;
    v.argmax_rows                         = &argmax_rows;
    v.ddim_step                           = &ddim_step;
    v.euler_step                          = &euler_step;
    v.dpmpp_2m_step                       = &dpmpp_2m_step;
    v.timestep_embedding                  = &timestep_embedding;
    v.nchw_to_sequence                    = &nchw_to_sequence;
    v.sequence_to_nchw                    = &sequence_to_nchw;
}

} // namespace brotensor::detail::cuda
