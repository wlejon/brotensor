#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"
#include "ops_gemm_mir.h"
#include "../kernels/mir_gemm_mma.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace fp16_internal {
void launch_matmul_ABT_batched_impl(const __half* A, const __half* B, __half* C,
                                    int batch, int M, int N, int K,
                                    std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                    const __half* bias, int act);
void launch_matmul_ABT_batched_impl(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                    __nv_bfloat16* C,
                                    int batch, int M, int N, int K,
                                    std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                    const __nv_bfloat16* bias, int act);
}  // namespace fp16_internal

namespace detail::cuda {

namespace {

int cdiv_i(int a, int b) {
    return (a + b - 1) / b;
}

int sm_count_and_major(int& major) {
    static int cached_sms[16] = {0}, cached_major[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    if (cached_sms[dev] == 0) {
        cudaDeviceGetAttribute(&cached_sms[dev], cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&cached_major[dev], cudaDevAttrComputeCapabilityMajor, dev);
        if (cached_sms[dev] <= 0) cached_sms[dev] = 1;
    }
    major = cached_major[dev];
    return cached_sms[dev];
}

std::uint8_t choose_nn_cfg(int m, int n, int sms) {
    const long long tiles0 = static_cast<long long>(cdiv_i(m, 128)) * cdiv_i(n, 128);
    if (tiles0 >= static_cast<long long>(sms)) return 0u;
    const long long tiles1 = static_cast<long long>(cdiv_i(m, 64)) * cdiv_i(n, 128);
    if (tiles1 >= 2LL * sms) return 1u;
    const long long tiles2 = static_cast<long long>(cdiv_i(m, 64)) * cdiv_i(n, 64);
    if (tiles2 >= static_cast<long long>(sms)) return 2u;
    const long long tiles3 = static_cast<long long>(cdiv_i(m, 32)) * cdiv_i(n, 64);
    return (tiles3 > tiles2) ? 3u : 2u;
}

bool launch_nn_mma(Dtype dt, const void* a, const void* b, void* c, int m, int n, int k) {
    if (dt != Dtype::FP16 && dt != Dtype::BF16) return false;
    if (static_cast<std::size_t>(m) * static_cast<std::size_t>(n) < 256u) return false;
    if (k < 16 || (k & 7) != 0 || (n & 7) != 0) return false;
    if ((reinterpret_cast<std::uintptr_t>(a) & 15u) != 0u ||
        (reinterpret_cast<std::uintptr_t>(b) & 15u) != 0u ||
        (reinterpret_cast<std::uintptr_t>(c) & 3u) != 0u) {
        return false;
    }
    int major = 0;
    const int sms = sm_count_and_major(major);
    if (major < 8) return false;

    const std::uint8_t cfg = choose_nn_cfg(m, n, sms);
    const kernels::GemmMmaCfg& tcfg = kernels::kGemmMmaCfgs[cfg];
    const int bm = static_cast<int>(tcfg.bm);
    const int bn = static_cast<int>(tcfg.bn);
    const int gx = cdiv_i(n, bn);
    const int gy = cdiv_i(m, bm);
    const bool aligned = (m % bm == 0) && (n % bn == 0) && (k % 32 == 0);
    const bool swizzle = (gy >= 8 && gx >= 4);
    const std::uint8_t flags = static_cast<std::uint8_t>((aligned ? 1u : 0u) | (swizzle ? 2u : 0u));

    CUfunction fn = mir_function({MirKind::GemmMmaNn, cfg, dt, dt, 1u, dt, flags});
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(a),
        reinterpret_cast<std::uintptr_t>(b),
        reinterpret_cast<std::uintptr_t>(c),
    };
    const std::int32_t ints[] = {m, n, k, gy};
    mir_dispatch_args_3d(fn, static_cast<std::uint32_t>(gx), static_cast<std::uint32_t>(gy), 1u,
                         tcfg.threads, ptrs, ints, {});
    return true;
}

}  // namespace

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
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
            C.data, 0, C.bytes(),
            reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream())));
        return;
    }

    if (!launch_nn_mma(A.dtype, A.data, B.data, C.data, M, N, K)) {
        dispatch_mir_gemm_simt(
            A.data, A.dtype, B.data, B.dtype, C.data, C.dtype,
            nullptr, C.dtype, 1, M, N, K, K, N, N, 0, 0, 0,
            false, true, 0, 0, A.device);
    }
}

namespace matmul_rm_internal {

int rm_wmma_grid_ctas(int M, int N) {
    return cdiv_i(N, 64) * cdiv_i(M, 64);
}

bool launch_matmul_rm_wmma(const __half* A, const __half* B, __half* C,
                           int M, int N, int K) {
    return launch_nn_mma(Dtype::FP16, A, B, C, M, N, K);
}

bool launch_matmul_rm_wmma_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                __nv_bfloat16* C, int M, int N, int K) {
    return launch_nn_mma(Dtype::BF16, A, B, C, M, N, K);
}

}  // namespace matmul_rm_internal

void matmul_abt(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                ::brotensor::Tensor& C,
                int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                const ::brotensor::Tensor* bias, int act) {
    if (A.dtype != B.dtype || A.dtype != C.dtype) {
        throw std::runtime_error("matmul_abt: A, B, C must share dtype");
    }
    if (A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_abt: dtype must be FP16 or BF16");
    }
    if (bias && bias->dtype != A.dtype) {
        throw std::runtime_error("matmul_abt: bias dtype must match operands");
    }
    if (batch <= 0 || M == 0 || N == 0) return;

    if (A.dtype == Dtype::FP16) {
        fp16_internal::launch_matmul_ABT_batched_impl(
            static_cast<const __half*>(A.data),
            static_cast<const __half*>(B.data),
            static_cast<__half*>(C.data),
            batch, M, N, K,
            static_cast<std::size_t>(strideA), static_cast<std::size_t>(strideB),
            static_cast<std::size_t>(strideC),
            bias ? static_cast<const __half*>(bias->data) : nullptr, act);
    } else {
        fp16_internal::launch_matmul_ABT_batched_impl(
            static_cast<const __nv_bfloat16*>(A.data),
            static_cast<const __nv_bfloat16*>(B.data),
            static_cast<__nv_bfloat16*>(C.data),
            batch, M, N, K,
            static_cast<std::size_t>(strideA), static_cast<std::size_t>(strideB),
            static_cast<std::size_t>(strideC),
            bias ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr, act);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

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
void rope_apply_perhead(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                        const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                        ::brotensor::Tensor& Y);
void rope_apply_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& cos_tbl,
                         const ::brotensor::Tensor& sin_tbl,
                         int head_dim, int num_heads, ::brotensor::Tensor& dX);
void softmax_forward(const ::brotensor::Tensor& logits,
                     ::brotensor::Tensor& probs, const float* mask);
void softmax_rows_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y,
                          int rows, int cols);
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
void rows_count_above(const ::brotensor::Tensor& X, float t_lo, float t_hi,
                      ::brotensor::Tensor& counts);
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
void nchw_to_sequence(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);
void sequence_to_nchw(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);

void fill_cuda_vtable_utils(::brotensor::detail::OpsVTable& v) {
    v.matmul                              = &matmul;
    v.matmul_abt                          = &matmul_abt;
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
    v.rope_apply_perhead                  = &rope_apply_perhead;
    v.rope_apply_backward                 = &rope_apply_backward;
    v.softmax_forward                     = &softmax_forward;
    v.softmax_rows_forward                = &softmax_rows_forward;
    v.softmax_backward                    = &softmax_backward;
    v.sgd_step                            = &sgd_step;
    v.adam_step                           = &adam_step;
    v.sum_rows                            = &sum_rows;
    v.sum_cols                            = &sum_cols;
    v.argmax_rows                         = &argmax_rows;
    v.rows_count_above                    = &rows_count_above;
    v.ddim_step                           = &ddim_step;
    v.euler_step                          = &euler_step;
    v.dpmpp_2m_step                       = &dpmpp_2m_step;
    v.timestep_embedding                  = &timestep_embedding;
    v.nchw_to_sequence                    = &nchw_to_sequence;
    v.sequence_to_nchw                    = &sequence_to_nchw;
}

}  // namespace detail::cuda
}  // namespace brotensor
