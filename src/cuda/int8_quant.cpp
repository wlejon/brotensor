// CUDA INT8 weight-only quantisation (W8A16) — pure C++ MIR dispatcher.

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor {
void* cuda_current_stream();
}

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

namespace linear_int8w_wmma_internal {
bool launch_linear_int8w_fp16_wmma(
        const __half* X, const int8_t* W, const float* scales,
        const __half* bias, __half* Y,
        int B, int M, int K);
bool launch_linear_int8w_bf16_wmma(
        const __nv_bfloat16* X, const int8_t* W, const float* scales,
        const __nv_bfloat16* bias, __nv_bfloat16* Y,
        int B, int M, int K);
}

void gemv_int8w_cuda(const Tensor& W_int8, const Tensor& scales, const Tensor* bias,
                     const Tensor& X, Tensor& Y);

namespace conv2d_int8w_wmma_internal {
bool launch_conv2d_int8w_implicit_gemm_wmma(
        const __half* X, const int8_t* W_int8, const float* scales,
        const __half* bias, __half* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out);
}

namespace {

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

}  // namespace

void matmul_int8w_fp16(const Tensor& W_int8,
                       const Tensor& scales,
                       const Tensor& X,
                       Tensor& Y) {
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("matmul_int8w_fp16: W_int8 must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("matmul_int8w_fp16: scales must be FP32");
    }
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("matmul_int8w_fp16: X must be FP16");
    }
    const int M = W_int8.rows;
    const int K = W_int8.cols;
    if (X.rows != K) {
        throw std::runtime_error("matmul_int8w_fp16: K mismatch (W.cols != X.rows)");
    }
    if (scales.rows != M || scales.cols != 1) {
        throw std::runtime_error("matmul_int8w_fp16: scales shape must be (out, 1)");
    }
    const int Nb = X.cols;
    if (Y.rows != M || Y.cols != Nb || Y.dtype != Dtype::FP16) {
        Y.resize(M, Nb, Dtype::FP16);
    }
    if (M == 0 || Nb == 0) return;
    if (K == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(Y.data, 0, Y.bytes(), cur_stream()));
        return;
    }

    CUfunction fn = mir_function({MirKind::Int8QuantAux, 0u, Dtype::FP16});
    const std::uintptr_t ptrs[] = {addr(W_int8.data), addr(scales.data), addr(X.data), addr(Y.data)};
    const std::int32_t ints[] = {M, Nb, K};
    mir_dispatch_args_3d(
        fn, static_cast<std::uint32_t>((Nb + 15) / 16), static_cast<std::uint32_t>((M + 15) / 16), 1u,
        256u, ptrs, ints, {});
}

void linear_forward_batched_int8w_fp16(const Tensor& W_int8,
                                       const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD,
                                       Tensor& Y_BD) {
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: scales must be FP32");
    }
    const Dtype dt = X_BD.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: X must be FP16 or BF16");
    }
    if (bias && bias->dtype != dt) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: bias dtype must match X");
    }
    const int B = X_BD.rows;
    const int in_dim = X_BD.cols;
    const int out_dim = W_int8.rows;
    if (W_int8.cols != in_dim) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (scales.rows != out_dim || scales.cols != 1) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: scales shape must be (out, 1)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != dt) {
        Y_BD.resize(B, out_dim, dt);
    }
    if (B == 0 || out_dim == 0) return;
    if (in_dim == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(Y_BD.data, 0, Y_BD.bytes(), cur_stream()));
        return;
    }
    const void* b_p = (bias && bias->size() > 0) ? bias->data : nullptr;
    const int8_t* w_p = static_cast<const int8_t*>(W_int8.data);
    const float* s_p = static_cast<const float*>(scales.data);

    const bool mma_hit = (dt == Dtype::FP16)
        ? linear_int8w_wmma_internal::launch_linear_int8w_fp16_wmma(
              static_cast<const __half*>(X_BD.data), w_p, s_p,
              static_cast<const __half*>(b_p),
              static_cast<__half*>(Y_BD.data),
              B, out_dim, in_dim)
        : linear_int8w_wmma_internal::launch_linear_int8w_bf16_wmma(
              static_cast<const __nv_bfloat16*>(X_BD.data), w_p, s_p,
              static_cast<const __nv_bfloat16*>(b_p),
              static_cast<__nv_bfloat16*>(Y_BD.data),
              B, out_dim, in_dim);
    if (mma_hit) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    gemv_int8w_cuda(W_int8, scales, bias, X_BD, Y_BD);
}

void conv2d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8,
                               const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups,
                               Tensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: X must be FP16");
    }
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: scales must be FP32");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_int8w_fp16_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    if (W_int8.rows != C_out || W_int8.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: W shape mismatch");
    }
    if (scales.rows != C_out || scales.cols != 1) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: scales shape mismatch");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;
    if (groups == 1 && dil_h == 1 && dil_w == 1 &&
        conv2d_int8w_wmma_internal::launch_conv2d_int8w_implicit_gemm_wmma(
            static_cast<const __half*>(X.data),
            static_cast<const int8_t*>(W_int8.data),
            static_cast<const float*>(scales.data),
            b_p,
            static_cast<__half*>(Y.data),
            N, C_in, H, W, C_out, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            H_out, W_out)) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::uint8_t flags = (b_p != nullptr) ? 1u : 0u;
    CUfunction fn = mir_function({MirKind::Int8QuantAux, 1u, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint32_t gx = static_cast<std::uint32_t>((total + 255) / 256);
    const std::uintptr_t ptrs[] = {addr(X.data), addr(W_int8.data), addr(scales.data), addr(b_p), addr(Y.data)};
    const std::int32_t ints[] = {
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        Cg_in, Cg_out, total};
    mir_dispatch_args_3d(fn, gx, 1u, 1u, 256u, ptrs, ints, {});
}

}  // namespace brotensor::detail::cuda
