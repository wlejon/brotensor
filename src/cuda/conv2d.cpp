// Pure C++ CUDA conv2d forward/backward dispatchers backed by Brass MIR,
// replacing src/cuda/conv2d.cu.

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace conv2d_wmma_internal {
bool launch_conv2d_implicit_gemm_wmma(
        const __half* X, const __half* Wt, const __half* bias, __half* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out);
bool launch_conv2d_implicit_gemm_wmma_bf16(
        const __nv_bfloat16* X, const __nv_bfloat16* Wt,
        const __nv_bfloat16* bias, __nv_bfloat16* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out);
}

namespace detail::cuda::matmul_rm_internal {
bool launch_matmul_rm_wmma(const __half* A, const __half* B, __half* C,
                           int M, int N, int K);
bool launch_matmul_rm_wmma_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                __nv_bfloat16* C, int M, int N, int K);
int rm_wmma_grid_ctas(int M, int N);
}

namespace detail::cuda {

bool conv2d_igemm_f32(const float* X, const float* Wt, const float* bias, float* Y, int N,
                      int Cin, int H, int W, int Cout, int kH, int kW, int sh, int sw, int ph,
                      int pw, int dh, int dw, int Ho, int Wo, cudaStream_t stream);

void dispatch_conv_direct(std::uint32_t mode, Dtype dt,
                          const void* x, const void* w, void* y, const void* bias,
                          int N, int Cin, int T, int H, int Wd,
                          int Cout, int To, int Ho, int Wo,
                          int kT, int kH, int kW,
                          int st, int sh, int sw,
                          int pt, int ph, int pw,
                          int dt_, int dh, int dw, int groups);

namespace {

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

}  // namespace

void conv2d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_forward: X must be FP16, BF16 or FP32");
    }
    if (Wt.dtype != X.dtype) throw std::runtime_error("conv2d_forward: Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) throw std::runtime_error("conv2d_forward: bias dtype must match X");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv2d_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    constexpr int kMinGemmCtas = 128;
    if (groups == 1 && bias == nullptr && kH == 1 && kW == 1 &&
        pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1 &&
        dil_h == 1 && dil_w == 1 &&
        (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) &&
        detail::cuda::matmul_rm_internal::rm_wmma_grid_ctas(C_out, H * W) >= kMinGemmCtas) {
        const int HW = H * W;
        const std::size_t x_stride = static_cast<std::size_t>(C_in) * HW;
        const std::size_t y_stride = static_cast<std::size_t>(C_out) * HW;
        bool taken = true;
        for (int n = 0; n < N && taken; ++n) {
            if (X.dtype == Dtype::FP16) {
                taken = detail::cuda::matmul_rm_internal::launch_matmul_rm_wmma(
                    static_cast<const __half*>(Wt.data),
                    static_cast<const __half*>(X.data) + n * x_stride,
                    static_cast<__half*>(Y.data) + n * y_stride,
                    C_out, HW, C_in);
            } else {
                taken = detail::cuda::matmul_rm_internal::launch_matmul_rm_wmma_bf16(
                    static_cast<const __nv_bfloat16*>(Wt.data),
                    static_cast<const __nv_bfloat16*>(X.data) + n * x_stride,
                    static_cast<__nv_bfloat16*>(Y.data) + n * y_stride,
                    C_out, HW, C_in);
            }
        }
        if (taken) {
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    if (X.dtype == Dtype::FP16) {
        const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;
        if (groups == 1 &&
            conv2d_wmma_internal::launch_conv2d_implicit_gemm_wmma(
                static_cast<const __half*>(X.data), static_cast<const __half*>(Wt.data),
                b_p, static_cast<__half*>(Y.data),
                N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out)) {
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            return;
        }
    } else if (X.dtype == Dtype::BF16) {
        const __nv_bfloat16* b_p = (bias && bias->size() > 0)
            ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr;
        if (groups == 1 &&
            conv2d_wmma_internal::launch_conv2d_implicit_gemm_wmma_bf16(
                static_cast<const __nv_bfloat16*>(X.data), static_cast<const __nv_bfloat16*>(Wt.data),
                b_p, static_cast<__nv_bfloat16*>(Y.data),
                N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out)) {
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            return;
        }
    } else {
        if (groups == 1 &&
            conv2d_igemm_f32(
                static_cast<const float*>(X.data), static_cast<const float*>(Wt.data),
                (bias && bias->size() > 0) ? static_cast<const float*>(bias->data) : nullptr,
                static_cast<float*>(Y.data),
                N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out, stream)) {
            return;
        }
    }
    dispatch_conv_direct(
        0u, X.dtype, X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, H_out, W_out, 1, kH, kW,
        1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           ::brotensor::Tensor& dX) {
    if (Wt.dtype != Dtype::FP16 && Wt.dtype != Dtype::FP32 && Wt.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_input: Wt must be FP16, BF16 or FP32");
    }
    if (dY.dtype != Wt.dtype) throw std::runtime_error("conv2d_backward_input: dY dtype must match Wt");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv2d_backward_input: groups must be >=1 and divide both C_in and C_out");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) throw std::runtime_error("conv2d_backward_input: non-positive output shape");
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    if (N == 0 || in_cols == 0) return;

    dispatch_conv_direct(
        1u, Wt.dtype, dY.data, Wt.data, dX.data, nullptr,
        N, C_out, 1, H_out, W_out, C_in, 1, H, W, 1, kH, kW,
        1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            ::brotensor::Tensor& dWt) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_weight: X must be FP16, BF16 or FP32");
    }
    if (dY.dtype != X.dtype || dWt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_backward_weight: X, dY, dWt dtype must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv2d_backward_weight: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) throw std::runtime_error("conv2d_backward_weight: non-positive output shape");
    if (dWt.rows != C_out || dWt.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight: dWt shape mismatch");
    }
    const int total = C_out * Cg_in * kH * kW;
    if (total == 0) return;

    const bool block_per_elem = (total < 65536);
    CUfunction fn = mir_function({MirKind::ConvAux, block_per_elem ? 0u : 1u, X.dtype});
    const std::uintptr_t ptrs[] = {addr(X.data), addr(dY.data), addr(dWt.data)};
    if (block_per_elem) {
        const std::int32_t ints[] = {
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, Cg_in, Cg_out};
        mir_dispatch_args_3d(fn, static_cast<std::uint32_t>(total), 1u, 1u, 256u, ptrs, ints, {});
    } else {
        const std::int32_t ints[] = {
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, Cg_in, Cg_out, total};
        mir_dispatch_args_3d(fn, static_cast<std::uint32_t>((total + 255) / 256), 1u, 1u, 256u, ptrs, ints, {});
    }
}

void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB) {
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_bias: dY must be FP16, BF16 or FP32");
    }
    if (dB.dtype != dY.dtype) throw std::runtime_error("conv2d_backward_bias: dB dtype must match dY");
    if (dB.rows != C_out || dB.cols != 1) throw std::runtime_error("conv2d_backward_bias: dB shape mismatch");
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    CUfunction fn = mir_function({MirKind::ConvAux, 2u, dY.dtype});
    const std::uintptr_t ptrs[] = {addr(dY.data), addr(dB.data)};
    const std::int32_t ints[] = {N, C_out, H_out, W_out};
    mir_dispatch_args_3d(fn, static_cast<std::uint32_t>(C_out), 1u, 1u, 256u, ptrs, ints, {});
}

void upsample_nearest_2x(const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y);
void upsample_bilinear_2x(const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y);
void downsample_avg_2x(const ::brotensor::Tensor& X, int N, int C, int H, int W, ::brotensor::Tensor& Y);
void upsample_nearest_2x_backward(const ::brotensor::Tensor& dY, int N, int C, int H, int W, ::brotensor::Tensor& dX);
void upsample_bilinear_2x_backward(const ::brotensor::Tensor& dY, int N, int C, int H, int W, ::brotensor::Tensor& dX);
void downsample_avg_2x_backward(const ::brotensor::Tensor& dY, int N, int C, int H, int W, ::brotensor::Tensor& dX);

void fill_cuda_vtable_conv(::brotensor::detail::OpsVTable& v) {
    v.conv2d_forward                = &conv2d_forward;
    v.conv2d_backward_input         = &conv2d_backward_input;
    v.conv2d_backward_weight        = &conv2d_backward_weight;
    v.conv2d_backward_bias          = &conv2d_backward_bias;
    v.upsample_nearest_2x           = &upsample_nearest_2x;
    v.upsample_bilinear_2x          = &upsample_bilinear_2x;
    v.downsample_avg_2x             = &downsample_avg_2x;
    v.upsample_nearest_2x_backward  = &upsample_nearest_2x_backward;
    v.upsample_bilinear_2x_backward = &upsample_bilinear_2x_backward;
    v.downsample_avg_2x_backward    = &downsample_avg_2x_backward;
}

}  // namespace detail::cuda
}  // namespace brotensor
