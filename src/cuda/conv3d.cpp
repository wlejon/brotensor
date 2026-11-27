// Pure C++ CUDA conv3d_forward + conv3d_int8w_fp16_forward backed by Brass MIR,
// replacing src/cuda/conv3d.cu.

#include "cu_mir_cache.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

void dispatch_conv_direct(std::uint32_t mode, Dtype dt,
                          const void* x, const void* w, void* y, const void* bias,
                          int N, int Cin, int T, int H, int Wd,
                          int Cout, int To, int Ho, int Wo,
                          int kT, int kH, int kW,
                          int st, int sh, int sw,
                          int pt, int ph, int pw,
                          int dt_, int dh, int dw, int groups);

void conv3d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int T_in, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv3d_forward: X must be FP16, BF16 or FP32");
    }
    if (Wt.dtype != X.dtype) throw std::runtime_error("conv3d_forward: Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) throw std::runtime_error("conv3d_forward: bias dtype must match X");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv3d_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_forward: non-positive output shape");
    }
    const int out_cols = C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || out_cols == 0) return;

    dispatch_conv_direct(
        0u, X.dtype, X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, T_in, H, W, C_out, T_out, H_out, W_out, kT, kH, kW,
        stride_t, stride_h, stride_w, pad_t, pad_h, pad_w, dil_t, dil_h, dil_w, groups);
}

void conv3d_int8w_fp16_forward(const ::brotensor::Tensor& X,
                               const ::brotensor::Tensor& W_int8,
                               const ::brotensor::Tensor& scales,
                               const ::brotensor::Tensor* bias,
                               int N, int C_in, int T_in, int H, int W,
                               int C_out, int kT, int kH, int kW,
                               int stride_t, int stride_h, int stride_w,
                               int pad_t, int pad_h, int pad_w,
                               int dil_t, int dil_h, int dil_w, int groups,
                               ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP16) throw std::runtime_error("conv3d_int8w_fp16_forward: X must be FP16");
    if (W_int8.dtype != Dtype::INT8) throw std::runtime_error("conv3d_int8w_fp16_forward: W must be INT8");
    if (scales.dtype != Dtype::FP32) throw std::runtime_error("conv3d_int8w_fp16_forward: scales must be FP32");
    if (bias && bias->dtype != Dtype::FP16) throw std::runtime_error("conv3d_int8w_fp16_forward: bias must be FP16");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in = C_in / groups;
    const int Cg_out = C_out / groups;
    if (W_int8.rows != C_out || W_int8.cols != Cg_in * kT * kH * kW) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: W shape mismatch");
    }
    if (scales.rows != C_out || scales.cols != 1) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: scales shape mismatch");
    }
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: non-positive output shape");
    }
    const int out_cols = C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    const void* b_p = (bias && bias->size() > 0) ? bias->data : nullptr;
    CUfunction fn = mir_function(
        {MirKind::ConvAux, 3u, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(b_p != nullptr ? 1u : 0u)});
    const std::uintptr_t ptrs[] = {
        reinterpret_cast<std::uintptr_t>(X.data),
        reinterpret_cast<std::uintptr_t>(W_int8.data),
        reinterpret_cast<std::uintptr_t>(scales.data),
        reinterpret_cast<std::uintptr_t>(b_p),
        reinterpret_cast<std::uintptr_t>(Y.data),
    };
    const std::int32_t ints[] = {
        N, C_in, T_in, H, W, C_out, kT, kH, kW, T_out, H_out, W_out,
        stride_t, stride_h, stride_w, pad_t, pad_h, pad_w,
        dil_t, dil_h, dil_w, Cg_in, Cg_out, total,
    };
    mir_dispatch_args_3d(fn, static_cast<std::uint32_t>((total + 255) / 256), 1u, 1u, 256u, ptrs, ints, {});
}

void fill_cuda_vtable_conv3d(::brotensor::detail::OpsVTable& v) {
    v.conv3d_forward            = &conv3d_forward;
    v.conv3d_int8w_fp16_forward = &conv3d_int8w_fp16_forward;
}

}  // namespace brotensor::detail::cuda
