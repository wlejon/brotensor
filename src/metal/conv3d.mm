// Metal conv3d_forward (GEMM when kernel covers full input, else MirKind::ConvDirect)
// and conv3d_int8w_fp16_forward (W8A16 patch embed).

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>

#include <cstdint>
#include <stdexcept>

#import "fp16_matmul.h"
#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

void conv3d_forward(const Tensor& X,
                    const Tensor& Wt,
                    const Tensor* bias,
                    int N, int C_in, int T_in, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w,
                    int groups,
                    Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv3d_forward: X must be FP16, BF16, or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: bias dtype must match X");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv3d_forward: groups must be >=1 and divide both C_in and C_out");
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

    // Patch-embedding fast path (kernel covers whole input volume): single GEMM.
    if (groups == 1 && kT == T_in && kH == H && kW == W &&
        pad_t == 0 && pad_h == 0 && pad_w == 0 &&
        dil_t == 1 && dil_h == 1 && dil_w == 1) {
        const int K = C_in * kT * kH * kW;
        metal_impl::AbtMixed g;
        g.A = buffer_for(X);  g.ofs_A = buffer_offset_for(X);  g.lda = static_cast<uint64_t>(K);
        g.B = buffer_for(Wt); g.ofs_B = buffer_offset_for(Wt); g.ldb = static_cast<uint64_t>(K);
        g.C = buffer_for(Y);  g.ofs_C = buffer_offset_for(Y);  g.ldc = static_cast<uint64_t>(C_out);
        if (bias) {
            g.bias = buffer_for(*bias);
            g.ofs_bias = buffer_offset_for(*bias);
        }
        g.M = N; g.N = C_out; g.K = K;
        g.in = g.out = metal_impl::abt_type(X.dtype);
        metal_impl::launch_matmul_abt_mixed(g);
        return;
    }

    mir_dispatch_conv_direct(
        0u, X.dtype, "conv3d_forward",
        X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, T_in, H, W, C_out, T_out, H_out, W_out,
        kT, kH, kW, stride_t, stride_h, stride_w, pad_t, pad_h, pad_w,
        dil_t, dil_h, dil_w, groups);
}

void conv3d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8,
                               const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int T_in, int H, int W,
                               int C_out, int kT, int kH, int kW,
                               int stride_t, int stride_h, int stride_w,
                               int pad_t, int pad_h, int pad_w,
                               int dil_t, int dil_h, int dil_w, int groups,
                               Tensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: X must be FP16");
    }
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: scales must be FP32");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv3d_int8w_fp16_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
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
    const std::uint32_t total = static_cast<std::uint32_t>(N) * static_cast<std::uint32_t>(out_cols);
    if (total == 0) return;

    const void* b_p = (bias && bias->size() > 0) ? bias->data : nullptr;
    const MirKernel mk = mir_kernel(
        {MirKind::ConvAux, 3u, Dtype::FP16, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(b_p != nullptr ? 1u : 0u)});
    const void* ptrs[] = {X.data, W_int8.data, scales.data, b_p, Y.data};
    const std::int32_t ints[] = {
        N, C_in, T_in, H, W, C_out, kT, kH, kW, T_out, H_out, W_out,
        stride_t, stride_h, stride_w, pad_t, pad_h, pad_w,
        dil_t, dil_h, dil_w, Cg_in, Cg_out, static_cast<std::int32_t>(total),
    };
    const std::uint32_t gx = static_cast<std::uint32_t>((static_cast<std::uint64_t>(total) + 255u) / 256u);
    mir_dispatch_args_3d(mk, gx, 1u, 1u, "conv3d_int8w_fp16_forward", ptrs, ints, {});
}

}  // namespace brotensor::detail::metal
