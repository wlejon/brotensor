// Metal W8A16 (INT8 weight, FP16/BF16 activation) matmul, batched linear, and
// conv2d — backed by MirKind::{GemvQ, WqRows} + Metal simdgroup GEMM / Conv2D.

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

void dequantize_rows(const Tensor& Wq, const Tensor& scales, Dtype out, Tensor& W);
void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, Tensor& Y);

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out) {
    if (out <= 0 || in <= 0) {
        for (int r = 0; r < out; ++r) scales_out[r] = 0.0f;
        return;
    }
    for (int r = 0; r < out; ++r) {
        const uint16_t* row = W_fp16 + static_cast<size_t>(r) * static_cast<size_t>(in);
        float amax = 0.0f;
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            const float a = std::fabs(v);
            if (a > amax) amax = a;
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        scales_out[r] = scale;
        int8_t* dst = W_int8_out + static_cast<size_t>(r) * static_cast<size_t>(in);
        for (int c = 0; c < in; ++c) {
            const float v = fp16_bits_to_fp32(row[c]);
            int q = static_cast<int>(std::lrint(v * inv));
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            dst[c] = static_cast<int8_t>(q);
        }
    }
}

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
        Y.zero();
        return;
    }

    Tensor Wf = Tensor::empty_on(Device::Metal, M, K, Dtype::FP32);
    dequantize_rows(W_int8, scales, Dtype::FP32, Wf);
    metal_impl::AbtMixed g;
    g.A = buffer_for(Wf); g.ofs_A = buffer_offset_for(Wf); g.lda = static_cast<uint64_t>(K);
    g.B = buffer_for(X);  g.ofs_B = buffer_offset_for(X);  g.ldb = static_cast<uint64_t>(Nb);
    g.transB = true;
    g.C = buffer_for(Y);  g.ofs_C = buffer_offset_for(Y);  g.ldc = static_cast<uint64_t>(Nb);
    g.M = M; g.N = Nb; g.K = K;
    g.in = metal_impl::kAbtF32;
    g.in_B = metal_impl::kAbtF16;
    g.out = metal_impl::kAbtF16;
    metal_impl::launch_matmul_abt_mixed(g);
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
    if (N == 0 || out_cols == 0) return;

    Tensor Wf = Tensor::empty_on(Device::Metal, C_out, Cg_in * kH * kW, Dtype::FP16);
    dequantize_rows(W_int8, scales, Dtype::FP16, Wf);
    conv2d_forward(X, Wf, bias, N, C_in, H, W, C_out, kH, kW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Y);
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
    if (bias && bias->size() > 0 && bias->dtype != dt) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: bias must share X's dtype");
    }
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
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
    if (in_dim == 0) { Y_BD.zero(); return; }

    if (B < kQuantPrefillMinB) {
        mir_dispatch_gemv_q(QF_INT8, dt, "linear_forward_batched_int8w_fp16",
                            X_BD.data, W_int8.data, Y_BD.data,
                            (bias && bias->size() > 0) ? bias->data : nullptr, scales.data,
                            out_dim, in_dim, B, in_dim, out_dim, in_dim, LACT_NONE, EPI_STORE);
        return;
    }

    Tensor Wf = Tensor::empty_on(Device::Metal, out_dim, in_dim, Dtype::FP32);
    dequantize_rows(W_int8, scales, Dtype::FP32, Wf);
    metal_impl::AbtMixed g;
    g.A = buffer_for(X_BD); g.ofs_A = buffer_offset_for(X_BD); g.lda = static_cast<uint64_t>(in_dim);
    g.B = buffer_for(Wf);   g.ofs_B = buffer_offset_for(Wf);   g.ldb = static_cast<uint64_t>(in_dim);
    g.C = buffer_for(Y_BD); g.ofs_C = buffer_offset_for(Y_BD); g.ldc = static_cast<uint64_t>(out_dim);
    if (bias && bias->size() > 0) {
        g.bias = buffer_for(*bias);
        g.ofs_bias = buffer_offset_for(*bias);
    }
    g.M = B; g.N = out_dim; g.K = in_dim;
    g.in = g.out = metal_impl::abt_type(dt);
    g.in_B = metal_impl::kAbtF32;
    metal_impl::launch_matmul_abt_mixed(g);
}

}  // namespace brotensor::detail::metal
