// Batched linear forward & backward on Metal backed by Brass MIR kernels
// (Gemm/GEMV via launch_matmul_abt_mixed and NORM_COL_SUM for dB).

#include "mtl_mir_cache.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#import "fp16_matmul.h"
#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

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

    metal_impl::AbtMixed g;
    g.A = buffer_for(X_BD); g.ofs_A = buffer_offset_for(X_BD); g.lda = static_cast<uint64_t>(in_dim);
    g.B = buffer_for(W);    g.ofs_B = buffer_offset_for(W);    g.ldb = static_cast<uint64_t>(in_dim);
    g.C = buffer_for(Y_BD); g.ofs_C = buffer_offset_for(Y_BD); g.ldc = static_cast<uint64_t>(out_dim);
    if (bias.size() > 0) {
        g.bias = buffer_for(bias);
        g.ofs_bias = buffer_offset_for(bias);
    }
    g.M = B; g.N = out_dim; g.K = in_dim;
    g.in = g.out = metal_impl::kAbtF32;
    g.in_B = metal_impl::abt_type(W.dtype);
    metal_impl::launch_matmul_abt_mixed(g);
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

    if (in_dim > 0 && out_dim > 0) {
        const metal_impl::AbtType ty = metal_impl::abt_type(W.dtype);
        metal_impl::AbtMixed g;
        g.A = buffer_for(dY_BD); g.ofs_A = buffer_offset_for(dY_BD); g.lda = static_cast<uint64_t>(out_dim);
        g.B = buffer_for(W);     g.ofs_B = buffer_offset_for(W);     g.ldb = static_cast<uint64_t>(in_dim);
        g.transB = true;
        g.C = buffer_for(dX_BD); g.ofs_C = buffer_offset_for(dX_BD); g.ldc = static_cast<uint64_t>(in_dim);
        g.M = B; g.N = in_dim; g.K = out_dim;
        g.in = g.out = ty;
        metal_impl::launch_matmul_abt_mixed(g);

        g.A = buffer_for(dY_BD); g.ofs_A = buffer_offset_for(dY_BD); g.lda = static_cast<uint64_t>(out_dim);
        g.transA = true;
        g.B = buffer_for(X_BD);  g.ofs_B = buffer_offset_for(X_BD);  g.ldb = static_cast<uint64_t>(in_dim);
        g.transB = true;
        g.C = buffer_for(dW);    g.ofs_C = buffer_offset_for(dW);    g.ldc = static_cast<uint64_t>(in_dim);
        g.M = out_dim; g.N = in_dim; g.K = B;
        g.epilogue = metal_impl::kAbtAccumulate;
        metal_impl::launch_matmul_abt_mixed(g);
    }
    if (out_dim > 0) {
        const std::uint8_t flags = pack_norm_flags(true, false, false, false, false);
        const MirKernel mk = mir_kernel(
            {MirKind::Norm, std::uint32_t(NORM_COL_SUM), W.dtype, W.dtype, 1, W.dtype, flags});
        const std::uint32_t groups = std::min<std::uint32_t>(
            (static_cast<std::uint32_t>(out_dim) + 31u) / 32u, 65535u);
        mir_dispatch_norm(mk, groups, "linear_backward_batched",
                          dY_BD.data, nullptr, dB.data, nullptr, nullptr, nullptr, nullptr,
                          static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(out_dim), 0.0f, 0.0f);
    }
}

} // namespace brotensor::detail::metal
