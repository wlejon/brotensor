// Batched linear forward & backward ops backed by Brass MIR kernels.

#include "cu_mir_cache.h"
#include "ops_gemm_mir.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

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
    const int in_dim = W.cols;
    const int B = X_BD.rows;
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
    const Dtype dt = W.dtype;
    const int out_dim = W.rows;
    const int in_dim = W.cols;
    const int B = X_BD.rows;

    if (dX_BD.rows != B || dX_BD.cols != in_dim || dX_BD.dtype != dt) {
        dX_BD.resize(B, in_dim, dt);
    }
    if (B == 0) return;

    if (in_dim > 0 && out_dim > 0) {
        dispatch_mir_gemm_simt(dY_BD.data, dt, W.data, dt, dX_BD.data, dt,
                               nullptr, dt, 1, B, in_dim, out_dim,
                               out_dim, in_dim, in_dim, 0, 0, 0,
                               /*ta=*/false, /*nb=*/true, EPI_STORE, 0, W.device);

        dispatch_mir_gemm_simt(dY_BD.data, dt, X_BD.data, dt, dW.data, dt,
                               nullptr, dt, 1, out_dim, in_dim, B,
                               out_dim, in_dim, in_dim, 0, 0, 0,
                               /*ta=*/true, /*nb=*/true, EPI_ACCUM, 0, W.device);
    }

    if (out_dim > 0) {
        const std::uint8_t flags = pack_norm_flags(true, false, false, false, false);
        CUfunction fn = mir_function(
            {MirKind::Norm, std::uint32_t(NORM_COL_SUM), dt, dt, 1, dt, flags});
        const std::uint32_t groups = std::min<std::uint32_t>(
            (static_cast<std::uint32_t>(out_dim) + 31u) / 32u, 65535u);
        mir_dispatch_norm(fn, groups, dY_BD.data, nullptr, dB.data, nullptr, nullptr, nullptr, nullptr,
                          static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(out_dim), 0.0f, 0.0f);
    }
}

}  // namespace brotensor::detail::cuda
