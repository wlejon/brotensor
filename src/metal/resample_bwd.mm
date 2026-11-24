// Metal NCHW spatial adjoints: interp2d_backward, upsample_bilinear_2x_backward,
// pad2d_backward, adaptive_avg_pool2d_backward, and max_pool2d_backward.
//
// Single-source brass MIR kernel (src/kernels/mir_resample_bwd.cpp) compiled via
// MslTarget and cached in mtl_mir_cache.mm (mirrors src/vulkan/ops_spatial_bwd.cpp).

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#import "internal.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "unsupported dtype (expected FP32/FP16/BF16)");
    }
}

bool prepare(const char* op, const Tensor& dY, int N, int C, int H, int W,
             int Ho, int Wo, Tensor& dX) {
    require_fp(dY.dtype, op);
    need(op, dY.rows == N && dY.cols == static_cast<long long>(C) * Ho * Wo,
         "dY shape must be (N, C*H_out*W_out)");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, cols <= 0x7fffffffLL && N * cols <= 0xffffffffLL &&
             N * static_cast<long long>(C) * Ho * Wo <= 0xffffffffLL,
         "tensor too large");
    if (dX.data == nullptr) dX.device = dY.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != dY.dtype) {
        dX.resize(N, static_cast<int>(cols), dY.dtype);
    }
    return N > 0 && cols > 0;
}

void run(const char* op, std::uint32_t mode, std::uint32_t imode,
         const Tensor& dY, int N, int C, int H, int W, int Ho, int Wo,
         Tensor& dX, int pt = 0, int pl = 0) {
    const std::uint32_t planes = static_cast<std::uint32_t>(N * C);
    const std::uint32_t total =
        planes * static_cast<std::uint32_t>(H) * static_cast<std::uint32_t>(W);
    const MirKernel mk = mir_kernel(
        {MirKind::ResampleBwd, mode, dY.dtype, dY.dtype, 1,
         Dtype::FP32, static_cast<std::uint8_t>(imode)});
    mir_dispatch_resample_bwd(mk, mir_groups_1d(total, 1), op,
                              dY.data, dX.data, nullptr, planes,
                              static_cast<std::uint32_t>(H),
                              static_cast<std::uint32_t>(W),
                              static_cast<std::uint32_t>(Ho),
                              static_cast<std::uint32_t>(Wo), pt, pl);
}

}  // namespace

void interp2d_backward(const Tensor& dY, int N, int C, int H_in, int W_in,
                       int H_out, int W_out, int mode, Tensor& dX) {
    const char* op = "interp2d_backward";
    need(op, N >= 0 && C >= 0 && H_in >= 0 && W_in >= 0 && H_out >= 0 && W_out >= 0,
         "N, C, H_in, W_in, H_out, W_out must be non-negative");
    need(op, mode >= 0 && mode <= 3,
         "mode must be 0 (nearest), 1 (bilinear), 2 or 3 (bicubic)");
    if (mode >= 2) {
        fail(op, "bicubic backward is not implemented — mode must be 0 (nearest) or 1 (bilinear)");
    }
    need(op, !((H_out > 0 && H_in == 0) || (W_out > 0 && W_in == 0)),
         "input spatial dims must be > 0 when output spatial dims are > 0");
    if (!prepare(op, dY, N, C, H_in, W_in, H_out, W_out, dX)) return;
    if (H_out == 0 || W_out == 0) {
        dX.zero();
        return;
    }
    run(op, RB_INTERP, static_cast<std::uint32_t>(mode), dY, N, C,
        H_in, W_in, H_out, W_out, dX);
}

void upsample_bilinear_2x_backward(const Tensor& dY, int N, int C, int H, int W,
                                   Tensor& dX) {
    const char* op = "upsample_bilinear_2x_backward";
    need(op, N >= 0 && C >= 0 && H >= 0 && W >= 0, "bad dimension");
    if (!prepare(op, dY, N, C, H, W, 2 * H, 2 * W, dX)) return;
    run(op, RB_INTERP, 1, dY, N, C, H, W, 2 * H, 2 * W, dX);
}

void pad2d_backward(const Tensor& dY, int N, int C, int H, int W,
                    int pad_top, int pad_bottom, int pad_left, int pad_right,
                    int mode, Tensor& dX) {
    const char* op = "pad2d_backward";
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, pad_top >= 0 && pad_bottom >= 0 && pad_left >= 0 && pad_right >= 0,
         "pad counts must be >=0");
    need(op, mode >= 0 && mode <= 2,
         "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (mode == 1) {
        need(op, pad_top < H && pad_bottom < H,
             "reflect padding requires pad_top and pad_bottom < H");
        need(op, pad_left < W && pad_right < W,
             "reflect padding requires pad_left and pad_right < W");
    }
    const int Ho = H + pad_top + pad_bottom, Wo = W + pad_left + pad_right;
    if (!prepare(op, dY, N, C, H, W, Ho, Wo, dX)) return;
    run(op, RB_PAD, static_cast<std::uint32_t>(mode), dY, N, C,
        H, W, Ho, Wo, dX, pad_top, pad_left);
}

void adaptive_avg_pool2d_backward(const Tensor& dY, int N, int C, int H, int W,
                                  int H_out, int W_out, Tensor& dX) {
    const char* op = "adaptive_avg_pool2d_backward";
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, H_out >= 1 && W_out >= 1, "H_out and W_out must be >= 1");
    if (!prepare(op, dY, N, C, H, W, H_out, W_out, dX)) return;
    run(op, RB_ADAPTIVE, 0, dY, N, C, H, W, H_out, W_out, dX);
}

void max_pool2d_backward(const Tensor& dY, const Tensor& Idx,
                         int N, int C, int H, int W, int H_out, int W_out,
                         Tensor& dX) {
    const char* op = "max_pool2d_backward";
    need(op, Idx.dtype == Dtype::INT32,
         "Idx must be INT32 (as produced by max_pool2d_forward)");
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, H_out >= 0 && W_out >= 0, "H_out and W_out must be >= 0");
    need(op, Idx.rows == N && Idx.cols == static_cast<long long>(C) * H_out * W_out,
         "Idx shape must be (N, C*H_out*W_out)");
    if (!prepare(op, dY, N, C, H, W, H_out, W_out, dX)) return;
    Tensor acc = (dY.dtype == Dtype::FP32)
                     ? Tensor()
                     : Tensor::empty_on(dY.device, N, C * H * W, Dtype::FP32);
    Tensor& out = (dY.dtype == Dtype::FP32) ? dX : acc;
    const std::uint32_t planes = static_cast<std::uint32_t>(N * C);
    const MirKernel mk = mir_kernel(
        {MirKind::ResampleBwd, static_cast<std::uint32_t>(RB_MAXPOOL),
         dY.dtype, dY.dtype, 1});
    mir_dispatch_resample_bwd(mk, mir_groups_1d(planes, 1), op,
                              dY.data, out.data, Idx.data, planes,
                              static_cast<std::uint32_t>(H),
                              static_cast<std::uint32_t>(W),
                              static_cast<std::uint32_t>(H_out),
                              static_cast<std::uint32_t>(W_out), 0, 0);
    if (dY.dtype != Dtype::FP32) ::brotensor::cast(acc, dX, dY.dtype);
}

}  // namespace brotensor::detail::metal
