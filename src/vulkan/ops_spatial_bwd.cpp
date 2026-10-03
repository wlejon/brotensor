// Vulkan adjoints of the NCHW resamples, padding and pooling:
// upsample_bilinear_2x_backward, interp2d_backward (nearest / bilinear),
// pad2d_backward, adaptive_avg_pool2d_backward and max_pool2d_backward
// (shaders/resample_bwd.comp). The first four are gathers (each input element
// sums the outputs that read it), max pooling walks each plane's outputs in
// order: no atomics, deterministic, FP32 sums. Contracts follow CUDA
// (src/cuda/resample.cu, interp2d.cu, pad2d.cu, pool2d.cu) and the CPU
// reference: dX resized to (N, C H W) in dY's dtype (FP32 / FP16 / BF16;
// CUDA's pooling backwards are FP32 only) and OVERWRITTEN; bicubic
// interp2d_backward throws, as on every backend.

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct RbPush {
    std::uint64_t dy, dx, aux;
    std::uint32_t planes, hi, wi, ho, wo;
    std::int32_t pt, pl;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

// dX (N, C H W) in dY's dtype; false when there is nothing to compute.
bool prepare(const char* op, const Tensor& dY, int N, int C, int H, int W, int Ho, int Wo, Tensor& dX) {
    dt_code(dY.dtype, op);
    need(op, dY.rows == N && dY.cols == static_cast<long long>(C) * Ho * Wo, "dY shape must be (N, C*H_out*W_out)");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, cols <= 0x7fffffffLL && N * cols <= 0xffffffffLL && N * static_cast<long long>(C) * Ho * Wo <= 0xffffffffLL,
         "tensor too large");
    if (dX.data == nullptr) dX.device = dY.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != dY.dtype) dX.resize(N, static_cast<int>(cols), dY.dtype);
    return N > 0 && cols > 0;
}

void run(const char* op, std::uint32_t mode, std::uint32_t imode, const Tensor& dY, int N, int C, int H, int W, int Ho,
         int Wo, Tensor& dX, int pt = 0, int pl = 0) {
    DeviceCtx& d = device_of(dY);
    RbPush pc{addr(dY.data), addr(dX.data), 0, static_cast<std::uint32_t>(N * C), static_cast<std::uint32_t>(H),
              static_cast<std::uint32_t>(W), static_cast<std::uint32_t>(Ho), static_cast<std::uint32_t>(Wo), pt, pl};
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::resample_bwd_f32, dY.dtype, op), {mode, imode});
    launch(d, k, pc, groups_1d(static_cast<std::uint64_t>(N) * C * H * W, k));
}

}  // namespace

void interp2d_backward(const Tensor& dY, int N, int C, int H_in, int W_in, int H_out, int W_out, int mode,
                       Tensor& dX) {
    const char* op = "interp2d_backward";
    need(op, N >= 0 && C >= 0 && H_in >= 0 && W_in >= 0 && H_out >= 0 && W_out >= 0,
         "N, C, H_in, W_in, H_out, W_out must be non-negative");
    need(op, mode >= 0 && mode <= 3, "mode must be 0 (nearest), 1 (bilinear), 2 or 3 (bicubic)");
    if (mode >= 2) fail(op, "bicubic backward is not implemented — mode must be 0 (nearest) or 1 (bilinear)");
    need(op, !((H_out > 0 && H_in == 0) || (W_out > 0 && W_in == 0)),
         "input spatial dims must be > 0 when output spatial dims are > 0");
    if (!prepare(op, dY, N, C, H_in, W_in, H_out, W_out, dX)) return;
    if (H_out == 0 || W_out == 0) {
        dX.zero();
        return;
    }
    run(op, RB_INTERP, static_cast<std::uint32_t>(mode), dY, N, C, H_in, W_in, H_out, W_out, dX);
}

void upsample_bilinear_2x_backward(const Tensor& dY, int N, int C, int H, int W, Tensor& dX) {
    const char* op = "upsample_bilinear_2x_backward";
    need(op, N >= 0 && C >= 0 && H >= 0 && W >= 0, "bad dimension");
    if (!prepare(op, dY, N, C, H, W, 2 * H, 2 * W, dX)) return;
    run(op, RB_INTERP, 1, dY, N, C, H, W, 2 * H, 2 * W, dX);
}

void pad2d_backward(const Tensor& dY, int N, int C, int H, int W, int pad_top, int pad_bottom, int pad_left,
                    int pad_right, int mode, Tensor& dX) {
    const char* op = "pad2d_backward";
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, pad_top >= 0 && pad_bottom >= 0 && pad_left >= 0 && pad_right >= 0, "pad counts must be >=0");
    need(op, mode >= 0 && mode <= 2, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (mode == 1) {
        need(op, pad_top < H && pad_bottom < H, "reflect padding requires pad_top and pad_bottom < H");
        need(op, pad_left < W && pad_right < W, "reflect padding requires pad_left and pad_right < W");
    }
    const int Ho = H + pad_top + pad_bottom, Wo = W + pad_left + pad_right;
    if (!prepare(op, dY, N, C, H, W, Ho, Wo, dX)) return;
    run(op, RB_PAD, static_cast<std::uint32_t>(mode), dY, N, C, H, W, Ho, Wo, dX, pad_top, pad_left);
}

void adaptive_avg_pool2d_backward(const Tensor& dY, int N, int C, int H, int W, int H_out, int W_out, Tensor& dX) {
    const char* op = "adaptive_avg_pool2d_backward";
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, H_out >= 1 && W_out >= 1, "H_out and W_out must be >= 1");
    if (!prepare(op, dY, N, C, H, W, H_out, W_out, dX)) return;
    run(op, RB_ADAPTIVE, 0, dY, N, C, H, W, H_out, W_out, dX);
}

void max_pool2d_backward(const Tensor& dY, const Tensor& Idx, int N, int C, int H, int W, int H_out, int W_out,
                         Tensor& dX) {
    const char* op = "max_pool2d_backward";
    need(op, Idx.dtype == Dtype::INT32, "Idx must be INT32 (as produced by max_pool2d_forward)");
    need(op, N >= 0 && C >= 1 && H >= 1 && W >= 1, "C/H/W must be >=1 and N >=0");
    need(op, H_out >= 0 && W_out >= 0, "H_out and W_out must be >= 0");
    need(op, Idx.rows == N && Idx.cols == static_cast<long long>(C) * H_out * W_out,
         "Idx shape must be (N, C*H_out*W_out)");
    if (!prepare(op, dY, N, C, H, W, H_out, W_out, dX)) return;
    DeviceCtx& d = device_of(dY);
    // The walk accumulates in FP32: dX itself, or a scratch cast into it.
    Tensor acc = dY.dtype == Dtype::FP32 ? Tensor() : Tensor::empty_on(dY.device, N, C * H * W, Dtype::FP32);
    Tensor& out = dY.dtype == Dtype::FP32 ? dX : acc;
    RbPush pc{addr(dY.data), addr(out.data), addr(Idx.data), static_cast<std::uint32_t>(N * C),
              static_cast<std::uint32_t>(H), static_cast<std::uint32_t>(W), static_cast<std::uint32_t>(H_out),
              static_cast<std::uint32_t>(W_out), 0, 0};
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::resample_bwd_f32, dY.dtype, op), {std::uint32_t(RB_MAXPOOL), 0u});
    launch(d, k, pc, groups_1d(static_cast<std::uint64_t>(N) * C, k));
    if (dY.dtype != Dtype::FP32) ::brotensor::cast(acc, dX, dY.dtype);
}

void fill_vulkan_vtable_spatial_bwd(::brotensor::detail::OpsVTable& v) {
    v.interp2d_backward = &vulkan::interp2d_backward;
    v.upsample_bilinear_2x_backward = &vulkan::upsample_bilinear_2x_backward;
    v.pad2d_backward = &vulkan::pad2d_backward;
    v.adaptive_avg_pool2d_backward = &vulkan::adaptive_avg_pool2d_backward;
    v.max_pool2d_backward = &vulkan::max_pool2d_backward;
}

}  // namespace brotensor::detail::vulkan
