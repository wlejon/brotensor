// Vulkan NCHW normalisations: GroupNorm forward (split statistics + apply)
// and backward (stats, per-channel sums, dX, dGamma / dBeta summed over
// the batch: no atomics; dX overwritten, dGamma / dBeta accumulated), BatchNorm
// inference with running statistics, per-pixel L2 normalise over channels, and
// training-mode BatchNorm forward / backward (FP32). Kernels: src/kernels/mir_gnorm.cpp.

#include "detail/kernels.h"
#include "detail/spatial.h"
#include "../gpu_common/gnorm.h"
#include "vk_mir_cache.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct GnArgs {
    std::uint64_t x = 0, y = 0, g = 0, b = 0, part = 0, rm = 0, rv = 0;
    std::uint32_t tile = 0, hw = 0, cpg = 0, groups = 0, splits = 0, chunk = 0, total = 0, c = 0;
    float eps = 0.0f;
};

void dispatch_gn(DeviceCtx& d, std::uint32_t mode, Dtype dt, bool silu, int lanes,
                 std::uint32_t gx, std::uint32_t gy, const GnArgs& a) {
    const MirKernel mk = mir_kernel(
        d, {MirKind::Gnorm, mode, dt, dt, static_cast<std::uint8_t>(lanes), Dtype::FP32,
            static_cast<std::uint8_t>(silu ? 1u : 0u)});
    mir_dispatch_gnorm(d, mk, gx, gy, a.x, a.y, a.g, a.b, a.part, a.rm, a.rv,
                       a.tile, a.hw, a.cpg, a.groups, a.splits, a.chunk, a.total, a.c, a.eps);
}

GnArgs gn_stats(DeviceCtx& d, std::uint64_t x, Dtype dt, int n, int c, int hw, int groups, float eps, Tensor& part,
                const char* op) {
    const gpu_common::GnStatsPlan sp = gpu_common::plan_gn_stats(op, n, c, hw, groups);
    part = Tensor::empty_on(::brotensor::Device::vulkan(d.index()),
                            static_cast<int>(sp.tiles * sp.splits * 3u), 1, Dtype::FP32);
    GnArgs pc{};
    pc.x = x;
    pc.part = addr(part.data);
    pc.tile = sp.tile;
    pc.hw = sp.hw;
    pc.cpg = sp.cpg;
    pc.groups = sp.groups;
    pc.splits = sp.splits;
    pc.chunk = sp.chunk;
    pc.eps = eps;
    const int lanes = gpu_common::gn_vec4_lanes(dt, pc.hw, pc.chunk, x);
    dispatch_gn(d, GN_STATS, dt, false, lanes, pc.splits, sp.tiles, pc);
    return pc;
}

}  // namespace

void group_norm(DeviceCtx& d, std::uint64_t x, std::uint64_t gamma, std::uint64_t beta, std::uint64_t y, Dtype dt,
                int n, int c, int hw, int groups, float eps, bool silu) {
    const std::uint64_t tiles = std::uint64_t(n) * groups;
    const std::uint64_t tile = std::uint64_t(c / groups) * hw;
    if (tiles == 0 || tile == 0) return;
    Tensor part;
    GnArgs pc = gn_stats(d, x, dt, n, c, hw, groups, eps, part, "group_norm_forward");
    pc.y = y;
    pc.g = gamma;
    pc.b = beta;
    const int lanes = gpu_common::gn_vec4_lanes(dt, pc.hw, pc.chunk, x, y);
    dispatch_gn(d, GN_APPLY, dt, silu, lanes, gpu_common::gn_apply_gx(tile),
                static_cast<std::uint32_t>(tiles), pc);
}

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, int N, int C, int H, int W,
                        int num_groups, float eps, Tensor& Y) {
    const char* op = "group_norm_forward";
    const long long cols = gpu_common::check_group_norm_fwd(op, X, gamma, beta, N, C, H, W, num_groups, Y);
    if (N == 0 || cols == 0) return;
    group_norm(device_of(X), addr(X.data), addr(gamma.data), addr(beta.data), addr(Y.data), X.dtype, N, C, H * W,
               num_groups, eps, false);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY, int N, int C, int H, int W,
                         int num_groups, float eps, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const char* op = "group_norm_backward";
    if (!gpu_common::prepare_group_norm_bwd(op, X, gamma, dY, N, C, H, W, num_groups, dX, dGamma, dBeta)) return;
    const Dtype dt = X.dtype;
    DeviceCtx& d = device_of(X);
    const int hw = H * W;
    Tensor part;
    GnArgs pc = gn_stats(d, addr(X.data), dt, N, C, hw, num_groups, eps, part, op);
    const ::brotensor::Device dev = ::brotensor::Device::vulkan(d.index());
    Tensor sums = Tensor::empty_on(dev, N * C * 2, 1, Dtype::FP32);
    Tensor stats = Tensor::empty_on(dev, N * num_groups * 2, 1, Dtype::FP32);
    pc.y = addr(dX.data);
    pc.g = addr(gamma.data);
    pc.b = addr(dY.data);
    pc.rm = addr(sums.data);
    pc.rv = addr(stats.data);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N);
    dispatch_gn(d, GN_BWD_CH, dt, false, 1, static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(N), pc);
    dispatch_gn(d, GN_BWD_DX, dt, false, 1, gpu_common::gn_apply_gx(pc.tile),
                static_cast<std::uint32_t>(N * num_groups), pc);
    pc.y = addr(dGamma.data);
    pc.g = addr(dBeta.data);
    dispatch_gn(d, GN_BWD_PARAM, dt, false, 1, mir_groups_1d(static_cast<std::uint32_t>(C), 1), 1, pc);
}

void batch_norm_inference(const Tensor& X, const Tensor& gamma, const Tensor& beta, const Tensor& running_mean,
                          const Tensor& running_var, int N, int C, int H, int W, float eps, Tensor& Y) {
    const char* op = "batch_norm_inference";
    const long long cols = gpu_common::prepare_batch_norm_infer(
        op, X, gamma, beta, running_mean, running_var, N, C, H, W, Y);
    if (N == 0 || cols == 0) return;
    DeviceCtx& d = device_of(X);
    GnArgs pc{};
    pc.x = addr(X.data); pc.y = addr(Y.data); pc.g = addr(gamma.data); pc.b = addr(beta.data);
    pc.rm = addr(running_mean.data); pc.rv = addr(running_var.data);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * cols);
    pc.eps = eps;
    dispatch_gn(d, GN_BN_INFER, X.dtype, false, 1, mir_groups_1d(pc.total, 1), 1, pc);
}

void l2_normalize_nchw_forward(const Tensor& X, int N, int C, int H, int W, float eps, Tensor& Y) {
    const char* op = "l2_normalize_nchw_forward";
    gpu_common::prepare_l2_norm_nchw(op, X, N, C, H, W, Y);
    if (N == 0) return;
    DeviceCtx& d = device_of(X);
    GnArgs pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * H * W);
    pc.eps = eps;
    dispatch_gn(d, GN_L2_NCHW, X.dtype, false, 1, mir_groups_1d(pc.total, 1), 1, pc);
}

void batch_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, Tensor& running_mean,
                        Tensor& running_var, int N, int C, int H, int W, float eps, float momentum, Tensor& Y,
                        Tensor& saved_mean, Tensor& saved_rstd) {
    const char* op = "batch_norm_forward";
    const long long cols = gpu_common::prepare_batch_norm_fwd(
        op, X, gamma, beta, running_mean, running_var, N, C, H, W, Y, saved_mean, saved_rstd);
    if (N == 0 || cols == 0) return;
    DeviceCtx& d = device_of(X);
    const MirKernel mk = mir_kernel(d, {MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_FWD)});
    mir_dispatch_bnorm(d, mk, gpu_common::bn_groups(C),
                       addr(X.data), addr(Y.data), addr(gamma.data), addr(beta.data),
                       addr(running_mean.data), addr(running_var.data),
                       addr(saved_mean.data), addr(saved_rstd.data),
                       0, 0, 0,
                       static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(C),
                       static_cast<std::uint32_t>(H * W), eps, momentum);
}

void batch_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& saved_mean, const Tensor& saved_rstd,
                         const Tensor& dY, int N, int C, int H, int W, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const char* op = "batch_norm_backward";
    const long long cols = gpu_common::prepare_batch_norm_bwd(
        op, X, gamma, saved_mean, saved_rstd, dY, N, C, H, W, dX, dGamma, dBeta);
    if (N == 0 || cols == 0) return;
    DeviceCtx& d = device_of(X);
    const MirKernel mk = mir_kernel(d, {MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_BWD)});
    mir_dispatch_bnorm(d, mk, gpu_common::bn_groups(C),
                       addr(X.data), addr(dX.data), addr(gamma.data), 0,
                       0, 0, addr(saved_mean.data), addr(saved_rstd.data),
                       addr(dY.data), addr(dGamma.data), addr(dBeta.data),
                       static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(C),
                       static_cast<std::uint32_t>(H * W), 0.0f, 0.0f);
}

void fill_vulkan_vtable_gnorm(::brotensor::detail::OpsVTable& v) {
    v.group_norm_forward = &group_norm_forward;
    v.group_norm_backward = &group_norm_backward;
    v.batch_norm_inference = &batch_norm_inference;
    v.batch_norm_forward = &batch_norm_forward;
    v.batch_norm_backward = &batch_norm_backward;
    v.l2_normalize_nchw_forward = &l2_normalize_nchw_forward;
}

}  // namespace brotensor::detail::vulkan
