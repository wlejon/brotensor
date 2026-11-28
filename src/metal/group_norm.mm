// Metal NCHW normalisations: GroupNorm forward/backward, BatchNorm inference
// and training forward/backward, and NCHW per-pixel L2 normalize.
//
// Single-source brass MIR kernels (src/kernels/mir_gnorm.cpp) compiled via
// MslTarget and cached in mtl_mir_cache.mm (mirrors src/vulkan/ops_gnorm.cpp).

#include "mtl_mir_cache.h"
#include "../gpu_common/gnorm.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

namespace {

std::uint64_t addr(const void* p) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p));
}

struct GnArgs {
    const void* x = nullptr;
    const void* y = nullptr;
    const void* g = nullptr;
    const void* b = nullptr;
    const void* part = nullptr;
    const void* rm = nullptr;
    const void* rv = nullptr;
    std::uint32_t tile = 0, hw = 0, cpg = 0, groups = 0, splits = 0, chunk = 0, total = 0, c = 0;
    float eps = 0.0f;
};

void dispatch_gn(const char* op, std::uint32_t mode, Dtype dt, bool silu, int lanes,
                 std::uint32_t gx, std::uint32_t gy, const GnArgs& a) {
    const MirKernel mk = mir_kernel(
        {MirKind::Gnorm, mode, dt, dt, static_cast<std::uint8_t>(lanes), Dtype::FP32,
         static_cast<std::uint8_t>(silu ? 1u : 0u)});
    mir_dispatch_gnorm(mk, gx, gy, op, a.x, a.y, a.g, a.b, a.part, a.rm, a.rv,
                       a.tile, a.hw, a.cpg, a.groups, a.splits, a.chunk,
                       a.total, a.c, a.eps);
}

GnArgs gn_stats(const void* x, Dtype dt, ::brotensor::Device dev,
                int n, int c, int hw, int groups, float eps,
                Tensor& part, const char* op) {
    const gpu_common::GnStatsPlan sp = gpu_common::plan_gn_stats(op, n, c, hw, groups);
    part = Tensor::empty_on(dev, static_cast<int>(sp.tiles * sp.splits * 3u), 1, Dtype::FP32);
    GnArgs pc{};
    pc.x = x;
    pc.part = part.data;
    pc.tile = sp.tile;
    pc.hw = sp.hw;
    pc.cpg = sp.cpg;
    pc.groups = sp.groups;
    pc.splits = sp.splits;
    pc.chunk = sp.chunk;
    pc.eps = eps;
    const int lanes = gpu_common::gn_vec4_lanes(dt, pc.hw, pc.chunk, addr(x));
    dispatch_gn(op, GN_STATS, dt, false, lanes, pc.splits, sp.tiles, pc);
    return pc;
}

}  // namespace

void group_norm_forward_impl(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                             int N, int C, int H, int W, int num_groups, float eps,
                             bool silu, const char* op, Tensor& Y) {
    const long long cols = gpu_common::check_group_norm_fwd(op, X, gamma, beta, N, C, H, W, num_groups, Y);
    if (N == 0 || cols == 0) return;
    const std::uint64_t tiles = std::uint64_t(N) * num_groups;
    const std::uint64_t tile = std::uint64_t(C / num_groups) * (H * W);
    if (tiles == 0 || tile == 0) return;
    Tensor part;
    GnArgs pc = gn_stats(X.data, X.dtype, X.device, N, C, H * W, num_groups, eps, part, op);
    pc.y = Y.data;
    pc.g = gamma.data;
    pc.b = beta.data;
    const int lanes = gpu_common::gn_vec4_lanes(X.dtype, pc.hw, pc.chunk, addr(X.data), addr(Y.data));
    dispatch_gn(op, GN_APPLY, X.dtype, silu, lanes, gpu_common::gn_apply_gx(tile),
                static_cast<std::uint32_t>(tiles), pc);
}

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        int N, int C, int H, int W, int num_groups, float eps,
                        Tensor& Y) {
    group_norm_forward_impl(X, gamma, beta, N, C, H, W, num_groups, eps, false,
                            "group_norm_forward", Y);
}

void group_norm_silu_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                             int N, int C, int H, int W, int num_groups, float eps,
                             Tensor& Y) {
    group_norm_forward_impl(X, gamma, beta, N, C, H, W, num_groups, eps, true,
                            "group_norm_silu_forward", Y);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const char* op = "group_norm_backward";
    if (!gpu_common::prepare_group_norm_bwd(op, X, gamma, dY, N, C, H, W, num_groups, dX, dGamma, dBeta)) return;
    const Dtype dt = X.dtype;
    const int hw = H * W;
    Tensor part;
    GnArgs pc = gn_stats(X.data, dt, X.device, N, C, hw, num_groups, eps, part, op);
    Tensor sums = Tensor::empty_on(X.device, N * C * 2, 1, Dtype::FP32);
    Tensor stats = Tensor::empty_on(X.device, N * num_groups * 2, 1, Dtype::FP32);
    pc.y = dX.data;
    pc.g = gamma.data;
    pc.b = dY.data;
    pc.rm = sums.data;
    pc.rv = stats.data;
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N);
    dispatch_gn(op, GN_BWD_CH, dt, false, 1,
                static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(N), pc);
    dispatch_gn(op, GN_BWD_DX, dt, false, 1, gpu_common::gn_apply_gx(pc.tile),
                static_cast<std::uint32_t>(N * num_groups), pc);
    pc.y = dGamma.data;
    pc.g = dBeta.data;
    dispatch_gn(op, GN_BWD_PARAM, dt, false, 1,
                mir_groups_1d(static_cast<std::uint32_t>(C), 1), 1, pc);
}

void batch_norm_inference(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                          const Tensor& running_mean, const Tensor& running_var,
                          int N, int C, int H, int W, float eps, Tensor& Y) {
    const char* op = "batch_norm_inference";
    const long long cols = gpu_common::prepare_batch_norm_infer(
        op, X, gamma, beta, running_mean, running_var, N, C, H, W, Y);
    if (N == 0 || cols == 0) return;
    GnArgs pc{};
    pc.x = X.data;
    pc.y = Y.data;
    pc.g = gamma.data;
    pc.b = beta.data;
    pc.rm = running_mean.data;
    pc.rv = running_var.data;
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * cols);
    pc.eps = eps;
    dispatch_gn(op, GN_BN_INFER, X.dtype, false, 1, mir_groups_1d(pc.total, 1), 1, pc);
}

void l2_normalize_nchw_forward(const Tensor& X, int N, int C, int H, int W,
                               float eps, Tensor& Y) {
    const char* op = "l2_normalize_nchw_forward";
    gpu_common::prepare_l2_norm_nchw(op, X, N, C, H, W, Y);
    if (N == 0) return;
    GnArgs pc{};
    pc.x = X.data;
    pc.y = Y.data;
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * H * W);
    pc.eps = eps;
    dispatch_gn(op, GN_L2_NCHW, X.dtype, false, 1, mir_groups_1d(pc.total, 1), 1, pc);
}

void batch_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        Tensor& running_mean, Tensor& running_var,
                        int N, int C, int H, int W, float eps, float momentum,
                        Tensor& Y, Tensor& saved_mean, Tensor& saved_rstd) {
    const char* op = "batch_norm_forward";
    const long long cols = gpu_common::prepare_batch_norm_fwd(
        op, X, gamma, beta, running_mean, running_var, N, C, H, W, Y, saved_mean, saved_rstd);
    if (N == 0 || cols == 0) return;
    const MirKernel mk = mir_kernel({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_FWD)});
    mir_dispatch_bnorm(mk, gpu_common::bn_groups(C), op,
                       X.data, Y.data, gamma.data, beta.data,
                       running_mean.data, running_var.data,
                       saved_mean.data, saved_rstd.data,
                       nullptr, nullptr, nullptr,
                       static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(C),
                       static_cast<std::uint32_t>(H * W), eps, momentum);
}

void batch_norm_backward(const Tensor& X, const Tensor& gamma,
                         const Tensor& saved_mean, const Tensor& saved_rstd,
                         const Tensor& dY, int N, int C, int H, int W,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const char* op = "batch_norm_backward";
    const long long cols = gpu_common::prepare_batch_norm_bwd(
        op, X, gamma, saved_mean, saved_rstd, dY, N, C, H, W, dX, dGamma, dBeta);
    if (N == 0 || cols == 0) return;
    const MirKernel mk = mir_kernel({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_BWD)});
    mir_dispatch_bnorm(mk, gpu_common::bn_groups(C), op,
                       X.data, dX.data, gamma.data, nullptr,
                       nullptr, nullptr, saved_mean.data, saved_rstd.data,
                       dY.data, dGamma.data, dBeta.data,
                       static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(C),
                       static_cast<std::uint32_t>(H * W), 0.0f, 0.0f);
}

}  // namespace brotensor::detail::metal
