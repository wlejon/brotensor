// CUDA NCHW normalisations backed by Brass MIR -> PTX: GroupNorm forward/backward,
// BatchNorm inference/training forward/backward, and per-pixel NCHW L2 normalize.

#include "cu_mir_cache.h"
#include "../gpu_common/gnorm.h"
#include "../kernels/mir_gnorm.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct GnArgs {
    const void* x = nullptr;
    void* y = nullptr;
    const void* g = nullptr;
    const void* b = nullptr;
    const void* part = nullptr;
    const void* rm = nullptr;
    const void* rv = nullptr;
    std::uint32_t tile = 0, hw = 0, cpg = 0, groups = 0, splits = 0, chunk = 0, total = 0, c = 0;
    float eps = 0.0f;
};

void dispatch_gn(std::uint32_t mode, Dtype dt, bool silu, int lanes,
                 std::uint32_t gx, std::uint32_t gy, const GnArgs& a) {
    CUfunction fn = mir_function(
        {MirKind::Gnorm, mode, dt, dt, static_cast<std::uint8_t>(lanes), Dtype::FP32,
         static_cast<std::uint8_t>(silu ? 1u : 0u)});
    mir_dispatch_gnorm(
        fn, gx, gy, a.x, a.y, a.g, a.b, a.part, a.rm, a.rv,
        a.tile, a.hw, a.cpg, a.groups, a.splits, a.chunk, a.total, a.c, a.eps);
}

int gn_vec4_lanes(Dtype dt, std::uint32_t hw, std::uint32_t chunk, const void* x, const void* y = nullptr) {
    return gpu_common::gn_vec4_lanes(
        dt, hw, chunk, reinterpret_cast<std::uintptr_t>(x), reinterpret_cast<std::uintptr_t>(y));
}

GnArgs gn_stats(const Tensor& X, int n, int c, int hw, int groups, float eps, Tensor& part, const char* op) {
    const gpu_common::GnStatsPlan sp = gpu_common::plan_gn_stats(op, n, c, hw, groups);
    part = Tensor::empty_on(X.device, static_cast<int>(sp.tiles * sp.splits * 3u), 1, Dtype::FP32);
    GnArgs pc{};
    pc.x = X.data;
    pc.part = part.data;
    pc.tile = sp.tile;
    pc.hw = sp.hw;
    pc.cpg = sp.cpg;
    pc.groups = sp.groups;
    pc.splits = sp.splits;
    pc.chunk = sp.chunk;
    pc.eps = eps;
    const int lanes = gn_vec4_lanes(X.dtype, pc.hw, pc.chunk, X.data);
    dispatch_gn(GN_STATS, X.dtype, false, lanes, pc.splits, sp.tiles, pc);
    return pc;
}

}  // namespace

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, int N, int C, int H, int W,
                        int num_groups, float eps, Tensor& Y) {
    constexpr const char* op = "group_norm_forward";
    const long long cols = gpu_common::check_group_norm_fwd(op, X, gamma, beta, N, C, H, W, num_groups, Y);
    if (N == 0 || cols == 0) return;

    const int hw = H * W;
    Tensor part;
    GnArgs pc = gn_stats(X, N, C, hw, num_groups, eps, part, op);
    pc.y = Y.data;
    pc.g = gamma.data;
    pc.b = beta.data;
    const int lanes = gn_vec4_lanes(X.dtype, pc.hw, pc.chunk, X.data, Y.data);
    dispatch_gn(GN_APPLY, X.dtype, false, lanes, gpu_common::gn_apply_gx(pc.tile),
                static_cast<std::uint32_t>(N * num_groups), pc);
}

void group_norm_silu_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, int N, int C, int H, int W,
                             int num_groups, float eps, Tensor& Y) {
    constexpr const char* op = "group_norm_silu_forward";
    const long long cols = gpu_common::check_group_norm_fwd(op, X, gamma, beta, N, C, H, W, num_groups, Y);
    if (N == 0 || cols == 0) return;

    const int hw = H * W;
    Tensor part;
    GnArgs pc = gn_stats(X, N, C, hw, num_groups, eps, part, op);
    pc.y = Y.data;
    pc.g = gamma.data;
    pc.b = beta.data;
    const int lanes = gn_vec4_lanes(X.dtype, pc.hw, pc.chunk, X.data, Y.data);
    dispatch_gn(GN_APPLY, X.dtype, true, lanes, gpu_common::gn_apply_gx(pc.tile),
                static_cast<std::uint32_t>(N * num_groups), pc);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY, int N, int C, int H, int W,
                         int num_groups, float eps, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "group_norm_backward";
    if (!gpu_common::prepare_group_norm_bwd(op, X, gamma, dY, N, C, H, W, num_groups, dX, dGamma, dBeta)) return;
    const Dtype dt = X.dtype;

    const int hw = H * W;
    Tensor part;
    GnArgs pc = gn_stats(X, N, C, hw, num_groups, eps, part, op);
    Tensor sums = Tensor::empty_on(X.device, N * C * 2, 1, Dtype::FP32);
    Tensor stats = Tensor::empty_on(X.device, N * num_groups * 2, 1, Dtype::FP32);
    pc.y = dX.data;
    pc.g = gamma.data;
    pc.b = dY.data;
    pc.rm = sums.data;
    pc.rv = stats.data;
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N);
    dispatch_gn(GN_BWD_CH, dt, false, 1, static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(N), pc);
    dispatch_gn(GN_BWD_DX, dt, false, 1, gpu_common::gn_apply_gx(pc.tile),
                static_cast<std::uint32_t>(N * num_groups), pc);
    pc.y = dGamma.data;
    pc.g = dBeta.data;
    dispatch_gn(GN_BWD_PARAM, dt, false, 1, mir_groups_1d(static_cast<std::uint32_t>(C), 1), 1, pc);
}

void batch_norm_inference(const Tensor& X, const Tensor& gamma, const Tensor& beta, const Tensor& running_mean,
                          const Tensor& running_var, int N, int C, int H, int W, float eps, Tensor& Y) {
    constexpr const char* op = "batch_norm_inference";
    const long long cols = gpu_common::prepare_batch_norm_infer(
        op, X, gamma, beta, running_mean, running_var, N, C, H, W, Y);
    if (N == 0 || cols == 0) return;

    GnArgs pc{};
    pc.x = X.data; pc.y = Y.data; pc.g = gamma.data; pc.b = beta.data;
    pc.rm = running_mean.data; pc.rv = running_var.data;
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * cols);
    pc.eps = eps;
    dispatch_gn(GN_BN_INFER, X.dtype, false, 1, mir_groups_1d(pc.total, 1), 1, pc);
}

void l2_normalize_nchw_forward(const Tensor& X, int N, int C, int H, int W, float eps, Tensor& Y) {
    constexpr const char* op = "l2_normalize_nchw_forward";
    gpu_common::prepare_l2_norm_nchw(op, X, N, C, H, W, Y);
    if (N == 0) return;

    GnArgs pc{};
    pc.x = X.data;
    pc.y = Y.data;
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * H * W);
    pc.eps = eps;
    dispatch_gn(GN_L2_NCHW, X.dtype, false, 1, mir_groups_1d(pc.total, 1), 1, pc);
}

void batch_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, Tensor& running_mean,
                        Tensor& running_var, int N, int C, int H, int W, float eps, float momentum, Tensor& Y,
                        Tensor& saved_mean, Tensor& saved_rstd) {
    constexpr const char* op = "batch_norm_forward";
    const long long cols = gpu_common::prepare_batch_norm_fwd(
        op, X, gamma, beta, running_mean, running_var, N, C, H, W, Y, saved_mean, saved_rstd);
    if (N == 0 || cols == 0) return;

    CUfunction fn = mir_function({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_FWD)});
    mir_dispatch_bnorm(
        fn, gpu_common::bn_groups(C),
        X.data, Y.data, gamma.data, beta.data,
        running_mean.data, running_var.data,
        saved_mean.data, saved_rstd.data,
        nullptr, nullptr, nullptr,
        static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(C),
        static_cast<std::uint32_t>(H * W), eps, momentum);
}

void batch_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& saved_mean, const Tensor& saved_rstd,
                         const Tensor& dY, int N, int C, int H, int W, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "batch_norm_backward";
    const long long cols = gpu_common::prepare_batch_norm_bwd(
        op, X, gamma, saved_mean, saved_rstd, dY, N, C, H, W, dX, dGamma, dBeta);
    if (N == 0 || cols == 0) return;

    CUfunction fn = mir_function({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_BWD)});
    mir_dispatch_bnorm(
        fn, gpu_common::bn_groups(C),
        X.data, dX.data, gamma.data, nullptr,
        nullptr, nullptr,
        saved_mean.data, saved_rstd.data,
        dY.data, dGamma.data, dBeta.data,
        static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(C),
        static_cast<std::uint32_t>(H * W), 0.0f, 0.0f);
}

void fill_cuda_vtable_batch_norm(::brotensor::detail::OpsVTable& v) {
    v.batch_norm_forward = &batch_norm_forward;
    v.batch_norm_inference = &batch_norm_inference;
    v.batch_norm_backward = &batch_norm_backward;
}

void fill_cuda_vtable_l2_normalize(::brotensor::detail::OpsVTable& v) {
    v.l2_normalize_nchw_forward = &l2_normalize_nchw_forward;
}

}  // namespace brotensor::detail::cuda
