// Vulkan NCHW normalisations: GroupNorm forward (split statistics + apply)
// and backward (stats, per-channel sums, dX, dGamma / dBeta summed over
// the batch: no atomics; dX overwritten, dGamma / dBeta accumulated), BatchNorm
// inference with running statistics, per-pixel L2 normalise over channels, and
// training-mode BatchNorm forward / backward (FP32). Kernels: src/kernels/mir_gnorm.cpp.

#include "detail/kernels.h"
#include "detail/spatial.h"
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

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

void check_param(const char* op, const Tensor& t, int C, Dtype dt, const char* name) {
    if (t.dtype != dt) fail(op, std::string(name) + " dtype must match X");
    if (t.size() != C) fail(op, std::string(name) + " must have C elements");
}

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

int gn_vec4_lanes(Dtype dt, std::uint32_t hw, std::uint32_t chunk, std::uint64_t x, std::uint64_t y = 0) {
    const std::uint64_t req = (dt == Dtype::FP32) ? 16u : 8u;
    if (hw % 4u != 0u || chunk % 4u != 0u) return 1;
    if ((x % req) != 0u || (y != 0 && (y % req) != 0u)) return 1;
    return 4;
}

GnArgs gn_stats(DeviceCtx& d, std::uint64_t x, Dtype dt, int n, int c, int hw, int groups, float eps, Tensor& part,
                const char* op) {
    const std::uint64_t tiles = std::uint64_t(n) * groups;
    const std::uint64_t tile = std::uint64_t(c / groups) * hw;
    if (tiles > 65535) fail(op, "N * num_groups above 65535");
    if (tile * tiles > 0xffffffffULL) fail(op, "tensor too large");
    const std::uint64_t splits = std::clamp<std::uint64_t>(cdiv(tile, 8192), 1, 65535);
    const std::uint64_t chunk = (cdiv(tile, splits) + 3) / 4 * 4;
    const std::uint64_t used = cdiv(tile, chunk);
    part = Tensor::empty_on(::brotensor::Device::vulkan(d.index()), static_cast<int>(tiles * used * 3), 1, Dtype::FP32);
    GnArgs pc{};
    pc.x = x;
    pc.part = addr(part.data);
    pc.tile = static_cast<std::uint32_t>(tile);
    pc.hw = static_cast<std::uint32_t>(hw);
    pc.cpg = static_cast<std::uint32_t>(c / groups);
    pc.groups = static_cast<std::uint32_t>(groups);
    pc.splits = static_cast<std::uint32_t>(used);
    pc.chunk = static_cast<std::uint32_t>(chunk);
    pc.eps = eps;
    const int lanes = gn_vec4_lanes(dt, pc.hw, pc.chunk, x);
    dispatch_gn(d, GN_STATS, dt, false, lanes, pc.splits, static_cast<std::uint32_t>(tiles), pc);
    return pc;
}

void need_f32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32");
}

void ensure_c1(Tensor& t, int C, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.size() != C || t.dtype != Dtype::FP32) t.resize(C, 1, Dtype::FP32);
}

std::uint32_t bn_groups(int C) { return static_cast<std::uint32_t>(std::min(C, 65535)); }

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
    const int lanes = gn_vec4_lanes(dt, pc.hw, pc.chunk, x, y);
    const std::uint32_t gx = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(cdiv(tile, 256 * 16), 1, 1024));
    dispatch_gn(d, GN_APPLY, dt, silu, lanes, gx, static_cast<std::uint32_t>(tiles), pc);
}

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, int N, int C, int H, int W,
                        int num_groups, float eps, Tensor& Y) {
    const char* op = "group_norm_forward";
    dt_code(X.dtype, op);
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    need(op, num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    check_param(op, gamma, C, X.dtype, "gamma");
    check_param(op, beta, C, X.dtype, "beta");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    need(op, cols <= 0x7fffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(cols), X.dtype);
    if (N == 0 || cols == 0) return;
    group_norm(device_of(X), addr(X.data), addr(gamma.data), addr(beta.data), addr(Y.data), X.dtype, N, C, H * W,
               num_groups, eps, false);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY, int N, int C, int H, int W,
                         int num_groups, float eps, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const char* op = "group_norm_backward";
    const Dtype dt = X.dtype;
    dt_code(dt, op);
    need(op, gamma.dtype == dt && dY.dtype == dt, "gamma/dY dtype must match X");
    need(op, dGamma.dtype == dt && dBeta.dtype == dt, "dGamma/dBeta dtype must match X");
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    need(op, num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    need(op, dGamma.rows == C && dGamma.cols == 1 && dBeta.rows == C && dBeta.cols == 1, "dGamma/dBeta must be (C,1)");
    need(op, gamma.size() == C, "gamma must have C elements");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, cols <= 0x7fffffffLL, "tensor too large");
    need(op, X.size() >= N * cols && dY.rows == N && dY.cols == cols, "X / dY shape mismatch");
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != dt) dX.resize(N, static_cast<int>(cols), dt);
    if (N == 0 || cols == 0) return;
    need(op, N <= 65535 && C <= 65535, "N and C must be at most 65535");
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
    const std::uint32_t gx = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(cdiv(pc.tile, 256 * 16), 1, 1024));
    dispatch_gn(d, GN_BWD_DX, dt, false, 1, gx, static_cast<std::uint32_t>(N * num_groups), pc);
    pc.y = addr(dGamma.data);
    pc.g = addr(dBeta.data);
    dispatch_gn(d, GN_BWD_PARAM, dt, false, 1, mir_groups_1d(static_cast<std::uint32_t>(C), 1), 1, pc);
}

void batch_norm_inference(const Tensor& X, const Tensor& gamma, const Tensor& beta, const Tensor& running_mean,
                          const Tensor& running_var, int N, int C, int H, int W, float eps, Tensor& Y) {
    const char* op = "batch_norm_inference";
    dt_code(X.dtype, op);
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    check_param(op, gamma, C, X.dtype, "gamma");
    check_param(op, beta, C, X.dtype, "beta");
    check_param(op, running_mean, C, X.dtype, "running_mean");
    check_param(op, running_var, C, X.dtype, "running_var");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(cols), X.dtype);
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
    dt_code(X.dtype, op);
    need(op, N >= 0 && C > 0 && H > 0 && W > 0, "C/H/W must be >= 1 and N >= 0");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape must be (N, C*H*W)");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(cols), X.dtype);
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
    for (const Tensor* t : std::initializer_list<const Tensor*>{&X, &gamma, &beta, &running_mean, &running_var}) need_f32(op, *t, "every operand");
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    check_param(op, gamma, C, Dtype::FP32, "gamma");
    check_param(op, beta, C, Dtype::FP32, "beta");
    check_param(op, running_mean, C, Dtype::FP32, "running_mean");
    check_param(op, running_var, C, Dtype::FP32, "running_var");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.data == nullptr) Y.device = X.device;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) Y.resize(N, static_cast<int>(cols), Dtype::FP32);
    ensure_c1(saved_mean, C, X.device);
    ensure_c1(saved_rstd, C, X.device);
    if (N == 0 || cols == 0) return;
    DeviceCtx& d = device_of(X);
    const MirKernel mk = mir_kernel(d, {MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_FWD)});
    mir_dispatch_bnorm(d, mk, bn_groups(C),
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
    for (const Tensor* t : std::initializer_list<const Tensor*>{&X, &gamma, &saved_mean, &saved_rstd, &dY, &dGamma, &dBeta}) need_f32(op, *t, "every operand");
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    for (const Tensor* t : std::initializer_list<const Tensor*>{&gamma, &saved_mean, &saved_rstd, &dGamma, &dBeta}) {
        need(op, t->size() == C, "gamma, saved_mean / rstd, dGamma and dBeta must have C elements");
    }
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape mismatch");
    need(op, dY.rows == N && dY.cols == cols, "dY shape mismatch");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) dX.resize(N, static_cast<int>(cols), Dtype::FP32);
    if (N == 0 || cols == 0) return;
    DeviceCtx& d = device_of(X);
    const MirKernel mk = mir_kernel(d, {MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_BWD)});
    mir_dispatch_bnorm(d, mk, bn_groups(C),
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
