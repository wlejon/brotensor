// Metal NCHW normalisations: GroupNorm forward/backward, BatchNorm inference
// and training forward/backward, and NCHW per-pixel L2 normalize.
//
// Single-source brass MIR kernels (src/kernels/mir_gnorm.cpp) compiled via
// MslTarget and cached in mtl_mir_cache.mm (mirrors src/vulkan/ops_gnorm.cpp).

#include "mtl_mir_cache.h"

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

std::uint64_t addr(const void* p) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p));
}

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

void check_param(const char* op, const Tensor& t, int C, Dtype dt, const char* name) {
    if (t.dtype != dt) fail(op, std::string(name) + " dtype must match X");
    if (t.size() != C) fail(op, std::string(name) + " must have C elements");
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

int gn_vec4_lanes(Dtype dt, std::uint32_t hw, std::uint32_t chunk,
                  const void* x, const void* y = nullptr) {
    const std::uint64_t req = (dt == Dtype::FP32) ? 16u : 8u;
    if (hw % 4u != 0u || chunk % 4u != 0u) return 1;
    if ((addr(x) % req) != 0u || (y != nullptr && (addr(y) % req) != 0u)) return 1;
    return 4;
}

GnArgs gn_stats(const void* x, Dtype dt, ::brotensor::Device dev,
                int n, int c, int hw, int groups, float eps,
                Tensor& part, const char* op) {
    const std::uint64_t tiles = std::uint64_t(n) * groups;
    const std::uint64_t tile = std::uint64_t(c / groups) * hw;
    if (tiles > 65535) fail(op, "N * num_groups above 65535");
    if (tile * tiles > 0xffffffffULL) fail(op, "tensor too large");
    const std::uint64_t splits = std::clamp<std::uint64_t>(cdiv(tile, 8192), 1, 65535);
    const std::uint64_t chunk = (cdiv(tile, splits) + 3) / 4 * 4;
    const std::uint64_t used = cdiv(tile, chunk);
    part = Tensor::empty_on(dev, static_cast<int>(tiles * used * 3), 1, Dtype::FP32);
    GnArgs pc{};
    pc.x = x;
    pc.part = part.data;
    pc.tile = static_cast<std::uint32_t>(tile);
    pc.hw = static_cast<std::uint32_t>(hw);
    pc.cpg = static_cast<std::uint32_t>(c / groups);
    pc.groups = static_cast<std::uint32_t>(groups);
    pc.splits = static_cast<std::uint32_t>(used);
    pc.chunk = static_cast<std::uint32_t>(chunk);
    pc.eps = eps;
    const int lanes = gn_vec4_lanes(dt, pc.hw, pc.chunk, x);
    dispatch_gn(op, GN_STATS, dt, false, lanes, pc.splits,
                static_cast<std::uint32_t>(tiles), pc);
    return pc;
}

void need_f32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32");
}

void ensure_c1(Tensor& t, int C, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.size() != C || t.dtype != Dtype::FP32) t.resize(C, 1, Dtype::FP32);
}

std::uint32_t bn_groups(int C) {
    return static_cast<std::uint32_t>(std::min(C, 65535));
}

}  // namespace

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        int N, int C, int H, int W, int num_groups, float eps,
                        Tensor& Y) {
    const char* op = "group_norm_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    need(op, num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    check_param(op, gamma, C, X.dtype, "gamma");
    check_param(op, beta, C, X.dtype, "beta");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    need(op, cols <= 0x7fffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
    if (N == 0 || cols == 0) return;
    const std::uint64_t tiles = std::uint64_t(N) * num_groups;
    const std::uint64_t tile = std::uint64_t(C / num_groups) * (H * W);
    if (tiles == 0 || tile == 0) return;
    Tensor part;
    GnArgs pc = gn_stats(X.data, X.dtype, X.device, N, C, H * W, num_groups, eps, part, op);
    pc.y = Y.data;
    pc.g = gamma.data;
    pc.b = beta.data;
    const int lanes = gn_vec4_lanes(X.dtype, pc.hw, pc.chunk, X.data, Y.data);
    const std::uint32_t gx =
        static_cast<std::uint32_t>(std::clamp<std::uint64_t>(cdiv(tile, 256 * 16), 1, 1024));
    dispatch_gn(op, GN_APPLY, X.dtype, false, lanes, gx,
                static_cast<std::uint32_t>(tiles), pc);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const char* op = "group_norm_backward";
    const Dtype dt = X.dtype;
    require_fp(dt, op);
    need(op, gamma.dtype == dt && dY.dtype == dt, "gamma/dY dtype must match X");
    need(op, dGamma.dtype == dt && dBeta.dtype == dt, "dGamma/dBeta dtype must match X");
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    need(op, num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    need(op, dGamma.rows == C && dGamma.cols == 1 && dBeta.rows == C && dBeta.cols == 1,
         "dGamma/dBeta must be (C,1)");
    need(op, gamma.size() == C, "gamma must have C elements");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, cols <= 0x7fffffffLL, "tensor too large");
    need(op, X.size() >= N * cols && dY.rows == N && dY.cols == cols,
         "X / dY shape mismatch");
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != dt) {
        dX.resize(N, static_cast<int>(cols), dt);
    }
    if (N == 0 || cols == 0) return;
    need(op, N <= 65535 && C <= 65535, "N and C must be at most 65535");
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
    const std::uint32_t gx =
        static_cast<std::uint32_t>(std::clamp<std::uint64_t>(cdiv(pc.tile, 256 * 16), 1, 1024));
    dispatch_gn(op, GN_BWD_DX, dt, false, 1, gx,
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
    require_fp(X.dtype, op);
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    check_param(op, gamma, C, X.dtype, "gamma");
    check_param(op, beta, C, X.dtype, "beta");
    check_param(op, running_mean, C, X.dtype, "running_mean");
    check_param(op, running_var, C, X.dtype, "running_var");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
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
    require_fp(X.dtype, op);
    need(op, N >= 0 && C > 0 && H > 0 && W > 0, "C/H/W must be >= 1 and N >= 0");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape must be (N, C*H*W)");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
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
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &X, &gamma, &beta, &running_mean, &running_var}) {
        need_f32(op, *t, "every operand");
    }
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    check_param(op, gamma, C, Dtype::FP32, "gamma");
    check_param(op, beta, C, Dtype::FP32, "beta");
    check_param(op, running_mean, C, Dtype::FP32, "running_mean");
    check_param(op, running_var, C, Dtype::FP32, "running_var");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.data == nullptr) Y.device = X.device;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, static_cast<int>(cols), Dtype::FP32);
    }
    ensure_c1(saved_mean, C, X.device);
    ensure_c1(saved_rstd, C, X.device);
    if (N == 0 || cols == 0) return;
    const MirKernel mk = mir_kernel({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_FWD)});
    mir_dispatch_bnorm(mk, bn_groups(C), op,
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
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &X, &gamma, &saved_mean, &saved_rstd, &dY, &dGamma, &dBeta}) {
        need_f32(op, *t, "every operand");
    }
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &gamma, &saved_mean, &saved_rstd, &dGamma, &dBeta}) {
        need(op, t->size() == C,
             "gamma, saved_mean / rstd, dGamma and dBeta must have C elements");
    }
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape mismatch");
    need(op, dY.rows == N && dY.cols == cols, "dY shape mismatch");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, static_cast<int>(cols), Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;
    const MirKernel mk = mir_kernel({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_BWD)});
    mir_dispatch_bnorm(mk, bn_groups(C), op,
                       X.data, dX.data, gamma.data, nullptr,
                       nullptr, nullptr, saved_mean.data, saved_rstd.data,
                       dY.data, dGamma.data, dBeta.data,
                       static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(C),
                       static_cast<std::uint32_t>(H * W), 0.0f, 0.0f);
}

}  // namespace brotensor::detail::metal
