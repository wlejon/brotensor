// CUDA NCHW normalisations backed by Brass MIR -> PTX: GroupNorm forward/backward,
// BatchNorm inference/training forward/backward, and per-pixel NCHW L2 normalize.

#include "cu_mir_cache.h"
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

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

void require_f32(Dtype dt, const char* op, const char* what) {
    if (dt != Dtype::FP32) fail(op, std::string(what) + " must be FP32");
}

void check_param(const char* op, const Tensor& t, int c, Dtype dt, const char* name) {
    if (t.dtype != dt) fail(op, std::string(name) + " dtype must match X");
    if (t.size() != c) fail(op, std::string(name) + " must have C elements");
}

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

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
    const std::uintptr_t req = (dt == Dtype::FP32) ? 16u : 8u;
    if (hw % 4u != 0u || chunk % 4u != 0u) return 1;
    if ((reinterpret_cast<std::uintptr_t>(x) % req) != 0u) return 1;
    if (y != nullptr && (reinterpret_cast<std::uintptr_t>(y) % req) != 0u) return 1;
    return 4;
}

GnArgs gn_stats(const Tensor& X, int n, int c, int hw, int groups, float eps, Tensor& part, const char* op) {
    const std::uint64_t tiles = static_cast<std::uint64_t>(n) * groups;
    const std::uint64_t tile = static_cast<std::uint64_t>(c / groups) * hw;
    if (tiles > 65535) fail(op, "N * num_groups above 65535");
    if (tile * tiles > 0xffffffffULL) fail(op, "tensor too large");
    const std::uint64_t splits = std::clamp<std::uint64_t>(cdiv(tile, 8192), 1, 65535);
    const std::uint64_t chunk = (cdiv(tile, splits) + 3) / 4 * 4;
    const std::uint64_t used = cdiv(tile, chunk);
    part = Tensor::empty_on(X.device, static_cast<int>(tiles * used * 3), 1, Dtype::FP32);
    GnArgs pc{};
    pc.x = X.data;
    pc.part = part.data;
    pc.tile = static_cast<std::uint32_t>(tile);
    pc.hw = static_cast<std::uint32_t>(hw);
    pc.cpg = static_cast<std::uint32_t>(c / groups);
    pc.groups = static_cast<std::uint32_t>(groups);
    pc.splits = static_cast<std::uint32_t>(used);
    pc.chunk = static_cast<std::uint32_t>(chunk);
    pc.eps = eps;
    const int lanes = gn_vec4_lanes(X.dtype, pc.hw, pc.chunk, X.data);
    dispatch_gn(GN_STATS, X.dtype, false, lanes, pc.splits, static_cast<std::uint32_t>(tiles), pc);
    return pc;
}

void ensure_c1(Tensor& t, int c, ::brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.size() != c || t.dtype != Dtype::FP32) t.resize(c, 1, Dtype::FP32);
}

std::uint32_t bn_groups(int c) { return static_cast<std::uint32_t>((std::min)(c, 65535)); }

}  // namespace

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta, int N, int C, int H, int W,
                        int num_groups, float eps, Tensor& Y) {
    constexpr const char* op = "group_norm_forward";
    require_fp(X.dtype, op);
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    need(op, num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    check_param(op, gamma, C, X.dtype, "gamma");
    check_param(op, beta, C, X.dtype, "beta");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    need(op, cols <= 0x7fffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(cols), X.dtype);
    if (N == 0 || cols == 0) return;

    const int hw = H * W;
    Tensor part;
    GnArgs pc = gn_stats(X, N, C, hw, num_groups, eps, part, op);
    pc.y = Y.data;
    pc.g = gamma.data;
    pc.b = beta.data;
    const int lanes = gn_vec4_lanes(X.dtype, pc.hw, pc.chunk, X.data, Y.data);
    const std::uint64_t tile = static_cast<std::uint64_t>(C / num_groups) * hw;
    const std::uint32_t gx = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(cdiv(tile, 256 * 16), 1, 1024));
    dispatch_gn(GN_APPLY, X.dtype, false, lanes, gx, static_cast<std::uint32_t>(N * num_groups), pc);
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY, int N, int C, int H, int W,
                         int num_groups, float eps, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "group_norm_backward";
    const Dtype dt = X.dtype;
    require_fp(dt, op);
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
    const std::uint32_t gx = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(cdiv(pc.tile, 256 * 16), 1, 1024));
    dispatch_gn(GN_BWD_DX, dt, false, 1, gx, static_cast<std::uint32_t>(N * num_groups), pc);
    pc.y = dGamma.data;
    pc.g = dBeta.data;
    dispatch_gn(GN_BWD_PARAM, dt, false, 1, mir_groups_1d(static_cast<std::uint32_t>(C), 1), 1, pc);
}

void batch_norm_inference(const Tensor& X, const Tensor& gamma, const Tensor& beta, const Tensor& running_mean,
                          const Tensor& running_var, int N, int C, int H, int W, float eps, Tensor& Y) {
    constexpr const char* op = "batch_norm_inference";
    require_fp(X.dtype, op);
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
    require_fp(X.dtype, op);
    need(op, N >= 0 && C > 0 && H > 0 && W > 0, "C/H/W must be >= 1 and N >= 0");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols, "X shape must be (N, C*H*W)");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) Y.resize(N, static_cast<int>(cols), X.dtype);
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
    for (const Tensor* t : std::initializer_list<const Tensor*>{&X, &gamma, &beta, &running_mean, &running_var}) {
        require_f32(t->dtype, op, "every operand");
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
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) Y.resize(N, static_cast<int>(cols), Dtype::FP32);
    ensure_c1(saved_mean, C, X.device);
    ensure_c1(saved_rstd, C, X.device);
    if (N == 0 || cols == 0) return;

    CUfunction fn = mir_function({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_FWD)});
    mir_dispatch_bnorm(
        fn, bn_groups(C),
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
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &X, &gamma, &saved_mean, &saved_rstd, &dY, &dGamma, &dBeta}) {
        require_f32(t->dtype, op, "every operand");
    }
    need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    check_param(op, gamma, C, Dtype::FP32, "gamma");
    check_param(op, saved_mean, C, Dtype::FP32, "saved_mean");
    check_param(op, saved_rstd, C, Dtype::FP32, "saved_rstd");
    need(op, dGamma.rows == C && dGamma.cols == 1 && dBeta.rows == C && dBeta.cols == 1, "dGamma/dBeta must be (C,1)");
    const long long cols = static_cast<long long>(C) * H * W;
    need(op, X.rows == N && X.cols == cols && dY.rows == N && dY.cols == cols, "X / dY shape mismatch");
    need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) dX.resize(N, static_cast<int>(cols), Dtype::FP32);
    if (N == 0 || cols == 0) return;

    CUfunction fn = mir_function({MirKind::Bnorm, static_cast<std::uint32_t>(BN_TRAIN_BWD)});
    mir_dispatch_bnorm(
        fn, bn_groups(C),
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
