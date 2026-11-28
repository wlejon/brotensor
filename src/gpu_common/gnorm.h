#pragma once

// Shared host-side validation, split statistics planning, and dispatch sizing
// for GroupNorm / BatchNorm / L2Normalize NCHW across CUDA, Vulkan, and Metal.

#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::gpu_common {

[[noreturn]] inline void gnorm_fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

inline void gnorm_need(const char* op, bool ok, const char* what) {
    if (!ok) gnorm_fail(op, what);
}

inline void gnorm_require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        gnorm_fail(op, "must be FP32, FP16, or BF16");
    }
}

inline void gnorm_require_f32(Dtype dt, const char* op, const char* what) {
    if (dt != Dtype::FP32) gnorm_fail(op, std::string(what) + " must be FP32");
}

inline std::uint64_t gnorm_cdiv(std::uint64_t a, std::uint64_t b) {
    return (a + b - 1u) / b;
}

inline void gnorm_check_param(const char* op, const Tensor& t, int C, Dtype dt, const char* name) {
    if (t.dtype != dt) gnorm_fail(op, std::string(name) + " dtype must match X");
    if (t.size() != C) gnorm_fail(op, std::string(name) + " must have C elements");
}

inline void gnorm_ensure_c1_f32(Tensor& t, int C, Device dev) {
    if (t.data == nullptr) t.device = dev;
    if (t.size() != C || t.dtype != Dtype::FP32) t.resize(C, 1, Dtype::FP32);
}

inline std::uint32_t bn_groups(int C) {
    return static_cast<std::uint32_t>(std::min(C, 65535));
}

inline int gn_vec4_lanes(Dtype dt, std::uint32_t hw, std::uint32_t chunk,
                         std::uint64_t x, std::uint64_t y = 0) {
    const std::uint64_t req = (dt == Dtype::FP32) ? 16u : 8u;
    if (hw % 4u != 0u || chunk % 4u != 0u) return 1;
    if ((x % req) != 0u || (y != 0 && (y % req) != 0u)) return 1;
    return 4;
}

inline std::uint32_t gn_apply_gx(std::uint64_t tile) {
    return static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>(gnorm_cdiv(tile, 256u * 16u), 1u, 1024u));
}

struct GnStatsPlan {
    std::uint32_t tile = 0;
    std::uint32_t hw = 0;
    std::uint32_t cpg = 0;
    std::uint32_t groups = 0;
    std::uint32_t splits = 0;
    std::uint32_t chunk = 0;
    std::uint32_t tiles = 0;
};

inline GnStatsPlan plan_gn_stats(const char* op, int n, int c, int hw, int groups) {
    const std::uint64_t tiles = static_cast<std::uint64_t>(n) * groups;
    const std::uint64_t tile = static_cast<std::uint64_t>(c / groups) * hw;
    if (tiles > 65535) gnorm_fail(op, "N * num_groups above 65535");
    if (tile * tiles > 0xffffffffULL) gnorm_fail(op, "tensor too large");
    const std::uint64_t splits = std::clamp<std::uint64_t>(gnorm_cdiv(tile, 8192), 1, 65535);
    const std::uint64_t chunk = (gnorm_cdiv(tile, splits) + 3) / 4 * 4;
    const std::uint64_t used = gnorm_cdiv(tile, chunk);
    GnStatsPlan s{};
    s.tile = static_cast<std::uint32_t>(tile);
    s.hw = static_cast<std::uint32_t>(hw);
    s.cpg = static_cast<std::uint32_t>(c / groups);
    s.groups = static_cast<std::uint32_t>(groups);
    s.splits = static_cast<std::uint32_t>(used);
    s.chunk = static_cast<std::uint32_t>(chunk);
    s.tiles = static_cast<std::uint32_t>(tiles);
    return s;
}

inline long long check_group_norm_fwd(const char* op, const Tensor& X,
                                      const Tensor& gamma, const Tensor& beta,
                                      int N, int C, int H, int W, int num_groups, Tensor& Y) {
    gnorm_require_fp(X.dtype, op);
    gnorm_need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    gnorm_need(op, num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    gnorm_check_param(op, gamma, C, X.dtype, "gamma");
    gnorm_check_param(op, beta, C, X.dtype, "beta");
    const long long cols = static_cast<long long>(C) * H * W;
    gnorm_need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    gnorm_need(op, cols <= 0x7fffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
    return cols;
}

inline bool prepare_group_norm_bwd(const char* op, const Tensor& X, const Tensor& gamma,
                                   const Tensor& dY, int N, int C, int H, int W, int num_groups,
                                   Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    const Dtype dt = X.dtype;
    gnorm_require_fp(dt, op);
    gnorm_need(op, gamma.dtype == dt && dY.dtype == dt, "gamma/dY dtype must match X");
    gnorm_need(op, dGamma.dtype == dt && dBeta.dtype == dt, "dGamma/dBeta dtype must match X");
    gnorm_need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    gnorm_need(op, num_groups > 0 && C % num_groups == 0, "num_groups must divide C");
    gnorm_need(op, dGamma.rows == C && dGamma.cols == 1 && dBeta.rows == C && dBeta.cols == 1,
               "dGamma/dBeta must be (C,1)");
    gnorm_need(op, gamma.size() == C, "gamma must have C elements");
    const long long cols = static_cast<long long>(C) * H * W;
    gnorm_need(op, cols <= 0x7fffffffLL, "tensor too large");
    gnorm_need(op, X.size() >= N * cols && dY.rows == N && dY.cols == cols,
               "X / dY shape mismatch");
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != dt) {
        dX.resize(N, static_cast<int>(cols), dt);
    }
    if (N == 0 || cols == 0) return false;
    gnorm_need(op, N <= 65535 && C <= 65535, "N and C must be at most 65535");
    return true;
}

inline long long prepare_batch_norm_infer(const char* op, const Tensor& X,
                                          const Tensor& gamma, const Tensor& beta,
                                          const Tensor& rm, const Tensor& rv,
                                          int N, int C, int H, int W, Tensor& Y) {
    gnorm_require_fp(X.dtype, op);
    gnorm_need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    gnorm_check_param(op, gamma, C, X.dtype, "gamma");
    gnorm_check_param(op, beta, C, X.dtype, "beta");
    gnorm_check_param(op, rm, C, X.dtype, "running_mean");
    gnorm_check_param(op, rv, C, X.dtype, "running_var");
    const long long cols = static_cast<long long>(C) * H * W;
    gnorm_need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    gnorm_need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
    return cols;
}

inline long long prepare_l2_norm_nchw(const char* op, const Tensor& X,
                                      int N, int C, int H, int W, Tensor& Y) {
    gnorm_require_fp(X.dtype, op);
    gnorm_need(op, N >= 0 && C > 0 && H > 0 && W > 0, "C/H/W must be >= 1 and N >= 0");
    const long long cols = static_cast<long long>(C) * H * W;
    gnorm_need(op, X.rows == N && X.cols == cols, "X shape must be (N, C*H*W)");
    gnorm_need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
    return cols;
}

inline long long prepare_batch_norm_fwd(const char* op, const Tensor& X,
                                        const Tensor& gamma, const Tensor& beta,
                                        const Tensor& rm, const Tensor& rv,
                                        int N, int C, int H, int W,
                                        Tensor& Y, Tensor& saved_mean, Tensor& saved_rstd) {
    for (const Tensor* t : std::initializer_list<const Tensor*>{&X, &gamma, &beta, &rm, &rv}) {
        gnorm_require_f32(t->dtype, op, "every operand");
    }
    gnorm_need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    gnorm_check_param(op, gamma, C, Dtype::FP32, "gamma");
    gnorm_check_param(op, beta, C, Dtype::FP32, "beta");
    gnorm_check_param(op, rm, C, Dtype::FP32, "running_mean");
    gnorm_check_param(op, rv, C, Dtype::FP32, "running_var");
    const long long cols = static_cast<long long>(C) * H * W;
    gnorm_need(op, X.size() >= N * cols, "X is smaller than N*C*H*W");
    gnorm_need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (Y.data == nullptr) Y.device = X.device;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, static_cast<int>(cols), Dtype::FP32);
    }
    gnorm_ensure_c1_f32(saved_mean, C, X.device);
    gnorm_ensure_c1_f32(saved_rstd, C, X.device);
    return cols;
}

inline long long prepare_batch_norm_bwd(const char* op, const Tensor& X, const Tensor& gamma,
                                        const Tensor& saved_mean, const Tensor& saved_rstd,
                                        const Tensor& dY, int N, int C, int H, int W,
                                        Tensor& dX, const Tensor& dGamma, const Tensor& dBeta) {
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &X, &gamma, &saved_mean, &saved_rstd, &dY, &dGamma, &dBeta}) {
        gnorm_require_f32(t->dtype, op, "every operand");
    }
    gnorm_need(op, N >= 0 && C > 0 && H >= 0 && W >= 0, "bad dimension");
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &gamma, &saved_mean, &saved_rstd, &dGamma, &dBeta}) {
        gnorm_need(op, t->size() == C,
                   "gamma, saved_mean / rstd, dGamma and dBeta must have C elements");
    }
    const long long cols = static_cast<long long>(C) * H * W;
    gnorm_need(op, X.rows == N && X.cols == cols, "X shape mismatch");
    gnorm_need(op, dY.rows == N && dY.cols == cols, "dY shape mismatch");
    gnorm_need(op, N * cols <= 0xffffffffLL, "tensor too large");
    if (dX.data == nullptr) dX.device = X.device;
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, static_cast<int>(cols), Dtype::FP32);
    }
    return cols;
}

}  // namespace brotensor::gpu_common
