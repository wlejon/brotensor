// Vulkan NCHW normalisations: GroupNorm forward (split statistics + apply,
// shaders/gnorm.comp), BatchNorm inference with running statistics, and the
// per-pixel L2 normalise over channels. Contracts follow the CUDA backend
// (src/cuda/group_norm.cu, batch_norm.cu, l2_normalize.cu): FP32 / FP16 /
// BF16 with gamma / beta / running statistics in X's dtype, Y resized to X's
// shape and dtype and overwritten.

#include "detail/kernels.h"
#include "detail/spatial.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct GnPush {
    std::uint64_t x, y, g, b, part, rm, rv;
    std::uint32_t tile, hw, cpg, groups, splits, chunk, total, c;
    float eps;
};

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

}  // namespace

void group_norm(DeviceCtx& d, std::uint64_t x, std::uint64_t gamma, std::uint64_t beta, std::uint64_t y, Dtype dt,
                int n, int c, int hw, int groups, float eps, bool silu) {
    const std::uint64_t tiles = std::uint64_t(n) * groups;
    const std::uint64_t tile = std::uint64_t(c / groups) * hw;
    if (tiles == 0 || tile == 0) return;
    if (tiles > 65535) fail("group_norm_forward", "N * num_groups above 65535");
    if (tile * tiles > 0xffffffffULL) fail("group_norm_forward", "tensor too large");
    // Chunks of ~8192 elements: enough workgroups that one sample fills the
    // GPU, and short ones (the partials combine in the apply pass).
    const std::uint64_t splits = std::clamp<std::uint64_t>(cdiv(tile, 8192), 1, 65535);
    const std::uint64_t chunk = (cdiv(tile, splits) + 3) / 4 * 4;   // whole 4-element groups
    const std::uint64_t used = cdiv(tile, chunk);
    Tensor part = Tensor::empty_on(::brotensor::Device::vulkan(d.index()), static_cast<int>(tiles * used * 3), 1,
                                   Dtype::FP32);
    GnPush pc{};
    pc.x = x; pc.y = y; pc.g = gamma; pc.b = beta; pc.part = addr(part.data);
    pc.tile = static_cast<std::uint32_t>(tile);
    pc.hw = static_cast<std::uint32_t>(hw);
    pc.cpg = static_cast<std::uint32_t>(c / groups);
    pc.groups = static_cast<std::uint32_t>(groups);
    pc.splits = static_cast<std::uint32_t>(used);
    pc.chunk = static_cast<std::uint32_t>(chunk);
    pc.eps = eps;
    const ShaderId id = dt_variant(ShaderId::gnorm_f32, dt, "group_norm_forward");
    const std::uint32_t v4 = hw % 4 == 0 && chunk % 4 == 0 && x % 16 == 0 && y % 16 == 0 ? 1u : 0u;
    launch(d, d.pipelines().get(id, {GN_STATS, 0u, v4}), pc, pc.splits, static_cast<std::uint32_t>(tiles));
    const std::uint32_t gx = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(cdiv(tile, 256 * 16), 1, 1024));
    launch(d, d.pipelines().get(id, {GN_APPLY, silu ? 1u : 0u, v4}), pc, gx, static_cast<std::uint32_t>(tiles));
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
    GnPush pc{};
    pc.x = addr(X.data); pc.y = addr(Y.data); pc.g = addr(gamma.data); pc.b = addr(beta.data);
    pc.rm = addr(running_mean.data); pc.rv = addr(running_var.data);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * cols);
    pc.eps = eps;
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::gnorm_f32, X.dtype, op), {GN_BN_INFER, 0u});
    launch(d, k, pc, groups_1d(pc.total, k));
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
    GnPush pc{};
    pc.x = addr(X.data); pc.y = addr(Y.data);
    pc.hw = static_cast<std::uint32_t>(H * W);
    pc.c = static_cast<std::uint32_t>(C);
    pc.total = static_cast<std::uint32_t>(N * H * W);
    pc.eps = eps;
    const Kernel& k = d.pipelines().get(dt_variant(ShaderId::gnorm_f32, X.dtype, op), {GN_L2_NCHW, 0u});
    launch(d, k, pc, groups_1d(pc.total, k));
}

void fill_vulkan_vtable_gnorm(::brotensor::detail::OpsVTable& v) {
    v.group_norm_forward = &group_norm_forward;
    v.batch_norm_inference = &batch_norm_inference;
    v.l2_normalize_nchw_forward = &l2_normalize_nchw_forward;
}

}  // namespace brotensor::detail::vulkan
