// Metal normalisations and softmax: LayerNorm (vector, batched inference,
// FP16 inference, training with caches, backwards), RMSNorm, pixel norm,
// per-head L2 norm, and vector / row softmax + backward.
//
// Single-source brass MIR kernels compiled via MslTarget and cached in
// mtl_mir_cache.mm (mirrors src/vulkan/ops_norm.cpp).

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::metal {

namespace {

struct NormArgs {
    const void* x = nullptr;
    const void* y = nullptr;
    const void* g = nullptr;
    const void* b = nullptr;
    const void* xh = nullptr;
    const void* s0 = nullptr;
    const void* s1 = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t d = 0;
    float eps = 0.0f;
    float rstd = 0.0f;
};

struct NormDtypes {
    Dtype x, g, y;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "unsupported dtype (expected FP32/FP16/BF16)");
    }
}

std::uint64_t addr(const void* p) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p));
}

NormDtypes norm_dtypes(Dtype dx, Dtype dg, Dtype dy, const char* op) {
    require_fp(dx, op);
    if ((dg == dx && dy == dx) ||
        (dg == Dtype::FP32 && dy == dx && (dx == Dtype::FP16 || dx == Dtype::BF16)) ||
        (dx == Dtype::FP32 && dg == Dtype::FP32 && dy == Dtype::FP16)) {
        return {dx, dg, dy};
    }
    fail(op, "unsupported dtype combination");
}

std::uint32_t u32(long long v, const char* op) {
    if (v < 0 || v > 0x7fffffffLL) fail(op, "dimension out of range");
    return static_cast<std::uint32_t>(v);
}

void ensure(Tensor& t, int r, int c, Dtype dt) {
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(r, c, dt);
}

void ensure_grad(Tensor& g, const Tensor& like, long long n, Dtype dt) {
    if (g.size() != n || g.dtype != dt) {
        g.resize(like.rows, like.cols, dt);
        g.zero();
    }
}

void run(const char* op, NormDtypes dts, std::uint32_t mode, const NormArgs& a) {
    if (a.rows == 0 || a.d == 0) return;
    const bool cols = mode >= NORM_COL_LN;
    const int lanes = cols ? 1
                           : mir_vec_lanes_norm(
                                 dts.x, dts.y, dts.g, a.d, addr(a.x), addr(a.y),
                                 addr(a.g), addr(a.b), addr(a.xh),
                                 (mode == NORM_SOFTMAX) ? addr(a.s0) : 0u);
    const std::uint8_t flags = pack_norm_flags(
        a.g != nullptr, a.b != nullptr, a.xh != nullptr,
        a.s0 != nullptr, a.s1 != nullptr);
    const MirKernel mk = mir_kernel(
        {MirKind::Norm, mode, dts.x, dts.y, static_cast<std::uint8_t>(lanes), dts.g, flags});
    const std::uint32_t groups = cols ? (a.d + 31u) / 32u : a.rows;
    mir_dispatch_norm(mk, std::min<std::uint32_t>(groups, 65535u), op,
                      a.x, a.y, a.g, a.b, a.xh, a.s0, a.s1,
                      a.rows, a.d, a.eps, a.rstd);
}

void column_grads(const char* op, NormDtypes dts, std::uint32_t mode,
                  const void* x, const void* xh, const void* s1,
                  Tensor& dG, Tensor* dB, std::uint32_t rows, std::uint32_t d) {
    NormArgs a{};
    a.x = x;
    a.xh = xh;
    a.s1 = s1;
    a.g = dG.data;
    a.b = dB ? dB->data : nullptr;
    a.rows = rows;
    a.d = d;
    run(op, dts, mode, a);
}

void layernorm_rows(const char* op, const Tensor& X, const Tensor& gamma,
                    const Tensor* beta, Tensor& Y, Tensor* Xhat,
                    Tensor* Mean, Tensor* Rstd, float eps) {
    const std::uint32_t R = u32(X.rows, op), D = u32(X.cols, op);
    if (gamma.size() != X.cols || (beta && beta->size() != X.cols)) {
        fail(op, "gamma / beta must have X.cols elements");
    }
    if (gamma.dtype != X.dtype || (beta && beta->dtype != X.dtype)) {
        fail(op, "gamma / beta dtype must match X");
    }
    NormArgs a{};
    a.x = X.data;
    a.y = Y.data;
    a.g = gamma.data;
    a.b = beta ? beta->data : nullptr;
    a.xh = Xhat ? Xhat->data : nullptr;
    a.s0 = Mean ? Mean->data : nullptr;
    a.s1 = Rstd ? Rstd->data : nullptr;
    a.rows = R;
    a.d = D;
    a.eps = eps;
    run(op, norm_dtypes(X.dtype, gamma.dtype, Y.dtype, op), NORM_LN, a);
}

}  // namespace

// ─── LayerNorm ─────────────────────────────────────────────────────────────

void layernorm_forward_inference_batched(const Tensor& X, const Tensor& gamma,
                                         const Tensor* beta, Tensor& Y, float eps) {
    constexpr const char* op = "layernorm_forward_inference_batched";
    require_fp(X.dtype, op);
    if (Y.rows != X.rows || Y.cols != X.cols ||
        (Y.dtype != X.dtype && !(X.dtype == Dtype::FP32 && Y.dtype == Dtype::FP16))) {
        Y.resize(X.rows, X.cols, X.dtype);
    }
    layernorm_rows(op, X, gamma, beta, Y, nullptr, nullptr, nullptr, eps);
}

void layernorm_forward_inference_batched_fp16(const Tensor& X, const Tensor& gamma,
                                              const Tensor* beta, Tensor& Y, float eps) {
    constexpr const char* op = "layernorm_forward_inference_batched_fp16";
    if (X.dtype != Dtype::FP16) fail(op, "all tensors must be FP16");
    ensure(Y, X.rows, X.cols, Dtype::FP16);
    layernorm_rows(op, X, gamma, beta, Y, nullptr, nullptr, nullptr, eps);
}

void layernorm_forward_batched_with_caches(const Tensor& X, const Tensor& gamma,
                                           const Tensor& beta, Tensor& Y,
                                           Tensor& Xhat, Tensor& Mean, Tensor& Rstd,
                                           float eps) {
    constexpr const char* op = "layernorm_forward_batched_with_caches";
    require_fp(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    ensure(Xhat, X.rows, X.cols, X.dtype);
    ensure(Mean, X.rows, 1, Dtype::FP32);
    ensure(Rstd, X.rows, 1, Dtype::FP32);
    layernorm_rows(op, X, gamma, &beta, Y, &Xhat, &Mean, &Rstd, eps);
}

void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       Tensor& y, Tensor& xhat, float& mean_out, float& rstd_out,
                       float eps) {
    constexpr const char* op = "layernorm_forward";
    require_fp(x.dtype, op);
    ensure(y, x.rows, x.cols, x.dtype);
    ensure(xhat, x.rows, x.cols, x.dtype);
    if (x.size() == 0) {
        mean_out = 0.0f;
        rstd_out = 0.0f;
        return;
    }
    const Tensor xr = Tensor::view(x.device, x.data, 1, x.size(), x.dtype);
    Tensor yr = Tensor::view(y.device, y.data, 1, x.size(), y.dtype);
    Tensor hr = Tensor::view(xhat.device, xhat.data, 1, x.size(), xhat.dtype);
    const Tensor gr = Tensor::view(gamma.device, gamma.data, 1, gamma.size(), gamma.dtype);
    const Tensor br = Tensor::view(beta.device, beta.data, 1, beta.size(), beta.dtype);
    Tensor stats = Tensor::empty_on(x.device, 2, 1, Dtype::FP32);
    Tensor m = Tensor::view(x.device, stats.data, 1, 1, Dtype::FP32);
    Tensor r = Tensor::view(x.device, static_cast<char*>(stats.data) + 4, 1, 1, Dtype::FP32);
    layernorm_rows(op, xr, gr, &br, yr, &hr, &m, &r, eps);
    const std::vector<float> h = stats.to_host_vector();
    mean_out = h[0];
    rstd_out = h[1];
}

void layernorm_backward_batched_with_caches(const Tensor& dY, const Tensor& Xhat,
                                            const Tensor& gamma, const Tensor& Rstd,
                                            Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "layernorm_backward_batched_with_caches";
    require_fp(dY.dtype, op);
    if (Xhat.dtype != dY.dtype || gamma.dtype != dY.dtype) {
        fail(op, "Xhat / gamma dtype must match dY");
    }
    if (Rstd.dtype != Dtype::FP32) fail(op, "Rstd_R must be FP32");
    const std::uint32_t R = u32(dY.rows, op), D = u32(dY.cols, op);
    if (Xhat.rows != dY.rows || Xhat.cols != dY.cols ||
        gamma.size() != dY.cols || Rstd.size() != dY.rows) {
        fail(op, "shape mismatch");
    }
    ensure(dX, dY.rows, dY.cols, dY.dtype);
    ensure_grad(dGamma, gamma, D, dY.dtype);
    ensure_grad(dBeta, gamma, D, dY.dtype);
    const NormDtypes id = norm_dtypes(dY.dtype, dY.dtype, dY.dtype, op);
    NormArgs a{};
    a.x = dY.data;
    a.y = dX.data;
    a.g = gamma.data;
    a.xh = Xhat.data;
    a.s1 = Rstd.data;
    a.rows = R;
    a.d = D;
    run(op, id, NORM_LN_BWD, a);
    column_grads(op, id, NORM_COL_LN, dY.data, Xhat.data, nullptr, dGamma, &dBeta, R, D);
}

void layernorm_backward(const Tensor& dY, const Tensor& xhat, const Tensor& gamma,
                        float rstd, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "layernorm_backward";
    require_fp(dY.dtype, op);
    if (xhat.dtype != dY.dtype || gamma.dtype != dY.dtype) {
        fail(op, "xhat / gamma dtype must match dY");
    }
    const std::uint32_t n = u32(dY.size(), op);
    if (xhat.size() != dY.size() || gamma.size() != dY.size()) fail(op, "size mismatch");
    ensure(dX, dY.rows, dY.cols, dY.dtype);
    ensure_grad(dGamma, gamma, n, dY.dtype);
    ensure_grad(dBeta, gamma, n, dY.dtype);
    const NormDtypes id = norm_dtypes(dY.dtype, dY.dtype, dY.dtype, op);
    NormArgs a{};
    a.x = dY.data;
    a.y = dX.data;
    a.g = gamma.data;
    a.xh = xhat.data;
    a.rows = 1;
    a.d = n;
    a.rstd = rstd;
    run(op, id, NORM_LN_BWD, a);
    column_grads(op, id, NORM_COL_LN, dY.data, xhat.data, nullptr, dGamma, &dBeta, 1, n);
}

// ─── RMSNorm, pixel norm, per-head L2 norm ─────────────────────────────────

void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y) {
    constexpr const char* op = "rms_norm_forward";
    require_fp(X.dtype, op);
    if (gamma.dtype != X.dtype && gamma.dtype != Dtype::FP32) {
        fail(op, "gamma.dtype must match X.dtype or be FP32");
    }
    if (gamma.size() != X.cols) fail(op, "gamma must have D elements");
    ensure(Y, X.rows, X.cols, X.dtype);
    NormArgs a{};
    a.x = X.data;
    a.y = Y.data;
    a.g = gamma.data;
    a.rows = u32(X.rows, op);
    a.d = u32(X.cols, op);
    a.eps = eps;
    run(op, norm_dtypes(X.dtype, gamma.dtype, X.dtype, op), NORM_RMS, a);
}

void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma) {
    constexpr const char* op = "rms_norm_backward";
    require_fp(X.dtype, op);
    if (dY.dtype != X.dtype) fail(op, "dY.dtype must match X.dtype");
    if (gamma.dtype != X.dtype && gamma.dtype != Dtype::FP32) {
        fail(op, "gamma.dtype must match X.dtype or be FP32");
    }
    if (dY.rows != X.rows || dY.cols != X.cols) fail(op, "dY shape mismatch");
    if (gamma.size() != X.cols) fail(op, "gamma must have D elements");
    ensure(dX, X.rows, X.cols, X.dtype);
    ensure_grad(dGamma, gamma, X.cols, gamma.dtype);
    if (X.rows == 0 || X.cols == 0) return;
    const NormDtypes id = norm_dtypes(X.dtype, gamma.dtype, X.dtype, op);
    Tensor rr = Tensor::empty_on(X.device, X.rows, 1, Dtype::FP32);
    NormArgs a{};
    a.x = X.data;
    a.xh = dY.data;
    a.y = dX.data;
    a.g = gamma.data;
    a.s1 = rr.data;
    a.rows = u32(X.rows, op);
    a.d = u32(X.cols, op);
    a.eps = eps;
    run(op, id, NORM_RMS_BWD, a);
    column_grads(op, id, NORM_COL_RMS, X.data, dY.data, rr.data, dGamma, nullptr,
                 a.rows, a.d);
}

void pixel_norm_forward(const Tensor& X, float eps, Tensor& Y) {
    constexpr const char* op = "pixel_norm_forward";
    require_fp(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    NormArgs a{};
    a.x = X.data;
    a.y = Y.data;
    a.rows = u32(X.rows, op);
    a.d = u32(X.cols, op);
    a.eps = eps;
    run(op, norm_dtypes(X.dtype, X.dtype, X.dtype, op), NORM_RMS, a);
}

void pixel_norm_backward(const Tensor& X, const Tensor& dY, float eps, Tensor& dX) {
    constexpr const char* op = "pixel_norm_backward";
    require_fp(X.dtype, op);
    if (dY.dtype != X.dtype || dY.rows != X.rows || dY.cols != X.cols) {
        fail(op, "dY must match X");
    }
    ensure(dX, X.rows, X.cols, X.dtype);
    NormArgs a{};
    a.x = X.data;
    a.xh = dY.data;
    a.y = dX.data;
    a.rows = u32(X.rows, op);
    a.d = u32(X.cols, op);
    a.eps = eps;
    run(op, norm_dtypes(X.dtype, X.dtype, X.dtype, op), NORM_RMS_BWD, a);
}

void l2_norm_forward(const Tensor& X, int head_dim, int num_heads, float eps, Tensor& Y) {
    constexpr const char* op = "l2_norm_forward";
    require_fp(X.dtype, op);
    if (head_dim <= 0 || num_heads <= 0 || X.cols != head_dim * num_heads) {
        fail(op, "X.cols must equal head_dim * num_heads");
    }
    ensure(Y, X.rows, X.cols, X.dtype);
    NormArgs a{};
    a.x = X.data;
    a.y = Y.data;
    a.rows = u32(static_cast<long long>(X.rows) * num_heads, op);
    a.d = u32(head_dim, op);
    a.eps = eps;
    run(op, norm_dtypes(X.dtype, X.dtype, X.dtype, op), NORM_L2, a);
}

void l2_norm_backward(const Tensor& X, int head_dim, int num_heads, float eps,
                      const Tensor& dY, Tensor& dX) {
    constexpr const char* op = "l2_norm_backward";
    require_fp(X.dtype, op);
    if (dY.dtype != X.dtype) fail(op, "dY must match X.dtype");
    if (head_dim <= 0 || num_heads <= 0 ||
        X.cols != head_dim * num_heads || dY.size() != X.size()) {
        fail(op, "X.cols must equal head_dim * num_heads and dY match X");
    }
    ensure(dX, X.rows, X.cols, X.dtype);
    NormArgs a{};
    a.x = X.data;
    a.xh = dY.data;
    a.y = dX.data;
    a.rows = u32(static_cast<long long>(X.rows) * num_heads, op);
    a.d = u32(head_dim, op);
    a.eps = eps;
    run(op, norm_dtypes(X.dtype, X.dtype, X.dtype, op), NORM_L2_BWD, a);
}

// ─── Softmax ───────────────────────────────────────────────────────────────

void softmax_forward(const Tensor& logits, Tensor& probs, const float* d_mask) {
    constexpr const char* op = "softmax_forward";
    require_fp(logits.dtype, op);
    ensure(probs, logits.rows, logits.cols, logits.dtype);
    if (d_mask) {
        NSUInteger ofs = 0;
        metal_impl::pool_require(d_mask, ofs, op, "mask");
    }
    NormArgs a{};
    a.x = logits.data;
    a.y = probs.data;
    a.s0 = d_mask;
    a.rows = 1;
    a.d = u32(logits.size(), op);
    run(op, norm_dtypes(logits.dtype, logits.dtype, logits.dtype, op), NORM_SOFTMAX, a);
}

void softmax_rows_forward(const Tensor& X, Tensor& Y, int rows, int cols) {
    constexpr const char* op = "softmax_rows_forward";
    require_fp(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    if (rows <= 0 || cols <= 0) return;
    if (static_cast<long long>(rows) * cols > X.size()) {
        fail(op, "rows * cols exceeds X");
    }
    NormArgs a{};
    a.x = X.data;
    a.y = Y.data;
    a.rows = u32(rows, op);
    a.d = u32(cols, op);
    run(op, norm_dtypes(X.dtype, X.dtype, X.dtype, op), NORM_SOFTMAX, a);
}

void softmax_backward(const Tensor& probs, const Tensor& dProbs, Tensor& dLogits) {
    constexpr const char* op = "softmax_backward";
    require_fp(probs.dtype, op);
    if (dProbs.dtype != probs.dtype || dProbs.size() != probs.size()) {
        fail(op, "dProbs must match probs");
    }
    ensure(dLogits, probs.rows, probs.cols, probs.dtype);
    NormArgs a{};
    a.x = probs.data;
    a.xh = dProbs.data;
    a.y = dLogits.data;
    a.rows = 1;
    a.d = u32(probs.size(), op);
    run(op, norm_dtypes(probs.dtype, probs.dtype, probs.dtype, op), NORM_SM_BWD, a);
}

}  // namespace brotensor::detail::metal
