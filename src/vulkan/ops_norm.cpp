// Vulkan normalisations and softmax: the LayerNorm family (vector, batched
// inference with and without beta, FP16 inference, training with caches and
// both backwards), RMSNorm, per-head L2 norm, pixel norm, and the vector /
// row softmax with its backward. One kernel (shaders/norm.comp), one
// workgroup per row, FP32 accumulation, two-pass variance. Contracts: FP32 / FP16 / BF16
// storage, LayerNorm caches (Mean_R, Rstd_R) in FP32, RMSNorm gamma in X's
// dtype or FP32, gradients of gamma / beta accumulated (+=).

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct NormPush {
    std::uint64_t x, y, g, b, xh, s0, s1;
    std::uint32_t rows, d;
    float eps, rstd;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

// norm.comp variant for (X-side, gamma, forward output) dtypes.
ShaderId norm_shader(Dtype dx, Dtype dg, Dtype dy, const char* op) {
    dt_code(dx, op);
    if (dg == dx && dy == dx) return dt_variant(ShaderId::norm_f32, dx, op);
    if (dg == Dtype::FP32 && dy == dx && dx == Dtype::FP16) return ShaderId::norm_f16_g32;
    if (dg == Dtype::FP32 && dy == dx && dx == Dtype::BF16) return ShaderId::norm_bf16_g32;
    if (dx == Dtype::FP32 && dg == Dtype::FP32 && dy == Dtype::FP16) return ShaderId::norm_f32_y16;
    fail(op, "unsupported dtype combination");
}

void run(const Tensor& on, ShaderId id, std::uint32_t mode, const NormPush& pc) {
    if (pc.rows == 0 || pc.d == 0) return;
    DeviceCtx& d = device_of(on);
    const Kernel& k = d.pipelines().get(id, {mode});
    const bool cols = mode >= NORM_COL_LN;
    const std::uint32_t groups = cols ? (pc.d + 31) / 32 : pc.rows;
    launch(d, k, pc, std::min<std::uint32_t>(groups, 65535));
}

std::uint32_t u32(long long v, const char* op) {
    if (v < 0 || v > 0x7fffffffLL) fail(op, "dimension out of range");
    return static_cast<std::uint32_t>(v);
}

void ensure(Tensor& t, int r, int c, Dtype dt) {
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(r, c, dt);
}

// A gradient accumulator: kept when it already has n elements of dt,
// otherwise reshaped like `like` and zeroed.
void ensure_grad(Tensor& g, const Tensor& like, long long n, Dtype dt) {
    if (g.size() != n || g.dtype != dt) {
        g.resize(like.rows, like.cols, dt);
        g.zero();
    }
}

// Column pass: dGamma (and dBeta) += sums over rows.
void column_grads(const Tensor& on, ShaderId id, std::uint32_t mode, std::uint64_t x, std::uint64_t xh,
                  std::uint64_t s1, Tensor& dG, Tensor* dB, std::uint32_t rows, std::uint32_t d) {
    NormPush pc{};
    pc.x = x;
    pc.xh = xh;
    pc.s1 = s1;
    pc.g = addr(dG.data);
    pc.b = dB ? addr(dB->data) : 0;
    pc.rows = rows;
    pc.d = d;
    run(on, id, mode, pc);
}

void layernorm_rows(const char* op, const Tensor& X, const Tensor& gamma, const Tensor* beta, Tensor& Y,
                    Tensor* Xhat, Tensor* Mean, Tensor* Rstd, float eps) {
    const std::uint32_t R = u32(X.rows, op), D = u32(X.cols, op);
    if (gamma.size() != X.cols || (beta && beta->size() != X.cols)) fail(op, "gamma / beta must have X.cols elements");
    if (gamma.dtype != X.dtype || (beta && beta->dtype != X.dtype)) fail(op, "gamma / beta dtype must match X");
    NormPush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.g = addr(gamma.data);
    pc.b = beta ? addr(beta->data) : 0;
    pc.xh = Xhat ? addr(Xhat->data) : 0;
    pc.s0 = Mean ? addr(Mean->data) : 0;
    pc.s1 = Rstd ? addr(Rstd->data) : 0;
    pc.rows = R;
    pc.d = D;
    pc.eps = eps;
    run(X, norm_shader(X.dtype, gamma.dtype, Y.dtype, op), NORM_LN, pc);
}

}  // namespace

// out (any shape, X.cols elements) += column sums of X, FP32 accumulation and
// one rounding into out's dtype (X's): bias gradients of the linear ops.
void column_sum_accumulate(const Tensor& X, Tensor& out) {
    constexpr const char* op = "column_sum";
    if (out.dtype != X.dtype || out.size() != X.cols) fail(op, "out must have X.cols elements of X's dtype");
    const ShaderId id = norm_shader(X.dtype, X.dtype, X.dtype, op);
    column_grads(X, id, NORM_COL_SUM, addr(X.data), 0, 0, out, nullptr, u32(X.rows, op), u32(X.cols, op));
}

// ─── LayerNorm ─────────────────────────────────────────────────────────────

void layernorm_forward_inference_batched(const Tensor& X, const Tensor& gamma, const Tensor* beta, Tensor& Y,
                                         float eps) {
    constexpr const char* op = "layernorm_forward_inference_batched";
    dt_code(X.dtype, op);
    // FP32 X may write a pre-typed FP16 Y; anything else gets X's dtype.
    if (Y.rows != X.rows || Y.cols != X.cols || (Y.dtype != X.dtype && !(X.dtype == Dtype::FP32 && Y.dtype == Dtype::FP16))) {
        Y.resize(X.rows, X.cols, X.dtype);
    }
    layernorm_rows(op, X, gamma, beta, Y, nullptr, nullptr, nullptr, eps);
}

void layernorm_forward_inference_batched_fp16(const Tensor& X, const Tensor& gamma, const Tensor* beta, Tensor& Y,
                                              float eps) {
    constexpr const char* op = "layernorm_forward_inference_batched_fp16";
    if (X.dtype != Dtype::FP16) fail(op, "all tensors must be FP16");
    ensure(Y, X.rows, X.cols, Dtype::FP16);
    layernorm_rows(op, X, gamma, beta, Y, nullptr, nullptr, nullptr, eps);
}

void layernorm_forward_batched_with_caches(const Tensor& X, const Tensor& gamma, const Tensor& beta, Tensor& Y,
                                           Tensor& Xhat, Tensor& Mean, Tensor& Rstd, float eps) {
    constexpr const char* op = "layernorm_forward_batched_with_caches";
    dt_code(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    ensure(Xhat, X.rows, X.cols, X.dtype);
    ensure(Mean, X.rows, 1, Dtype::FP32);
    ensure(Rstd, X.rows, 1, Dtype::FP32);
    layernorm_rows(op, X, gamma, &beta, Y, &Xhat, &Mean, &Rstd, eps);
}

void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta, Tensor& y, Tensor& xhat,
                       float& mean_out, float& rstd_out, float eps) {
    constexpr const char* op = "layernorm_forward";
    dt_code(x.dtype, op);
    ensure(y, x.rows, x.cols, x.dtype);
    ensure(xhat, x.rows, x.cols, x.dtype);
    if (x.size() == 0) { mean_out = 0.0f; rstd_out = 0.0f; return; }
    // The vector as one row; mean and rstd come back to the host (a sync).
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

void layernorm_backward_batched_with_caches(const Tensor& dY, const Tensor& Xhat, const Tensor& gamma,
                                            const Tensor& Rstd, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "layernorm_backward_batched_with_caches";
    dt_code(dY.dtype, op);
    if (Xhat.dtype != dY.dtype || gamma.dtype != dY.dtype) fail(op, "Xhat / gamma dtype must match dY");
    if (Rstd.dtype != Dtype::FP32) fail(op, "Rstd_R must be FP32");
    const std::uint32_t R = u32(dY.rows, op), D = u32(dY.cols, op);
    if (Xhat.rows != dY.rows || Xhat.cols != dY.cols || gamma.size() != dY.cols || Rstd.size() != dY.rows) fail(op, "shape mismatch");
    ensure(dX, dY.rows, dY.cols, dY.dtype);
    ensure_grad(dGamma, gamma, D, dY.dtype);
    ensure_grad(dBeta, gamma, D, dY.dtype);
    const ShaderId id = norm_shader(dY.dtype, dY.dtype, dY.dtype, op);
    NormPush pc{};
    pc.x = addr(dY.data);
    pc.y = addr(dX.data);
    pc.g = addr(gamma.data);
    pc.xh = addr(Xhat.data);
    pc.s1 = addr(Rstd.data);
    pc.rows = R;
    pc.d = D;
    run(dY, id, NORM_LN_BWD, pc);
    column_grads(dY, id, NORM_COL_LN, addr(dY.data), addr(Xhat.data), 0, dGamma, &dBeta, R, D);
}

void layernorm_backward(const Tensor& dY, const Tensor& xhat, const Tensor& gamma, float rstd, Tensor& dX,
                        Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "layernorm_backward";
    dt_code(dY.dtype, op);
    if (xhat.dtype != dY.dtype || gamma.dtype != dY.dtype) fail(op, "xhat / gamma dtype must match dY");
    const std::uint32_t n = u32(dY.size(), op);
    if (xhat.size() != dY.size() || gamma.size() != dY.size()) fail(op, "size mismatch");
    ensure(dX, dY.rows, dY.cols, dY.dtype);
    ensure_grad(dGamma, gamma, n, dY.dtype);
    ensure_grad(dBeta, gamma, n, dY.dtype);
    const ShaderId id = norm_shader(dY.dtype, dY.dtype, dY.dtype, op);
    NormPush pc{};
    pc.x = addr(dY.data);
    pc.y = addr(dX.data);
    pc.g = addr(gamma.data);
    pc.xh = addr(xhat.data);
    pc.rows = 1;
    pc.d = n;
    pc.rstd = rstd;
    run(dY, id, NORM_LN_BWD, pc);
    column_grads(dY, id, NORM_COL_LN, addr(dY.data), addr(xhat.data), 0, dGamma, &dBeta, 1, n);
}

// ─── RMSNorm, L2 norm, pixel norm ──────────────────────────────────────────

void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y) {
    constexpr const char* op = "rms_norm_forward";
    dt_code(X.dtype, op);
    if (gamma.dtype != X.dtype && gamma.dtype != Dtype::FP32) fail(op, "gamma.dtype must match X.dtype or be FP32");
    if (gamma.size() != X.cols) fail(op, "gamma must have D elements");
    ensure(Y, X.rows, X.cols, X.dtype);
    NormPush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.g = addr(gamma.data);
    pc.rows = u32(X.rows, op);
    pc.d = u32(X.cols, op);
    pc.eps = eps;
    run(X, norm_shader(X.dtype, gamma.dtype, X.dtype, op), NORM_RMS, pc);
}

void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY, float eps, Tensor& dX,
                       Tensor& dGamma) {
    constexpr const char* op = "rms_norm_backward";
    dt_code(X.dtype, op);
    if (dY.dtype != X.dtype) fail(op, "dY.dtype must match X.dtype");
    if (gamma.dtype != X.dtype && gamma.dtype != Dtype::FP32) fail(op, "gamma.dtype must match X.dtype or be FP32");
    if (dY.rows != X.rows || dY.cols != X.cols) fail(op, "dY shape mismatch");
    if (gamma.size() != X.cols) fail(op, "gamma must have D elements");
    ensure(dX, X.rows, X.cols, X.dtype);
    ensure_grad(dGamma, gamma, X.cols, gamma.dtype);
    if (X.rows == 0 || X.cols == 0) return;
    const ShaderId id = norm_shader(X.dtype, gamma.dtype, X.dtype, op);
    Tensor rr = Tensor::empty_on(X.device, X.rows, 1, Dtype::FP32);
    NormPush pc{};
    pc.x = addr(X.data);
    pc.xh = addr(dY.data);
    pc.y = addr(dX.data);
    pc.g = addr(gamma.data);
    pc.s1 = addr(rr.data);
    pc.rows = u32(X.rows, op);
    pc.d = u32(X.cols, op);
    pc.eps = eps;
    run(X, id, NORM_RMS_BWD, pc);
    column_grads(X, id, NORM_COL_RMS, addr(X.data), addr(dY.data), addr(rr.data), dGamma, nullptr, pc.rows, pc.d);
}

void pixel_norm_forward(const Tensor& X, float eps, Tensor& Y) {
    constexpr const char* op = "pixel_norm_forward";
    dt_code(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    NormPush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.rows = u32(X.rows, op);
    pc.d = u32(X.cols, op);
    pc.eps = eps;
    run(X, norm_shader(X.dtype, X.dtype, X.dtype, op), NORM_RMS, pc);
}

void pixel_norm_backward(const Tensor& X, const Tensor& dY, float eps, Tensor& dX) {
    constexpr const char* op = "pixel_norm_backward";
    dt_code(X.dtype, op);
    if (dY.dtype != X.dtype || dY.rows != X.rows || dY.cols != X.cols) fail(op, "dY must match X");
    ensure(dX, X.rows, X.cols, X.dtype);
    NormPush pc{};
    pc.x = addr(X.data);
    pc.xh = addr(dY.data);
    pc.y = addr(dX.data);
    pc.rows = u32(X.rows, op);
    pc.d = u32(X.cols, op);
    pc.eps = eps;
    run(X, norm_shader(X.dtype, X.dtype, X.dtype, op), NORM_RMS_BWD, pc);
}

void l2_norm_forward(const Tensor& X, int head_dim, int num_heads, float eps, Tensor& Y) {
    constexpr const char* op = "l2_norm_forward";
    dt_code(X.dtype, op);
    if (head_dim <= 0 || num_heads <= 0 || X.cols != head_dim * num_heads) {
        fail(op, "X.cols must equal head_dim * num_heads");
    }
    ensure(Y, X.rows, X.cols, X.dtype);
    NormPush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.rows = u32(static_cast<long long>(X.rows) * num_heads, op);
    pc.d = u32(head_dim, op);
    pc.eps = eps;
    run(X, norm_shader(X.dtype, X.dtype, X.dtype, op), NORM_L2, pc);
}

void l2_norm_backward(const Tensor& X, int head_dim, int num_heads, float eps, const Tensor& dY, Tensor& dX) {
    constexpr const char* op = "l2_norm_backward";
    dt_code(X.dtype, op);
    if (dY.dtype != X.dtype) fail(op, "dY must match X.dtype");
    if (head_dim <= 0 || num_heads <= 0 || X.cols != head_dim * num_heads || dY.size() != X.size()) {
        fail(op, "X.cols must equal head_dim * num_heads and dY match X");
    }
    ensure(dX, X.rows, X.cols, X.dtype);
    NormPush pc{};
    pc.x = addr(X.data);
    pc.xh = addr(dY.data);
    pc.y = addr(dX.data);
    pc.rows = u32(static_cast<long long>(X.rows) * num_heads, op);
    pc.d = u32(head_dim, op);
    pc.eps = eps;
    run(X, norm_shader(X.dtype, X.dtype, X.dtype, op), NORM_L2_BWD, pc);
}

// ─── softmax ───────────────────────────────────────────────────────────────

void softmax_forward(const Tensor& logits, Tensor& probs, const float* d_mask) {
    constexpr const char* op = "softmax_forward";
    dt_code(logits.dtype, op);
    ensure(probs, logits.rows, logits.cols, logits.dtype);
    NormPush pc{};
    pc.x = addr(logits.data);
    pc.y = addr(probs.data);
    pc.s0 = addr(d_mask);
    pc.rows = 1;
    pc.d = u32(logits.size(), op);
    run(logits, norm_shader(logits.dtype, logits.dtype, logits.dtype, op), NORM_SOFTMAX, pc);
}

void softmax_rows_forward(const Tensor& X, Tensor& Y, int rows, int cols) {
    constexpr const char* op = "softmax_rows_forward";
    dt_code(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    if (rows <= 0 || cols <= 0) return;
    if (static_cast<long long>(rows) * cols > X.size()) fail(op, "rows * cols exceeds X");
    NormPush pc{};
    pc.x = addr(X.data);
    pc.y = addr(Y.data);
    pc.rows = u32(rows, op);
    pc.d = u32(cols, op);
    run(X, norm_shader(X.dtype, X.dtype, X.dtype, op), NORM_SOFTMAX, pc);
}

void softmax_backward(const Tensor& probs, const Tensor& dProbs, Tensor& dLogits) {
    constexpr const char* op = "softmax_backward";
    dt_code(probs.dtype, op);
    if (dProbs.dtype != probs.dtype || dProbs.size() != probs.size()) fail(op, "dProbs must match probs");
    ensure(dLogits, probs.rows, probs.cols, probs.dtype);
    NormPush pc{};
    pc.x = addr(probs.data);
    pc.xh = addr(dProbs.data);
    pc.y = addr(dLogits.data);
    pc.rows = 1;
    pc.d = u32(probs.size(), op);
    run(probs, norm_shader(probs.dtype, probs.dtype, probs.dtype, op), NORM_SM_BWD, pc);
}

void fill_vulkan_vtable_norm(::brotensor::detail::OpsVTable& v) {
    v.layernorm_forward = &layernorm_forward;
    v.layernorm_backward = &layernorm_backward;
    v.layernorm_forward_inference_batched = &layernorm_forward_inference_batched;
    v.layernorm_forward_inference_batched_fp16 = &layernorm_forward_inference_batched_fp16;
    v.layernorm_forward_batched_with_caches = &layernorm_forward_batched_with_caches;
    v.layernorm_backward_batched_with_caches = &layernorm_backward_batched_with_caches;
    v.rms_norm_forward = &rms_norm_forward;
    v.rms_norm_backward = &rms_norm_backward;
    v.pixel_norm_forward = &pixel_norm_forward;
    v.pixel_norm_backward = &pixel_norm_backward;
    v.l2_norm_forward = &l2_norm_forward;
    v.l2_norm_backward = &l2_norm_backward;
    v.softmax_forward = &softmax_forward;
    v.softmax_rows_forward = &softmax_rows_forward;
    v.softmax_backward = &softmax_backward;
}

}  // namespace brotensor::detail::vulkan
