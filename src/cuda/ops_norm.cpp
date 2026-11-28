// CUDA normalizations, L2/pixel norm, softmax, and column-sum ops backed by Brass MIR -> PTX.

#include "cu_mir_cache.h"
#include "cuda_driver.h"
#include "../kernels/mir_norm.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct NormLaunch {
    const void* x = nullptr;
    const void* y = nullptr;
    const void* g = nullptr;
    const void* b = nullptr;
    const void* xh = nullptr;
    const void* s0 = nullptr;
    const void* s1 = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t d = 0;
    float eps = 1e-5f;
    float rstd = 0.0f;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

inline std::uintptr_t addr(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "unsupported dtype (expected FP32/FP16/BF16)");
    }
}

struct NormDtypes {
    Dtype x, g, y;
};

NormDtypes norm_dts(Dtype dx, Dtype dg, Dtype dy, const char* op) {
    require_fp(dx, op);
    if ((dg == dx && dy == dx) ||
        (dg == Dtype::FP32 && dy == dx && (dx == Dtype::FP16 || dx == Dtype::BF16)) ||
        (dx == Dtype::FP32 && dg == Dtype::FP32 && dy == Dtype::FP16)) {
        return {dx, dg, dy};
    }
    fail(op, "unsupported dtype combination");
}

void run_norm(NormDtypes dts, std::uint32_t mode, const NormLaunch& nl) {
    if (nl.rows == 0 || nl.d == 0) return;
    const bool cols = (mode >= NORM_COL_LN);
    const int lanes = cols
                          ? 1
                          : mir_vec_lanes_norm(
                                dts.x, dts.y, dts.g, nl.d, addr(nl.x), addr(nl.y),
                                addr(nl.g), addr(nl.b), addr(nl.xh),
                                (mode == NORM_SOFTMAX) ? addr(nl.s0) : 0u);
    const std::uint8_t flags = pack_norm_flags(
        nl.g != nullptr, nl.b != nullptr, nl.xh != nullptr, nl.s0 != nullptr, nl.s1 != nullptr);
    CUfunction fn = mir_function(
        {MirKind::Norm, mode, dts.x, dts.y, static_cast<std::uint8_t>(lanes), dts.g, flags});
    const std::uint32_t groups = cols ? (nl.d + 31u) / 32u : nl.rows;
    mir_dispatch_norm(
        fn, (std::min)(groups, 65535u),
        nl.x, nl.y, nl.g, nl.b, nl.xh, nl.s0, nl.s1,
        nl.rows, nl.d, nl.eps, nl.rstd);
}

std::uint32_t u32(long long v, const char* op) {
    if (v < 0 || v > 0x7fffffffLL) fail(op, "dimension out of range");
    return static_cast<std::uint32_t>(v);
}

void ensure(Tensor& t, int r, int c, Dtype dt) {
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(r, c, dt);
}

void ensure_grad(Tensor& g, const Tensor& like, long long n, Dtype dt, const char* op) {
    if (g.size() != n) {
        g.resize(like.rows, like.cols, dt);
        g.zero();
    } else if (g.dtype != dt) {
        fail(op, "gradient dtype mismatch");
    }
}

void column_grads(NormDtypes dts, std::uint32_t mode, const void* x, const void* xh,
                  const void* s1, Tensor& dG, Tensor* dB, std::uint32_t rows, std::uint32_t d) {
    NormLaunch nl{};
    nl.x = x;
    nl.xh = xh;
    nl.s1 = s1;
    nl.g = dG.data;
    nl.b = dB ? dB->data : nullptr;
    nl.rows = rows;
    nl.d = d;
    run_norm(dts, mode, nl);
}

void layernorm_rows(const char* op, const Tensor& X, const Tensor& gamma, const Tensor* beta, Tensor& Y,
                    Tensor* Xhat, Tensor* Mean, Tensor* Rstd, float eps) {
    const std::uint32_t R = u32(X.rows, op);
    const std::uint32_t D = u32(X.cols, op);
    if (gamma.size() != X.cols || (beta && beta->size() != X.cols)) {
        fail(op, "gamma / beta must have X.cols elements");
    }
    if (gamma.dtype != X.dtype || (beta && beta->dtype != X.dtype)) {
        fail(op, "gamma / beta dtype must match X");
    }
    NormLaunch nl{};
    nl.x = X.data;
    nl.y = Y.data;
    nl.g = gamma.data;
    nl.b = beta ? beta->data : nullptr;
    nl.xh = Xhat ? Xhat->data : nullptr;
    nl.s0 = Mean ? Mean->data : nullptr;
    nl.s1 = Rstd ? Rstd->data : nullptr;
    nl.rows = R;
    nl.d = D;
    nl.eps = eps;
    run_norm(norm_dts(X.dtype, gamma.dtype, Y.dtype, op), NORM_LN, nl);
}

}  // namespace

// ─── LayerNorm ─────────────────────────────────────────────────────────────

void layernorm_forward_inference_batched(const Tensor& X_RD, const Tensor& gamma,
                                         const Tensor* beta, Tensor& Y_RD, float eps) {
    constexpr const char* op = "layernorm_forward_inference_batched";
    require_fp(X_RD.dtype, op);
    if (gamma.dtype != X_RD.dtype || (beta && beta->dtype != X_RD.dtype)) {
        fail(op, "gamma/beta dtype must match X.dtype");
    }
    const bool mixed = (X_RD.dtype == Dtype::FP32 && Y_RD.dtype == Dtype::FP16);
    if (Y_RD.rows != X_RD.rows || Y_RD.cols != X_RD.cols || (Y_RD.dtype != X_RD.dtype && !mixed)) {
        Y_RD.resize(X_RD.rows, X_RD.cols, X_RD.dtype);
    }
    layernorm_rows(op, X_RD, gamma, beta, Y_RD, nullptr, nullptr, nullptr, eps);
}

void layernorm_forward_inference_batched_fp16(const Tensor& X_RD, const Tensor& gamma,
                                              const Tensor* beta, Tensor& Y_RD, float eps) {
    constexpr const char* op = "layernorm_forward_inference_batched_fp16";
    if (X_RD.dtype != Dtype::FP16 || gamma.dtype != Dtype::FP16 ||
        (beta && beta->dtype != Dtype::FP16)) {
        fail(op, "all tensors must be FP16");
    }
    ensure(Y_RD, X_RD.rows, X_RD.cols, Dtype::FP16);
    layernorm_rows(op, X_RD, gamma, beta, Y_RD, nullptr, nullptr, nullptr, eps);
}

void layernorm_forward_batched_with_caches(const Tensor& X_RD, const Tensor& gamma,
                                           const Tensor& beta, Tensor& Y_RD,
                                           Tensor& Xhat_RD, Tensor& Mean_R,
                                           Tensor& Rstd_R, float eps) {
    constexpr const char* op = "layernorm_forward_batched_with_caches";
    require_fp(X_RD.dtype, op);
    ensure(Y_RD, X_RD.rows, X_RD.cols, X_RD.dtype);
    ensure(Xhat_RD, X_RD.rows, X_RD.cols, X_RD.dtype);
    ensure(Mean_R, X_RD.rows, 1, Dtype::FP32);
    ensure(Rstd_R, X_RD.rows, 1, Dtype::FP32);
    layernorm_rows(op, X_RD, gamma, &beta, Y_RD, &Xhat_RD, &Mean_R, &Rstd_R, eps);
}

void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       Tensor& y, Tensor& xhat,
                       float& mean_out, float& rstd_out, float eps) {
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

void layernorm_backward_batched_with_caches(const Tensor& dY_RD, const Tensor& Xhat_RD,
                                            const Tensor& gamma, const Tensor& Rstd_R,
                                            Tensor& dX_RD, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "layernorm_backward_batched_with_caches";
    require_fp(dY_RD.dtype, op);
    if (Xhat_RD.dtype != dY_RD.dtype || gamma.dtype != dY_RD.dtype) {
        fail(op, "dY/Xhat/gamma must share dtype");
    }
    if (Rstd_R.dtype != Dtype::FP32) fail(op, "Rstd_R must be FP32");
    const std::uint32_t R = u32(dY_RD.rows, op);
    const std::uint32_t D = u32(dY_RD.cols, op);
    if (Xhat_RD.rows != dY_RD.rows || Xhat_RD.cols != dY_RD.cols ||
        gamma.size() != dY_RD.cols || Rstd_R.size() != dY_RD.rows) {
        fail(op, "shape mismatch");
    }
    ensure(dX_RD, dY_RD.rows, dY_RD.cols, dY_RD.dtype);
    ensure_grad(dGamma, gamma, D, dY_RD.dtype, op);
    ensure_grad(dBeta, gamma, D, dY_RD.dtype, op);
    if (R == 0 || D == 0) return;
    const NormDtypes id = norm_dts(dY_RD.dtype, dY_RD.dtype, dY_RD.dtype, op);
    NormLaunch nl{};
    nl.x = dY_RD.data;
    nl.y = dX_RD.data;
    nl.g = gamma.data;
    nl.xh = Xhat_RD.data;
    nl.s1 = Rstd_R.data;
    nl.rows = R;
    nl.d = D;
    run_norm(id, NORM_LN_BWD, nl);
    column_grads(id, NORM_COL_LN, dY_RD.data, Xhat_RD.data, nullptr, dGamma, &dBeta, R, D);
}

void layernorm_backward(const Tensor& dY, const Tensor& xhat, const Tensor& gamma,
                        float rstd, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    constexpr const char* op = "layernorm_backward";
    require_fp(dY.dtype, op);
    if (xhat.dtype != dY.dtype || gamma.dtype != dY.dtype) {
        fail(op, "all tensors must share dtype");
    }
    const std::uint32_t n = u32(dY.size(), op);
    if (xhat.size() != dY.size() || gamma.size() != dY.size()) {
        fail(op, "size mismatch");
    }
    ensure(dX, dY.rows, dY.cols, dY.dtype);
    ensure_grad(dGamma, gamma, n, dY.dtype, op);
    ensure_grad(dBeta, gamma, n, dY.dtype, op);
    if (n == 0) return;
    const NormDtypes id = norm_dts(dY.dtype, dY.dtype, dY.dtype, op);
    NormLaunch nl{};
    nl.x = dY.data;
    nl.y = dX.data;
    nl.g = gamma.data;
    nl.xh = xhat.data;
    nl.rows = 1;
    nl.d = n;
    nl.rstd = rstd;
    run_norm(id, NORM_LN_BWD, nl);
    column_grads(id, NORM_COL_LN, dY.data, xhat.data, nullptr, dGamma, &dBeta, 1, n);
}

// ─── RMSNorm, pixel_norm, l2_norm ──────────────────────────────────────────

void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y) {
    constexpr const char* op = "rms_norm_forward";
    require_fp(X.dtype, op);
    if (gamma.dtype != X.dtype && gamma.dtype != Dtype::FP32) {
        fail(op, "gamma.dtype must match X.dtype or be FP32");
    }
    if (gamma.size() != X.cols) fail(op, "gamma must have D elements");
    ensure(Y, X.rows, X.cols, X.dtype);
    NormLaunch nl{};
    nl.x = X.data;
    nl.y = Y.data;
    nl.g = gamma.data;
    nl.rows = u32(X.rows, op);
    nl.d = u32(X.cols, op);
    nl.eps = eps;
    run_norm(norm_dts(X.dtype, gamma.dtype, X.dtype, op), NORM_RMS, nl);
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
    ensure_grad(dGamma, gamma, X.cols, gamma.dtype, op);
    if (X.rows == 0 || X.cols == 0) return;
    const NormDtypes id = norm_dts(X.dtype, gamma.dtype, X.dtype, op);
    Tensor rr = Tensor::empty_on(X.device, X.rows, 1, Dtype::FP32);
    NormLaunch nl{};
    nl.x = X.data;
    nl.xh = dY.data;
    nl.y = dX.data;
    nl.g = gamma.data;
    nl.s1 = rr.data;
    nl.rows = u32(X.rows, op);
    nl.d = u32(X.cols, op);
    nl.eps = eps;
    run_norm(id, NORM_RMS_BWD, nl);
    column_grads(id, NORM_COL_RMS, X.data, dY.data, rr.data, dGamma, nullptr, nl.rows, nl.d);
}

void pixel_norm_forward(const Tensor& X, float eps, Tensor& Y) {
    constexpr const char* op = "pixel_norm_forward";
    require_fp(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    NormLaunch nl{};
    nl.x = X.data;
    nl.y = Y.data;
    nl.rows = u32(X.rows, op);
    nl.d = u32(X.cols, op);
    nl.eps = eps;
    run_norm(norm_dts(X.dtype, X.dtype, X.dtype, op), NORM_RMS, nl);
}

void pixel_norm_backward(const Tensor& X, const Tensor& dY, float eps, Tensor& dX) {
    constexpr const char* op = "pixel_norm_backward";
    require_fp(X.dtype, op);
    if (dY.dtype != X.dtype || dY.rows != X.rows || dY.cols != X.cols) {
        fail(op, "dY must match X");
    }
    ensure(dX, X.rows, X.cols, X.dtype);
    NormLaunch nl{};
    nl.x = X.data;
    nl.xh = dY.data;
    nl.y = dX.data;
    nl.rows = u32(X.rows, op);
    nl.d = u32(X.cols, op);
    nl.eps = eps;
    run_norm(norm_dts(X.dtype, X.dtype, X.dtype, op), NORM_RMS_BWD, nl);
}

void l2_norm_forward(const Tensor& X, int head_dim, int num_heads, float eps, Tensor& Y) {
    constexpr const char* op = "l2_norm_forward";
    require_fp(X.dtype, op);
    if (head_dim <= 0 || num_heads <= 0 || X.cols != head_dim * num_heads) {
        fail(op, "X.cols must equal head_dim * num_heads");
    }
    ensure(Y, X.rows, X.cols, X.dtype);
    NormLaunch nl{};
    nl.x = X.data;
    nl.y = Y.data;
    nl.rows = u32(static_cast<long long>(X.rows) * num_heads, op);
    nl.d = u32(head_dim, op);
    nl.eps = eps;
    run_norm(norm_dts(X.dtype, X.dtype, X.dtype, op), NORM_L2, nl);
}

void l2_norm_backward(const Tensor& X, int head_dim, int num_heads, float eps,
                      const Tensor& dY, Tensor& dX) {
    constexpr const char* op = "l2_norm_backward";
    require_fp(X.dtype, op);
    if (dY.dtype != X.dtype) fail(op, "dY must match X.dtype");
    if (head_dim <= 0 || num_heads <= 0 || X.cols != head_dim * num_heads || dY.size() != X.size()) {
        fail(op, "X.cols must equal head_dim * num_heads and dY match X");
    }
    ensure(dX, X.rows, X.cols, X.dtype);
    NormLaunch nl{};
    nl.x = X.data;
    nl.xh = dY.data;
    nl.y = dX.data;
    nl.rows = u32(static_cast<long long>(X.rows) * num_heads, op);
    nl.d = u32(head_dim, op);
    nl.eps = eps;
    run_norm(norm_dts(X.dtype, X.dtype, X.dtype, op), NORM_L2_BWD, nl);
}

// ─── softmax ───────────────────────────────────────────────────────────────

void softmax_forward(const Tensor& logits, Tensor& probs, const float* d_mask) {
    constexpr const char* op = "softmax_forward";
    require_fp(logits.dtype, op);
    if (d_mask != nullptr && !is_device_or_managed_ptr(d_mask)) {
        fail(op, "d_mask must be a device pointer");
    }
    ensure(probs, logits.rows, logits.cols, logits.dtype);
    NormLaunch nl{};
    nl.x = logits.data;
    nl.y = probs.data;
    nl.s0 = d_mask;
    nl.rows = 1;
    nl.d = u32(logits.size(), op);
    run_norm(norm_dts(logits.dtype, logits.dtype, logits.dtype, op), NORM_SOFTMAX, nl);
}

void softmax_rows_forward(const Tensor& X, Tensor& Y, int rows, int cols) {
    constexpr const char* op = "softmax_rows_forward";
    require_fp(X.dtype, op);
    ensure(Y, X.rows, X.cols, X.dtype);
    if (rows <= 0 || cols <= 0) return;
    if (static_cast<long long>(rows) * cols > X.size()) fail(op, "rows * cols exceeds X");
    NormLaunch nl{};
    nl.x = X.data;
    nl.y = Y.data;
    nl.rows = u32(rows, op);
    nl.d = u32(cols, op);
    run_norm(norm_dts(X.dtype, X.dtype, X.dtype, op), NORM_SOFTMAX, nl);
}

void softmax_backward(const Tensor& probs, const Tensor& dProbs, Tensor& dLogits) {
    constexpr const char* op = "softmax_backward";
    require_fp(probs.dtype, op);
    if (dProbs.dtype != probs.dtype || dProbs.size() != probs.size()) {
        fail(op, "dProbs must match probs");
    }
    ensure(dLogits, probs.rows, probs.cols, probs.dtype);
    NormLaunch nl{};
    nl.x = probs.data;
    nl.xh = dProbs.data;
    nl.y = dLogits.data;
    nl.rows = 1;
    nl.d = u32(probs.size(), op);
    run_norm(norm_dts(probs.dtype, probs.dtype, probs.dtype, op), NORM_SM_BWD, nl);
}

// ─── vtable registration ───────────────────────────────────────────────────

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        int N, int C, int H, int W, int num_groups, float eps, Tensor& Y);
void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta);
void resblock_forward(const Tensor& X,
                      const Tensor& gamma1, const Tensor& beta1,
                      const Tensor& W1, const Tensor* b1,
                      const Tensor* t_emb_shift,
                      const Tensor& gamma2, const Tensor& beta2,
                      const Tensor& W2, const Tensor* b2,
                      const Tensor* Wskip, const Tensor* bskip,
                      int N, int C_in, int C_out, int H, int Wd,
                      int num_groups, float eps,
                      Tensor& Y);
void resblock_forward_int8w_fp16(const Tensor& X,
                                 const Tensor& gamma1, const Tensor& beta1,
                                 const Tensor& W1_int8, const Tensor& s1,
                                 const Tensor* b1,
                                 const Tensor* t_emb_shift,
                                 const Tensor& gamma2, const Tensor& beta2,
                                 const Tensor& W2_int8, const Tensor& s2,
                                 const Tensor* b2,
                                 const Tensor* Wskip_int8, const Tensor* sskip,
                                 const Tensor* bskip,
                                 int N, int C_in, int C_out, int H, int Wd,
                                 int num_groups, float eps,
                                 Tensor& Y);
void resblock_backward(const Tensor& X,
                       const Tensor& gamma1, const Tensor& beta1,
                       const Tensor& W1, const Tensor* b1,
                       const Tensor* t_emb_shift,
                       const Tensor& gamma2, const Tensor& beta2,
                       const Tensor& W2, const Tensor* b2,
                       const Tensor* Wskip, const Tensor* bskip,
                       int N, int C_in, int C_out, int H, int Wd,
                       int num_groups, float eps,
                       const Tensor& dY,
                       Tensor& dX,
                       Tensor& dGamma1, Tensor& dBeta1,
                       Tensor& dW1, Tensor* db1,
                       Tensor* dt_emb_shift,
                       Tensor& dGamma2, Tensor& dBeta2,
                       Tensor& dW2, Tensor* db2,
                       Tensor* dWskip, Tensor* dbskip);
void masked_mean_pool_forward(const Tensor& X, const float* d_mask, Tensor& y);
void masked_mean_pool_backward(const Tensor& dY, const float* d_mask, int K, Tensor& dX);

void fill_cuda_vtable_norms(::brotensor::detail::OpsVTable& v) {
    v.layernorm_forward                        = &layernorm_forward;
    v.layernorm_backward                       = &layernorm_backward;
    v.layernorm_forward_inference_batched      = &layernorm_forward_inference_batched;
    v.layernorm_forward_inference_batched_fp16 = &layernorm_forward_inference_batched_fp16;
    v.layernorm_forward_batched_with_caches    = &layernorm_forward_batched_with_caches;
    v.layernorm_backward_batched_with_caches   = &layernorm_backward_batched_with_caches;
    v.group_norm_forward                       = &group_norm_forward;
    v.group_norm_backward                      = &group_norm_backward;
    v.resblock_forward                         = &resblock_forward;
    v.resblock_backward                        = &resblock_backward;
    v.resblock_forward_int8w_fp16              = &resblock_forward_int8w_fp16;
    v.masked_mean_pool_forward                 = &masked_mean_pool_forward;
    v.masked_mean_pool_backward                = &masked_mean_pool_backward;
}

void fill_cuda_vtable_l2_norm(::brotensor::detail::OpsVTable& v) {
    v.l2_norm_forward  = &l2_norm_forward;
    v.l2_norm_backward = &l2_norm_backward;
}

}  // namespace brotensor::detail::cuda
