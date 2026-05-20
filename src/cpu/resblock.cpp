// CPU backend — diffusion ResBlock (CHUNK 6).
//
// Ground truth: src/cuda/resblock.cu — the FP32-weight resblock_forward and
// resblock_backward (NOT resblock_forward_int8w_fp16).
//
// DTYPE DECISIONS
//   The CUDA resblock runs FP16 internally (GroupNorm/SiLU/conv2d all FP16,
//   FP32 accumulators). The CPU backend is FP32-only (per CLAUDE.md). Both
//   CPU impls run the FP32 scalar math; the parity test quantises inputs
//   through FP16 so both backends start identical and compares with a loose
//   FP16-scale tolerance (the test_resblock.cpp smoke-test envelope:
//   atol=1e-2, rtol=1e-2). A diffusion ResBlock is a long composite chain
//   (two GroupNorms + two 3x3 convs), so FP16 rounding compounds — a tight
//   FP32 tolerance would not survive the FP16/FP32 cross-backend comparison.
//
// EXACT OP ORDER (verified against resblock.cu)
//   forward:
//     h1 = SiLU(GroupNorm(X, gamma1, beta1))                      [C_in]
//     h2 = conv2d(h1, W1, b1, 3x3 stride-1 pad-1)                 [C_out]
//     if t_emb_shift: h2 += broadcast(t_emb_shift)  — applied AFTER conv1,
//          BEFORE GroupNorm2; (N, C_out) per-(n,c) or (C_out,) per-c shift.
//     h3 = SiLU(GroupNorm(h2, gamma2, beta2))                     [C_out]
//     Y  = conv2d(h3, W2, b2, 3x3 stride-1 pad-1)                 [C_out]
//     skip = (Wskip ? conv2d(X, Wskip, bskip, 1x1) : X)
//     Y += skip
//   Wskip is required when C_in != C_out; null skip is the identity (C_in
//   must equal C_out).
//
// ACCUMULATION (verified against resblock.cu's composition of the conv2d /
// group_norm backward kernels)
//   * dX                 — OVERWRITTEN by the GN1 backward, then the skip
//                           path is added on top (dX += dY or dX += dX_skip).
//   * dGamma1/dBeta1/2    — ACCUMULATE (+=)   (group_norm_backward contract).
//   * dW1/dW2/dWskip      — ACCUMULATE (+=)   (conv2d_backward_weight).
//   * db1/db2/dbskip      — ACCUMULATE (+=)   (conv2d_backward_bias).
//   * dt_emb_shift        — ACCUMULATES (+=)  (CUDA folds prev value when
//                           reducing dh2 over the spatial / batch axes).
//   All grad buffers are caller-zeroed (or pre-seeded to test accumulation).

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::cpu {

// CPU primitives from sibling chunk files.
void group_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        int N, int C, int H, int W, int num_groups,
                        float eps, ::brotensor::Tensor& Y);
void group_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         ::brotensor::Tensor& dX, ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta);
void silu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void silu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX);
void conv2d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, ::brotensor::Tensor& Y);
void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups,
                           ::brotensor::Tensor& dX);
void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int dil_h, int dil_w, int groups,
                            ::brotensor::Tensor& dWt);
void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB);
void add_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x);

namespace {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

// Resolve a t_emb_shift tensor's broadcast mode. Returns true and sets has_N
// if the shape is (N, C_out); false (per-channel) if (C_out,) / (C_out,1) /
// (1,C_out). Throws on any other shape — mirrors resblock.cu.
bool temb_has_N(const Tensor& t, int N, int C_out) {
    if (t.rows == N && t.cols == C_out) return true;
    if ((t.rows == C_out && t.cols == 1) ||
        (t.rows == 1 && t.cols == C_out) ||
        t.size() == C_out)
        return false;
    throw std::runtime_error(
        "resblock: t_emb_shift shape must be (N, C_out) or (C_out,)");
}

// h2[n, c, p] += t_emb_shift broadcast over the spatial axis.
void add_temb_shift(float* h2, const float* shift,
                    int N, int C_out, int spatial, bool has_N) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_out; ++c) {
            const float sv = has_N ? shift[n * C_out + c] : shift[c];
            float* row = h2 + (static_cast<std::size_t>(n) * C_out + c) * spatial;
            for (int p = 0; p < spatial; ++p) row[p] += sv;
        }
    }
}

} // namespace

// ─── resblock_forward ──────────────────────────────────────────────────────

void resblock_forward(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& gamma1,
                      const ::brotensor::Tensor& beta1,
                      const ::brotensor::Tensor& W1,
                      const ::brotensor::Tensor* b1,
                      const ::brotensor::Tensor* t_emb_shift,
                      const ::brotensor::Tensor& gamma2,
                      const ::brotensor::Tensor& beta2,
                      const ::brotensor::Tensor& W2,
                      const ::brotensor::Tensor* b2,
                      const ::brotensor::Tensor* Wskip,
                      const ::brotensor::Tensor* bskip,
                      int N, int C_in, int C_out, int H, int W,
                      int num_groups, float eps,
                      ::brotensor::Tensor& Y) {
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0)
        throw std::runtime_error("resblock_forward: num_groups must divide C_in and C_out");
    if (Wskip == nullptr && C_in != C_out)
        throw std::runtime_error("resblock_forward: Wskip required when C_in != C_out");
    const int spatial = H * W;
    const int out_cols = C_out * spatial;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP32)
        Y.resize(N, out_cols, Dtype::FP32);
    if (N == 0 || spatial == 0) return;

    // Leg 1: GN1 → SiLU.
    Tensor h1;
    group_norm_forward(X, gamma1, beta1, N, C_in, H, W, num_groups, eps, h1);
    silu_forward(h1, h1);

    // Conv1: 3x3 same.
    Tensor h2;
    conv2d_forward(h1, W1, b1, N, C_in, H, W, C_out, 3, 3,
                   1, 1, 1, 1, 1, 1, 1, h2);

    // Optional t_emb shift (after conv1, before GN2).
    if (t_emb_shift) {
        const bool has_N = temb_has_N(*t_emb_shift, N, C_out);
        add_temb_shift(h2.host_f32_mut(), t_emb_shift->host_f32(),
                       N, C_out, spatial, has_N);
    }

    // Leg 2: GN2 → SiLU.
    Tensor h3;
    group_norm_forward(h2, gamma2, beta2, N, C_out, H, W, num_groups, eps, h3);
    silu_forward(h3, h3);

    // Conv2: 3x3 same → Y.
    conv2d_forward(h3, W2, b2, N, C_out, H, W, C_out, 3, 3,
                   1, 1, 1, 1, 1, 1, 1, Y);

    // Skip path.
    if (Wskip == nullptr) {
        add_inplace(Y, X);
    } else {
        Tensor skip;
        conv2d_forward(X, *Wskip, bskip, N, C_in, H, W, C_out, 1, 1,
                       1, 1, 0, 0, 1, 1, 1, skip);
        add_inplace(Y, skip);
    }
}

// ─── resblock_backward ─────────────────────────────────────────────────────

void resblock_backward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& gamma1,
                       const ::brotensor::Tensor& beta1,
                       const ::brotensor::Tensor& W1,
                       const ::brotensor::Tensor* b1,
                       const ::brotensor::Tensor* t_emb_shift,
                       const ::brotensor::Tensor& gamma2,
                       const ::brotensor::Tensor& beta2,
                       const ::brotensor::Tensor& W2,
                       const ::brotensor::Tensor* b2,
                       const ::brotensor::Tensor* Wskip,
                       const ::brotensor::Tensor* bskip,
                       int N, int C_in, int C_out, int H, int W,
                       int num_groups, float eps,
                       const ::brotensor::Tensor& dY,
                       ::brotensor::Tensor& dX,
                       ::brotensor::Tensor& dGamma1,
                       ::brotensor::Tensor& dBeta1,
                       ::brotensor::Tensor& dW1,
                       ::brotensor::Tensor* db1,
                       ::brotensor::Tensor* dt_emb_shift,
                       ::brotensor::Tensor& dGamma2,
                       ::brotensor::Tensor& dBeta2,
                       ::brotensor::Tensor& dW2,
                       ::brotensor::Tensor* db2,
                       ::brotensor::Tensor* dWskip,
                       ::brotensor::Tensor* dbskip) {
    // Forward biases b2/bskip do not affect any backward quantity (the conv2
    // and skip-conv outputs are never recomputed in the backward, and a bias
    // grad is a pure reduction of the downstream grad); accepted for API
    // symmetry with the forward and resblock.cu. b1 IS used — conv1 is
    // recomputed to obtain h2.
    (void)b2; (void)bskip;
    if (Wskip == nullptr && C_in != C_out)
        throw std::runtime_error("resblock_backward: Wskip required when C_in != C_out");
    const int spatial = H * W;
    if (dY.rows != N || dY.cols != C_out * spatial)
        throw std::runtime_error("resblock_backward: dY shape mismatch");
    if (dX.rows != N || dX.cols != C_in * spatial || dX.dtype != Dtype::FP32)
        dX.resize(N, C_in * spatial, Dtype::FP32);
    if (N == 0 || spatial == 0) return;

    // ── Recompute forward intermediates. ──
    Tensor h1_pre_silu, h1;
    group_norm_forward(X, gamma1, beta1, N, C_in, H, W, num_groups, eps,
                       h1_pre_silu);
    silu_forward(h1_pre_silu, h1);

    Tensor h2;
    conv2d_forward(h1, W1, b1, N, C_in, H, W, C_out, 3, 3,
                   1, 1, 1, 1, 1, 1, 1, h2);
    bool temb_N = false;
    if (t_emb_shift) {
        temb_N = temb_has_N(*t_emb_shift, N, C_out);
        add_temb_shift(h2.host_f32_mut(), t_emb_shift->host_f32(),
                       N, C_out, spatial, temb_N);
    }

    Tensor h3_pre_silu, h3;
    group_norm_forward(h2, gamma2, beta2, N, C_out, H, W, num_groups, eps,
                       h3_pre_silu);
    silu_forward(h3_pre_silu, h3);

    // ── Conv2 backward: dh3 (input grad), dW2 +=, db2 +=. ──
    Tensor dh3;
    conv2d_backward_input(W2, dY, N, C_out, H, W, C_out, 3, 3,
                          1, 1, 1, 1, 1, 1, 1, dh3);
    conv2d_backward_weight(h3, dY, N, C_out, H, W, C_out, 3, 3,
                           1, 1, 1, 1, 1, 1, 1, dW2);
    if (db2) conv2d_backward_bias(dY, N, C_out, H, W, *db2);

    // ── SiLU2 backward over h3_pre_silu. ──
    Tensor dh3_pre_silu;
    silu_backward(h3_pre_silu, dh3, dh3_pre_silu);

    // ── GN2 backward: dh2 (overwritten), dGamma2/dBeta2 accumulate. ──
    Tensor dh2;
    group_norm_backward(h2, gamma2, dh3_pre_silu, N, C_out, H, W,
                        num_groups, eps, dh2, dGamma2, dBeta2);

    // ── t_emb_shift backward: reduce dh2 over the spatial (and, for the
    //    per-channel case, batch) axes; accumulate into dt_emb_shift. ──
    if (t_emb_shift && dt_emb_shift) {
        const float* dh2p = dh2.host_f32();
        float* dtp = dt_emb_shift->host_f32_mut();
        if (temb_N) {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C_out; ++c) {
                    float acc = 0.0f;
                    const float* row =
                        dh2p + (static_cast<std::size_t>(n) * C_out + c) *
                               spatial;
                    for (int p = 0; p < spatial; ++p) acc += row[p];
                    dtp[n * C_out + c] += acc;  // accumulate
                }
            }
        } else {
            for (int c = 0; c < C_out; ++c) {
                float acc = 0.0f;
                for (int n = 0; n < N; ++n) {
                    const float* row =
                        dh2p + (static_cast<std::size_t>(n) * C_out + c) *
                               spatial;
                    for (int p = 0; p < spatial; ++p) acc += row[p];
                }
                dtp[c] += acc;  // accumulate
            }
        }
    }

    // ── Conv1 backward: dh1, dW1 +=, db1 +=. ──
    Tensor dh1;
    conv2d_backward_input(W1, dh2, N, C_in, H, W, C_out, 3, 3,
                          1, 1, 1, 1, 1, 1, 1, dh1);
    conv2d_backward_weight(h1, dh2, N, C_in, H, W, C_out, 3, 3,
                           1, 1, 1, 1, 1, 1, 1, dW1);
    if (db1) conv2d_backward_bias(dh2, N, C_out, H, W, *db1);

    // ── SiLU1 backward over h1_pre_silu. ──
    Tensor dh1_pre_silu;
    silu_backward(h1_pre_silu, dh1, dh1_pre_silu);

    // ── GN1 backward: dX (overwritten), dGamma1/dBeta1 accumulate. ──
    group_norm_backward(X, gamma1, dh1_pre_silu, N, C_in, H, W,
                        num_groups, eps, dX, dGamma1, dBeta1);

    // ── Skip path backward, then sum into dX. ──
    if (Wskip == nullptr) {
        add_inplace(dX, dY);  // identity skip: dX += dY.
    } else {
        Tensor dX_skip;
        conv2d_backward_input(*Wskip, dY, N, C_in, H, W, C_out, 1, 1,
                              1, 1, 0, 0, 1, 1, 1, dX_skip);
        if (dWskip)
            conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, 1, 1,
                                   1, 1, 0, 0, 1, 1, 1, *dWskip);
        if (dbskip)
            conv2d_backward_bias(dY, N, C_out, H, W, *dbskip);
        add_inplace(dX, dX_skip);
    }
}

} // namespace brotensor::detail::cpu
