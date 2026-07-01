// CPU backend — cross-attention family (CHUNK 5).
//
// Ground truth: src/cuda/cross_attention.cu and
// src/cuda/cross_attention_with_attn.cu.
//
// Four ops:
//   * cross_attention_forward            — public forward, O only.
//   * cross_attention_forward_with_attn  — forward + head-averaged AttnAvg,
//                                          optional pre-softmax logit bias.
//   * cross_attention_forward_train      — forward, emits per-head caches.
//   * cross_attention_backward           — backward, accumulates dW*.
//
// DTYPE DECISIONS
//   The op_table groups these as "FP16 inference + FP32 train". The CPU
//   backend is FP32-only (per CLAUDE.md). All four CPU impls run the FP32
//   scalar math identical to cross_attention.cu's FP32 train-core kernels.
//     - cross_attention_forward:  the CUDA op delegates to the FP16 flash
//       path for FP16 inputs and to the FP32 train-core for FP32 inputs. The
//       CPU op always runs the FP32 train-core math. The parity test feeds
//       FP32 to CPU and FP16 to GPU, comparing with a loose FP16-scale
//       tolerance.
//     - cross_attention_forward_with_attn: CUDA op is FP16-only. CPU runs
//       FP32; parity test quantises through FP16 and uses loose tolerance.
//     - cross_attention_forward_train / cross_attention_backward: FP32 on
//       both backends — straightforward FP32<->FP32 parity.
//
// CONVENTIONS (verified against cross_attention.cu)
//   * Weight layout: Wq/Wo are (D, D); Wk/Wv are (D, D_ctx). Projection is
//     out(hh,i,j) = sum_k In(i,k) * W(hh*dh + j, k)  — i.e. In @ W^T with the
//     head slice taken as contiguous rows hh*dh..hh*dh+dh of W.
//   * Per-head split: Qh/Kh/Vh are (H*L, dh), row-major by (head, token).
//     Yconcat is (Lq, D) with head hh occupying columns hh*dh..hh*dh+dh.
//   * Softmax scale: 1/sqrt(dh).
//   * Mask: length-Lk key-validity buffer (1=valid, 0=invalid). Masked keys
//     are excluded from the softmax denominator and forced to 0 probability.
//     Query-side gating is enabled only when Lq == Lk (gate_query): an
//     invalid query row produces a zero Attn row and a zero O row.
//     cross_attention_forward_with_attn does NOT do query-side gating (it
//     follows cxa_row_softmax_kernel which only masks keys).
//   * Intermediates for the *_train pair:
//       Qh      (H*Lq, dh)  per-head Q projection of X
//       Kh      (H*Lk, dh)  per-head K projection of Ctx
//       Vh      (H*Lk, dh)  per-head V projection of Ctx
//       Attnh   (H*Lq, Lk)  per-head softmax probabilities
//       Yconcat (Lq, D)     per-head Attn@V written into head-strided columns
//       O       (Lq, D)     Yconcat @ Wo^T (query-gated)
//   * Backward accumulation: dWq/dWk/dWv/dWo ACCUMULATE (+=). dX/dCtx are
//     OVERWRITTEN. Matches cx_dW_proj_kernel (+=) and cx_dX/cx_dCtx kernels.

#include <brotensor/tensor.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <cmath>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

// out(hh,i,j) = sum_k In(i,k) * W(hh*dh + j, k).  In: (L, Din). W: (D, Din).
// Out: (H*L, dh). Each hh owns rows [hh*L, hh*L+L) of Out exclusively (In/W
// read-only), so the head axis parallelizes with no cross-thread writes.
void cx_proj(const float* In, const float* W, float* Out,
             int L, int Din, int H, int dh) {
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        const int row_off = hh * dh;
        for (int i = 0; i < L; ++i) {
            const float* xr = In + static_cast<std::size_t>(i) * Din;
            const std::size_t out_row =
                (static_cast<std::size_t>(hh) * L + i) * dh;
            for (int j = 0; j < dh; ++j) {
                const float* wr = W + static_cast<std::size_t>(row_off + j) * Din;
                float acc = 0.0f;
                for (int k = 0; k < Din; ++k) acc += xr[k] * wr[k];
                Out[out_row + j] = acc;
            }
        }
    });
}

// Core FP32 forward shared by cross_attention_forward and
// cross_attention_forward_train.
void cross_attention_forward_core(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor& Ctx,
                                  const ::brotensor::Tensor& Wq,
                                  const ::brotensor::Tensor& Wk,
                                  const ::brotensor::Tensor& Wv,
                                  const ::brotensor::Tensor& Wo,
                                  const float* d_mask,
                                  int num_heads,
                                  ::brotensor::Tensor& Qh,
                                  ::brotensor::Tensor& Kh,
                                  ::brotensor::Tensor& Vh,
                                  ::brotensor::Tensor& Attnh,
                                  ::brotensor::Tensor& Yconcat,
                                  ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;

    if (Qh.rows != H * Lq || Qh.cols != dh || Qh.dtype != Dtype::FP32)
        Qh.resize(H * Lq, dh, Dtype::FP32);
    if (Kh.rows != H * Lk || Kh.cols != dh || Kh.dtype != Dtype::FP32)
        Kh.resize(H * Lk, dh, Dtype::FP32);
    if (Vh.rows != H * Lk || Vh.cols != dh || Vh.dtype != Dtype::FP32)
        Vh.resize(H * Lk, dh, Dtype::FP32);
    if (Attnh.rows != H * Lq || Attnh.cols != Lk || Attnh.dtype != Dtype::FP32)
        Attnh.resize(H * Lq, Lk, Dtype::FP32);
    if (Yconcat.rows != Lq || Yconcat.cols != D || Yconcat.dtype != Dtype::FP32)
        Yconcat.resize(Lq, D, Dtype::FP32);
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP32)
        O.resize(Lq, D, Dtype::FP32);
    if (Lq == 0 || Lk == 0 || D == 0 || H == 0) return;

    const int gate_query = (Lq == Lk) ? 1 : 0;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    const float* Xp   = X.host_f32();
    const float* Ctxp = Ctx.host_f32();
    const float* Wqp  = Wq.host_f32();
    const float* Wkp  = Wk.host_f32();
    const float* Wvp  = Wv.host_f32();
    const float* Wop  = Wo.host_f32();
    float* Qp = Qh.host_f32_mut();
    float* Kp = Kh.host_f32_mut();
    float* Vp = Vh.host_f32_mut();
    float* Ap = Attnh.host_f32_mut();
    float* Yp = Yconcat.host_f32_mut();
    float* Op = O.host_f32_mut();

    // Q from X (D, D); K, V from Ctx (D, Dctx).
    cx_proj(Xp,   Wqp, Qp, Lq, D,    H, dh);
    cx_proj(Ctxp, Wkp, Kp, Lk, Dctx, H, dh);
    cx_proj(Ctxp, Wvp, Vp, Lk, Dctx, H, dh);

    // Scores -> masked row softmax -> Attnh. Each hh owns rows
    // [hh*Lq, hh*Lq+Lq) of Attnh exclusively, so parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < Lq; ++i) {
            float* arow = Ap + (static_cast<std::size_t>(hh) * Lq + i) * Lk;
            if (gate_query && d_mask && d_mask[i] < 0.5f) {
                for (int j = 0; j < Lk; ++j) arow[j] = 0.0f;
                continue;
            }
            const float* qr = Qp + (static_cast<std::size_t>(hh) * Lq + i) * dh;
            float m = -1e30f;
            for (int j = 0; j < Lk; ++j) {
                if (d_mask && d_mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
                const float* kr =
                    Kp + (static_cast<std::size_t>(hh) * Lk + j) * dh;
                float s = 0.0f;
                for (int k = 0; k < dh; ++k) s += qr[k] * kr[k];
                s *= inv_sqrtdh;
                arow[j] = s;
                if (s > m) m = s;
            }
            float sum = 0.0f;
            for (int j = 0; j < Lk; ++j) {
                if (d_mask && d_mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
                const float e = std::exp(arow[j] - m);
                arow[j] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int j = 0; j < Lk; ++j) arow[j] *= inv;
        }
    });

    // Attn @ V -> Yconcat(i, hh*dh+k). Each hh writes only its own column
    // range [hh*dh, hh*dh+dh) of every row of Yconcat, so parallelizes
    // over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < Lq; ++i) {
            const float* arow =
                Ap + (static_cast<std::size_t>(hh) * Lq + i) * Lk;
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < Lk; ++j) {
                    const float vv =
                        Vp[(static_cast<std::size_t>(hh) * Lk + j) * dh + k];
                    acc += arow[j] * vv;
                }
                Yp[static_cast<std::size_t>(i) * D + (hh * dh + k)] = acc;
            }
        }
    });

    // Output projection O = Yconcat @ Wo^T, query-gated. Each i owns row i
    // of O exclusively, so parallelizes over i.
    parallel_for(static_cast<std::size_t>(Lq), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        float* orow = Op + static_cast<std::size_t>(i) * D;
        if (gate_query && d_mask && d_mask[i] < 0.5f) {
            for (int c = 0; c < D; ++c) orow[c] = 0.0f;
            return;
        }
        const float* yr = Yp + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<std::size_t>(c) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            orow[c] = acc;
        }
    });
}

} // namespace

// ─── cross_attention_forward (FP32 scalar; O only) ─────────────────────────

void cross_attention_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Ctx,
                             const ::brotensor::Tensor& Wq,
                             const ::brotensor::Tensor& Wk,
                             const ::brotensor::Tensor& Wv,
                             const ::brotensor::Tensor& Wo,
                             const float* d_mask,
                             int num_heads,
                             ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Dtype;
    Tensor Qh, Kh, Vh, Attnh, Yconcat;
    cross_attention_forward_core(X, Ctx, Wq, Wk, Wv, Wo, d_mask, num_heads,
                                 Qh, Kh, Vh, Attnh, Yconcat, O);
}

// ─── cross_attention_forward_train (FP32; emits per-head caches) ────────────

void cross_attention_forward_train(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor& Ctx,
                                   const ::brotensor::Tensor& Wq,
                                   const ::brotensor::Tensor& Wk,
                                   const ::brotensor::Tensor& Wv,
                                   const ::brotensor::Tensor& Wo,
                                   const float* d_mask,
                                   int num_heads,
                                   ::brotensor::Tensor& Qh,
                                   ::brotensor::Tensor& Kh,
                                   ::brotensor::Tensor& Vh,
                                   ::brotensor::Tensor& Attnh,
                                   ::brotensor::Tensor& Yconcat,
                                   ::brotensor::Tensor& O) {
    cross_attention_forward_core(X, Ctx, Wq, Wk, Wv, Wo, d_mask, num_heads,
                                 Qh, Kh, Vh, Attnh, Yconcat, O);
}

// ─── cross_attention_forward_with_attn (FP32; + head-averaged AttnAvg) ─────

void cross_attention_forward_with_attn(const ::brotensor::Tensor& X,
                                       const ::brotensor::Tensor& Ctx,
                                       const ::brotensor::Tensor& Wq,
                                       const ::brotensor::Tensor& Wk,
                                       const ::brotensor::Tensor& Wv,
                                       const ::brotensor::Tensor& Wo,
                                       const float* d_mask,
                                       const ::brotensor::Tensor* attn_logit_bias,
                                       int num_heads,
                                       ::brotensor::Tensor& O,
                                       ::brotensor::Tensor& AttnAvg) {
    using ::brotensor::Dtype;
    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;

    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP32)
        O.resize(Lq, D, Dtype::FP32);
    if (AttnAvg.rows != Lq || AttnAvg.cols != Lk || AttnAvg.dtype != Dtype::FP32)
        AttnAvg.resize(Lq, Lk, Dtype::FP32);
    if (Lq == 0 || Lk == 0 || D == 0 || H == 0) return;

    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    const float* Xp   = X.host_f32();
    const float* Ctxp = Ctx.host_f32();
    const float* Wqp  = Wq.host_f32();
    const float* Wkp  = Wk.host_f32();
    const float* Wvp  = Wv.host_f32();
    const float* Wop  = Wo.host_f32();
    const float* bias = attn_logit_bias ? attn_logit_bias->host_f32() : nullptr;
    float* Op   = O.host_f32_mut();
    float* AAp  = AttnAvg.host_f32_mut();

    std::vector<float> Q(static_cast<std::size_t>(H) * Lq * dh, 0.0f);
    std::vector<float> Kbuf(static_cast<std::size_t>(H) * Lk * dh, 0.0f);
    std::vector<float> Vbuf(static_cast<std::size_t>(H) * Lk * dh, 0.0f);
    std::vector<float> Attn(static_cast<std::size_t>(H) * Lq * Lk, 0.0f);
    std::vector<float> Yc(static_cast<std::size_t>(Lq) * D, 0.0f);

    cx_proj(Xp,   Wqp, Q.data(),    Lq, D,    H, dh);
    cx_proj(Ctxp, Wkp, Kbuf.data(), Lk, Dctx, H, dh);
    cx_proj(Ctxp, Wvp, Vbuf.data(), Lk, Dctx, H, dh);

    // Scores (+ optional bias) -> key-masked row softmax. No query gating —
    // matches cxa_row_softmax_kernel which only masks keys. Each hh owns
    // rows [hh*Lq, hh*Lq+Lq) of Attn exclusively, so parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < Lq; ++i) {
            float* arow = Attn.data() + (static_cast<std::size_t>(hh) * Lq + i) * Lk;
            const float* qr = Q.data() + (static_cast<std::size_t>(hh) * Lq + i) * dh;
            float m = -1e30f;
            for (int j = 0; j < Lk; ++j) {
                const float* kr =
                    Kbuf.data() + (static_cast<std::size_t>(hh) * Lk + j) * dh;
                float s = 0.0f;
                for (int k = 0; k < dh; ++k) s += qr[k] * kr[k];
                s *= inv_sqrtdh;
                if (bias) s += bias[static_cast<std::size_t>(i) * Lk + j];
                arow[j] = s;
                if (d_mask && d_mask[j] < 0.5f) continue;
                if (s > m) m = s;
            }
            float sum = 0.0f;
            for (int j = 0; j < Lk; ++j) {
                if (d_mask && d_mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
                const float e = std::exp(arow[j] - m);
                arow[j] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int j = 0; j < Lk; ++j) arow[j] *= inv;
        }
    });

    // AttnAvg(i, j) = (1/H) sum_h Attn(h, i, j). Each (i, j) writes exactly
    // once from a private serial sum over hh, so parallelizes over i (each i
    // owns row i of AttnAvg exclusively).
    parallel_for(static_cast<std::size_t>(Lq), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        for (int j = 0; j < Lk; ++j) {
            float acc = 0.0f;
            for (int hh = 0; hh < H; ++hh)
                acc += Attn[(static_cast<std::size_t>(hh) * Lq + i) * Lk + j];
            AAp[static_cast<std::size_t>(i) * Lk + j] =
                acc / static_cast<float>(H);
        }
    });

    // Attn @ V -> Yconcat(i, hh*dh+k). Each hh writes only its own column
    // range of every row of Yc, so parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < Lq; ++i) {
            const float* arow =
                Attn.data() + (static_cast<std::size_t>(hh) * Lq + i) * Lk;
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < Lk; ++j) {
                    const float vv =
                        Vbuf[(static_cast<std::size_t>(hh) * Lk + j) * dh + k];
                    acc += arow[j] * vv;
                }
                Yc[static_cast<std::size_t>(i) * D + (hh * dh + k)] = acc;
            }
        }
    });

    // Output projection O = Yconcat @ Wo^T (no query gating). Each i owns
    // row i of O exclusively, so parallelizes over i.
    parallel_for(static_cast<std::size_t>(Lq), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        const float* yr = Yc.data() + static_cast<std::size_t>(i) * D;
        float* orow = Op + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<std::size_t>(c) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            orow[c] = acc;
        }
    });
}

// ─── cross_attention_backward (FP32; accumulates dW*, overwrites dX/dCtx) ──

void cross_attention_backward(const ::brotensor::Tensor& dO,
                              const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Ctx,
                              const ::brotensor::Tensor& Qh,
                              const ::brotensor::Tensor& Kh,
                              const ::brotensor::Tensor& Vh,
                              const ::brotensor::Tensor& Attnh,
                              const ::brotensor::Tensor& Yconcat,
                              const ::brotensor::Tensor& Wq,
                              const ::brotensor::Tensor& Wk,
                              const ::brotensor::Tensor& Wv,
                              const ::brotensor::Tensor& Wo,
                              const float* d_mask,
                              int num_heads,
                              ::brotensor::Tensor& dX,
                              ::brotensor::Tensor& dCtx,
                              ::brotensor::Tensor& dWq,
                              ::brotensor::Tensor& dWk,
                              ::brotensor::Tensor& dWv,
                              ::brotensor::Tensor& dWo) {
    using ::brotensor::Dtype;
    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;

    if (dX.rows != Lq || dX.cols != D || dX.dtype != Dtype::FP32)
        dX.resize(Lq, D, Dtype::FP32);
    if (dCtx.rows != Lk || dCtx.cols != Dctx || dCtx.dtype != Dtype::FP32)
        dCtx.resize(Lk, Dctx, Dtype::FP32);
    if (Lq == 0 || Lk == 0 || D == 0 || H == 0) return;

    const int gate_query = (Lq == Lk) ? 1 : 0;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    const float* dOp = dO.host_f32();
    const float* Xp  = X.host_f32();
    const float* Cp  = Ctx.host_f32();
    const float* Qp  = Qh.host_f32();
    const float* Kp  = Kh.host_f32();
    const float* Vp  = Vh.host_f32();
    const float* Ap  = Attnh.host_f32();
    const float* Yp  = Yconcat.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* dXp   = dX.host_f32_mut();
    float* dCp   = dCtx.host_f32_mut();
    float* dWqp  = dWq.host_f32_mut();
    float* dWkp  = dWk.host_f32_mut();
    float* dWvp  = dWv.host_f32_mut();
    float* dWop  = dWo.host_f32_mut();

    std::vector<float> dYc(static_cast<std::size_t>(Lq) * D, 0.0f);
    std::vector<float> dAttn(static_cast<std::size_t>(H) * Lq * Lk, 0.0f);
    std::vector<float> dVh(static_cast<std::size_t>(H) * Lk * dh, 0.0f);
    std::vector<float> dScores(static_cast<std::size_t>(H) * Lq * Lk, 0.0f);
    std::vector<float> dQh(static_cast<std::size_t>(H) * Lq * dh, 0.0f);
    std::vector<float> dKh(static_cast<std::size_t>(H) * Lk * dh, 0.0f);

    // dYconcat = dO @ Wo (query-gated; overwrite). Each i owns row i of dYc
    // exclusively, so parallelizes over i.
    parallel_for(static_cast<std::size_t>(Lq), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        const bool valid = !(gate_query && d_mask && d_mask[i] < 0.5f);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            if (valid) {
                for (int c = 0; c < D; ++c)
                    acc += Wop[static_cast<std::size_t>(c) * D + k] *
                           dOp[static_cast<std::size_t>(i) * D + c];
            }
            dYc[static_cast<std::size_t>(i) * D + k] = acc;
        }
    });
    // dWo accumulates. c-outer with a private serial sum over i — each
    // (c, k) cell is written exactly once, so this parallelizes over c
    // (unlike ops_impl.cpp's mha_backward/attention_backward dWo, which use
    // a token-outermost accumulate-in-place order and are left
    // single-threaded there).
    parallel_for(static_cast<std::size_t>(D), [&](std::size_t ci) {
        const int c = static_cast<int>(ci);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int i = 0; i < Lq; ++i) {
                if (gate_query && d_mask && d_mask[i] < 0.5f) continue;
                acc += dOp[static_cast<std::size_t>(i) * D + c] *
                       Yp[static_cast<std::size_t>(i) * D + k];
            }
            dWop[static_cast<std::size_t>(c) * D + k] += acc;
        }
    });

    // Per-head dAttn and dVh. Each hh owns rows [hh*Lq, hh*Lq+Lq) of dAttn
    // and [hh*Lk, hh*Lk+Lk) of dVh exclusively (dVh's inner reduction over i
    // sums only into this head's own private slice), so parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < Lq; ++i) {
            for (int j = 0; j < Lk; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < dh; ++k) {
                    const float dy =
                        dYc[static_cast<std::size_t>(i) * D + (hh * dh + k)];
                    const float vv =
                        Vp[(static_cast<std::size_t>(hh) * Lk + j) * dh + k];
                    acc += dy * vv;
                }
                dAttn[(static_cast<std::size_t>(hh) * Lq + i) * Lk + j] = acc;
            }
        }
        for (int j = 0; j < Lk; ++j) {
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int i = 0; i < Lq; ++i) {
                    const float a =
                        Ap[(static_cast<std::size_t>(hh) * Lq + i) * Lk + j];
                    const float dy =
                        dYc[static_cast<std::size_t>(i) * D + (hh * dh + k)];
                    acc += a * dy;
                }
                dVh[(static_cast<std::size_t>(hh) * Lk + j) * dh + k] = acc;
            }
        }
    });

    // Per-head row-softmax backward -> dScores. Each hh owns rows
    // [hh*Lq, hh*Lq+Lq) of dScores exclusively, so parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < Lq; ++i) {
            const float* prow =
                Ap + (static_cast<std::size_t>(hh) * Lq + i) * Lk;
            const float* dprow =
                dAttn.data() + (static_cast<std::size_t>(hh) * Lq + i) * Lk;
            float* drow =
                dScores.data() + (static_cast<std::size_t>(hh) * Lq + i) * Lk;
            if (gate_query && d_mask && d_mask[i] < 0.5f) {
                for (int j = 0; j < Lk; ++j) drow[j] = 0.0f;
                continue;
            }
            float dot = 0.0f;
            for (int j = 0; j < Lk; ++j) dot += dprow[j] * prow[j];
            for (int j = 0; j < Lk; ++j) {
                if (d_mask && d_mask[j] < 0.5f) drow[j] = 0.0f;
                else drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
            }
        }
    });

    // Per-head dQh, dKh. Each hh owns rows [hh*Lq, hh*Lq+Lq) of dQh and
    // [hh*Lk, hh*Lk+Lk) of dKh exclusively, so parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < Lq; ++i) {
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < Lk; ++j) {
                    const float ds =
                        dScores[(static_cast<std::size_t>(hh) * Lq + i) * Lk + j];
                    const float kk =
                        Kp[(static_cast<std::size_t>(hh) * Lk + j) * dh + k];
                    acc += ds * kk;
                }
                dQh[(static_cast<std::size_t>(hh) * Lq + i) * dh + k] = acc;
            }
        }
        for (int j = 0; j < Lk; ++j) {
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int i = 0; i < Lq; ++i) {
                    const float ds =
                        dScores[(static_cast<std::size_t>(hh) * Lq + i) * Lk + j];
                    const float qq =
                        Qp[(static_cast<std::size_t>(hh) * Lq + i) * dh + k];
                    acc += ds * qq;
                }
                dKh[(static_cast<std::size_t>(hh) * Lk + j) * dh + k] = acc;
            }
        }
    });

    // dWq (D, D) accumulates against X.
    //   dWq(hh*dh+j, k) += sum_i dQh(hh,i,j) * X(i,k).
    // wrow-outer with a private serial sum over i — each (wrow, k) cell is
    // written exactly once, so this parallelizes over wrow.
    parallel_for(static_cast<std::size_t>(D), [&](std::size_t wrowi) {
        const int wrow = static_cast<int>(wrowi);
        const int hh = wrow / dh;
        const int j  = wrow % dh;
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int i = 0; i < Lq; ++i) {
                const float xv = Xp[static_cast<std::size_t>(i) * D + k];
                acc += dQh[(static_cast<std::size_t>(hh) * Lq + i) * dh + j] * xv;
            }
            dWqp[static_cast<std::size_t>(wrow) * D + k] += acc;
        }
    });
    // dWk, dWv (D, Dctx) accumulate against Ctx. Same wrow-outer
    // private-sum-inner pattern, so parallelizes over wrow.
    parallel_for(static_cast<std::size_t>(D), [&](std::size_t wrowi) {
        const int wrow = static_cast<int>(wrowi);
        const int hh = wrow / dh;
        const int j  = wrow % dh;
        for (int k = 0; k < Dctx; ++k) {
            float ak = 0.0f, av = 0.0f;
            for (int i = 0; i < Lk; ++i) {
                const float cv = Cp[static_cast<std::size_t>(i) * Dctx + k];
                ak += dKh[(static_cast<std::size_t>(hh) * Lk + i) * dh + j] * cv;
                av += dVh[(static_cast<std::size_t>(hh) * Lk + i) * dh + j] * cv;
            }
            const std::size_t idx = static_cast<std::size_t>(wrow) * Dctx + k;
            dWkp[idx] += ak;
            dWvp[idx] += av;
        }
    });

    // dX(i,k) = sum over heads, j: dQh(hh,i,j) * Wq(hh*dh+j, k). Overwrite.
    // Each i owns row i of dX exclusively, so parallelizes over i.
    parallel_for(static_cast<std::size_t>(Lq), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int hh = 0; hh < H; ++hh) {
                for (int j = 0; j < dh; ++j) {
                    const int wrow = hh * dh + j;
                    const float gq =
                        dQh[(static_cast<std::size_t>(hh) * Lq + i) * dh + j];
                    acc += gq * Wqp[static_cast<std::size_t>(wrow) * D + k];
                }
            }
            dXp[static_cast<std::size_t>(i) * D + k] = acc;
        }
    });

    // dCtx(j,k) = sum over heads, m: dKh*Wk + dVh*Wv at (hh*dh+m, k).
    // Overwrite. Each j owns row j of dCtx exclusively, so parallelizes
    // over j.
    parallel_for(static_cast<std::size_t>(Lk), [&](std::size_t ji) {
        const int j = static_cast<int>(ji);
        for (int k = 0; k < Dctx; ++k) {
            float acc = 0.0f;
            for (int hh = 0; hh < H; ++hh) {
                for (int m = 0; m < dh; ++m) {
                    const int wrow = hh * dh + m;
                    const std::size_t widx =
                        static_cast<std::size_t>(wrow) * Dctx + k;
                    const float gk =
                        dKh[(static_cast<std::size_t>(hh) * Lk + j) * dh + m];
                    const float gv =
                        dVh[(static_cast<std::size_t>(hh) * Lk + j) * dh + m];
                    acc += gk * Wkp[widx] + gv * Wvp[widx];
                }
            }
            dCp[static_cast<std::size_t>(j) * Dctx + k] = acc;
        }
    });
}

} // namespace brotensor::detail::cpu
