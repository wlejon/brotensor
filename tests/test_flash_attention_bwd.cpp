// Backward parity for the flash-attention projection ops (self / cross,
// with biases, causal) against mha_backward / cross_attention_backward or a
// CPU reference. Part of brotensor_test_flash_attention (main in
// test_flash_attention.cpp).

#include "test_flash_attention_common.h"

// ─── Backward tests ────────────────────────────────────────────────────────

// Self-attention backward parity with mha_backward (FP32 reference).
// Build identical X / Wq/Wk/Wv/Wo (no biases — mha has no bias path) and
// random dO; run FP32 mha fwd+bwd to get the reference grads; run flash
// FP16 bwd on the same inputs; compare element-wise with FP16 tolerance.
void run_bwd_self_vs_mha(const char* label, int L, int D, int nh) {
    std::printf("  %s self-bwd-vs-mha L=%d D=%d nh=%d\n", label, L, D, nh);
    std::mt19937 rng(0xD06E);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(L*D), Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D), dO(L*D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X); fill(Wq); fill(Wk); fill(Wv); fill(Wo); fill(dO);

    // FP32 reference path through mha. Round inputs through FP16 so the
    // reference sees the exact same values the flash bwd will work with.
    auto Xq = rq(X);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);
    auto dOq = rq(dO);

    Tensor X_f32  = Tensor::from_host_on(Device::CUDA, Xq.data(),  L, D);
    Tensor Wq_f32 = Tensor::from_host_on(Device::CUDA, Wqq.data(), D, D);
    Tensor Wk_f32 = Tensor::from_host_on(Device::CUDA, Wkq.data(), D, D);
    Tensor Wv_f32 = Tensor::from_host_on(Device::CUDA, Wvq.data(), D, D);
    Tensor Wo_f32 = Tensor::from_host_on(Device::CUDA, Woq.data(), D, D);
    Tensor dO_f32 = Tensor::from_host_on(Device::CUDA, dOq.data(), L, D);

    Tensor Qh, Kh, Vh, Attnh, Yconcat, O_ref;
    brotensor::mha_forward(X_f32, Wq_f32, Wk_f32, Wv_f32, Wo_f32,
                           nullptr, nh, Qh, Kh, Vh, Attnh, Yconcat, O_ref);

    Tensor dX_ref  = Tensor::zeros_on(Device::CUDA, L, D);
    Tensor dWq_ref = Tensor::zeros_on(Device::CUDA, D, D);
    Tensor dWk_ref = Tensor::zeros_on(Device::CUDA, D, D);
    Tensor dWv_ref = Tensor::zeros_on(Device::CUDA, D, D);
    Tensor dWo_ref = Tensor::zeros_on(Device::CUDA, D, D);
    brotensor::mha_backward(dO_f32, X_f32, Qh, Kh, Vh, Attnh, Yconcat,
                            Wq_f32, Wk_f32, Wv_f32, Wo_f32, nullptr, nh,
                            dX_ref, dWq_ref, dWk_ref, dWv_ref, dWo_ref);
    std::vector<float> dX_ref_h(L*D), dWq_ref_h(D*D), dWk_ref_h(D*D),
                       dWv_ref_h(D*D), dWo_ref_h(D*D);
    dX_ref.copy_to_host(dX_ref_h.data());
    dWq_ref.copy_to_host(dWq_ref_h.data());
    dWk_ref.copy_to_host(dWk_ref_h.data());
    dWv_ref.copy_to_host(dWv_ref_h.data());
    dWo_ref.copy_to_host(dWo_ref_h.data());
    brotensor::sync_all();

    // Flash FP16 bwd path.
    auto Xh = to_fp16(X), Wqh = to_fp16(Wq), Wkh = to_fp16(Wk),
         Wvh = to_fp16(Wv), Woh = to_fp16(Wo), dOh = to_fp16(dO);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  L, D);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);
    Tensor dOg = Tensor::from_host_fp16_on(Device::CUDA, dOh.data(), L, D);

    Tensor dXg  = Tensor::empty_on(Device::CUDA, L, D, Dtype::FP16);
    Tensor dWqg = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWkg = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWvg = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWog = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);

    brotensor::flash_attention_qkvo_backward(
        Xg, /*Ctx=*/nullptr,
        Wqg, /*bq=*/nullptr,
        Wkg, /*bk=*/nullptr,
        Wvg, /*bv=*/nullptr,
        Wog, /*bo=*/nullptr,
        /*d_mask=*/nullptr, nh, /*causal=*/false,
        dOg,
        dXg, /*dCtx=*/nullptr,
        dWqg, /*dbq=*/nullptr,
        dWkg, /*dbk=*/nullptr,
        dWvg, /*dbv=*/nullptr,
        dWog, /*dbo=*/nullptr);

    std::vector<uint16_t> dX_got(L*D), dWq_got(D*D), dWk_got(D*D),
                          dWv_got(D*D), dWo_got(D*D);
    dXg.copy_to_host_fp16(dX_got.data());
    dWqg.copy_to_host_fp16(dWq_got.data());
    dWkg.copy_to_host_fp16(dWk_got.data());
    dWvg.copy_to_host_fp16(dWv_got.data());
    dWog.copy_to_host_fp16(dWo_got.data());
    brotensor::sync_all();

    check_fp16(dX_got,  dX_ref_h,  "self-bwd dX");
    check_fp16(dWq_got, dWq_ref_h, "self-bwd dWq");
    check_fp16(dWk_got, dWk_ref_h, "self-bwd dWk");
    check_fp16(dWv_got, dWv_ref_h, "self-bwd dWv");
    check_fp16(dWo_got, dWo_ref_h, "self-bwd dWo");
}

// Cross-attention backward parity vs cross_attention_backward (FP32),
// with rectangular Wk/Wv (D_ctx != D).
void run_bwd_cross_vs_cx(const char* label, int Lq, int Lk, int D,
                         int D_ctx, int nh) {
    std::printf("  %s cross-bwd-vs-cx Lq=%d Lk=%d D=%d D_ctx=%d nh=%d\n",
                label, Lq, Lk, D, D_ctx, nh);
    std::mt19937 rng(0xC0BA17);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq*D), Ctx(Lk*D_ctx);
    std::vector<float> Wq(D*D), Wk(D*D_ctx), Wv(D*D_ctx), Wo(D*D);
    std::vector<float> dO(Lq*D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X); fill(Ctx); fill(Wq); fill(Wk); fill(Wv); fill(Wo); fill(dO);

    auto Xq = rq(X), Ctxq = rq(Ctx);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);
    auto dOq = rq(dO);

    Tensor X_f32   = Tensor::from_host_on(Device::CUDA, Xq.data(),   Lq, D);
    Tensor Ctx_f32 = Tensor::from_host_on(Device::CUDA, Ctxq.data(), Lk, D_ctx);
    Tensor Wq_f32  = Tensor::from_host_on(Device::CUDA, Wqq.data(),  D, D);
    Tensor Wk_f32  = Tensor::from_host_on(Device::CUDA, Wkq.data(),  D, D_ctx);
    Tensor Wv_f32  = Tensor::from_host_on(Device::CUDA, Wvq.data(),  D, D_ctx);
    Tensor Wo_f32  = Tensor::from_host_on(Device::CUDA, Woq.data(),  D, D);
    Tensor dO_f32  = Tensor::from_host_on(Device::CUDA, dOq.data(),  Lq, D);

    Tensor Qh, Kh, Vh, Attnh, Yconcat, O_ref;
    brotensor::cross_attention_forward_train(
        X_f32, Ctx_f32, Wq_f32, Wk_f32, Wv_f32, Wo_f32,
        nullptr, nh, Qh, Kh, Vh, Attnh, Yconcat, O_ref);

    Tensor dX_ref   = Tensor::zeros_on(Device::CUDA, Lq, D);
    Tensor dCtx_ref = Tensor::zeros_on(Device::CUDA, Lk, D_ctx);
    Tensor dWq_ref  = Tensor::zeros_on(Device::CUDA, D, D);
    Tensor dWk_ref  = Tensor::zeros_on(Device::CUDA, D, D_ctx);
    Tensor dWv_ref  = Tensor::zeros_on(Device::CUDA, D, D_ctx);
    Tensor dWo_ref  = Tensor::zeros_on(Device::CUDA, D, D);
    brotensor::cross_attention_backward(
        dO_f32, X_f32, Ctx_f32, Qh, Kh, Vh, Attnh, Yconcat,
        Wq_f32, Wk_f32, Wv_f32, Wo_f32, nullptr, nh,
        dX_ref, dCtx_ref, dWq_ref, dWk_ref, dWv_ref, dWo_ref);

    std::vector<float> dX_ref_h(Lq*D), dCtx_ref_h(Lk*D_ctx);
    std::vector<float> dWq_ref_h(D*D), dWk_ref_h(D*D_ctx),
                       dWv_ref_h(D*D_ctx), dWo_ref_h(D*D);
    dX_ref.copy_to_host(dX_ref_h.data());
    dCtx_ref.copy_to_host(dCtx_ref_h.data());
    dWq_ref.copy_to_host(dWq_ref_h.data());
    dWk_ref.copy_to_host(dWk_ref_h.data());
    dWv_ref.copy_to_host(dWv_ref_h.data());
    dWo_ref.copy_to_host(dWo_ref_h.data());
    brotensor::sync_all();

    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto dOh = to_fp16(dO);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  Lq, D);
    Tensor Cg  = Tensor::from_host_fp16_on(Device::CUDA, Ch.data(),  Lk, D_ctx);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D_ctx);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D_ctx);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);
    Tensor dOg = Tensor::from_host_fp16_on(Device::CUDA, dOh.data(), Lq, D);

    Tensor dXg   = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    Tensor dCtxg = Tensor::empty_on(Device::CUDA, Lk, D_ctx, Dtype::FP16);
    Tensor dWqg  = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWkg  = Tensor::zeros_on(Device::CUDA, D, D_ctx, Dtype::FP16);
    Tensor dWvg  = Tensor::zeros_on(Device::CUDA, D, D_ctx, Dtype::FP16);
    Tensor dWog  = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);

    brotensor::flash_attention_qkvo_backward(
        Xg, &Cg,
        Wqg, nullptr, Wkg, nullptr, Wvg, nullptr, Wog, nullptr,
        nullptr, nh, /*causal=*/false,
        dOg,
        dXg, &dCtxg,
        dWqg, nullptr, dWkg, nullptr, dWvg, nullptr, dWog, nullptr);

    std::vector<uint16_t> dX_got(Lq*D), dCtx_got(Lk*D_ctx);
    std::vector<uint16_t> dWq_got(D*D), dWk_got(D*D_ctx),
                          dWv_got(D*D_ctx), dWo_got(D*D);
    dXg.copy_to_host_fp16(dX_got.data());
    dCtxg.copy_to_host_fp16(dCtx_got.data());
    dWqg.copy_to_host_fp16(dWq_got.data());
    dWkg.copy_to_host_fp16(dWk_got.data());
    dWvg.copy_to_host_fp16(dWv_got.data());
    dWog.copy_to_host_fp16(dWo_got.data());
    brotensor::sync_all();

    check_fp16(dX_got,   dX_ref_h,   "cross-bwd dX");
    check_fp16(dCtx_got, dCtx_ref_h, "cross-bwd dCtx");
    check_fp16(dWq_got,  dWq_ref_h,  "cross-bwd dWq");
    check_fp16(dWk_got,  dWk_ref_h,  "cross-bwd dWk");
    check_fp16(dWv_got,  dWv_ref_h,  "cross-bwd dWv");
    check_fp16(dWo_got,  dWo_ref_h,  "cross-bwd dWo");
}

// Backward with all four biases present. CPU reference: hand-compute the
// per-projection backwards from a CPU-side forward then a CPU-side reverse
// over the attention math. Small shape so the O(Lq Lk D^2) doesn't hurt.
void run_bwd_cross_with_biases_cpu(const char* label, int Lq, int Lk,
                                   int D, int D_ctx, int nh) {
    std::printf("  %s cross-bwd+biases (CPU ref) Lq=%d Lk=%d D=%d D_ctx=%d nh=%d\n",
                label, Lq, Lk, D, D_ctx, nh);
    std::mt19937 rng(0xB1AB1A);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq*D), Ctx(Lk*D_ctx);
    std::vector<float> Wq(D*D), Wk(D*D_ctx), Wv(D*D_ctx), Wo(D*D);
    std::vector<float> bq(D), bk(D), bv(D), bo(D), dO(Lq*D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X); fill(Ctx); fill(Wq); fill(Wk); fill(Wv); fill(Wo);
    fill(bq); fill(bk); fill(bv); fill(bo); fill(dO);

    auto Xq = rq(X), Ctxq = rq(Ctx);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);
    auto bqq = rq(bq), bkq = rq(bk), bvq = rq(bv), boq = rq(bo);
    auto dOq = rq(dO);

    // CPU forward + backward.
    auto proj_with_bias = [](const std::vector<float>& A,
                             const std::vector<float>& W,
                             const std::vector<float>& b,
                             int M, int Kin, int Nout,
                             std::vector<float>& Out) {
        Out.assign(static_cast<size_t>(M) * Nout, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < Nout; ++n) {
                double s = b[n];
                for (int k = 0; k < Kin; ++k)
                    s += static_cast<double>(A[m*Kin + k]) * W[n*Kin + k];
                Out[m*Nout + n] = static_cast<float>(s);
            }
    };
    std::vector<float> Q, K, V, P, O_attn;
    proj_with_bias(Xq,  Wqq, bqq, Lq, D,     D, Q);
    proj_with_bias(Ctxq, Wkq, bkq, Lk, D_ctx, D, K);
    proj_with_bias(Ctxq, Wvq, bvq, Lk, D_ctx, D, V);

    const int hd = D / nh;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    P.assign(static_cast<size_t>(nh) * Lq * Lk, 0.0f);
    O_attn.assign(static_cast<size_t>(Lq) * D, 0.0f);
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < nh; ++h) {
            const int off = h * hd;
            std::vector<float> srow(Lk);
            float maxv = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q*D + off + d]) * K[k*D + off + d];
                float s = static_cast<float>(dot) * inv_sqrt;
                srow[k] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) { srow[k] = std::exp(srow[k] - maxv); sum += srow[k]; }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k < Lk; ++k) {
                P[(h*Lq + q) * Lk + k] = srow[k] * inv;
            }
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = 0; k < Lk; ++k)
                    a += static_cast<double>(P[(h*Lq + q) * Lk + k]) * V[k*D + off + d];
                O_attn[q*D + off + d] = static_cast<float>(a);
            }
        }
    }

    // CPU backward.
    // Wo + bo backward:  dO_attn = dO · Wo, dWo[c,k] = sum_q dO[q,c] · O_attn[q,k], dbo[c] = sum_q dO[q,c]
    std::vector<float> dO_attn(Lq*D, 0.0f), dWo(D*D, 0.0f), dbo(D, 0.0f);
    for (int q = 0; q < Lq; ++q) {
        for (int k = 0; k < D; ++k) {
            double a = 0.0;
            for (int c = 0; c < D; ++c) a += static_cast<double>(dOq[q*D + c]) * Woq[c*D + k];
            dO_attn[q*D + k] = static_cast<float>(a);
        }
    }
    for (int c = 0; c < D; ++c) {
        double db = 0.0;
        for (int q = 0; q < Lq; ++q) db += dOq[q*D + c];
        dbo[c] = static_cast<float>(db);
        for (int k = 0; k < D; ++k) {
            double dw = 0.0;
            for (int q = 0; q < Lq; ++q) dw += static_cast<double>(dOq[q*D + c]) * O_attn[q*D + k];
            dWo[c*D + k] = static_cast<float>(dw);
        }
    }

    // Per-head attention backward.
    std::vector<float> dQ(Lq*D, 0.0f), dK(Lk*D, 0.0f), dV(Lk*D, 0.0f);
    for (int h = 0; h < nh; ++h) {
        const int off = h * hd;
        // dV[k, off+d] += sum_q P[h,q,k] dO_attn[q, off+d]
        for (int k = 0; k < Lk; ++k)
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int q = 0; q < Lq; ++q)
                    a += static_cast<double>(P[(h*Lq + q)*Lk + k]) * dO_attn[q*D + off + d];
                dV[k*D + off + d] = static_cast<float>(a);
            }
        // dP[q, k] = sum_d dO_attn[q,off+d] V[k, off+d]; D_q = sum_k P dP; dS = P*(dP-D_q)*inv_sqrt
        std::vector<float> dP(Lq*Lk, 0.0f), dS(Lq*Lk, 0.0f);
        for (int q = 0; q < Lq; ++q)
            for (int k = 0; k < Lk; ++k) {
                double a = 0.0;
                for (int d = 0; d < hd; ++d)
                    a += static_cast<double>(dO_attn[q*D + off + d]) * V[k*D + off + d];
                dP[q*Lk + k] = static_cast<float>(a);
            }
        for (int q = 0; q < Lq; ++q) {
            double Dq = 0.0;
            for (int k = 0; k < Lk; ++k)
                Dq += static_cast<double>(P[(h*Lq + q)*Lk + k]) * dP[q*Lk + k];
            for (int k = 0; k < Lk; ++k) {
                const float p = P[(h*Lq + q)*Lk + k];
                dS[q*Lk + k] = p * (dP[q*Lk + k] - static_cast<float>(Dq)) * inv_sqrt;
            }
        }
        // dQ[q,off+d] = sum_k dS[q,k] K[k,off+d]; dK[k,off+d] = sum_q dS[q,k] Q[q,off+d]
        for (int q = 0; q < Lq; ++q)
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = 0; k < Lk; ++k)
                    a += static_cast<double>(dS[q*Lk + k]) * K[k*D + off + d];
                dQ[q*D + off + d] = static_cast<float>(a);
            }
        for (int k = 0; k < Lk; ++k)
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int q = 0; q < Lq; ++q)
                    a += static_cast<double>(dS[q*Lk + k]) * Q[q*D + off + d];
                dK[k*D + off + d] = static_cast<float>(a);
            }
    }

    // Projection backwards: dX (Lq,D) = dQ · Wq (Wq is (D, D)) ; dCtx = dK · Wk + dV · Wv
    std::vector<float> dX(Lq*D, 0.0f), dCtx(Lk*D_ctx, 0.0f);
    std::vector<float> dWq(D*D, 0.0f), dWk(D*D_ctx, 0.0f),
                       dWv(D*D_ctx, 0.0f);
    std::vector<float> dbq(D, 0.0f), dbk(D, 0.0f), dbv(D, 0.0f);
    auto proj_bwd_full = [&](const std::vector<float>& In, int Lin, int Din,
                             const std::vector<float>& W,
                             const std::vector<float>& dOut,
                             std::vector<float>& dIn_acc,  // (Lin, Din) ACCUM
                             std::vector<float>& dW_acc,
                             std::vector<float>& db_acc) {
        // dIn[i, k] = sum_c W[c, k] dOut[i, c]
        for (int i = 0; i < Lin; ++i)
            for (int k = 0; k < Din; ++k) {
                double a = 0.0;
                for (int c = 0; c < D; ++c) a += static_cast<double>(W[c*Din + k]) * dOut[i*D + c];
                dIn_acc[i*Din + k] += static_cast<float>(a);
            }
        for (int c = 0; c < D; ++c) {
            double db = 0.0;
            for (int i = 0; i < Lin; ++i) db += dOut[i*D + c];
            db_acc[c] = static_cast<float>(db);
            for (int k = 0; k < Din; ++k) {
                double dw = 0.0;
                for (int i = 0; i < Lin; ++i)
                    dw += static_cast<double>(dOut[i*D + c]) * In[i*Din + k];
                dW_acc[c*Din + k] = static_cast<float>(dw);
            }
        }
    };
    proj_bwd_full(Xq,   Lq, D,     Wqq, dQ, dX,   dWq, dbq);
    proj_bwd_full(Ctxq, Lk, D_ctx, Wkq, dK, dCtx, dWk, dbk);
    proj_bwd_full(Ctxq, Lk, D_ctx, Wvq, dV, dCtx, dWv, dbv);

    // GPU side: flash bwd with biases.
    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    auto dOh = to_fp16(dO);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  Lq, D);
    Tensor Cg  = Tensor::from_host_fp16_on(Device::CUDA, Ch.data(),  Lk, D_ctx);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D_ctx);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D_ctx);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);
    Tensor bqg = Tensor::from_host_fp16_on(Device::CUDA, bqh.data(), D, 1);
    Tensor bkg = Tensor::from_host_fp16_on(Device::CUDA, bkh.data(), D, 1);
    Tensor bvg = Tensor::from_host_fp16_on(Device::CUDA, bvh.data(), D, 1);
    Tensor bog = Tensor::from_host_fp16_on(Device::CUDA, boh.data(), D, 1);
    Tensor dOg = Tensor::from_host_fp16_on(Device::CUDA, dOh.data(), Lq, D);

    Tensor dXg   = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    Tensor dCtxg = Tensor::empty_on(Device::CUDA, Lk, D_ctx, Dtype::FP16);
    Tensor dWqg  = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWkg  = Tensor::zeros_on(Device::CUDA, D, D_ctx, Dtype::FP16);
    Tensor dWvg  = Tensor::zeros_on(Device::CUDA, D, D_ctx, Dtype::FP16);
    Tensor dWog  = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dbqg  = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);
    Tensor dbkg  = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);
    Tensor dbvg  = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);
    Tensor dbog  = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);

    brotensor::flash_attention_qkvo_backward(
        Xg, &Cg,
        Wqg, &bqg, Wkg, &bkg, Wvg, &bvg, Wog, &bog,
        nullptr, nh, /*causal=*/false,
        dOg,
        dXg, &dCtxg,
        dWqg, &dbqg, dWkg, &dbkg, dWvg, &dbvg, dWog, &dbog);

    std::vector<uint16_t> dX_got(Lq*D), dCtx_got(Lk*D_ctx);
    std::vector<uint16_t> dWq_got(D*D), dWk_got(D*D_ctx),
                          dWv_got(D*D_ctx), dWo_got(D*D);
    std::vector<uint16_t> dbq_got(D), dbk_got(D), dbv_got(D), dbo_got(D);
    dXg.copy_to_host_fp16(dX_got.data());
    dCtxg.copy_to_host_fp16(dCtx_got.data());
    dWqg.copy_to_host_fp16(dWq_got.data());
    dWkg.copy_to_host_fp16(dWk_got.data());
    dWvg.copy_to_host_fp16(dWv_got.data());
    dWog.copy_to_host_fp16(dWo_got.data());
    dbqg.copy_to_host_fp16(dbq_got.data());
    dbkg.copy_to_host_fp16(dbk_got.data());
    dbvg.copy_to_host_fp16(dbv_got.data());
    dbog.copy_to_host_fp16(dbo_got.data());
    brotensor::sync_all();

    check_fp16(dX_got,   dX,   "cross-bwd+b dX");
    check_fp16(dCtx_got, dCtx, "cross-bwd+b dCtx");
    check_fp16(dWq_got,  dWq,  "cross-bwd+b dWq");
    check_fp16(dWk_got,  dWk,  "cross-bwd+b dWk");
    check_fp16(dWv_got,  dWv,  "cross-bwd+b dWv");
    check_fp16(dWo_got,  dWo,  "cross-bwd+b dWo");
    check_fp16(dbq_got,  dbq,  "cross-bwd+b dbq");
    check_fp16(dbk_got,  dbk,  "cross-bwd+b dbk");
    check_fp16(dbv_got,  dbv,  "cross-bwd+b dbv");
    check_fp16(dbo_got,  dbo,  "cross-bwd+b dbo");
}

// Causal self-attention bwd. CPU reference (full-precision through FP16
// inputs) since mha doesn't have a causal flag.
void run_bwd_causal_cpu(const char* label, int L, int D, int nh) {
    std::printf("  %s causal-bwd (CPU ref) L=%d D=%d nh=%d\n", label, L, D, nh);
    std::mt19937 rng(0xC0A11);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(L*D), Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D);
    std::vector<float> bq(D), bk(D), bv(D), bo(D), dO(L*D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X); fill(Wq); fill(Wk); fill(Wv); fill(Wo);
    fill(bq); fill(bk); fill(bv); fill(bo); fill(dO);

    auto Xq = rq(X);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);
    auto bqq = rq(bq), bkq = rq(bk), bvq = rq(bv), boq = rq(bo);
    auto dOq = rq(dO);

    auto proj_with_bias = [](const std::vector<float>& A,
                             const std::vector<float>& W,
                             const std::vector<float>& b,
                             int M, int Kin, int Nout,
                             std::vector<float>& Out) {
        Out.assign(static_cast<size_t>(M) * Nout, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < Nout; ++n) {
                double s = b[n];
                for (int k = 0; k < Kin; ++k)
                    s += static_cast<double>(A[m*Kin + k]) * W[n*Kin + k];
                Out[m*Nout + n] = static_cast<float>(s);
            }
    };
    std::vector<float> Q, K, V;
    proj_with_bias(Xq, Wqq, bqq, L, D, D, Q);
    proj_with_bias(Xq, Wkq, bkq, L, D, D, K);
    proj_with_bias(Xq, Wvq, bvq, L, D, D, V);

    const int hd = D / nh;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> P(static_cast<size_t>(nh)*L*L, 0.0f), O_attn(L*D, 0.0f);
    for (int q = 0; q < L; ++q)
      for (int h = 0; h < nh; ++h) {
        const int off = h * hd;
        std::vector<float> srow(L, 0.0f);
        float maxv = -1e30f;
        for (int k = 0; k <= q; ++k) {
            double dot = 0.0;
            for (int d = 0; d < hd; ++d)
                dot += static_cast<double>(Q[q*D + off + d]) * K[k*D + off + d];
            float s = static_cast<float>(dot) * inv_sqrt;
            srow[k] = s;
            if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (int k = 0; k <= q; ++k) { srow[k] = std::exp(srow[k] - maxv); sum += srow[k]; }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int k = 0; k <= q; ++k) P[(h*L + q)*L + k] = srow[k] * inv;
        for (int d = 0; d < hd; ++d) {
            double a = 0.0;
            for (int k = 0; k <= q; ++k)
                a += static_cast<double>(P[(h*L + q)*L + k]) * V[k*D + off + d];
            O_attn[q*D + off + d] = static_cast<float>(a);
        }
      }

    // Wo bwd
    std::vector<float> dO_attn(L*D, 0.0f), dWo(D*D, 0.0f), dbo(D, 0.0f);
    for (int q = 0; q < L; ++q)
      for (int k = 0; k < D; ++k) {
        double a = 0.0;
        for (int c = 0; c < D; ++c) a += static_cast<double>(dOq[q*D + c]) * Woq[c*D + k];
        dO_attn[q*D + k] = static_cast<float>(a);
      }
    for (int c = 0; c < D; ++c) {
        double db = 0.0;
        for (int q = 0; q < L; ++q) db += dOq[q*D + c];
        dbo[c] = static_cast<float>(db);
        for (int k = 0; k < D; ++k) {
            double dw = 0.0;
            for (int q = 0; q < L; ++q) dw += static_cast<double>(dOq[q*D + c]) * O_attn[q*D + k];
            dWo[c*D + k] = static_cast<float>(dw);
        }
    }

    // Attention bwd (causal: only k<=q contribute).
    std::vector<float> dQ(L*D, 0.0f), dK(L*D, 0.0f), dV(L*D, 0.0f);
    for (int h = 0; h < nh; ++h) {
        const int off = h * hd;
        for (int k = 0; k < L; ++k)
          for (int d = 0; d < hd; ++d) {
            double a = 0.0;
            for (int q = k; q < L; ++q)  // P[q,k]==0 for k>q
                a += static_cast<double>(P[(h*L + q)*L + k]) * dO_attn[q*D + off + d];
            dV[k*D + off + d] = static_cast<float>(a);
          }
        std::vector<float> dP(L*L, 0.0f), dS(L*L, 0.0f);
        for (int q = 0; q < L; ++q)
          for (int k = 0; k <= q; ++k) {
            double a = 0.0;
            for (int d = 0; d < hd; ++d)
                a += static_cast<double>(dO_attn[q*D + off + d]) * V[k*D + off + d];
            dP[q*L + k] = static_cast<float>(a);
          }
        for (int q = 0; q < L; ++q) {
            double Dq = 0.0;
            for (int k = 0; k <= q; ++k)
                Dq += static_cast<double>(P[(h*L + q)*L + k]) * dP[q*L + k];
            for (int k = 0; k <= q; ++k) {
                const float p = P[(h*L + q)*L + k];
                dS[q*L + k] = p * (dP[q*L + k] - static_cast<float>(Dq)) * inv_sqrt;
            }
        }
        for (int q = 0; q < L; ++q)
          for (int d = 0; d < hd; ++d) {
            double a = 0.0;
            for (int k = 0; k <= q; ++k)
                a += static_cast<double>(dS[q*L + k]) * K[k*D + off + d];
            dQ[q*D + off + d] = static_cast<float>(a);
          }
        for (int k = 0; k < L; ++k)
          for (int d = 0; d < hd; ++d) {
            double a = 0.0;
            for (int q = k; q < L; ++q)
                a += static_cast<double>(dS[q*L + k]) * Q[q*D + off + d];
            dK[k*D + off + d] = static_cast<float>(a);
          }
    }

    // Projection bwd: self-attn → dX absorbs all three. Bias gradients = colsum.
    std::vector<float> dX(L*D, 0.0f), dWq(D*D, 0.0f), dWk(D*D, 0.0f), dWv(D*D, 0.0f);
    std::vector<float> dbq(D, 0.0f), dbk(D, 0.0f), dbv(D, 0.0f);
    auto proj_bwd_self = [&](const std::vector<float>& W,
                             const std::vector<float>& dOut,
                             std::vector<float>& dW_acc,
                             std::vector<float>& db_acc) {
        for (int i = 0; i < L; ++i)
          for (int k = 0; k < D; ++k) {
            double a = 0.0;
            for (int c = 0; c < D; ++c) a += static_cast<double>(W[c*D + k]) * dOut[i*D + c];
            dX[i*D + k] += static_cast<float>(a);
          }
        for (int c = 0; c < D; ++c) {
            double db = 0.0;
            for (int i = 0; i < L; ++i) db += dOut[i*D + c];
            db_acc[c] = static_cast<float>(db);
            for (int k = 0; k < D; ++k) {
                double dw = 0.0;
                for (int i = 0; i < L; ++i)
                    dw += static_cast<double>(dOut[i*D + c]) * Xq[i*D + k];
                dW_acc[c*D + k] = static_cast<float>(dw);
            }
        }
    };
    proj_bwd_self(Wqq, dQ, dWq, dbq);
    proj_bwd_self(Wkq, dK, dWk, dbk);
    proj_bwd_self(Wvq, dV, dWv, dbv);

    // GPU side.
    auto Xh = to_fp16(X);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    auto dOh = to_fp16(dO);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  L, D);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);
    Tensor bqg = Tensor::from_host_fp16_on(Device::CUDA, bqh.data(), D, 1);
    Tensor bkg = Tensor::from_host_fp16_on(Device::CUDA, bkh.data(), D, 1);
    Tensor bvg = Tensor::from_host_fp16_on(Device::CUDA, bvh.data(), D, 1);
    Tensor bog = Tensor::from_host_fp16_on(Device::CUDA, boh.data(), D, 1);
    Tensor dOg = Tensor::from_host_fp16_on(Device::CUDA, dOh.data(), L, D);

    Tensor dXg  = Tensor::empty_on(Device::CUDA, L, D, Dtype::FP16);
    Tensor dWqg = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWkg = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWvg = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dWog = Tensor::zeros_on(Device::CUDA, D, D, Dtype::FP16);
    Tensor dbqg = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);
    Tensor dbkg = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);
    Tensor dbvg = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);
    Tensor dbog = Tensor::zeros_on(Device::CUDA, D, 1, Dtype::FP16);

    brotensor::flash_attention_qkvo_backward(
        Xg, nullptr,
        Wqg, &bqg, Wkg, &bkg, Wvg, &bvg, Wog, &bog,
        nullptr, nh, /*causal=*/true,
        dOg,
        dXg, nullptr,
        dWqg, &dbqg, dWkg, &dbkg, dWvg, &dbvg, dWog, &dbog);

    std::vector<uint16_t> dX_got(L*D), dWq_got(D*D), dWk_got(D*D),
                          dWv_got(D*D), dWo_got(D*D);
    std::vector<uint16_t> dbq_got(D), dbk_got(D), dbv_got(D), dbo_got(D);
    dXg.copy_to_host_fp16(dX_got.data());
    dWqg.copy_to_host_fp16(dWq_got.data());
    dWkg.copy_to_host_fp16(dWk_got.data());
    dWvg.copy_to_host_fp16(dWv_got.data());
    dWog.copy_to_host_fp16(dWo_got.data());
    dbqg.copy_to_host_fp16(dbq_got.data());
    dbkg.copy_to_host_fp16(dbk_got.data());
    dbvg.copy_to_host_fp16(dbv_got.data());
    dbog.copy_to_host_fp16(dbo_got.data());
    brotensor::sync_all();

    // Tolerance: dWq/dWk/dWv accumulate over (L * D) terms each ≈ 0.05 magnitude,
    // and the dO scale is ~0.3 * 0.3 = 0.1; for L=77 D=768 we expect grads up to
    // a few. The 1e-2 atol + 1e-2 rtol policy still holds at this scale.
    check_fp16(dX_got,  dX,  "causal-bwd dX");
    check_fp16(dWq_got, dWq, "causal-bwd dWq");
    check_fp16(dWk_got, dWk, "causal-bwd dWk");
    check_fp16(dWv_got, dWv, "causal-bwd dWv");
    check_fp16(dWo_got, dWo, "causal-bwd dWo");
    check_fp16(dbq_got, dbq, "causal-bwd dbq");
    check_fp16(dbk_got, dbk, "causal-bwd dbk");
    check_fp16(dbv_got, dbv, "causal-bwd dbv");
    check_fp16(dbo_got, dbo, "causal-bwd dbo");
}
