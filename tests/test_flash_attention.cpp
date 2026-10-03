// Parity for flash_attention_forward (tiled, online-softmax) against the
// classic cross_attention_forward impl at small sizes, plus a stress
// shape (Lk=8192) that the dynamic-shared-mem cross-attn cannot run. The backward cases are in
// test_flash_attention_bwd.cpp.

#include "test_flash_attention_common.h"

// CPU reference for QKV-already-projected attention.
static void attn_qkv_cpu(const std::vector<float>& Q,
                         const std::vector<float>& K,
                         const std::vector<float>& V,
                         const std::vector<float>* mask,
                         int Lq, int Lk, int D, int nh,
                         std::vector<float>& O) {
    const int hd = D / nh;
    O.assign(static_cast<size_t>(Lq) * D, 0.0f);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> scores(Lk);
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < nh; ++h) {
            const int off = h * hd;
            float maxv = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q*D + off + d]) * K[k*D + off + d];
                float s = static_cast<float>(dot) * inv_sqrt;
                if (mask && (*mask)[k] <= 0.5f) s = -1e30f;
                scores[k] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                scores[k] = std::exp(scores[k] - maxv);
                sum += scores[k];
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = 0; k < Lk; ++k)
                    a += static_cast<double>(scores[k]) * inv * V[k*D + off + d];
                O[q*D + off + d] = static_cast<float>(a);
            }
        }
    }
}


static void run_one(const char* label, int Lq, int Lk, int D, int nh,
                    bool use_mask) {
    std::printf("  %s  Lq=%d Lk=%d D=%d nh=%d mask=%d\n",
                label, Lq, Lk, D, nh, (int)use_mask);
    std::mt19937 rng(0xF1A5);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(Lq*D), K(Lk*D), V(Lk*D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    auto Qq = rq(Q), Kq = rq(K), Vq = rq(V);
    std::vector<float> mask_host;
    const std::vector<float>* mask_ptr = nullptr;
    if (use_mask) {
        mask_host.assign(Lk, 1.0f);
        for (int k = 3*Lk/4; k < Lk; ++k) mask_host[k] = 0.0f;
        mask_ptr = &mask_host;
    }
    std::vector<float> O_ref;
    attn_qkv_cpu(Qq, Kq, Vq, mask_ptr, Lq, Lk, D, nh, O_ref);

    auto Qh = to_fp16(Q), Kh = to_fp16(K), Vh = to_fp16(V);
    Tensor Qg = Tensor::from_host_fp16_on(Device::CUDA, Qh.data(), Lq, D);
    Tensor Kg = Tensor::from_host_fp16_on(Device::CUDA, Kh.data(), Lk, D);
    Tensor Vg = Tensor::from_host_fp16_on(Device::CUDA, Vh.data(), Lk, D);
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor Og;
    brotensor::flash_attention_forward(Qg, Kg, Vg, d_mask, nh, /*causal=*/false, Og);
    CHECK(Og.rows == Lq && Og.cols == D && Og.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Og.size());
    Og.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, O_ref, label);
}

static std::vector<uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_bf16_bits(v[i]);
    return o;
}
static std::vector<float> rqb(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v[i]));
    return o;
}
static void check_bf16(const std::vector<uint16_t>& got,
                       const std::vector<float>& ref,
                       const char* label,
                       float atol = 4e-2f, float rtol = 4e-2f) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::bf16_bits_to_fp32(got[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(ref[i])) {
            if (bad < 3)
                std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                            label, i, g, ref[i], e);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

// BF16 twin of run_one — exercises flash_attention_forward's BF16 per-head
// path (the one VAE mid-block self-attention hits: a single head with
// head_dim > 72, too wide for the fused kernel). Covers the Lk-padding the
// WMMA matmul now uses for BF16.
static void run_one_bf16(const char* label, int Lq, int Lk, int D, int nh,
                         bool use_mask) {
    std::printf("  %s [bf16]  Lq=%d Lk=%d D=%d nh=%d mask=%d\n",
                label, Lq, Lk, D, nh, (int)use_mask);
    std::mt19937 rng(0xB16F);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(Lq*D), K(Lk*D), V(Lk*D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    auto Qq = rqb(Q), Kq = rqb(K), Vq = rqb(V);
    std::vector<float> mask_host;
    const std::vector<float>* mask_ptr = nullptr;
    if (use_mask) {
        mask_host.assign(Lk, 1.0f);
        for (int k = 3*Lk/4; k < Lk; ++k) mask_host[k] = 0.0f;
        mask_ptr = &mask_host;
    }
    std::vector<float> O_ref;
    attn_qkv_cpu(Qq, Kq, Vq, mask_ptr, Lq, Lk, D, nh, O_ref);

    auto Qb = to_bf16(Q), Kb = to_bf16(K), Vb = to_bf16(V);
    Tensor Qg = Tensor::from_host_bf16_on(Device::CUDA, Qb.data(), Lq, D);
    Tensor Kg = Tensor::from_host_bf16_on(Device::CUDA, Kb.data(), Lk, D);
    Tensor Vg = Tensor::from_host_bf16_on(Device::CUDA, Vb.data(), Lk, D);
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor Og;
    brotensor::flash_attention_forward(Qg, Kg, Vg, d_mask, nh, /*causal=*/false, Og);
    CHECK(Og.rows == Lq && Og.cols == D && Og.dtype == Dtype::BF16);
    std::vector<uint16_t> got = Og.to_host_vector_bf16();
    brotensor::sync_all();
    check_bf16(got, O_ref, label);
}

static void run_qkvo(const char* label, int Lq, int Lk, int D, int nh) {
    std::printf("  %s qkvo Lq=%d Lk=%d D=%d nh=%d\n", label, Lq, Lk, D, nh);
    std::mt19937 rng(0xC0DE);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq*D), Ctx(Lk*D);
    std::vector<float> Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D);
    for (auto& v : X)   v = dist(rng);
    for (auto& v : Ctx) v = dist(rng);
    for (auto& v : Wq)  v = dist(rng);
    for (auto& v : Wk)  v = dist(rng);
    for (auto& v : Wv)  v = dist(rng);
    for (auto& v : Wo)  v = dist(rng);

    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), Lq, D);
    Tensor Cg  = Tensor::from_host_fp16_on(Device::CUDA, Ch.data(), Lk, D);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);

    Tensor O_ref_g;
    brotensor::cross_attention_forward(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                       nullptr, nh, O_ref_g);
    Tensor O_flash_g;
    brotensor::flash_attention_qkvo_forward(Xg, &Cg,
                                            Wqg, nullptr, Wkg, nullptr,
                                            Wvg, nullptr, Wog, nullptr,
                                            nullptr, nh, /*causal=*/false, O_flash_g);

    std::vector<uint16_t> ref_h(O_ref_g.size()), flash_h(O_flash_g.size());
    O_ref_g.copy_to_host_fp16(ref_h.data());
    O_flash_g.copy_to_host_fp16(flash_h.data());
    brotensor::sync_all();
    std::vector<float> ref(ref_h.size());
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = brotensor::fp16_bits_to_fp32(ref_h[i]);
    check_fp16(flash_h, ref, label);
}

// Verify the optional bias path of flash_attention_qkvo_forward against a
// CPU reference. We project Q/K/V/O with explicit biases on the host and
// compare to the GPU op called with all four biases. Tests the case
// brodiffusion needs for CLIP attention (all biases present).
static void run_qkvo_with_biases(const char* label, int Lq, int Lk, int D, int nh) {
    std::printf("  %s qkvo+biases Lq=%d Lk=%d D=%d nh=%d\n", label, Lq, Lk, D, nh);
    std::mt19937 rng(0xB1A5);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq*D), Ctx(Lk*D);
    std::vector<float> Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D);
    std::vector<float> bq(D), bk(D), bv(D), bo(D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X); fill(Ctx);
    fill(Wq); fill(Wk); fill(Wv); fill(Wo);
    fill(bq); fill(bk); fill(bv); fill(bo);

    auto Xq = rq(X), Ctxq = rq(Ctx);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);
    auto bqq = rq(bq), bkq = rq(bk), bvq = rq(bv), boq = rq(bo);

    // CPU: project Q = X @ Wq^T + bq, etc., then attn, then O = Op @ Wo^T + bo.
    auto proj = [](const std::vector<float>& A, const std::vector<float>& W,
                   const std::vector<float>& b, int M, int Dim,
                   std::vector<float>& Out) {
        Out.assign(static_cast<size_t>(M) * Dim, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < Dim; ++n) {
                double s = b[n];
                for (int k = 0; k < Dim; ++k)
                    s += static_cast<double>(A[m*Dim + k]) * W[n*Dim + k];
                Out[m*Dim + n] = static_cast<float>(s);
            }
    };
    std::vector<float> Qp, Kp, Vp, Op, O_ref;
    proj(Xq,   Wqq, bqq, Lq, D, Qp);
    proj(Ctxq, Wkq, bkq, Lk, D, Kp);
    proj(Ctxq, Wvq, bvq, Lk, D, Vp);
    attn_qkv_cpu(Qp, Kp, Vp, nullptr, Lq, Lk, D, nh, Op);
    proj(Op, Woq, boq, Lq, D, O_ref);

    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  Lq, D);
    Tensor Cg  = Tensor::from_host_fp16_on(Device::CUDA, Ch.data(),  Lk, D);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);
    Tensor bqg = Tensor::from_host_fp16_on(Device::CUDA, bqh.data(), D, 1);
    Tensor bkg = Tensor::from_host_fp16_on(Device::CUDA, bkh.data(), D, 1);
    Tensor bvg = Tensor::from_host_fp16_on(Device::CUDA, bvh.data(), D, 1);
    Tensor bog = Tensor::from_host_fp16_on(Device::CUDA, boh.data(), D, 1);

    Tensor Og;
    brotensor::flash_attention_qkvo_forward(Xg, &Cg,
                                            Wqg, &bqg, Wkg, &bkg,
                                            Wvg, &bvg, Wog, &bog,
                                            nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    Og.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, O_ref, label);
}

// Cross-attention with rectangular Wk/Wv: Ctx has a different width than X
// (the SD1.5 case — Q from image tokens at D, K/V from CLIP text tokens at
// D_ctx=768). Verifies the shape-check relaxation accepts Wk/Wv as
// (D, D_ctx) and the projection math matches a CPU reference.
static void run_qkvo_rect_ctx(const char* label, int Lq, int Lk, int D,
                              int D_ctx, int nh) {
    std::printf("  %s qkvo rect-ctx Lq=%d Lk=%d D=%d D_ctx=%d nh=%d\n",
                label, Lq, Lk, D, D_ctx, nh);
    std::mt19937 rng(0xCAFEBABE);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq*D), Ctx(Lk*D_ctx);
    std::vector<float> Wq(D*D), Wk(D*D_ctx), Wv(D*D_ctx), Wo(D*D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X); fill(Ctx); fill(Wq); fill(Wk); fill(Wv); fill(Wo);

    auto Xq = rq(X), Ctxq = rq(Ctx);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);

    // CPU reference: rectangular projection for K/V.
    auto proj = [](const std::vector<float>& A, const std::vector<float>& W,
                   int M, int Kin, int Nout, std::vector<float>& Out) {
        Out.assign(static_cast<size_t>(M) * Nout, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < Nout; ++n) {
                double s = 0.0;
                for (int k = 0; k < Kin; ++k)
                    s += static_cast<double>(A[m*Kin + k]) * W[n*Kin + k];
                Out[m*Nout + n] = static_cast<float>(s);
            }
    };
    std::vector<float> Qp, Kp, Vp, Op, O_ref;
    proj(Xq,   Wqq, Lq, D,     D, Qp);
    proj(Ctxq, Wkq, Lk, D_ctx, D, Kp);
    proj(Ctxq, Wvq, Lk, D_ctx, D, Vp);
    attn_qkv_cpu(Qp, Kp, Vp, nullptr, Lq, Lk, D, nh, Op);
    proj(Op, Woq, Lq, D, D, O_ref);

    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  Lq, D);
    Tensor Cg  = Tensor::from_host_fp16_on(Device::CUDA, Ch.data(),  Lk, D_ctx);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D_ctx);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D_ctx);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);

    Tensor Og;
    brotensor::flash_attention_qkvo_forward(Xg, &Cg,
                                            Wqg, nullptr, Wkg, nullptr,
                                            Wvg, nullptr, Wog, nullptr,
                                            nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    Og.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, O_ref, label);
}

// Self-attention QKVO compared against a CPU reference (not against
// cross_attention_forward, which itself dispatches to flash for FP16 —
// that would be flash-vs-flash). Square weights, no context tensor.
static void run_qkvo_self_cpu(const char* label, int Lq, int D, int nh) {
    std::printf("  %s qkvo self-vs-cpu Lq=%d D=%d nh=%d\n", label, Lq, D, nh);
    std::mt19937 rng(0xA5A5);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq*D);
    std::vector<float> Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X); fill(Wq); fill(Wk); fill(Wv); fill(Wo);

    auto Xq = rq(X);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);

    auto proj = [](const std::vector<float>& A, const std::vector<float>& W,
                   int M, int Kin, int Nout, std::vector<float>& Out) {
        Out.assign(static_cast<size_t>(M) * Nout, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < Nout; ++n) {
                double s = 0.0;
                for (int k = 0; k < Kin; ++k)
                    s += static_cast<double>(A[m*Kin + k]) * W[n*Kin + k];
                Out[m*Nout + n] = static_cast<float>(s);
            }
    };
    std::vector<float> Qp, Kp, Vp, Op, O_ref;
    proj(Xq, Wqq, Lq, D, D, Qp);
    proj(Xq, Wkq, Lq, D, D, Kp);
    proj(Xq, Wvq, Lq, D, D, Vp);
    attn_qkv_cpu(Qp, Kp, Vp, nullptr, Lq, Lq, D, nh, Op);
    proj(Op, Woq, Lq, D, D, O_ref);

    auto Xh = to_fp16(X);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  Lq, D);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);

    Tensor Og;
    brotensor::flash_attention_qkvo_forward(Xg, nullptr,
                                            Wqg, nullptr, Wkg, nullptr,
                                            Wvg, nullptr, Wog, nullptr,
                                            nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    Og.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, O_ref, label);
}

// Causal self-attention with all four biases — CLIP text encoder shape.
static void run_qkvo_causal(const char* label, int L, int D, int nh) {
    std::printf("  %s qkvo causal+biases L=%d D=%d nh=%d\n", label, L, D, nh);
    std::mt19937 rng(0xC11C);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(L*D);
    std::vector<float> Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D);
    std::vector<float> bq(D), bk(D), bv(D), bo(D);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
    fill(X);
    fill(Wq); fill(Wk); fill(Wv); fill(Wo);
    fill(bq); fill(bk); fill(bv); fill(bo);

    auto Xq = rq(X);
    auto Wqq = rq(Wq), Wkq = rq(Wk), Wvq = rq(Wv), Woq = rq(Wo);
    auto bqq = rq(bq), bkq = rq(bk), bvq = rq(bv), boq = rq(bo);

    auto proj = [](const std::vector<float>& A, const std::vector<float>& W,
                   const std::vector<float>& b, int M, int Dim,
                   std::vector<float>& Out) {
        Out.assign(static_cast<size_t>(M) * Dim, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < Dim; ++n) {
                double s = b[n];
                for (int k = 0; k < Dim; ++k)
                    s += static_cast<double>(A[m*Dim + k]) * W[n*Dim + k];
                Out[m*Dim + n] = static_cast<float>(s);
            }
    };
    std::vector<float> Qp, Kp, Vp;
    proj(Xq, Wqq, bqq, L, D, Qp);
    proj(Xq, Wkq, bkq, L, D, Kp);
    proj(Xq, Wvq, bvq, L, D, Vp);

    // Causal attention CPU reference: same as attn_qkv_cpu but k > q is masked.
    const int hd = D / nh;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> Op(static_cast<size_t>(L) * D, 0.0f);
    std::vector<float> scores(L);
    for (int q = 0; q < L; ++q) {
        for (int h = 0; h < nh; ++h) {
            const int off = h * hd;
            float maxv = -1e30f;
            for (int k = 0; k <= q; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Qp[q*D + off + d]) * Kp[k*D + off + d];
                float s = static_cast<float>(dot) * inv_sqrt;
                scores[k] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int k = 0; k <= q; ++k) {
                scores[k] = std::exp(scores[k] - maxv);
                sum += scores[k];
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = 0; k <= q; ++k)
                    a += static_cast<double>(scores[k]) * inv * Vp[k*D + off + d];
                Op[q*D + off + d] = static_cast<float>(a);
            }
        }
    }
    std::vector<float> O_ref;
    proj(Op, Woq, boq, L, D, O_ref);

    auto Xh = to_fp16(X);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    Tensor Xg  = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),  L, D);
    Tensor Wqg = Tensor::from_host_fp16_on(Device::CUDA, Wqh.data(), D, D);
    Tensor Wkg = Tensor::from_host_fp16_on(Device::CUDA, Wkh.data(), D, D);
    Tensor Wvg = Tensor::from_host_fp16_on(Device::CUDA, Wvh.data(), D, D);
    Tensor Wog = Tensor::from_host_fp16_on(Device::CUDA, Woh.data(), D, D);
    Tensor bqg = Tensor::from_host_fp16_on(Device::CUDA, bqh.data(), D, 1);
    Tensor bkg = Tensor::from_host_fp16_on(Device::CUDA, bkh.data(), D, 1);
    Tensor bvg = Tensor::from_host_fp16_on(Device::CUDA, bvh.data(), D, 1);
    Tensor bog = Tensor::from_host_fp16_on(Device::CUDA, boh.data(), D, 1);

    Tensor Og;
    brotensor::flash_attention_qkvo_forward(Xg, nullptr,
                                            Wqg, &bqg, Wkg, &bkg,
                                            Wvg, &bvg, Wog, &bog,
                                            nullptr, nh, /*causal=*/true, Og);
    std::vector<uint16_t> got(Og.size());
    Og.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    check_fp16(got, O_ref, label);
}

// ─── flash_attention_windowed_forward (FP32, CPU + CUDA) ────────────────────

static void check_f32(const std::vector<float>& got, const std::vector<float>& ref,
                      const char* label, float atol = 1e-4f, float rtol = 1e-4f) {
    int bad = 0; float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(got[i] - ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(ref[i])) {
            if (bad < 3)
                std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                            label, i, got[i], ref[i], e);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

// Naive sliding-window causal reference: query q attends keys [lo, q] with
// lo = max(0, q-window+1) (window <= 0 => lo = 0, full causal).
static void windowed_ref(const std::vector<float>& Q, const std::vector<float>& K,
                         const std::vector<float>& V, int L, int D, int nh,
                         int window, std::vector<float>& O) {
    const int hd = D / nh;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    O.assign(static_cast<size_t>(L) * D, 0.0f);
    std::vector<float> sc(L);
    for (int q = 0; q < L; ++q) {
        const int lo = (window > 0) ? std::max(0, q - window + 1) : 0;
        for (int h = 0; h < nh; ++h) {
            const int off = h * hd;
            float mx = -1e30f;
            for (int k = lo; k <= q; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q*D + off + d]) * K[k*D + off + d];
                sc[k] = static_cast<float>(dot) * inv_sqrt;
                if (sc[k] > mx) mx = sc[k];
            }
            float sum = 0.0f;
            for (int k = lo; k <= q; ++k) { sc[k] = std::exp(sc[k] - mx); sum += sc[k]; }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = lo; k <= q; ++k)
                    a += static_cast<double>(sc[k]) * inv * V[k*D + off + d];
                O[q*D + off + d] = static_cast<float>(a);
            }
        }
    }
}

static void run_windowed(const char* label, int L, int D, int nh, int window) {
    std::printf("  %s windowed L=%d D=%d nh=%d window=%d\n", label, L, D, nh, window);
    std::mt19937 rng(0x5117 + window);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(L*D), K(L*D), V(L*D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    std::vector<float> O_ref;
    windowed_ref(Q, K, V, L, D, nh, window, O_ref);

    // CPU FP32.
    Tensor Qc = Tensor::from_host_on(Device::CPU, Q.data(), L, D);
    Tensor Kc = Tensor::from_host_on(Device::CPU, K.data(), L, D);
    Tensor Vc = Tensor::from_host_on(Device::CPU, V.data(), L, D);
    Tensor Oc;
    brotensor::flash_attention_windowed_forward(Qc, Kc, Vc, nullptr, nh, window, Oc);
    CHECK(Oc.rows == L && Oc.cols == D && Oc.dtype == Dtype::FP32);
    std::vector<float> cpu_got(static_cast<size_t>(L) * D);
    Oc.copy_to_host(cpu_got.data());
    check_f32(cpu_got, O_ref, (std::string(label) + " cpu").c_str());

    // CUDA FP32.
    Tensor Qg = Tensor::from_host_on(Device::CUDA, Q.data(), L, D);
    Tensor Kg = Tensor::from_host_on(Device::CUDA, K.data(), L, D);
    Tensor Vg = Tensor::from_host_on(Device::CUDA, V.data(), L, D);
    Tensor Og;
    brotensor::flash_attention_windowed_forward(Qg, Kg, Vg, nullptr, nh, window, Og);
    std::vector<float> cuda_got(static_cast<size_t>(L) * D);
    Og.copy_to_host(cuda_got.data());
    brotensor::sync_all();
    check_f32(cuda_got, O_ref, (std::string(label) + " cuda").c_str());
}

// Decode reference: Lq queries at the last Lq positions of a length-Lk causal
// sequence (q_offset = Lk - Lq), each attending [max(0,pos-window+1), pos].
static void decode_ref(const std::vector<float>& Q, const std::vector<float>& K,
                       const std::vector<float>& V, int Lq, int Lk, int D, int nh,
                       int window, std::vector<float>& O) {
    const int hd = D / nh;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    const int q_off = Lk - Lq;
    O.assign(static_cast<size_t>(Lq) * D, 0.0f);
    std::vector<float> sc(Lk);
    for (int q = 0; q < Lq; ++q) {
        const int aq = q + q_off;
        const int lo = (window > 0) ? std::max(0, aq - window + 1) : 0;
        for (int h = 0; h < nh; ++h) {
            const int off = h * hd;
            float mx = -1e30f;
            for (int k = lo; k <= aq; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q*D + off + d]) * K[k*D + off + d];
                sc[k] = static_cast<float>(dot) * inv_sqrt;
                if (sc[k] > mx) mx = sc[k];
            }
            float sum = 0.0f;
            for (int k = lo; k <= aq; ++k) { sc[k] = std::exp(sc[k] - mx); sum += sc[k]; }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = lo; k <= aq; ++k)
                    a += static_cast<double>(sc[k]) * inv * V[k*D + off + d];
                O[q*D + off + d] = static_cast<float>(a);
            }
        }
    }
}

// Incremental-decode case: an Lq-token query block (Lq < Lk) attends a length-Lk
// K/V cache — the AR step that replaces the varlen path in the TTS loops.
static void run_windowed_decode(const char* label, int Lq, int Lk, int D, int nh,
                                int window) {
    std::printf("  %s decode Lq=%d Lk=%d D=%d nh=%d window=%d\n",
                label, Lq, Lk, D, nh, window);
    std::mt19937 rng(0xDEC0 + window + Lk);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(static_cast<size_t>(Lq)*D), K(static_cast<size_t>(Lk)*D),
                       V(static_cast<size_t>(Lk)*D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    std::vector<float> O_ref;
    decode_ref(Q, K, V, Lq, Lk, D, nh, window, O_ref);

    Tensor Qc = Tensor::from_host_on(Device::CPU, Q.data(), Lq, D);
    Tensor Kc = Tensor::from_host_on(Device::CPU, K.data(), Lk, D);
    Tensor Vc = Tensor::from_host_on(Device::CPU, V.data(), Lk, D);
    Tensor Oc;
    brotensor::flash_attention_windowed_forward(Qc, Kc, Vc, nullptr, nh, window, Oc);
    CHECK(Oc.rows == Lq && Oc.cols == D && Oc.dtype == Dtype::FP32);
    std::vector<float> cpu_got(static_cast<size_t>(Lq) * D);
    Oc.copy_to_host(cpu_got.data());
    check_f32(cpu_got, O_ref, (std::string(label) + " cpu").c_str());

    Tensor Qg = Tensor::from_host_on(Device::CUDA, Q.data(), Lq, D);
    Tensor Kg = Tensor::from_host_on(Device::CUDA, K.data(), Lk, D);
    Tensor Vg = Tensor::from_host_on(Device::CUDA, V.data(), Lk, D);
    Tensor Og;
    brotensor::flash_attention_windowed_forward(Qg, Kg, Vg, nullptr, nh, window, Og);
    std::vector<float> cuda_got(static_cast<size_t>(Lq) * D);
    Og.copy_to_host(cuda_got.data());
    brotensor::sync_all();
    check_f32(cuda_got, O_ref, (std::string(label) + " cuda").c_str());
}

// GQA decode reference: Q has nh heads (Dq), K/V have n_kv heads (Dkv); query
// head h reads K/V head h/(nh/n_kv). Queries at the last Lq causal positions.
static void gqa_decode_ref(const std::vector<float>& Q, const std::vector<float>& K,
                           const std::vector<float>& V, int Lq, int Lk, int Dq,
                           int Dkv, int nh, int n_kv, int window,
                           std::vector<float>& O) {
    const int hd = Dq / nh;
    const int group = nh / n_kv;
    const int q_off = Lk - Lq;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    O.assign(static_cast<size_t>(Lq) * Dq, 0.0f);
    std::vector<float> sc(Lk);
    for (int q = 0; q < Lq; ++q) {
        const int aq = q + q_off;
        const int lo = (window > 0) ? std::max(0, aq - window + 1) : 0;
        for (int h = 0; h < nh; ++h) {
            const int off    = h * hd;
            const int off_kv = (h / group) * hd;
            float mx = -1e30f;
            for (int k = lo; k <= aq; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q*Dq + off + d]) * K[k*Dkv + off_kv + d];
                sc[k] = static_cast<float>(dot) * inv_sqrt;
                if (sc[k] > mx) mx = sc[k];
            }
            float sum = 0.0f;
            for (int k = lo; k <= aq; ++k) { sc[k] = std::exp(sc[k] - mx); sum += sc[k]; }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = lo; k <= aq; ++k)
                    a += static_cast<double>(sc[k]) * inv * V[k*Dkv + off_kv + d];
                O[q*Dq + off + d] = static_cast<float>(a);
            }
        }
    }
}

// Grouped-query windowed attention: K/V carry n_kv < nh heads (inferred from
// K.cols), as in the Qwen3-TTS AR loop's cache. CPU + CUDA vs reference.
static void run_gqa_decode(const char* label, int Lq, int Lk, int Dq, int nh,
                           int n_kv, int window) {
    const int hd = Dq / nh, Dkv = n_kv * hd;
    std::printf("  %s gqa Lq=%d Lk=%d Dq=%d nh=%d n_kv=%d window=%d\n",
                label, Lq, Lk, Dq, nh, n_kv, window);
    std::mt19937 rng(0x6CA9 + n_kv + window);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(static_cast<size_t>(Lq)*Dq), K(static_cast<size_t>(Lk)*Dkv),
                       V(static_cast<size_t>(Lk)*Dkv);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    std::vector<float> O_ref;
    gqa_decode_ref(Q, K, V, Lq, Lk, Dq, Dkv, nh, n_kv, window, O_ref);

    Tensor Qc = Tensor::from_host_on(Device::CPU, Q.data(), Lq, Dq);
    Tensor Kc = Tensor::from_host_on(Device::CPU, K.data(), Lk, Dkv);
    Tensor Vc = Tensor::from_host_on(Device::CPU, V.data(), Lk, Dkv);
    Tensor Oc;
    brotensor::flash_attention_windowed_forward(Qc, Kc, Vc, nullptr, nh, window, Oc);
    CHECK(Oc.rows == Lq && Oc.cols == Dq);
    std::vector<float> cpu_got(static_cast<size_t>(Lq) * Dq);
    Oc.copy_to_host(cpu_got.data());
    check_f32(cpu_got, O_ref, (std::string(label) + " cpu").c_str());

    Tensor Qg = Tensor::from_host_on(Device::CUDA, Q.data(), Lq, Dq);
    Tensor Kg = Tensor::from_host_on(Device::CUDA, K.data(), Lk, Dkv);
    Tensor Vg = Tensor::from_host_on(Device::CUDA, V.data(), Lk, Dkv);
    Tensor Og;
    brotensor::flash_attention_windowed_forward(Qg, Kg, Vg, nullptr, nh, window, Og);
    std::vector<float> cuda_got(static_cast<size_t>(Lq) * Dq);
    Og.copy_to_host(cuda_got.data());
    brotensor::sync_all();
    check_f32(cuda_got, O_ref, (std::string(label) + " cuda").c_str());
}

// window <= 0 (and window >= L) must reproduce plain causal — cross-check the
// windowed op against flash_attention_forward(causal=true) on CPU FP32.
static void run_windowed_eq_causal(const char* label, int L, int D, int nh) {
    std::printf("  %s windowed==causal L=%d D=%d nh=%d\n", label, L, D, nh);
    std::mt19937 rng(0xCA05A1);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(L*D), K(L*D), V(L*D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    Tensor Qc = Tensor::from_host_on(Device::CPU, Q.data(), L, D);
    Tensor Kc = Tensor::from_host_on(Device::CPU, K.data(), L, D);
    Tensor Vc = Tensor::from_host_on(Device::CPU, V.data(), L, D);
    Tensor O_causal, O_win0, O_winL;
    brotensor::flash_attention_forward(Qc, Kc, Vc, nullptr, nh, /*causal=*/true, O_causal);
    brotensor::flash_attention_windowed_forward(Qc, Kc, Vc, nullptr, nh, /*window=*/0, O_win0);
    brotensor::flash_attention_windowed_forward(Qc, Kc, Vc, nullptr, nh, /*window=*/L, O_winL);
    std::vector<float> ref(static_cast<size_t>(L)*D), w0(ref.size()), wL(ref.size());
    O_causal.copy_to_host(ref.data());
    O_win0.copy_to_host(w0.data());
    O_winL.copy_to_host(wL.data());
    check_f32(w0, ref, (std::string(label) + " window=0").c_str());
    check_f32(wL, ref, (std::string(label) + " window=L").c_str());
}

static void run_stress() {
    // Lk = 8192 is too big for the dynamic-shmem cross-attention but fine for
    // flash. We can't compare against CPU full-precision reasonably at this
    // size, so we just sanity-check that the output has finite values and a
    // plausible magnitude.
    const int Lq = 4, Lk = 8192, D = 64, nh = 4;
    std::printf("  stress Lq=%d Lk=%d D=%d nh=%d\n", Lq, Lk, D, nh);
    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> Q(Lq*D), K(Lk*D), V(Lk*D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);
    auto Qh = to_fp16(Q), Kh = to_fp16(K), Vh = to_fp16(V);
    Tensor Qg = Tensor::from_host_fp16_on(Device::CUDA, Qh.data(), Lq, D);
    Tensor Kg = Tensor::from_host_fp16_on(Device::CUDA, Kh.data(), Lk, D);
    Tensor Vg = Tensor::from_host_fp16_on(Device::CUDA, Vh.data(), Lk, D);
    Tensor Og;
    brotensor::flash_attention_forward(Qg, Kg, Vg, nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    Og.copy_to_host_fp16(got.data());
    brotensor::sync_all();
    int finite = 0;
    for (auto h : got) {
        const float f = brotensor::fp16_bits_to_fp32(h);
        if (std::isfinite(f) && std::fabs(f) < 1.0f) ++finite;
    }
    std::printf("    finite_and_small=%d / %zu\n", finite, got.size());
    CHECK(finite == static_cast<int>(got.size()));
}

int main() {
    brotensor::init();
    if (!bt_test::has_gpu()) {
        std::printf("no GPU backend - skipping\n");
        return 0;
    }
    std::printf("test_flash_attention\n");

    run_one("tiny",        4, 5, 16, 2, false);
    run_one("multi-head",  6, 7, 32, 4, false);
    run_one("with mask",   4, 8, 16, 2, true);
    run_one("ktile-cross", 4, 130, 32, 4, false);   // forces multiple Lk tiles

    // Fused FlashAttention-2 path (head_dim == 64). Exercise: exact tile
    // multiples, q-tile tails (Lq % 128), k-tile tails (Lk % 64), masking,
    // and a single-head case.
    run_one("fused hd64 exact", 128, 128, 256, 4, false);
    run_one("fused hd64 tails", 129,  65, 192, 3, false);
    run_one("fused hd64 mask",   70, 200, 128, 2, true);
    run_one("fused hd64 1head", 300, 333,  64, 1, false);

    // Fused path for head_dim == 72 (PixArt-Sigma DiT: D=1152, nh=16). 72 is
    // not a multiple of 16, so this exercises the HD_PAD=80 zero-padded WMMA
    // tiling and the decoupled softmax(BC) / output(HD_PAD) column ownership.
    // Same exact/tails/mask/1-head coverage as hd64; BR=64 here.
    run_one("fused hd72 exact",  64, 128, 1152, 16, false);
    run_one("fused hd72 tails",  65,  70, 1152, 16, false);
    run_one("fused hd72 mask",   70, 200,  720, 10, true);
    run_one("fused hd72 1head", 300, 333,   72,  1, false);

    // Fused path for head_dim == 128 (Krea 2 / Flux-class DiT self-attention).
    // PAD == HD here (no zero-pad columns) but BR drops to 48 for the shared-
    // memory cap, so exercise q-tile tails (Lq % 48), k-tile tails (Lk % 64),
    // masking, and a single-head case.
    run_one("fused hd128 exact",  96, 128, 512, 4, false);
    run_one("fused hd128 tails", 100,  70, 256, 2, false);
    run_one("fused hd128 mask",   70, 200, 384, 3, true);
    run_one("fused hd128 1head", 300, 333, 128, 1, false);

    // BF16 per-head WMMA path (the route VAE mid-block self-attention takes:
    // single head, head_dim > 72). Includes unaligned Lk to exercise the
    // 8-multiple padding the BF16 matmul now relies on, plus a mask case.
    run_one_bf16("bf16 multihead",   6,   7,  32, 4, false);
    run_one_bf16("bf16 1head hd128", 64, 128, 128, 1, false);
    run_one_bf16("bf16 1head tails", 70,  70, 128, 1, false);   // Lk not mult of 8
    run_one_bf16("bf16 1head mask",  48, 200, 128, 1, true);
    run_one_bf16("bf16 hd512 vae",   80,  80, 512, 1, false);   // VAE head_dim

    run_qkvo("qkvo small", 6, 7, 32, 4);
    run_qkvo("qkvo self",  8, 8, 32, 4);
    run_qkvo_with_biases("qkvo biases", 6, 7, 32, 4);
    run_qkvo_rect_ctx("qkvo rect-ctx",   8, 11, 32, 24, 4);
    run_qkvo_rect_ctx("qkvo SD1.5-like", 16, 77, 64, 48, 4);

    // SD1.5 U-Net cross-attention (Lk=77 CLIP, D_ctx=768, nh=8).
    run_qkvo_rect_ctx("SD1.5 xattn s1 hd40",  16, 77, 320,  768, 8);
    run_qkvo_rect_ctx("SD1.5 xattn s2 hd80",  16, 77, 640,  768, 8);
    run_qkvo_rect_ctx("SD1.5 xattn s3 hd160",  8, 77, 1280, 768, 8);

    // SD1.5 U-Net self-attention (square, no Ctx) vs CPU reference.
    run_qkvo_self_cpu("SD1.5 selfattn s1 hd40",  16, 320,  8);
    run_qkvo_self_cpu("SD1.5 selfattn s2 hd80",  16, 640,  8);
    run_qkvo_self_cpu("SD1.5 selfattn s3 hd160",  8, 1280, 8);

    // CLIP text encoder: D=768, nh=12 → hd=64, causal, all four biases.
    run_qkvo_causal("clip text", 77, 768, 12);

    run_stress();

    // ── Sliding-window causal attention (FP32, CPU + CUDA) ──────────────────
    run_windowed("win tiny",        8, 16, 2, 3);     // window < L, single tile
    run_windowed("win multi-head", 10, 32, 4, 4);
    run_windowed("win ktile",     200, 32, 4, 72);    // codec-like: spans Lk tiles
    run_windowed("win ge-L",       16, 32, 4, 64);    // window >= L => full causal
    run_windowed("win unbounded",  16, 32, 4, 0);     // window <= 0 => full causal
    run_windowed_eq_causal("win identity", 40, 64, 8);

    // ── Incremental-decode windowed attention (Lq < Lk) ─────────────────────
    run_windowed_decode("dec single",      1,  1, 32, 4, 0);    // first step, cache=1
    run_windowed_decode("dec one-of-many", 1, 50, 32, 4, 0);    // 1 query, full cache
    run_windowed_decode("dec windowed",    1, 90, 32, 4, 72);   // cache beyond window
    run_windowed_decode("dec block",       4, 40, 64, 8, 0);    // multi-token block
    run_windowed_decode("dec block win",   3, 80, 64, 8, 16);   // block + window

    // ── Grouped-query windowed attention (K/V have n_kv < nh heads) ─────────
    run_gqa_decode("gqa decode",   1, 50, 256, 8, 2, 0);    // 1 query, group 4
    run_gqa_decode("gqa prefill", 16, 16, 256, 8, 2, 0);    // Lq == Lk, GQA
    run_gqa_decode("gqa win",      1, 90, 256, 8, 4, 72);   // group 2, windowed
    run_gqa_decode("gqa block",    3, 40, 128, 4, 1, 0);    // n_kv==nh (MHA degenerate)

    // ── Backward tests ────────────────────────────────────────────────────
    run_bwd_self_vs_mha("bwd-self small", 6, 32, 4);
    run_bwd_self_vs_mha("bwd-self med",   8, 64, 8);
    run_bwd_cross_vs_cx("bwd-cross small", 6, 7, 32, 24, 4);
    run_bwd_cross_vs_cx("bwd-cross SD-ish", 8, 16, 64, 48, 8);
    run_bwd_cross_with_biases_cpu("bwd-cross+biases", 6, 7, 32, 24, 4);
    run_bwd_causal_cpu("bwd-causal clipish", 16, 64, 8);
    // SD1.5-ish mid-shape sanity (D=160 to keep CPU/FP32 ref fast).
    run_bwd_cross_vs_cx("bwd-cross SD mid", 16, 77, 160, 768, 8);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll flash-attention checks passed.\n");
    return 0;
}
