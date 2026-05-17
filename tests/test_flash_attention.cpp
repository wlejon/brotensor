// Parity for flash_attention_forward_gpu (tiled, online-softmax) against the
// classic cross_attention_forward_gpu impl at small sizes, plus a stress
// shape (Lk=8192) that the dynamic-shared-mem cross-attn cannot run.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}
static std::vector<float> rq(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v[i]));
    return o;
}

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

static void check_fp16(const std::vector<uint16_t>& got,
                       const std::vector<float>& ref,
                       const char* label,
                       float atol = 1e-2f, float rtol = 1e-2f) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
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

    GpuTensor Qg, Kg, Vg, Og;
    auto Qh = to_fp16(Q), Kh = to_fp16(K), Vh = to_fp16(V);
    brotensor::upload_fp16(Qh.data(), Lq, D, Qg);
    brotensor::upload_fp16(Kh.data(), Lk, D, Kg);
    brotensor::upload_fp16(Vh.data(), Lk, D, Vg);
    GpuTensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        brotensor::upload(mask_host.data(), Lk, 1, mg);
        d_mask = mg.data;
    }
    brotensor::flash_attention_forward_gpu(Qg, Kg, Vg, d_mask, nh, /*causal=*/false, Og);
    CHECK(Og.rows == Lq && Og.cols == D && Og.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Og.size());
    brotensor::download_fp16(Og, got.data());
    brotensor::cuda_sync();
    check_fp16(got, O_ref, label);
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

    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog;
    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    brotensor::upload_fp16(Xh.data(), Lq, D, Xg);
    brotensor::upload_fp16(Ch.data(), Lk, D, Cg);
    brotensor::upload_fp16(Wqh.data(), D, D, Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D, Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D, Wvg);
    brotensor::upload_fp16(Woh.data(), D, D, Wog);

    GpuTensor O_ref_g;
    brotensor::cross_attention_forward_gpu(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                           nullptr, nh, O_ref_g);
    GpuTensor O_flash_g;
    brotensor::flash_attention_qkvo_forward_gpu(Xg, &Cg,
                                                Wqg, nullptr, Wkg, nullptr,
                                                Wvg, nullptr, Wog, nullptr,
                                                nullptr, nh, /*causal=*/false, O_flash_g);

    std::vector<uint16_t> ref_h(O_ref_g.size()), flash_h(O_flash_g.size());
    brotensor::download_fp16(O_ref_g, ref_h.data());
    brotensor::download_fp16(O_flash_g, flash_h.data());
    brotensor::cuda_sync();
    std::vector<float> ref(ref_h.size());
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = brotensor::fp16_bits_to_fp32(ref_h[i]);
    check_fp16(flash_h, ref, label);
}

// Verify the optional bias path of flash_attention_qkvo_forward_gpu against a
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

    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog, bqg, bkg, bvg, bog;
    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    brotensor::upload_fp16(Xh.data(),  Lq, D, Xg);
    brotensor::upload_fp16(Ch.data(),  Lk, D, Cg);
    brotensor::upload_fp16(Wqh.data(), D, D, Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D, Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D, Wvg);
    brotensor::upload_fp16(Woh.data(), D, D, Wog);
    brotensor::upload_fp16(bqh.data(), D, 1, bqg);
    brotensor::upload_fp16(bkh.data(), D, 1, bkg);
    brotensor::upload_fp16(bvh.data(), D, 1, bvg);
    brotensor::upload_fp16(boh.data(), D, 1, bog);

    GpuTensor Og;
    brotensor::flash_attention_qkvo_forward_gpu(Xg, &Cg,
                                                Wqg, &bqg, Wkg, &bkg,
                                                Wvg, &bvg, Wog, &bog,
                                                nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    brotensor::download_fp16(Og, got.data());
    brotensor::cuda_sync();
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

    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog;
    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    brotensor::upload_fp16(Xh.data(),  Lq, D,     Xg);
    brotensor::upload_fp16(Ch.data(),  Lk, D_ctx, Cg);
    brotensor::upload_fp16(Wqh.data(), D, D,     Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D_ctx, Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D_ctx, Wvg);
    brotensor::upload_fp16(Woh.data(), D, D,     Wog);

    GpuTensor Og;
    brotensor::flash_attention_qkvo_forward_gpu(Xg, &Cg,
                                                Wqg, nullptr, Wkg, nullptr,
                                                Wvg, nullptr, Wog, nullptr,
                                                nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    brotensor::download_fp16(Og, got.data());
    brotensor::cuda_sync();
    check_fp16(got, O_ref, label);
}

// Self-attention QKVO compared against a CPU reference (not against
// cross_attention_forward_gpu, which itself dispatches to flash for FP16 —
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

    GpuTensor Xg, Wqg, Wkg, Wvg, Wog;
    auto Xh = to_fp16(X);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    brotensor::upload_fp16(Xh.data(),  Lq, D, Xg);
    brotensor::upload_fp16(Wqh.data(), D, D, Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D, Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D, Wvg);
    brotensor::upload_fp16(Woh.data(), D, D, Wog);

    GpuTensor Og;
    brotensor::flash_attention_qkvo_forward_gpu(Xg, nullptr,
                                                Wqg, nullptr, Wkg, nullptr,
                                                Wvg, nullptr, Wog, nullptr,
                                                nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    brotensor::download_fp16(Og, got.data());
    brotensor::cuda_sync();
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

    GpuTensor Xg, Wqg, Wkg, Wvg, Wog, bqg, bkg, bvg, bog;
    auto Xh = to_fp16(X);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    brotensor::upload_fp16(Xh.data(),  L, D, Xg);
    brotensor::upload_fp16(Wqh.data(), D, D, Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D, Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D, Wvg);
    brotensor::upload_fp16(Woh.data(), D, D, Wog);
    brotensor::upload_fp16(bqh.data(), D, 1, bqg);
    brotensor::upload_fp16(bkh.data(), D, 1, bkg);
    brotensor::upload_fp16(bvh.data(), D, 1, bvg);
    brotensor::upload_fp16(boh.data(), D, 1, bog);

    GpuTensor Og;
    brotensor::flash_attention_qkvo_forward_gpu(Xg, nullptr,
                                                Wqg, &bqg, Wkg, &bkg,
                                                Wvg, &bvg, Wog, &bog,
                                                nullptr, nh, /*causal=*/true, Og);
    std::vector<uint16_t> got(Og.size());
    brotensor::download_fp16(Og, got.data());
    brotensor::cuda_sync();
    check_fp16(got, O_ref, label);
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
    GpuTensor Qg, Kg, Vg, Og;
    auto Qh = to_fp16(Q), Kh = to_fp16(K), Vh = to_fp16(V);
    brotensor::upload_fp16(Qh.data(), Lq, D, Qg);
    brotensor::upload_fp16(Kh.data(), Lk, D, Kg);
    brotensor::upload_fp16(Vh.data(), Lk, D, Vg);
    brotensor::flash_attention_forward_gpu(Qg, Kg, Vg, nullptr, nh, /*causal=*/false, Og);
    std::vector<uint16_t> got(Og.size());
    brotensor::download_fp16(Og, got.data());
    brotensor::cuda_sync();
    int finite = 0;
    for (auto h : got) {
        const float f = brotensor::fp16_bits_to_fp32(h);
        if (std::isfinite(f) && std::fabs(f) < 1.0f) ++finite;
    }
    std::printf("    finite_and_small=%d / %zu\n", finite, got.size());
    CHECK(finite == static_cast<int>(got.size()));
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_flash_attention\n");

    run_one("tiny",        4, 5, 16, 2, false);
    run_one("multi-head",  6, 7, 32, 4, false);
    run_one("with mask",   4, 8, 16, 2, true);
    run_one("ktile-cross", 4, 130, 32, 4, false);   // forces multiple Lk tiles
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

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll flash-attention checks passed.\n");
    return 0;
}
