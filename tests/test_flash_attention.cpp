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

// ─── Backward tests ────────────────────────────────────────────────────────

// Self-attention backward parity with mha_backward_gpu (FP32 reference).
// Build identical X / Wq/Wk/Wv/Wo (no biases — mha has no bias path) and
// random dO; run FP32 mha fwd+bwd to get the reference grads; run flash
// FP16 bwd on the same inputs; compare element-wise with FP16 tolerance.
static void run_bwd_self_vs_mha(const char* label, int L, int D, int nh) {
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

    GpuTensor X_f32, Wq_f32, Wk_f32, Wv_f32, Wo_f32, dO_f32;
    brotensor::upload(Xq.data(),   L, D, X_f32);
    brotensor::upload(Wqq.data(),  D, D, Wq_f32);
    brotensor::upload(Wkq.data(),  D, D, Wk_f32);
    brotensor::upload(Wvq.data(),  D, D, Wv_f32);
    brotensor::upload(Woq.data(),  D, D, Wo_f32);
    brotensor::upload(dOq.data(),  L, D, dO_f32);

    GpuTensor Qh, Kh, Vh, Attnh, Yconcat, O_ref;
    brotensor::mha_forward_gpu(X_f32, Wq_f32, Wk_f32, Wv_f32, Wo_f32,
                               nullptr, nh, Qh, Kh, Vh, Attnh, Yconcat, O_ref);

    GpuTensor dX_ref(L, D), dWq_ref(D, D), dWk_ref(D, D),
              dWv_ref(D, D), dWo_ref(D, D);
    dWq_ref.zero(); dWk_ref.zero(); dWv_ref.zero(); dWo_ref.zero();
    brotensor::mha_backward_gpu(dO_f32, X_f32, Qh, Kh, Vh, Attnh, Yconcat,
                                Wq_f32, Wk_f32, Wv_f32, Wo_f32, nullptr, nh,
                                dX_ref, dWq_ref, dWk_ref, dWv_ref, dWo_ref);
    std::vector<float> dX_ref_h(L*D), dWq_ref_h(D*D), dWk_ref_h(D*D),
                       dWv_ref_h(D*D), dWo_ref_h(D*D);
    brotensor::download(dX_ref,  dX_ref_h.data());
    brotensor::download(dWq_ref, dWq_ref_h.data());
    brotensor::download(dWk_ref, dWk_ref_h.data());
    brotensor::download(dWv_ref, dWv_ref_h.data());
    brotensor::download(dWo_ref, dWo_ref_h.data());
    brotensor::cuda_sync();

    // Flash FP16 bwd path.
    GpuTensor Xg, Wqg, Wkg, Wvg, Wog, dOg;
    auto Xh = to_fp16(X), Wqh = to_fp16(Wq), Wkh = to_fp16(Wk),
         Wvh = to_fp16(Wv), Woh = to_fp16(Wo), dOh = to_fp16(dO);
    brotensor::upload_fp16(Xh.data(),  L, D, Xg);
    brotensor::upload_fp16(Wqh.data(), D, D, Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D, Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D, Wvg);
    brotensor::upload_fp16(Woh.data(), D, D, Wog);
    brotensor::upload_fp16(dOh.data(), L, D, dOg);

    GpuTensor dXg(L, D, Dtype::FP16);
    GpuTensor dWqg(D, D, Dtype::FP16); dWqg.zero();
    GpuTensor dWkg(D, D, Dtype::FP16); dWkg.zero();
    GpuTensor dWvg(D, D, Dtype::FP16); dWvg.zero();
    GpuTensor dWog(D, D, Dtype::FP16); dWog.zero();

    brotensor::flash_attention_qkvo_backward_gpu(
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
    brotensor::download_fp16(dXg,  dX_got.data());
    brotensor::download_fp16(dWqg, dWq_got.data());
    brotensor::download_fp16(dWkg, dWk_got.data());
    brotensor::download_fp16(dWvg, dWv_got.data());
    brotensor::download_fp16(dWog, dWo_got.data());
    brotensor::cuda_sync();

    check_fp16(dX_got,  dX_ref_h,  "self-bwd dX");
    check_fp16(dWq_got, dWq_ref_h, "self-bwd dWq");
    check_fp16(dWk_got, dWk_ref_h, "self-bwd dWk");
    check_fp16(dWv_got, dWv_ref_h, "self-bwd dWv");
    check_fp16(dWo_got, dWo_ref_h, "self-bwd dWo");
}

// Cross-attention backward parity vs cross_attention_backward_gpu (FP32),
// with rectangular Wk/Wv (D_ctx != D).
static void run_bwd_cross_vs_cx(const char* label, int Lq, int Lk, int D,
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

    GpuTensor X_f32, Ctx_f32, Wq_f32, Wk_f32, Wv_f32, Wo_f32, dO_f32;
    brotensor::upload(Xq.data(),   Lq, D,     X_f32);
    brotensor::upload(Ctxq.data(), Lk, D_ctx, Ctx_f32);
    brotensor::upload(Wqq.data(),  D, D,      Wq_f32);
    brotensor::upload(Wkq.data(),  D, D_ctx,  Wk_f32);
    brotensor::upload(Wvq.data(),  D, D_ctx,  Wv_f32);
    brotensor::upload(Woq.data(),  D, D,      Wo_f32);
    brotensor::upload(dOq.data(),  Lq, D,     dO_f32);

    GpuTensor Qh, Kh, Vh, Attnh, Yconcat, O_ref;
    brotensor::cross_attention_forward_train_gpu(
        X_f32, Ctx_f32, Wq_f32, Wk_f32, Wv_f32, Wo_f32,
        nullptr, nh, Qh, Kh, Vh, Attnh, Yconcat, O_ref);

    GpuTensor dX_ref(Lq, D), dCtx_ref(Lk, D_ctx);
    GpuTensor dWq_ref(D, D), dWk_ref(D, D_ctx),
              dWv_ref(D, D_ctx), dWo_ref(D, D);
    dWq_ref.zero(); dWk_ref.zero(); dWv_ref.zero(); dWo_ref.zero();
    brotensor::cross_attention_backward_gpu(
        dO_f32, X_f32, Ctx_f32, Qh, Kh, Vh, Attnh, Yconcat,
        Wq_f32, Wk_f32, Wv_f32, Wo_f32, nullptr, nh,
        dX_ref, dCtx_ref, dWq_ref, dWk_ref, dWv_ref, dWo_ref);

    std::vector<float> dX_ref_h(Lq*D), dCtx_ref_h(Lk*D_ctx);
    std::vector<float> dWq_ref_h(D*D), dWk_ref_h(D*D_ctx),
                       dWv_ref_h(D*D_ctx), dWo_ref_h(D*D);
    brotensor::download(dX_ref,   dX_ref_h.data());
    brotensor::download(dCtx_ref, dCtx_ref_h.data());
    brotensor::download(dWq_ref,  dWq_ref_h.data());
    brotensor::download(dWk_ref,  dWk_ref_h.data());
    brotensor::download(dWv_ref,  dWv_ref_h.data());
    brotensor::download(dWo_ref,  dWo_ref_h.data());
    brotensor::cuda_sync();

    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog, dOg;
    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto dOh = to_fp16(dO);
    brotensor::upload_fp16(Xh.data(),  Lq, D,     Xg);
    brotensor::upload_fp16(Ch.data(),  Lk, D_ctx, Cg);
    brotensor::upload_fp16(Wqh.data(), D, D,      Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D_ctx,  Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D_ctx,  Wvg);
    brotensor::upload_fp16(Woh.data(), D, D,      Wog);
    brotensor::upload_fp16(dOh.data(), Lq, D,     dOg);

    GpuTensor dXg(Lq, D, Dtype::FP16);
    GpuTensor dCtxg(Lk, D_ctx, Dtype::FP16);
    GpuTensor dWqg(D, D, Dtype::FP16);         dWqg.zero();
    GpuTensor dWkg(D, D_ctx, Dtype::FP16);     dWkg.zero();
    GpuTensor dWvg(D, D_ctx, Dtype::FP16);     dWvg.zero();
    GpuTensor dWog(D, D, Dtype::FP16);         dWog.zero();

    brotensor::flash_attention_qkvo_backward_gpu(
        Xg, &Cg,
        Wqg, nullptr, Wkg, nullptr, Wvg, nullptr, Wog, nullptr,
        nullptr, nh, /*causal=*/false,
        dOg,
        dXg, &dCtxg,
        dWqg, nullptr, dWkg, nullptr, dWvg, nullptr, dWog, nullptr);

    std::vector<uint16_t> dX_got(Lq*D), dCtx_got(Lk*D_ctx);
    std::vector<uint16_t> dWq_got(D*D), dWk_got(D*D_ctx),
                          dWv_got(D*D_ctx), dWo_got(D*D);
    brotensor::download_fp16(dXg,   dX_got.data());
    brotensor::download_fp16(dCtxg, dCtx_got.data());
    brotensor::download_fp16(dWqg,  dWq_got.data());
    brotensor::download_fp16(dWkg,  dWk_got.data());
    brotensor::download_fp16(dWvg,  dWv_got.data());
    brotensor::download_fp16(dWog,  dWo_got.data());
    brotensor::cuda_sync();

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
static void run_bwd_cross_with_biases_cpu(const char* label, int Lq, int Lk,
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
    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog, bqg, bkg, bvg, bog, dOg;
    auto Xh = to_fp16(X), Ch = to_fp16(Ctx);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    auto dOh = to_fp16(dO);
    brotensor::upload_fp16(Xh.data(),  Lq, D,     Xg);
    brotensor::upload_fp16(Ch.data(),  Lk, D_ctx, Cg);
    brotensor::upload_fp16(Wqh.data(), D, D,      Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D_ctx,  Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D_ctx,  Wvg);
    brotensor::upload_fp16(Woh.data(), D, D,      Wog);
    brotensor::upload_fp16(bqh.data(), D, 1,      bqg);
    brotensor::upload_fp16(bkh.data(), D, 1,      bkg);
    brotensor::upload_fp16(bvh.data(), D, 1,      bvg);
    brotensor::upload_fp16(boh.data(), D, 1,      bog);
    brotensor::upload_fp16(dOh.data(), Lq, D,     dOg);

    GpuTensor dXg(Lq, D, Dtype::FP16);
    GpuTensor dCtxg(Lk, D_ctx, Dtype::FP16);
    GpuTensor dWqg(D, D, Dtype::FP16);         dWqg.zero();
    GpuTensor dWkg(D, D_ctx, Dtype::FP16);     dWkg.zero();
    GpuTensor dWvg(D, D_ctx, Dtype::FP16);     dWvg.zero();
    GpuTensor dWog(D, D, Dtype::FP16);         dWog.zero();
    GpuTensor dbqg(D, 1, Dtype::FP16);         dbqg.zero();
    GpuTensor dbkg(D, 1, Dtype::FP16);         dbkg.zero();
    GpuTensor dbvg(D, 1, Dtype::FP16);         dbvg.zero();
    GpuTensor dbog(D, 1, Dtype::FP16);         dbog.zero();

    brotensor::flash_attention_qkvo_backward_gpu(
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
    brotensor::download_fp16(dXg,   dX_got.data());
    brotensor::download_fp16(dCtxg, dCtx_got.data());
    brotensor::download_fp16(dWqg,  dWq_got.data());
    brotensor::download_fp16(dWkg,  dWk_got.data());
    brotensor::download_fp16(dWvg,  dWv_got.data());
    brotensor::download_fp16(dWog,  dWo_got.data());
    brotensor::download_fp16(dbqg,  dbq_got.data());
    brotensor::download_fp16(dbkg,  dbk_got.data());
    brotensor::download_fp16(dbvg,  dbv_got.data());
    brotensor::download_fp16(dbog,  dbo_got.data());
    brotensor::cuda_sync();

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
static void run_bwd_causal_cpu(const char* label, int L, int D, int nh) {
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
    GpuTensor Xg, Wqg, Wkg, Wvg, Wog, bqg, bkg, bvg, bog, dOg;
    auto Xh = to_fp16(X);
    auto Wqh = to_fp16(Wq), Wkh = to_fp16(Wk), Wvh = to_fp16(Wv), Woh = to_fp16(Wo);
    auto bqh = to_fp16(bq), bkh = to_fp16(bk), bvh = to_fp16(bv), boh = to_fp16(bo);
    auto dOh = to_fp16(dO);
    brotensor::upload_fp16(Xh.data(),  L, D, Xg);
    brotensor::upload_fp16(Wqh.data(), D, D, Wqg);
    brotensor::upload_fp16(Wkh.data(), D, D, Wkg);
    brotensor::upload_fp16(Wvh.data(), D, D, Wvg);
    brotensor::upload_fp16(Woh.data(), D, D, Wog);
    brotensor::upload_fp16(bqh.data(), D, 1, bqg);
    brotensor::upload_fp16(bkh.data(), D, 1, bkg);
    brotensor::upload_fp16(bvh.data(), D, 1, bvg);
    brotensor::upload_fp16(boh.data(), D, 1, bog);
    brotensor::upload_fp16(dOh.data(), L, D, dOg);

    GpuTensor dXg(L, D, Dtype::FP16);
    GpuTensor dWqg(D, D, Dtype::FP16); dWqg.zero();
    GpuTensor dWkg(D, D, Dtype::FP16); dWkg.zero();
    GpuTensor dWvg(D, D, Dtype::FP16); dWvg.zero();
    GpuTensor dWog(D, D, Dtype::FP16); dWog.zero();
    GpuTensor dbqg(D, 1, Dtype::FP16); dbqg.zero();
    GpuTensor dbkg(D, 1, Dtype::FP16); dbkg.zero();
    GpuTensor dbvg(D, 1, Dtype::FP16); dbvg.zero();
    GpuTensor dbog(D, 1, Dtype::FP16); dbog.zero();

    brotensor::flash_attention_qkvo_backward_gpu(
        Xg, nullptr,
        Wqg, &bqg, Wkg, &bkg, Wvg, &bvg, Wog, &bog,
        nullptr, nh, /*causal=*/true,
        dOg,
        dXg, nullptr,
        dWqg, &dbqg, dWkg, &dbkg, dWvg, &dbvg, dWog, &dbog);

    std::vector<uint16_t> dX_got(L*D), dWq_got(D*D), dWk_got(D*D),
                          dWv_got(D*D), dWo_got(D*D);
    std::vector<uint16_t> dbq_got(D), dbk_got(D), dbv_got(D), dbo_got(D);
    brotensor::download_fp16(dXg,  dX_got.data());
    brotensor::download_fp16(dWqg, dWq_got.data());
    brotensor::download_fp16(dWkg, dWk_got.data());
    brotensor::download_fp16(dWvg, dWv_got.data());
    brotensor::download_fp16(dWog, dWo_got.data());
    brotensor::download_fp16(dbqg, dbq_got.data());
    brotensor::download_fp16(dbkg, dbk_got.data());
    brotensor::download_fp16(dbvg, dbv_got.data());
    brotensor::download_fp16(dbog, dbo_got.data());
    brotensor::cuda_sync();

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
