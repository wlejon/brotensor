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
    brotensor::flash_attention_forward_gpu(Qg, Kg, Vg, d_mask, nh, Og);
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
    brotensor::flash_attention_qkvo_forward_gpu(Xg, &Cg, Wqg, Wkg, Wvg, Wog,
                                                nullptr, nh, O_flash_g);

    std::vector<uint16_t> ref_h(O_ref_g.size()), flash_h(O_flash_g.size());
    brotensor::download_fp16(O_ref_g, ref_h.data());
    brotensor::download_fp16(O_flash_g, flash_h.data());
    brotensor::cuda_sync();
    std::vector<float> ref(ref_h.size());
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = brotensor::fp16_bits_to_fp32(ref_h[i]);
    check_fp16(flash_h, ref, label);
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
    brotensor::flash_attention_forward_gpu(Qg, Kg, Vg, nullptr, nh, Og);
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
    run_stress();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll flash-attention checks passed.\n");
    return 0;
}
