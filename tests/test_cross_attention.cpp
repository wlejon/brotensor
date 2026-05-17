// CPU↔GPU parity for cross_attention_forward_gpu (FP16).

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

// Reference: C(M, N) = A(M, K) @ B(N, K)^T.
static void matmul_ABT(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C, int M, int N, int K) {
    C.assign(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += static_cast<double>(A[m * K + k]) * B[n * K + k];
            C[m * N + n] = static_cast<float>(s);
        }
}

static void cross_attn_cpu(const std::vector<float>& X,
                           const std::vector<float>& Ctx,
                           const std::vector<float>& Wq,
                           const std::vector<float>& Wk,
                           const std::vector<float>& Wv,
                           const std::vector<float>& Wo,
                           const std::vector<float>* mask,
                           int Lq, int Lk, int D, int nh,
                           std::vector<float>& O) {
    const int hd = D / nh;
    std::vector<float> Q, K, V, Op;
    matmul_ABT(X, Wq, Q, Lq, D, D);
    matmul_ABT(Ctx, Wk, K, Lk, D, D);
    matmul_ABT(Ctx, Wv, V, Lk, D, D);

    Op.assign(static_cast<size_t>(Lq) * D, 0.0f);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> scores(Lk);
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < nh; ++h) {
            const int off = h * hd;
            float max_v = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(Q[q * D + off + d]) * K[k * D + off + d];
                float s = static_cast<float>(dot) * inv_sqrt;
                if (mask && (*mask)[k] <= 0.5f) s = -1e30f;
                scores[k] = s;
                if (s > max_v) max_v = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                scores[k] = std::exp(scores[k] - max_v);
                sum += scores[k];
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                double a = 0.0;
                for (int k = 0; k < Lk; ++k)
                    a += static_cast<double>(scores[k]) * inv * V[k * D + off + d];
                Op[q * D + off + d] = static_cast<float>(a);
            }
        }
    }
    matmul_ABT(Op, Wo, O, Lq, D, D);
}

static void run_one(const char* label, int Lq, int Lk, int D, int nh,
                    bool use_mask) {
    std::printf("  %s  Lq=%d Lk=%d D=%d nh=%d mask=%d\n",
                label, Lq, Lk, D, nh, (int)use_mask);
    std::mt19937 rng(0xABCD);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    std::vector<float> X(Lq * D), Ctx(Lk * D);
    std::vector<float> Wq(D * D), Wk(D * D), Wv(D * D), Wo(D * D);
    for (auto& v : X)   v = dist(rng);
    for (auto& v : Ctx) v = dist(rng);
    for (auto& v : Wq)  v = dist(rng);
    for (auto& v : Wk)  v = dist(rng);
    for (auto& v : Wv)  v = dist(rng);
    for (auto& v : Wo)  v = dist(rng);

    auto X_q   = rq(X);
    auto Ctx_q = rq(Ctx);
    auto Wq_q  = rq(Wq);
    auto Wk_q  = rq(Wk);
    auto Wv_q  = rq(Wv);
    auto Wo_q  = rq(Wo);

    std::vector<float> mask_host;
    const std::vector<float>* mask_ptr = nullptr;
    if (use_mask) {
        mask_host.assign(Lk, 1.0f);
        // Invalidate the last quarter of keys.
        for (int k = 3 * Lk / 4; k < Lk; ++k) mask_host[k] = 0.0f;
        mask_ptr = &mask_host;
    }

    std::vector<float> O_ref;
    cross_attn_cpu(X_q, Ctx_q, Wq_q, Wk_q, Wv_q, Wo_q,
                   mask_ptr, Lq, Lk, D, nh, O_ref);

    auto X_h   = to_fp16(X);
    auto Ctx_h = to_fp16(Ctx);
    auto Wq_h  = to_fp16(Wq);
    auto Wk_h  = to_fp16(Wk);
    auto Wv_h  = to_fp16(Wv);
    auto Wo_h  = to_fp16(Wo);

    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog, Og;
    brotensor::upload_fp16(X_h.data(),   Lq, D, Xg);
    brotensor::upload_fp16(Ctx_h.data(), Lk, D, Cg);
    brotensor::upload_fp16(Wq_h.data(),  D,  D, Wqg);
    brotensor::upload_fp16(Wk_h.data(),  D,  D, Wkg);
    brotensor::upload_fp16(Wv_h.data(),  D,  D, Wvg);
    brotensor::upload_fp16(Wo_h.data(),  D,  D, Wog);

    // Device mask buffer if needed.
    GpuTensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        brotensor::upload(mask_host.data(), Lk, 1, mg);
        d_mask = mg.data;
    }

    brotensor::cross_attention_forward_gpu(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                           d_mask, nh, Og);
    CHECK(Og.rows == Lq && Og.cols == D && Og.dtype == Dtype::FP16);

    std::vector<uint16_t> O_h(static_cast<size_t>(Og.size()), 0);
    brotensor::download_fp16(Og, O_h.data());
    brotensor::cuda_sync();

    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < O_ref.size(); ++i) {
        const float got = brotensor::fp16_bits_to_fp32(O_h[i]);
        const float ref = O_ref[i];
        const float e   = std::fabs(got - ref);
        if (e > max_err) max_err = e;
        // Cross-attn output is a chain of three matmuls + softmax; FP16
        // accumulation error compounds. Loosen rtol slightly vs other ops.
        if (e > 2e-2f + 3e-2f * std::fabs(ref)) {
            if (bad < 5)
                std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, got, ref, e);
            ++bad;
        }
    }
    std::printf("    max_err=%g bad=%d / %zu\n", max_err, bad, O_ref.size());
    CHECK(bad == 0);
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_cross_attention\n");

    run_one("tiny single-head", 4, 5, 8,  1, false);
    run_one("multi-head",       6, 7, 16, 4, false);
    run_one("with mask",        4, 8, 16, 2, true);
    run_one("Lq=Lk",            8, 8, 32, 4, false);
    run_one("SD-realistic D",   16, 32, 640, 10, false);
    // Regression coverage. A prior hand-rolled core kernel failed silently at
    // (Lq * num_heads) ≳ 400 with head_dim ≥ 64 on this architecture; the
    // public op now delegates to the flash-attention path, which is correct
    // at all SD-scale shapes including the largest U-Net cross-attn site.
    run_one("SD bench shape",   64, 77, 1280, 20, false);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll cross-attention checks passed.\n");
    return 0;
}
