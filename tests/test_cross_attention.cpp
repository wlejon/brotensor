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

// ─── FP32 training-path parity tests ──────────────────────────────────────

// CPU reference for cross-attention FP32 with rectangular Wk/Wv (D, D_ctx)
// and arbitrary Lq, Lk, D, D_ctx. Returns O and the per-head caches used to
// validate backward.
static void cross_attn_fp32_cpu(const std::vector<float>& X,    // (Lq, D)
                                const std::vector<float>& Ctx,  // (Lk, Dctx)
                                const std::vector<float>& Wq,   // (D, D)
                                const std::vector<float>& Wk,   // (D, Dctx)
                                const std::vector<float>& Wv,   // (D, Dctx)
                                const std::vector<float>& Wo,   // (D, D)
                                const std::vector<float>* mask,
                                int Lq, int Lk, int D, int Dctx, int nh,
                                std::vector<float>& O) {
    const int dh = D / nh;
    std::vector<float> Q(Lq * D, 0.0f);
    std::vector<float> K(Lk * D, 0.0f);
    std::vector<float> V(Lk * D, 0.0f);
    // Q = X @ Wq^T
    for (int i = 0; i < Lq; ++i)
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += static_cast<double>(X[i * D + k]) * Wq[j * D + k];
            Q[i * D + j] = static_cast<float>(s);
        }
    // K = Ctx @ Wk^T, V = Ctx @ Wv^T  (Wk, Wv are (D, Dctx))
    for (int i = 0; i < Lk; ++i)
        for (int j = 0; j < D; ++j) {
            double sk = 0.0, sv = 0.0;
            for (int k = 0; k < Dctx; ++k) {
                sk += static_cast<double>(Ctx[i * Dctx + k]) * Wk[j * Dctx + k];
                sv += static_cast<double>(Ctx[i * Dctx + k]) * Wv[j * Dctx + k];
            }
            K[i * D + j] = static_cast<float>(sk);
            V[i * D + j] = static_cast<float>(sv);
        }

    std::vector<float> Yp(Lq * D, 0.0f);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dh));
    std::vector<float> scores(Lk);
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < nh; ++h) {
            const int off = h * dh;
            float max_v = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                double dot = 0.0;
                for (int d = 0; d < dh; ++d)
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
            for (int d = 0; d < dh; ++d) {
                double a = 0.0;
                for (int k = 0; k < Lk; ++k)
                    a += static_cast<double>(scores[k]) * inv * V[k * D + off + d];
                Yp[q * D + off + d] = static_cast<float>(a);
            }
        }
    }
    // O = Yp @ Wo^T
    O.assign(static_cast<size_t>(Lq) * D, 0.0f);
    for (int i = 0; i < Lq; ++i)
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += static_cast<double>(Yp[i * D + k]) * Wo[j * D + k];
            O[i * D + j] = static_cast<float>(s);
        }
}

// CPU backward reference using analytic chain rule on the same forward.
static void cross_attn_fp32_bwd_cpu(const std::vector<float>& dO,  // (Lq, D)
                                    const std::vector<float>& X,
                                    const std::vector<float>& Ctx,
                                    const std::vector<float>& Wq,
                                    const std::vector<float>& Wk,
                                    const std::vector<float>& Wv,
                                    const std::vector<float>& Wo,
                                    const std::vector<float>* mask,
                                    int Lq, int Lk, int D, int Dctx, int nh,
                                    std::vector<float>& dX,
                                    std::vector<float>& dCtx,
                                    std::vector<float>& dWq,
                                    std::vector<float>& dWk,
                                    std::vector<float>& dWv,
                                    std::vector<float>& dWo) {
    const int dh = D / nh;
    // Forward: Q, K, V, attn, Yp.
    std::vector<float> Q(Lq * D, 0.0f), K(Lk * D, 0.0f), V(Lk * D, 0.0f);
    for (int i = 0; i < Lq; ++i)
        for (int j = 0; j < D; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += static_cast<double>(X[i * D + k]) * Wq[j * D + k];
            Q[i * D + j] = static_cast<float>(s);
        }
    for (int i = 0; i < Lk; ++i)
        for (int j = 0; j < D; ++j) {
            double sk = 0.0, sv = 0.0;
            for (int k = 0; k < Dctx; ++k) {
                sk += static_cast<double>(Ctx[i * Dctx + k]) * Wk[j * Dctx + k];
                sv += static_cast<double>(Ctx[i * Dctx + k]) * Wv[j * Dctx + k];
            }
            K[i * D + j] = static_cast<float>(sk);
            V[i * D + j] = static_cast<float>(sv);
        }
    // Per (q, h) softmax attn: P(q, h, k).
    std::vector<float> Pmat(static_cast<size_t>(Lq) * nh * Lk, 0.0f);
    std::vector<float> Yp(Lq * D, 0.0f);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dh));
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < nh; ++h) {
            const int off = h * dh;
            float max_v = -1e30f;
            std::vector<float> sc(Lk);
            for (int k = 0; k < Lk; ++k) {
                double dot = 0.0;
                for (int d = 0; d < dh; ++d)
                    dot += static_cast<double>(Q[q * D + off + d]) * K[k * D + off + d];
                float s = static_cast<float>(dot) * inv_sqrt;
                if (mask && (*mask)[k] <= 0.5f) s = -1e30f;
                sc[k] = s;
                if (s > max_v) max_v = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) { sc[k] = std::exp(sc[k] - max_v); sum += sc[k]; }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k < Lk; ++k) {
                sc[k] *= inv;
                Pmat[(static_cast<size_t>(q) * nh + h) * Lk + k] = sc[k];
            }
            for (int d = 0; d < dh; ++d) {
                double a = 0.0;
                for (int k = 0; k < Lk; ++k)
                    a += static_cast<double>(sc[k]) * V[k * D + off + d];
                Yp[q * D + off + d] = static_cast<float>(a);
            }
        }
    }

    // dYp = dO @ Wo (Wo is (D, D), Yp @ Wo^T => dYp = dO @ Wo).
    std::vector<float> dYp(Lq * D, 0.0f);
    for (int i = 0; i < Lq; ++i)
        for (int k = 0; k < D; ++k) {
            double s = 0.0;
            for (int c = 0; c < D; ++c)
                s += static_cast<double>(dO[i * D + c]) * Wo[c * D + k];
            dYp[i * D + k] = static_cast<float>(s);
        }
    // dWo += dO^T @ Yp.
    for (int c = 0; c < D; ++c)
        for (int k = 0; k < D; ++k) {
            double s = 0.0;
            for (int i = 0; i < Lq; ++i)
                s += static_cast<double>(dO[i * D + c]) * Yp[i * D + k];
            dWo[c * D + k] += static_cast<float>(s);
        }

    // dV and dAttn per (q, h, k).
    std::vector<float> dV(Lk * D, 0.0f);
    std::vector<float> dAttn(static_cast<size_t>(Lq) * nh * Lk, 0.0f);
    for (int q = 0; q < Lq; ++q)
        for (int h = 0; h < nh; ++h) {
            const int off = h * dh;
            for (int k = 0; k < Lk; ++k) {
                float p = Pmat[(static_cast<size_t>(q) * nh + h) * Lk + k];
                float da = 0.0f;
                for (int d = 0; d < dh; ++d) {
                    dV[k * D + off + d] += p * dYp[q * D + off + d];
                    da += dYp[q * D + off + d] * V[k * D + off + d];
                }
                dAttn[(static_cast<size_t>(q) * nh + h) * Lk + k] = da;
            }
        }

    // dScores via softmax backward, then dQ, dK.
    std::vector<float> dQ(Lq * D, 0.0f), dK(Lk * D, 0.0f);
    for (int q = 0; q < Lq; ++q)
        for (int h = 0; h < nh; ++h) {
            const int off = h * dh;
            double dot = 0.0;
            for (int k = 0; k < Lk; ++k) {
                float p = Pmat[(static_cast<size_t>(q) * nh + h) * Lk + k];
                float da = dAttn[(static_cast<size_t>(q) * nh + h) * Lk + k];
                dot += static_cast<double>(p) * da;
            }
            for (int k = 0; k < Lk; ++k) {
                float p = Pmat[(static_cast<size_t>(q) * nh + h) * Lk + k];
                float da = dAttn[(static_cast<size_t>(q) * nh + h) * Lk + k];
                float ds = (mask && (*mask)[k] <= 0.5f)
                               ? 0.0f
                               : p * (da - static_cast<float>(dot)) * inv_sqrt;
                for (int d = 0; d < dh; ++d) {
                    dQ[q * D + off + d] += ds * K[k * D + off + d];
                    dK[k * D + off + d] += ds * Q[q * D + off + d];
                }
            }
        }

    // dWq += dQ^T @ X (Wq is (D, D)); dX from dQ @ Wq.
    for (int j = 0; j < D; ++j)
        for (int k = 0; k < D; ++k) {
            double s = 0.0;
            for (int i = 0; i < Lq; ++i)
                s += static_cast<double>(dQ[i * D + j]) * X[i * D + k];
            dWq[j * D + k] += static_cast<float>(s);
        }
    dX.assign(static_cast<size_t>(Lq) * D, 0.0f);
    for (int i = 0; i < Lq; ++i)
        for (int k = 0; k < D; ++k) {
            double s = 0.0;
            for (int j = 0; j < D; ++j)
                s += static_cast<double>(dQ[i * D + j]) * Wq[j * D + k];
            dX[i * D + k] = static_cast<float>(s);
        }
    // dWk, dWv (D, Dctx) and dCtx.
    for (int j = 0; j < D; ++j)
        for (int k = 0; k < Dctx; ++k) {
            double sk = 0.0, sv = 0.0;
            for (int i = 0; i < Lk; ++i) {
                sk += static_cast<double>(dK[i * D + j]) * Ctx[i * Dctx + k];
                sv += static_cast<double>(dV[i * D + j]) * Ctx[i * Dctx + k];
            }
            dWk[j * Dctx + k] += static_cast<float>(sk);
            dWv[j * Dctx + k] += static_cast<float>(sv);
        }
    dCtx.assign(static_cast<size_t>(Lk) * Dctx, 0.0f);
    for (int i = 0; i < Lk; ++i)
        for (int k = 0; k < Dctx; ++k) {
            double s = 0.0;
            for (int j = 0; j < D; ++j) {
                s += static_cast<double>(dK[i * D + j]) * Wk[j * Dctx + k];
                s += static_cast<double>(dV[i * D + j]) * Wv[j * Dctx + k];
            }
            dCtx[i * Dctx + k] = static_cast<float>(s);
        }
}

static void compare_close(const char* label,
                          const std::vector<float>& got,
                          const std::vector<float>& ref,
                          float atol, float rtol) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(got[i] - ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(ref[i])) {
            if (bad < 5) std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                                     label, i, got[i], ref[i], e);
            ++bad;
        }
    }
    std::printf("    %-12s max_err=%g bad=%d / %zu\n",
                label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

static void run_self_attn_train_parity(int L, int D, int nh, bool use_mask) {
    std::printf("  self-attn train parity  L=%d D=%d nh=%d mask=%d\n",
                L, D, nh, (int)use_mask);
    std::mt19937 rng(0x1234);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(L * D), Wq(D * D), Wk(D * D), Wv(D * D), Wo(D * D);
    std::vector<float> dO(L * D);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : Wq) v = dist(rng);
    for (auto& v : Wk) v = dist(rng);
    for (auto& v : Wv) v = dist(rng);
    for (auto& v : Wo) v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    std::vector<float> mask_host;
    const float* d_mask = nullptr;
    GpuTensor mg;
    if (use_mask) {
        mask_host.assign(L, 1.0f);
        for (int k = 3 * L / 4; k < L; ++k) mask_host[k] = 0.0f;
    }

    GpuTensor Xg, Wqg, Wkg, Wvg, Wog;
    brotensor::upload(X.data(), L, D, Xg);
    brotensor::upload(Wq.data(), D, D, Wqg);
    brotensor::upload(Wk.data(), D, D, Wkg);
    brotensor::upload(Wv.data(), D, D, Wvg);
    brotensor::upload(Wo.data(), D, D, Wog);
    if (use_mask) {
        brotensor::upload(mask_host.data(), L, 1, mg);
        d_mask = mg.data;
    }

    // mha reference forward + backward.
    GpuTensor Qh_r, Kh_r, Vh_r, Ah_r, Yc_r, O_r;
    brotensor::mha_forward_gpu(Xg, Wqg, Wkg, Wvg, Wog, d_mask, nh,
                               Qh_r, Kh_r, Vh_r, Ah_r, Yc_r, O_r);
    GpuTensor dOg; brotensor::upload(dO.data(), L, D, dOg);
    GpuTensor dX_r(L, D), dWq_r(D, D), dWk_r(D, D), dWv_r(D, D), dWo_r(D, D);
    dWq_r.zero(); dWk_r.zero(); dWv_r.zero(); dWo_r.zero();
    brotensor::mha_backward_gpu(dOg, Xg, Qh_r, Kh_r, Vh_r, Ah_r, Yc_r,
                                Wqg, Wkg, Wvg, Wog, d_mask, nh,
                                dX_r, dWq_r, dWk_r, dWv_r, dWo_r);

    // self_attention_*_train: same path through wrappers.
    GpuTensor Qh, Kh, Vh, Ah, Yc, O;
    brotensor::self_attention_forward_train_gpu(Xg, Wqg, Wkg, Wvg, Wog,
                                                d_mask, nh,
                                                Qh, Kh, Vh, Ah, Yc, O);
    GpuTensor dX(L, D), dWq(D, D), dWk(D, D), dWv(D, D), dWo(D, D);
    dWq.zero(); dWk.zero(); dWv.zero(); dWo.zero();
    brotensor::self_attention_backward_gpu(dOg, Xg, Qh, Kh, Vh, Ah, Yc,
                                           Wqg, Wkg, Wvg, Wog, d_mask, nh,
                                           dX, dWq, dWk, dWv, dWo);
    brotensor::cuda_sync();

    auto dl = [](const GpuTensor& g) {
        std::vector<float> v(g.size(), 0.0f);
        brotensor::download(g, v.data());
        return v;
    };
    compare_close("O",   dl(O),   dl(O_r),   1e-5f, 1e-5f);
    compare_close("dX",  dl(dX),  dl(dX_r),  1e-5f, 1e-5f);
    compare_close("dWq", dl(dWq), dl(dWq_r), 1e-5f, 1e-5f);
    compare_close("dWk", dl(dWk), dl(dWk_r), 1e-5f, 1e-5f);
    compare_close("dWv", dl(dWv), dl(dWv_r), 1e-5f, 1e-5f);
    compare_close("dWo", dl(dWo), dl(dWo_r), 1e-5f, 1e-5f);
}

static void run_cross_attn_train_parity(int Lq, int Lk, int D, int Dctx,
                                        int nh, bool use_mask) {
    std::printf("  cross-attn train parity  Lq=%d Lk=%d D=%d Dctx=%d nh=%d mask=%d\n",
                Lq, Lk, D, Dctx, nh, (int)use_mask);
    std::mt19937 rng(0x9876);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(Lq * D), Ctx(Lk * Dctx);
    std::vector<float> Wq(D * D), Wk(D * Dctx), Wv(D * Dctx), Wo(D * D);
    std::vector<float> dO(Lq * D);
    for (auto& v : X)   v = dist(rng);
    for (auto& v : Ctx) v = dist(rng);
    for (auto& v : Wq)  v = dist(rng);
    for (auto& v : Wk)  v = dist(rng);
    for (auto& v : Wv)  v = dist(rng);
    for (auto& v : Wo)  v = dist(rng);
    for (auto& v : dO)  v = dist(rng);

    std::vector<float> mask_host;
    const std::vector<float>* mask_ptr = nullptr;
    const float* d_mask = nullptr;
    GpuTensor mg;
    if (use_mask) {
        mask_host.assign(Lk, 1.0f);
        for (int k = 3 * Lk / 4; k < Lk; ++k) mask_host[k] = 0.0f;
        mask_ptr = &mask_host;
    }

    std::vector<float> O_ref;
    cross_attn_fp32_cpu(X, Ctx, Wq, Wk, Wv, Wo, mask_ptr,
                        Lq, Lk, D, Dctx, nh, O_ref);
    std::vector<float> dX_ref, dCtx_ref;
    std::vector<float> dWq_ref(D * D, 0.0f), dWk_ref(D * Dctx, 0.0f),
                       dWv_ref(D * Dctx, 0.0f), dWo_ref(D * D, 0.0f);
    cross_attn_fp32_bwd_cpu(dO, X, Ctx, Wq, Wk, Wv, Wo, mask_ptr,
                            Lq, Lk, D, Dctx, nh,
                            dX_ref, dCtx_ref,
                            dWq_ref, dWk_ref, dWv_ref, dWo_ref);

    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog, dOg;
    brotensor::upload(X.data(), Lq, D, Xg);
    brotensor::upload(Ctx.data(), Lk, Dctx, Cg);
    brotensor::upload(Wq.data(), D, D, Wqg);
    brotensor::upload(Wk.data(), D, Dctx, Wkg);
    brotensor::upload(Wv.data(), D, Dctx, Wvg);
    brotensor::upload(Wo.data(), D, D, Wog);
    brotensor::upload(dO.data(), Lq, D, dOg);
    if (use_mask) {
        brotensor::upload(mask_host.data(), Lk, 1, mg);
        d_mask = mg.data;
    }
    GpuTensor Qh, Kh, Vh, Ah, Yc, O;
    brotensor::cross_attention_forward_train_gpu(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                                 d_mask, nh,
                                                 Qh, Kh, Vh, Ah, Yc, O);
    GpuTensor dX(Lq, D), dCtx(Lk, Dctx);
    GpuTensor dWq(D, D), dWk(D, Dctx), dWv(D, Dctx), dWo(D, D);
    dWq.zero(); dWk.zero(); dWv.zero(); dWo.zero();
    brotensor::cross_attention_backward_gpu(dOg, Xg, Cg, Qh, Kh, Vh, Ah, Yc,
                                            Wqg, Wkg, Wvg, Wog, d_mask, nh,
                                            dX, dCtx, dWq, dWk, dWv, dWo);
    brotensor::cuda_sync();

    auto dl = [](const GpuTensor& g) {
        std::vector<float> v(g.size(), 0.0f);
        brotensor::download(g, v.data());
        return v;
    };
    compare_close("O",    dl(O),    O_ref,    1e-4f, 1e-4f);
    compare_close("dX",   dl(dX),   dX_ref,   1e-4f, 1e-4f);
    compare_close("dCtx", dl(dCtx), dCtx_ref, 1e-4f, 1e-4f);
    compare_close("dWq",  dl(dWq),  dWq_ref,  1e-4f, 1e-4f);
    compare_close("dWk",  dl(dWk),  dWk_ref,  1e-4f, 1e-4f);
    compare_close("dWv",  dl(dWv),  dWv_ref,  1e-4f, 1e-4f);
    compare_close("dWo",  dl(dWo),  dWo_ref,  1e-4f, 1e-4f);
}

static void run_cross_eq_self_degenerate(int L, int D, int nh) {
    std::printf("  cross == self when Ctx=X, Dctx=D  L=%d D=%d nh=%d\n", L, D, nh);
    std::mt19937 rng(0xC0FE);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> X(L * D), Wq(D * D), Wk(D * D), Wv(D * D), Wo(D * D);
    std::vector<float> dO(L * D);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : Wq) v = dist(rng);
    for (auto& v : Wk) v = dist(rng);
    for (auto& v : Wv) v = dist(rng);
    for (auto& v : Wo) v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    GpuTensor Xg, Cg, Wqg, Wkg, Wvg, Wog, dOg;
    brotensor::upload(X.data(), L, D, Xg);
    brotensor::upload(X.data(), L, D, Cg);   // Ctx aliased to X values
    brotensor::upload(Wq.data(), D, D, Wqg);
    brotensor::upload(Wk.data(), D, D, Wkg);
    brotensor::upload(Wv.data(), D, D, Wvg);
    brotensor::upload(Wo.data(), D, D, Wog);
    brotensor::upload(dO.data(), L, D, dOg);

    // self_attention_*_train path.
    GpuTensor Qh_s, Kh_s, Vh_s, Ah_s, Yc_s, O_s;
    brotensor::self_attention_forward_train_gpu(Xg, Wqg, Wkg, Wvg, Wog,
                                                nullptr, nh,
                                                Qh_s, Kh_s, Vh_s, Ah_s, Yc_s, O_s);
    GpuTensor dX_s(L, D);
    GpuTensor dWq_s(D, D), dWk_s(D, D), dWv_s(D, D), dWo_s(D, D);
    dWq_s.zero(); dWk_s.zero(); dWv_s.zero(); dWo_s.zero();
    brotensor::self_attention_backward_gpu(dOg, Xg, Qh_s, Kh_s, Vh_s, Ah_s, Yc_s,
                                           Wqg, Wkg, Wvg, Wog, nullptr, nh,
                                           dX_s, dWq_s, dWk_s, dWv_s, dWo_s);

    // cross_attention_*_train path with Ctx == X-values, Dctx == D.
    GpuTensor Qh_c, Kh_c, Vh_c, Ah_c, Yc_c, O_c;
    brotensor::cross_attention_forward_train_gpu(Xg, Cg, Wqg, Wkg, Wvg, Wog,
                                                 nullptr, nh,
                                                 Qh_c, Kh_c, Vh_c, Ah_c, Yc_c, O_c);
    GpuTensor dX_c(L, D), dCtx_c(L, D);
    GpuTensor dWq_c(D, D), dWk_c(D, D), dWv_c(D, D), dWo_c(D, D);
    dWq_c.zero(); dWk_c.zero(); dWv_c.zero(); dWo_c.zero();
    brotensor::cross_attention_backward_gpu(dOg, Xg, Cg, Qh_c, Kh_c, Vh_c, Ah_c, Yc_c,
                                            Wqg, Wkg, Wvg, Wog, nullptr, nh,
                                            dX_c, dCtx_c,
                                            dWq_c, dWk_c, dWv_c, dWo_c);
    brotensor::cuda_sync();

    auto dl = [](const GpuTensor& g) {
        std::vector<float> v(g.size(), 0.0f);
        brotensor::download(g, v.data());
        return v;
    };
    // Cross dWk/dWv are the contributions from K/V path against Ctx (= X).
    // Self mha's dWk/dWv also accumulate K/V grads against X. Should match.
    // dX_self should equal dX_cross + dCtx_cross (since Ctx grad collapses
    // back into X in the self-attn graph where Ctx aliases X).
    auto vO_s = dl(O_s),   vO_c = dl(O_c);
    compare_close("O",   vO_c, vO_s, 1e-5f, 1e-5f);
    auto vdX_s = dl(dX_s);
    auto vdX_c = dl(dX_c), vdCtx_c = dl(dCtx_c);
    std::vector<float> dX_sum(vdX_c.size());
    for (size_t i = 0; i < vdX_c.size(); ++i) dX_sum[i] = vdX_c[i] + vdCtx_c[i];
    compare_close("dX(self)==dX+dCtx(cross)", dX_sum, vdX_s, 1e-5f, 1e-5f);
    compare_close("dWq", dl(dWq_c), dl(dWq_s), 1e-5f, 1e-5f);
    compare_close("dWk", dl(dWk_c), dl(dWk_s), 1e-5f, 1e-5f);
    compare_close("dWv", dl(dWv_c), dl(dWv_s), 1e-5f, 1e-5f);
    compare_close("dWo", dl(dWo_c), dl(dWo_s), 1e-5f, 1e-5f);
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_cross_attention\n");

    // FP16 inference parity (unchanged).
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

    // FP32 training-path parity tests.
    run_self_attn_train_parity(8,  32, 4, false);
    run_self_attn_train_parity(8,  32, 4, true);
    run_cross_attn_train_parity(8,  8,  32, 32, 4, false);
    run_cross_attn_train_parity(6,  10, 48, 24, 6, false);
    run_cross_attn_train_parity(6,  10, 48, 24, 6, true);
    run_cross_eq_self_degenerate(8, 32, 4);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll cross-attention checks passed.\n");
    return 0;
}
