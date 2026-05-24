// Standalone CPU coverage for the linear-attention text-path ops:
//   * l2_norm_forward / l2_norm_backward          (per-head L2 normalisation)
//   * gated_delta_rule_chunked / gated_delta_rule_step
//   * flash_attention_decode GQA (num_kv_heads < num_q_heads)
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

static bool near_(double a, double b, double abs_eps, double rel_eps) {
    const double d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

#define EXPECT_NEAR(actual, expected, abs_eps, rel_eps, ctx)                    \
    do {                                                                       \
        const double _a = (actual);                                            \
        const double _e = (expected);                                          \
        if (!near_(_a, _e, (abs_eps), (rel_eps))) {                            \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.9g expected=%.9g\n",     \
                        __FILE__, __LINE__, (ctx), _a, _e);                     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",          \
                        __FILE__, __LINE__, (ctx), #cond);                      \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    float next() {  // uniform in [-1, 1)
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 11) /
                                  static_cast<double>(1ULL << 53)) * 2.0f - 1.0f;
    }
};

static Tensor cpu_rand(Rng& rng, int r, int c, float scale = 1.0f) {
    Tensor t = Tensor::zeros_on(Device::CPU, r, c);
    for (int i = 0; i < r * c; ++i) t.host_f32_mut()[i] = scale * rng.next();
    return t;
}

// ─── l2_norm ────────────────────────────────────────────────────────────────
static void test_l2_norm_forward_reference() {
    const char* ctx = "l2_norm_forward reference";
    Rng rng(0xDEADBEEFULL);
    const int L = 5, num_heads = 3, head_dim = 4;
    const float eps = 1e-6f;
    Tensor X = cpu_rand(rng, L, num_heads * head_dim);
    Tensor Y;
    brotensor::l2_norm_forward(X, head_dim, num_heads, eps, Y);
    EXPECT_TRUE(Y.rows == L && Y.cols == num_heads * head_dim, ctx);

    // Compare to hand-rolled reference: each head slice divided by its L2 norm.
    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < num_heads; ++h) {
            const int off = r * (num_heads * head_dim) + h * head_dim;
            double sumsq = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                const float v = X.host_f32()[off + d];
                sumsq += v * v;
            }
            const double inv = 1.0 / std::sqrt(sumsq + eps);
            for (int d = 0; d < head_dim; ++d) {
                EXPECT_NEAR(Y.host_f32()[off + d],
                            X.host_f32()[off + d] * inv,
                            1e-6, 1e-5, ctx);
            }
        }
    }

    // Norm of every output head row should be ~1 (modulo eps).
    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < num_heads; ++h) {
            const int off = r * (num_heads * head_dim) + h * head_dim;
            double sumsq = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                const float v = Y.host_f32()[off + d];
                sumsq += v * v;
            }
            // ||y||^2 = ||x||^2 / (||x||^2 + eps) ≈ 1 for non-trivial x.
            EXPECT_NEAR(std::sqrt(sumsq), 1.0, 1e-4, 1e-4,
                        "l2_norm_forward unit-norm");
        }
    }
}

static void test_l2_norm_backward_fd() {
    const char* ctx = "l2_norm_backward FD";
    Rng rng(0xC0FFEEULL);
    const int L = 3, num_heads = 2, head_dim = 5;
    const float eps = 1e-6f;
    Tensor X  = cpu_rand(rng, L, num_heads * head_dim);
    Tensor dY = cpu_rand(rng, L, num_heads * head_dim);
    Tensor dX;
    brotensor::l2_norm_backward(X, head_dim, num_heads, eps, dY, dX);
    EXPECT_TRUE(dX.rows == L && dX.cols == num_heads * head_dim,
                "l2_norm_backward shape");

    auto loss = [&](const Tensor& Xin) -> double {
        Tensor Yt;
        brotensor::l2_norm_forward(Xin, head_dim, num_heads, eps, Yt);
        double sm = 0.0;
        for (int i = 0; i < Yt.rows * Yt.cols; ++i)
            sm += static_cast<double>(Yt.host_f32()[i]) * dY.host_f32()[i];
        return sm;
    };
    const double h = 1e-3;
    for (int i = 0; i < X.rows * X.cols; ++i) {
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (loss(Xp) - loss(Xm)) / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 3e-3, 3e-3, ctx);
    }
}

// ─── gated_delta_rule ───────────────────────────────────────────────────────

// Hand-rolled per-token reference scan, FP32. Matches the spec exactly.
static Tensor ref_gated_delta_rule(const Tensor& Q, const Tensor& K,
                                   const Tensor& V,
                                   const Tensor& a_raw, const Tensor& beta,
                                   const Tensor& log_A,
                                   int num_heads, int d_k, int d_v,
                                   Tensor& state /* mutated */) {
    const int L  = Q.rows;
    const int Dv = num_heads * d_v;
    Tensor O = Tensor::zeros_on(Device::CPU, L, Dv);
    const float* Qp = Q.host_f32();
    const float* Kp = K.host_f32();
    const float* Vp = V.host_f32();
    const float* Ap = a_raw.host_f32();
    const float* Bp = beta.host_f32();
    const float* logA = log_A.host_f32();
    float* Sp = state.host_f32_mut();
    float* Op = O.host_f32_mut();
    const int Dq = num_heads * d_k;

    auto softplus = [](float x) {
        return std::max(x, 0.0f) + std::log1p(std::exp(-std::abs(x)));
    };
    auto sigmoid = [](float x) {
        return x >= 0.0f
            ? 1.0f / (1.0f + std::exp(-x))
            : std::exp(x) / (1.0f + std::exp(x));
    };

    for (int t = 0; t < L; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const float* qt = Qp + t * Dq + h * d_k;
            const float* kt = Kp + t * Dq + h * d_k;
            const float* vt = Vp + t * Dv + h * d_v;
            const float a_raw_t = Ap[t * num_heads + h];
            const float beta_t  = sigmoid(Bp[t * num_heads + h]);
            const float alpha = std::exp(-softplus(a_raw_t) * std::exp(logA[h]));
            float* S = Sp + h * d_v * d_k;

            // FLA / HF Qwen3.5 ordering: decay first, then read u, then write.
            for (int v = 0; v < d_v; ++v)
                for (int k = 0; k < d_k; ++k)
                    S[v * d_k + k] *= alpha;
            std::vector<float> u(d_v, 0.0f);
            for (int v = 0; v < d_v; ++v)
                for (int k = 0; k < d_k; ++k)
                    u[v] += S[v * d_k + k] * kt[k];
            for (int v = 0; v < d_v; ++v) {
                const float delta = vt[v] - u[v];
                const float scale = beta_t * delta;
                for (int k = 0; k < d_k; ++k)
                    S[v * d_k + k] += scale * kt[k];
            }
            float* orow = Op + t * Dv + h * d_v;
            for (int v = 0; v < d_v; ++v) {
                float o = 0.0f;
                for (int k = 0; k < d_k; ++k) o += S[v * d_k + k] * qt[k];
                orow[v] = o;
            }
        }
    }
    return O;
}

static void test_gated_delta_rule_matches_reference() {
    const char* ctx = "gated_delta_rule_chunked == reference scan";
    Rng rng(0x9A1B2C3DULL);
    const int L = 7, num_heads = 3, d_k = 4, d_v = 5;
    Tensor Q = cpu_rand(rng, L, num_heads * d_k);
    Tensor K = cpu_rand(rng, L, num_heads * d_k);
    Tensor V = cpu_rand(rng, L, num_heads * d_v);
    Tensor a_raw = cpu_rand(rng, L, num_heads, 0.5f);
    Tensor beta  = cpu_rand(rng, L, num_heads, 1.0f);
    Tensor log_A = cpu_rand(rng, num_heads, 1, 0.3f);

    Tensor state_ref = Tensor::zeros_on(Device::CPU, num_heads, d_v * d_k);
    Tensor O_ref = ref_gated_delta_rule(Q, K, V, a_raw, beta, log_A,
                                        num_heads, d_k, d_v, state_ref);

    Tensor state = Tensor::zeros_on(Device::CPU, num_heads, d_v * d_k);
    Tensor O;
    brotensor::gated_delta_rule_chunked(Q, K, V, a_raw, beta, log_A,
                                        num_heads, d_k, d_v, state, O);
    EXPECT_TRUE(O.rows == L && O.cols == num_heads * d_v, "chunked shape");
    for (int i = 0; i < O.rows * O.cols; ++i) {
        EXPECT_NEAR(O.host_f32()[i], O_ref.host_f32()[i], 1e-5, 1e-5, ctx);
    }
    for (int i = 0; i < state.rows * state.cols; ++i) {
        EXPECT_NEAR(state.host_f32()[i], state_ref.host_f32()[i],
                    1e-5, 1e-5, "chunked state == reference state");
    }
}

static void test_gated_delta_rule_step_vs_chunked() {
    // The chunked op (called once with L tokens) and the step op (called
    // L times with one token each) must produce identical outputs and final
    // state. This is the streaming-continuation contract.
    const char* ctx = "step concatenated == chunked";
    Rng rng(0xBEEFCAFEULL);
    const int L = 6, num_heads = 2, d_k = 3, d_v = 4;
    Tensor Q = cpu_rand(rng, L, num_heads * d_k);
    Tensor K = cpu_rand(rng, L, num_heads * d_k);
    Tensor V = cpu_rand(rng, L, num_heads * d_v);
    Tensor a_raw = cpu_rand(rng, L, num_heads, 0.5f);
    Tensor beta  = cpu_rand(rng, L, num_heads, 1.0f);
    Tensor log_A = cpu_rand(rng, num_heads, 1, 0.3f);

    Tensor state_ch = Tensor::zeros_on(Device::CPU, num_heads, d_v * d_k);
    Tensor O_ch;
    brotensor::gated_delta_rule_chunked(Q, K, V, a_raw, beta, log_A,
                                        num_heads, d_k, d_v, state_ch, O_ch);

    // Streaming: one row per step.
    Tensor state_st = Tensor::zeros_on(Device::CPU, num_heads, d_v * d_k);
    Tensor O_streamed = Tensor::zeros_on(Device::CPU, L, num_heads * d_v);
    for (int t = 0; t < L; ++t) {
        Tensor q1 = Tensor::zeros_on(Device::CPU, 1, num_heads * d_k);
        Tensor k1 = Tensor::zeros_on(Device::CPU, 1, num_heads * d_k);
        Tensor v1 = Tensor::zeros_on(Device::CPU, 1, num_heads * d_v);
        Tensor a1 = Tensor::zeros_on(Device::CPU, 1, num_heads);
        Tensor b1 = Tensor::zeros_on(Device::CPU, 1, num_heads);
        for (int j = 0; j < num_heads * d_k; ++j) {
            q1.host_f32_mut()[j] = Q.host_f32()[t * num_heads * d_k + j];
            k1.host_f32_mut()[j] = K.host_f32()[t * num_heads * d_k + j];
        }
        for (int j = 0; j < num_heads * d_v; ++j)
            v1.host_f32_mut()[j] = V.host_f32()[t * num_heads * d_v + j];
        for (int j = 0; j < num_heads; ++j) {
            a1.host_f32_mut()[j] = a_raw.host_f32()[t * num_heads + j];
            b1.host_f32_mut()[j] = beta .host_f32()[t * num_heads + j];
        }
        Tensor o1;
        brotensor::gated_delta_rule_step(q1, k1, v1, a1, b1, log_A,
                                         num_heads, d_k, d_v, state_st, o1);
        for (int j = 0; j < num_heads * d_v; ++j)
            O_streamed.host_f32_mut()[t * num_heads * d_v + j] =
                o1.host_f32()[j];
    }

    for (int i = 0; i < O_ch.rows * O_ch.cols; ++i) {
        EXPECT_NEAR(O_streamed.host_f32()[i], O_ch.host_f32()[i],
                    1e-5, 1e-5, ctx);
    }
    for (int i = 0; i < state_ch.rows * state_ch.cols; ++i) {
        EXPECT_NEAR(state_st.host_f32()[i], state_ch.host_f32()[i],
                    1e-5, 1e-5, "step final state == chunked final state");
    }
}

static void test_gated_delta_rule_zero_beta_is_pure_decay() {
    // sigmoid(beta_raw=-50) ≈ 0 → write strength ~0 → state decays without
    // updating. After L tokens state == alpha^L * S_0. Initial S_0 chosen
    // non-zero so the decay is observable; output o_t = alpha^t * S_0 q_t.
    const char* ctx = "beta=0 -> pure decay";
    Rng rng(0x111111ULL);
    const int L = 4, num_heads = 1, d_k = 3, d_v = 3;
    Tensor Q = cpu_rand(rng, L, num_heads * d_k);
    Tensor K = cpu_rand(rng, L, num_heads * d_k);
    Tensor V = cpu_rand(rng, L, num_heads * d_v);
    Tensor a_raw = Tensor::zeros_on(Device::CPU, L, num_heads);
    Tensor beta  = Tensor::zeros_on(Device::CPU, L, num_heads);
    for (int i = 0; i < L * num_heads; ++i) {
        a_raw.host_f32_mut()[i] = 0.5f;   // some fixed decay
        beta .host_f32_mut()[i] = -50.0f; // sigmoid ≈ 0
    }
    Tensor log_A = Tensor::zeros_on(Device::CPU, num_heads, 1);
    log_A.host_f32_mut()[0] = 0.0f;       // exp(log_A) = 1

    // Initial state: identity-ish, non-zero.
    Tensor state = Tensor::zeros_on(Device::CPU, num_heads, d_v * d_k);
    for (int v = 0; v < d_v; ++v)
        for (int k = 0; k < d_k; ++k)
            state.host_f32_mut()[v * d_k + k] = (v == k) ? 1.0f : 0.0f;

    // alpha per token = exp(-softplus(0.5) * 1).
    const float a_raw_v = 0.5f;
    const float softplus_a = std::max(a_raw_v, 0.0f) +
                             std::log1p(std::exp(-std::abs(a_raw_v)));
    const float alpha = std::exp(-softplus_a);

    Tensor O;
    brotensor::gated_delta_rule_chunked(Q, K, V, a_raw, beta, log_A,
                                        num_heads, d_k, d_v, state, O);

    // Expected output at token t: alpha^(t+1) * I * q_t = alpha^(t+1) * q_t
    // (mapped from R^{d_k} to R^{d_v} by truncation/padding via the identity).
    for (int t = 0; t < L; ++t) {
        const double a_pow = std::pow(static_cast<double>(alpha), t + 1);
        for (int v = 0; v < d_v; ++v) {
            const double expected = (v < d_k)
                ? a_pow * Q.host_f32()[t * d_k + v]
                : 0.0;
            EXPECT_NEAR(O.host_f32()[t * d_v + v], expected,
                        1e-5, 1e-5, ctx);
        }
    }
    // Final state == alpha^L * I.
    const double a_pow_L = std::pow(static_cast<double>(alpha), L);
    for (int v = 0; v < d_v; ++v) {
        for (int k = 0; k < d_k; ++k) {
            const double expected = (v == k) ? a_pow_L : 0.0;
            EXPECT_NEAR(state.host_f32()[v * d_k + k], expected,
                        1e-5, 1e-5, "final state");
        }
    }
}

// ─── flash_attention_decode GQA ─────────────────────────────────────────────

static void test_decode_mha_unchanged() {
    // num_kv_heads == num_q_heads must reproduce the existing MHA decode.
    const char* ctx = "decode GQA: MHA equivalence";
    Rng rng(0xA1B2C3D4ULL);
    const int Lq = 3, num_heads = 2, head_dim = 4, L_max = 6;
    const int D = num_heads * head_dim;
    Tensor Q = cpu_rand(rng, Lq, D);
    Tensor K_cache = cpu_rand(rng, L_max, D);
    Tensor V_cache = cpu_rand(rng, L_max, D);
    const int valid_len = 5;

    Tensor O_old, O_new;
    brotensor::flash_attention_decode(Q, K_cache, V_cache, valid_len,
                                      num_heads, O_old);
    brotensor::flash_attention_decode(Q, K_cache, V_cache, valid_len,
                                      num_heads, /*num_kv_heads=*/num_heads,
                                      O_new);
    EXPECT_TRUE(O_old.rows == Lq && O_old.cols == D, "old overload shape");
    EXPECT_TRUE(O_new.rows == Lq && O_new.cols == D, "new signature shape");
    for (int i = 0; i < Lq * D; ++i)
        EXPECT_NEAR(O_old.host_f32()[i], O_new.host_f32()[i],
                    1e-6, 1e-6, ctx);
}

static void test_decode_gqa_matches_repeated_kv() {
    // GQA: 4 Q heads, 2 KV heads (each KV head serves 2 consecutive Q heads).
    // The GQA decode must match a plain MHA decode against a K/V cache where
    // each KV head row has been duplicated to match the Q head count.
    const char* ctx = "decode GQA == MHA on repeated KV";
    Rng rng(0xFEEDFACEULL);
    const int num_q_heads = 4, num_kv_heads = 2, head_dim = 5;
    const int q_per_kv = num_q_heads / num_kv_heads;  // 2
    const int Lq = 2, L_max = 7;
    const int Dq  = num_q_heads  * head_dim;
    const int Dkv = num_kv_heads * head_dim;
    Tensor Q       = cpu_rand(rng, Lq,    Dq);
    Tensor K_cache = cpu_rand(rng, L_max, Dkv);
    Tensor V_cache = cpu_rand(rng, L_max, Dkv);
    const int valid_len = 6;

    // GQA decode.
    Tensor O_gqa;
    brotensor::flash_attention_decode(Q, K_cache, V_cache, valid_len,
                                      num_q_heads, num_kv_heads, O_gqa);

    // Expanded MHA: duplicate each KV head q_per_kv times to width Dq.
    Tensor K_exp = Tensor::zeros_on(Device::CPU, L_max, Dq);
    Tensor V_exp = Tensor::zeros_on(Device::CPU, L_max, Dq);
    for (int r = 0; r < L_max; ++r) {
        for (int hq = 0; hq < num_q_heads; ++hq) {
            const int hkv = hq / q_per_kv;
            for (int d = 0; d < head_dim; ++d) {
                K_exp.host_f32_mut()[r * Dq + hq * head_dim + d] =
                    K_cache.host_f32()[r * Dkv + hkv * head_dim + d];
                V_exp.host_f32_mut()[r * Dq + hq * head_dim + d] =
                    V_cache.host_f32()[r * Dkv + hkv * head_dim + d];
            }
        }
    }
    Tensor O_mha;
    brotensor::flash_attention_decode(Q, K_exp, V_exp, valid_len,
                                      num_q_heads, num_q_heads, O_mha);

    EXPECT_TRUE(O_gqa.rows == Lq && O_gqa.cols == Dq, "GQA output shape");
    for (int i = 0; i < Lq * Dq; ++i)
        EXPECT_NEAR(O_gqa.host_f32()[i], O_mha.host_f32()[i],
                    1e-6, 1e-6, ctx);
}

int main() {
    brotensor::init();
  try {
    test_l2_norm_forward_reference();
    test_l2_norm_backward_fd();
    test_gated_delta_rule_matches_reference();
    test_gated_delta_rule_step_vs_chunked();
    test_gated_delta_rule_zero_beta_is_pure_decay();
    test_decode_mha_unchanged();
    test_decode_gqa_matches_repeated_kv();
  } catch (const std::exception& e) {
    std::printf("test_gated_delta_rule: uncaught exception: %s\n", e.what());
    return 2;
  }

    if (g_failures == 0) {
        std::printf("test_gated_delta_rule: all checks passed\n");
        return 0;
    }
    std::printf("test_gated_delta_rule: %d FAILURE(S)\n", g_failures);
    return 1;
}
