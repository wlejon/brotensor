// Tests for ModernBERT & Laya inference operations:
// 1. Bidirectional Sliding-Window Flash Attention (flash_attention_windowed_forward with causal = false)
// 2. Bias-Free LayerNorm (layernorm_forward_inference_batched / _fp16 without beta)
// 3. Row-Broadcast In-Place Addition (add_row_bias_inplace for FP32, FP16, BF16)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

#define CHECK_THROWS(expr) do {                                             \
    bool caught = false;                                                    \
    try {                                                                   \
        expr;                                                               \
    } catch (const std::exception&) {                                       \
        caught = true;                                                      \
    }                                                                       \
    if (!caught) {                                                          \
        std::printf("  FAIL  %s:%d  expected exception: %s\n",              \
                    __FILE__, __LINE__, #expr);                             \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool close(float a, float b, float atol = 1e-5f, float rtol = 1e-4f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

static void fill_random_f32(Tensor& t, uint64_t seed, float low = -1.0f, float high = 1.0f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.size(); ++i) {
        p[i] = dist(rng);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Bidirectional Sliding-Window Flash Attention Reference & Tests
// ─────────────────────────────────────────────────────────────────────────────

static void attention_reference(const Tensor& Q,
                                const Tensor& K,
                                const Tensor& V,
                                const float* mask,
                                int num_heads,
                                int window,
                                bool causal,
                                Tensor& O) {
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D = Q.cols;
    const int hd = D / num_heads;
    const int q_offset = Lk - Lq;
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));

    O.resize(Lq, D, Dtype::FP32);
    float* o_ptr = O.host_f32_mut();
    std::fill(o_ptr, o_ptr + Lq * D, 0.0f);

    const float* q_ptr = Q.host_f32();
    const float* k_ptr = K.host_f32();
    const float* v_ptr = V.host_f32();

    std::vector<float> scores(Lk);

    for (int r = 0; r < Lq; ++r) {
        const int aq = r + q_offset;
        int k_lo = 0;
        int k_hi = Lk - 1;

        if (causal) {
            k_lo = (window > 0) ? std::max(0, aq - window + 1) : 0;
            k_hi = aq;
        } else {
            k_lo = (window > 0) ? std::max(0, aq - window / 2) : 0;
            k_hi = (window > 0) ? std::min(Lk - 1, aq + window / 2) : (Lk - 1);
        }

        for (int h = 0; h < num_heads; ++h) {
            const int head_off = h * hd;
            float max_score = -1e30f;

            for (int k = 0; k < Lk; ++k) {
                if (k < k_lo || k > k_hi || (mask && mask[k] <= 0.5f)) {
                    scores[k] = -1e30f;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) {
                    dot += q_ptr[r * D + head_off + d] * k_ptr[k * D + head_off + d];
                }
                scores[k] = dot * inv_sqrt_hd;
                if (scores[k] > max_score) {
                    max_score = scores[k];
                }
            }

            if (max_score <= -1e29f) {
                // All keys masked
                for (int d = 0; d < hd; ++d) {
                    o_ptr[r * D + head_off + d] = 0.0f;
                }
                continue;
            }

            float sum_exp = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                if (scores[k] <= -1e29f) {
                    scores[k] = 0.0f;
                } else {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum_exp += scores[k];
                }
            }

            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < Lk; ++k) {
                    if (scores[k] > 0.0f) {
                        acc += (scores[k] * inv_sum) * v_ptr[k * D + head_off + d];
                    }
                }
                o_ptr[r * D + head_off + d] = acc;
            }
        }
    }
}

static void test_flash_attention_windowed() {
    std::printf("test_flash_attention_windowed...\n");
    const int Lq = 8, Lk = 8, D = 32, num_heads = 2;
    Tensor Q = Tensor::mat(Lq, D);
    Tensor K = Tensor::mat(Lk, D);
    Tensor V = Tensor::mat(Lk, D);
    fill_random_f32(Q, 0x101);
    fill_random_f32(K, 0x102);
    fill_random_f32(V, 0x103);

    // 1. Bidirectional with window = 4 (attends [aq - 2, aq + 2])
    {
        Tensor O_op, O_ref;
        brotensor::flash_attention_windowed_forward(Q, K, V, nullptr, num_heads, 4, O_op, false);
        attention_reference(Q, K, V, nullptr, num_heads, 4, false, O_ref);

        CHECK(O_op.rows == Lq && O_op.cols == D);
        for (int i = 0; i < Lq * D; ++i) {
            CHECK(close(O_op.host_f32()[i], O_ref.host_f32()[i], 1e-4f, 1e-4f));
        }
    }

    // 2. Strict window verification: keys outside [aq - window/2, aq + window/2] must be dropped
    {
        // For aq = 4 and window = 4, attended range is [4 - 2, 4 + 2] = [2, 6].
        // Modifying V at key 0 or 7 must NOT alter query 4 output.
        Tensor V_mod = V;
        V_mod.host_f32_mut()[0 * D + 0] += 50.0f;
        V_mod.host_f32_mut()[7 * D + 0] += 50.0f;

        Tensor O_orig, O_mod;
        brotensor::flash_attention_windowed_forward(Q, K, V, nullptr, num_heads, 4, O_orig, false);
        brotensor::flash_attention_windowed_forward(Q, K, V_mod, nullptr, num_heads, 4, O_mod, false);

        // Query row 4 must be identical
        for (int d = 0; d < D; ++d) {
            CHECK(O_orig.host_f32()[4 * D + d] == O_mod.host_f32()[4 * D + d]);
        }

        // Modifying V inside the window (key 3) MUST alter query row 4
        Tensor V_mod_inside = V;
        V_mod_inside.host_f32_mut()[3 * D + 0] += 50.0f;
        Tensor O_mod_inside;
        brotensor::flash_attention_windowed_forward(Q, K, V_mod_inside, nullptr, num_heads, 4, O_mod_inside, false);
        CHECK(O_orig.host_f32()[4 * D + 0] != O_mod_inside.host_f32()[4 * D + 0]);
    }

    // 3. Bidirectional unbounded (window = 0)
    {
        Tensor O_op, O_ref;
        brotensor::flash_attention_windowed_forward(Q, K, V, nullptr, num_heads, 0, O_op, false);
        attention_reference(Q, K, V, nullptr, num_heads, 0, false, O_ref);

        for (int i = 0; i < Lq * D; ++i) {
            CHECK(close(O_op.host_f32()[i], O_ref.host_f32()[i], 1e-4f, 1e-4f));
        }
    }

    // 4. Default parameter check: causal = true by default (backwards compatibility)
    {
        Tensor O_default, O_causal_explicit;
        brotensor::flash_attention_windowed_forward(Q, K, V, nullptr, num_heads, 4, O_default);
        brotensor::flash_attention_windowed_forward(Q, K, V, nullptr, num_heads, 4, O_causal_explicit, true);

        for (int i = 0; i < Lq * D; ++i) {
            CHECK(O_default.host_f32()[i] == O_causal_explicit.host_f32()[i]);
        }
    }

    // 5. Rectangular decode shape: Lq = 4, Lk = 8 (q_offset = 4)
    {
        Tensor Q_dec = Tensor::mat(4, D);
        fill_random_f32(Q_dec, 0x104);
        Tensor O_op, O_ref;
        brotensor::flash_attention_windowed_forward(Q_dec, K, V, nullptr, num_heads, 4, O_op, false);
        attention_reference(Q_dec, K, V, nullptr, num_heads, 4, false, O_ref);

        for (int i = 0; i < 4 * D; ++i) {
            CHECK(close(O_op.host_f32()[i], O_ref.host_f32()[i], 1e-4f, 1e-4f));
        }
    }

    // 6. With key mask
    {
        std::vector<float> mask(Lk, 1.0f);
        mask[2] = 0.0f;
        mask[5] = 0.0f;
        Tensor O_op, O_ref;
        brotensor::flash_attention_windowed_forward(Q, K, V, mask.data(), num_heads, 4, O_op, false);
        attention_reference(Q, K, V, mask.data(), num_heads, 4, false, O_ref);

        for (int i = 0; i < Lq * D; ++i) {
            CHECK(close(O_op.host_f32()[i], O_ref.host_f32()[i], 1e-4f, 1e-4f));
        }
    }

    // 7. GPU Parity (if GPU available)
    if (brotensor::is_available(Device::CUDA) || brotensor::is_available(Device::Metal)) {
        Device gpu_dev = brotensor::is_available(Device::CUDA) ? Device::CUDA : Device::Metal;
        Tensor gQ = Q.to(gpu_dev);
        Tensor gK = K.to(gpu_dev);
        Tensor gV = V.to(gpu_dev);
        Tensor gO;

        // Bidirectional windowed on GPU
        brotensor::flash_attention_windowed_forward(gQ, gK, gV, nullptr, num_heads, 4, gO, false);
        Tensor cpu_from_gpu = gO.to(Device::CPU);

        Tensor cpu_expected;
        brotensor::flash_attention_windowed_forward(Q, K, V, nullptr, num_heads, 4, cpu_expected, false);

        for (int i = 0; i < Lq * D; ++i) {
            CHECK(close(cpu_from_gpu.host_f32()[i], cpu_expected.host_f32()[i], 1e-3f, 1e-3f));
        }

        // Causal windowed on GPU
        brotensor::flash_attention_windowed_forward(gQ, gK, gV, nullptr, num_heads, 4, gO, true);
        cpu_from_gpu = gO.to(Device::CPU);
        brotensor::flash_attention_windowed_forward(Q, K, V, nullptr, num_heads, 4, cpu_expected, true);

        for (int i = 0; i < Lq * D; ++i) {
            CHECK(close(cpu_from_gpu.host_f32()[i], cpu_expected.host_f32()[i], 1e-3f, 1e-3f));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Bias-Free LayerNorm Tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_layernorm_bias_free() {
    std::printf("test_layernorm_bias_free...\n");
    const int R = 8, D = 64;
    const float eps = 1e-5f;

    // --- FP32 CPU ---
    {
        Tensor X = Tensor::mat(R, D);
        Tensor gamma = Tensor::vec(D);
        Tensor beta_zeros = Tensor::vec(D);
        fill_random_f32(X, 0x201);
        fill_random_f32(gamma, 0x202, 0.5f, 1.5f);
        // beta_zeros is already all zeros

        Tensor Y_nobias, Y_with_zeros;
        // 4-arg bias-free
        brotensor::layernorm_forward_inference_batched(X, gamma, Y_nobias, eps);
        // 5-arg with zeros beta
        brotensor::layernorm_forward_inference_batched(X, gamma, beta_zeros, Y_with_zeros, eps);

        CHECK(Y_nobias.rows == R && Y_nobias.cols == D);
        for (int i = 0; i < R * D; ++i) {
            CHECK(close(Y_nobias.host_f32()[i], Y_with_zeros.host_f32()[i], 1e-6f, 1e-6f));
        }

        // Mathematical definition check: y = gamma * (x - mean) / sqrt(var + eps)
        for (int r = 0; r < R; ++r) {
            double sum = 0.0;
            for (int d = 0; d < D; ++d) {
                sum += X.host_f32()[r * D + d];
            }
            float mean = static_cast<float>(sum / D);

            double sum_sq = 0.0;
            for (int d = 0; d < D; ++d) {
                float diff = X.host_f32()[r * D + d] - mean;
                sum_sq += diff * diff;
            }
            float var = static_cast<float>(sum_sq / D);
            float rstd = 1.0f / std::sqrt(var + eps);

            for (int d = 0; d < D; ++d) {
                float expected = gamma.host_f32()[d] * ((X.host_f32()[r * D + d] - mean) * rstd);
                CHECK(close(Y_nobias.host_f32()[r * D + d], expected, 1e-5f, 1e-5f));
            }
        }
    }

    // --- FP16 CPU ---
    {
        Tensor X_f32 = Tensor::mat(R, D);
        Tensor gamma_f32 = Tensor::vec(D);
        Tensor beta_zeros_f32 = Tensor::vec(D);
        fill_random_f32(X_f32, 0x203);
        fill_random_f32(gamma_f32, 0x204, 0.5f, 1.5f);

        std::vector<uint16_t> x_fp16(R * D), g_fp16(D), b_fp16(D, 0);
        for (int i = 0; i < R * D; ++i) x_fp16[i] = brotensor::fp32_to_fp16_bits(X_f32.host_f32()[i]);
        for (int d = 0; d < D; ++d) g_fp16[d] = brotensor::fp32_to_fp16_bits(gamma_f32.host_f32()[d]);

        Tensor X_16 = Tensor::from_host_fp16_on(Device::CPU, x_fp16.data(), R, D);
        Tensor gamma_16 = Tensor::from_host_fp16_on(Device::CPU, g_fp16.data(), D, 1);
        Tensor beta_zeros_16 = Tensor::from_host_fp16_on(Device::CPU, b_fp16.data(), D, 1);

        Tensor Y_nobias_16, Y_with_zeros_16;
        brotensor::layernorm_forward_inference_batched_fp16(X_16, gamma_16, Y_nobias_16, eps);
        brotensor::layernorm_forward_inference_batched_fp16(X_16, gamma_16, beta_zeros_16, Y_with_zeros_16, eps);

        CHECK(Y_nobias_16.rows == R && Y_nobias_16.cols == D);
        CHECK(Y_nobias_16.dtype == Dtype::FP16);
        for (int i = 0; i < R * D; ++i) {
            float v1 = brotensor::fp16_bits_to_fp32(Y_nobias_16.host_fp16()[i]);
            float v2 = brotensor::fp16_bits_to_fp32(Y_with_zeros_16.host_fp16()[i]);
            CHECK(close(v1, v2, 1e-3f, 1e-3f));
        }
    }

    // --- GPU Parity (if GPU available) ---
    if (brotensor::is_available(Device::CUDA) || brotensor::is_available(Device::Metal)) {
        Device gpu_dev = brotensor::is_available(Device::CUDA) ? Device::CUDA : Device::Metal;

        // FP32 GPU
        {
            Tensor X = Tensor::mat(R, D);
            Tensor gamma = Tensor::vec(D);
            Tensor beta_zeros = Tensor::vec(D);
            fill_random_f32(X, 0x205);
            fill_random_f32(gamma, 0x206, 0.5f, 1.5f);

            Tensor gX = X.to(gpu_dev);
            Tensor ggamma = gamma.to(gpu_dev);
            Tensor gbeta_zeros = beta_zeros.to(gpu_dev);

            Tensor gY_nobias, gY_with_zeros;
            brotensor::layernorm_forward_inference_batched(gX, ggamma, gY_nobias, eps);
            brotensor::layernorm_forward_inference_batched(gX, ggamma, gbeta_zeros, gY_with_zeros, eps);

            Tensor Y_gpu_nobias = gY_nobias.to(Device::CPU);
            Tensor Y_gpu_zeros = gY_with_zeros.to(Device::CPU);

            Tensor Y_cpu_nobias;
            brotensor::layernorm_forward_inference_batched(X, gamma, Y_cpu_nobias, eps);

            for (int i = 0; i < R * D; ++i) {
                CHECK(close(Y_gpu_nobias.host_f32()[i], Y_gpu_zeros.host_f32()[i], 1e-5f, 1e-5f));
                CHECK(close(Y_gpu_nobias.host_f32()[i], Y_cpu_nobias.host_f32()[i], 1e-4f, 1e-4f));
            }
        }

        // FP16 GPU
        {
            Tensor X_f32 = Tensor::mat(R, D);
            Tensor gamma_f32 = Tensor::vec(D);
            fill_random_f32(X_f32, 0x207);
            fill_random_f32(gamma_f32, 0x208, 0.5f, 1.5f);

            std::vector<uint16_t> x_fp16(R * D), g_fp16(D), b_fp16(D, 0);
            for (int i = 0; i < R * D; ++i) x_fp16[i] = brotensor::fp32_to_fp16_bits(X_f32.host_f32()[i]);
            for (int d = 0; d < D; ++d) g_fp16[d] = brotensor::fp32_to_fp16_bits(gamma_f32.host_f32()[d]);

            Tensor X_16 = Tensor::from_host_fp16_on(Device::CPU, x_fp16.data(), R, D);
            Tensor gamma_16 = Tensor::from_host_fp16_on(Device::CPU, g_fp16.data(), D, 1);
            Tensor beta_zeros_16 = Tensor::from_host_fp16_on(Device::CPU, b_fp16.data(), D, 1);

            Tensor gX_16 = X_16.to(gpu_dev);
            Tensor ggamma_16 = gamma_16.to(gpu_dev);
            Tensor gbeta_zeros_16 = beta_zeros_16.to(gpu_dev);

            Tensor gY_nobias_16, gY_with_zeros_16;
            brotensor::layernorm_forward_inference_batched_fp16(gX_16, ggamma_16, gY_nobias_16, eps);
            brotensor::layernorm_forward_inference_batched_fp16(gX_16, ggamma_16, gbeta_zeros_16, gY_with_zeros_16, eps);

            Tensor Y_gpu_nobias_16 = gY_nobias_16.to(Device::CPU);
            Tensor Y_gpu_zeros_16 = gY_with_zeros_16.to(Device::CPU);

            for (int i = 0; i < R * D; ++i) {
                float v1 = brotensor::fp16_bits_to_fp32(Y_gpu_nobias_16.host_fp16()[i]);
                float v2 = brotensor::fp16_bits_to_fp32(Y_gpu_zeros_16.host_fp16()[i]);
                CHECK(close(v1, v2, 1e-3f, 1e-3f));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Row-Broadcast In-Place Addition Tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_add_row_bias_inplace() {
    std::printf("test_add_row_bias_inplace...\n");

    // --- FP32 CPU ---
    {
        const int R = 4, D = 32;
        Tensor Y = Tensor::mat(R, D);
        Tensor bias = Tensor::vec(D);
        fill_random_f32(Y, 0x301);
        fill_random_f32(bias, 0x302);

        std::vector<float> Y_before(Y.host_f32(), Y.host_f32() + R * D);
        std::vector<float> b(bias.host_f32(), bias.host_f32() + D);

        brotensor::add_row_bias_inplace(Y, bias);

        for (int r = 0; r < R; ++r) {
            for (int d = 0; d < D; ++d) {
                float expected = Y_before[r * D + d] + b[d];
                CHECK(close(Y.host_f32()[r * D + d], expected, 1e-6f, 1e-6f));
            }
        }

        // Test non-power-of-2 / odd shape
        const int R_odd = 7, D_odd = 65;
        Tensor Y_odd = Tensor::mat(R_odd, D_odd);
        Tensor bias_odd = Tensor::vec(D_odd);
        fill_random_f32(Y_odd, 0x303);
        fill_random_f32(bias_odd, 0x304);

        std::vector<float> Y_odd_before(Y_odd.host_f32(), Y_odd.host_f32() + R_odd * D_odd);
        brotensor::add_row_bias_inplace(Y_odd, bias_odd);

        for (int r = 0; r < R_odd; ++r) {
            for (int d = 0; d < D_odd; ++d) {
                float expected = Y_odd_before[r * D_odd + d] + bias_odd.host_f32()[d];
                CHECK(close(Y_odd.host_f32()[r * D_odd + d], expected, 1e-6f, 1e-6f));
            }
        }

        // Test zero bias is no-op
        Tensor Y_zero = Y;
        Tensor bias_zero = Tensor::zeros_on(Device::CPU, D, 1, Dtype::FP32);
        brotensor::add_row_bias_inplace(Y_zero, bias_zero);
        for (int i = 0; i < R * D; ++i) {
            CHECK(Y_zero.host_f32()[i] == Y.host_f32()[i]);
        }

        // Test error checking: bias size mismatch
        Tensor bias_bad = Tensor::vec(D + 1);
        CHECK_THROWS(brotensor::add_row_bias_inplace(Y, bias_bad));
    }

    // --- FP16 CPU ---
    {
        const int R = 4, D = 32;
        Tensor Y_f32 = Tensor::mat(R, D);
        Tensor bias_f32 = Tensor::vec(D);
        fill_random_f32(Y_f32, 0x305);
        fill_random_f32(bias_f32, 0x306);

        std::vector<uint16_t> y_16(R * D), b_16(D);
        for (int i = 0; i < R * D; ++i) y_16[i] = brotensor::fp32_to_fp16_bits(Y_f32.host_f32()[i]);
        for (int d = 0; d < D; ++d) b_16[d] = brotensor::fp32_to_fp16_bits(bias_f32.host_f32()[d]);

        Tensor Y = Tensor::from_host_fp16_on(Device::CPU, y_16.data(), R, D);
        Tensor bias = Tensor::from_host_fp16_on(Device::CPU, b_16.data(), D, 1);

        brotensor::add_row_bias_inplace(Y, bias);

        for (int r = 0; r < R; ++r) {
            for (int d = 0; d < D; ++d) {
                float y_val = brotensor::fp16_bits_to_fp32(Y.host_fp16()[r * D + d]);
                float orig = brotensor::fp16_bits_to_fp32(y_16[r * D + d]);
                float b_val = brotensor::fp16_bits_to_fp32(b_16[d]);
                CHECK(close(y_val, orig + b_val, 1e-3f, 1e-3f));
            }
        }
    }

    // --- BF16 CPU ---
    {
        const int R = 4, D = 32;
        Tensor Y_f32 = Tensor::mat(R, D);
        Tensor bias_f32 = Tensor::vec(D);
        fill_random_f32(Y_f32, 0x307);
        fill_random_f32(bias_f32, 0x308);

        std::vector<uint16_t> y_bf16(R * D), b_bf16(D);
        for (int i = 0; i < R * D; ++i) y_bf16[i] = brotensor::fp32_to_bf16_bits(Y_f32.host_f32()[i]);
        for (int d = 0; d < D; ++d) b_bf16[d] = brotensor::fp32_to_bf16_bits(bias_f32.host_f32()[d]);

        Tensor Y = Tensor::from_host_bf16_on(Device::CPU, y_bf16.data(), R, D);
        Tensor bias = Tensor::from_host_bf16_on(Device::CPU, b_bf16.data(), D, 1);

        brotensor::add_row_bias_inplace(Y, bias);

        for (int r = 0; r < R; ++r) {
            for (int d = 0; d < D; ++d) {
                float y_val = brotensor::bf16_bits_to_fp32(Y.host_bf16()[r * D + d]);
                float orig = brotensor::bf16_bits_to_fp32(y_bf16[r * D + d]);
                float b_val = brotensor::bf16_bits_to_fp32(b_bf16[d]);
                CHECK(close(y_val, orig + b_val, 2e-2f, 2e-2f));
            }
        }
    }

    // --- GPU Parity (FP32, FP16, BF16) ---
    if (brotensor::is_available(Device::CUDA) || brotensor::is_available(Device::Metal)) {
        Device gpu_dev = brotensor::is_available(Device::CUDA) ? Device::CUDA : Device::Metal;
        const int R = 8, D = 64;

        // FP32 GPU
        {
            Tensor Y_cpu = Tensor::mat(R, D);
            Tensor bias_cpu = Tensor::vec(D);
            fill_random_f32(Y_cpu, 0x309);
            fill_random_f32(bias_cpu, 0x30A);

            Tensor Y_gpu = Y_cpu.to(gpu_dev);
            Tensor bias_gpu = bias_cpu.to(gpu_dev);

            brotensor::add_row_bias_inplace(Y_cpu, bias_cpu);
            brotensor::add_row_bias_inplace(Y_gpu, bias_gpu);

            Tensor Y_from_gpu = Y_gpu.to(Device::CPU);
            for (int i = 0; i < R * D; ++i) {
                CHECK(close(Y_from_gpu.host_f32()[i], Y_cpu.host_f32()[i], 1e-6f, 1e-6f));
            }
        }

        // FP16 GPU
        {
            Tensor Y_f32 = Tensor::mat(R, D);
            Tensor bias_f32 = Tensor::vec(D);
            fill_random_f32(Y_f32, 0x30B);
            fill_random_f32(bias_f32, 0x30C);

            std::vector<uint16_t> y_16(R * D), b_16(D);
            for (int i = 0; i < R * D; ++i) y_16[i] = brotensor::fp32_to_fp16_bits(Y_f32.host_f32()[i]);
            for (int d = 0; d < D; ++d) b_16[d] = brotensor::fp32_to_fp16_bits(bias_f32.host_f32()[d]);

            Tensor Y_cpu = Tensor::from_host_fp16_on(Device::CPU, y_16.data(), R, D);
            Tensor bias_cpu = Tensor::from_host_fp16_on(Device::CPU, b_16.data(), D, 1);

            Tensor Y_gpu = Y_cpu.to(gpu_dev);
            Tensor bias_gpu = bias_cpu.to(gpu_dev);

            brotensor::add_row_bias_inplace(Y_cpu, bias_cpu);
            brotensor::add_row_bias_inplace(Y_gpu, bias_gpu);

            Tensor Y_from_gpu = Y_gpu.to(Device::CPU);
            for (int i = 0; i < R * D; ++i) {
                CHECK(Y_from_gpu.host_fp16()[i] == Y_cpu.host_fp16()[i]);
            }
        }

        // BF16 GPU
        {
            Tensor Y_f32 = Tensor::mat(R, D);
            Tensor bias_f32 = Tensor::vec(D);
            fill_random_f32(Y_f32, 0x30D);
            fill_random_f32(bias_f32, 0x30E);

            std::vector<uint16_t> y_bf16(R * D), b_bf16(D);
            for (int i = 0; i < R * D; ++i) y_bf16[i] = brotensor::fp32_to_bf16_bits(Y_f32.host_f32()[i]);
            for (int d = 0; d < D; ++d) b_bf16[d] = brotensor::fp32_to_bf16_bits(bias_f32.host_f32()[d]);

            Tensor Y_cpu = Tensor::from_host_bf16_on(Device::CPU, y_bf16.data(), R, D);
            Tensor bias_cpu = Tensor::from_host_bf16_on(Device::CPU, b_bf16.data(), D, 1);

            Tensor Y_gpu = Y_cpu.to(gpu_dev);
            Tensor bias_gpu = bias_cpu.to(gpu_dev);

            brotensor::add_row_bias_inplace(Y_cpu, bias_cpu);
            brotensor::add_row_bias_inplace(Y_gpu, bias_gpu);

            Tensor Y_from_gpu = Y_gpu.to(Device::CPU);
            for (int i = 0; i < R * D; ++i) {
                CHECK(Y_from_gpu.host_bf16()[i] == Y_cpu.host_bf16()[i]);
            }
        }
    }
}

int main() {
    std::printf("=== test_modernbert_ops ===\n");
    std::fflush(stdout);
    brotensor::init();
    std::printf("init done\n");
    std::fflush(stdout);

    try {
        test_flash_attention_windowed();
        std::printf("flash attention windowed tests passed\n");
        std::fflush(stdout);

        test_layernorm_bias_free();
        std::printf("layernorm bias-free tests passed\n");
        std::fflush(stdout);

        test_add_row_bias_inplace();
        std::printf("add_row_bias_inplace tests passed\n");
        std::fflush(stdout);
    } catch (const std::exception& e) {
        std::printf("FATAL EXCEPTION: %s\n", e.what());
        std::fflush(stdout);
        return 1;
    } catch (...) {
        std::printf("FATAL UNKNOWN EXCEPTION\n");
        std::fflush(stdout);
        return 1;
    }

    if (g_failures == 0) {
        std::printf("ALL modernbert ops tests PASSED\n");
        return 0;
    } else {
        std::printf("%d modernbert ops test(s) FAILED\n", g_failures);
        return 1;
    }
}
