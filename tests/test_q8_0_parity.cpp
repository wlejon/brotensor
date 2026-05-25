// Q8_0 dequant + GEMV + GEMM parity vs a host reference. GPU-only.
// Q8_0 layout: { fp16 d; int8 qs[32]; } = 34 bytes / 32 elems.
// Quantisation: pick d = max(|x|) / 127, qs[i] = round(x[i] / d) clamped to
// [-128, 127]; dequant = d * qs[i].

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#if defined(BROTENSOR_HAS_CUDA)
#include <cuda_runtime.h>
#else
#include <cstring>
static inline void cudaMemcpy(void* dst, const void* src, size_t n, int) {
    std::memcpy(dst, src, n);
}
#define cudaMemcpyHostToDevice 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while(0)

static constexpr int Q8_BLOCK = 32;
static constexpr int Q8_BYTES = 34;

struct Q8Block {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(Q8Block) == 34, "Q8Block must be 34 bytes");

static void quantize_q8_0_block(const float* src, Q8Block& out) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float a = std::fabs(src[i]);
        if (a > amax) amax = a;
    }
    const float d = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    out.d = brotensor::fp32_to_fp16_bits(d);
    const float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
    for (int i = 0; i < 32; ++i) {
        int q = static_cast<int>(std::lround(src[i] * inv_d));
        q = std::clamp(q, -128, 127);
        out.qs[i] = static_cast<int8_t>(q);
    }
}

static void dequant_q8_0_block(const Q8Block& blk, float* dst) {
    const float d = brotensor::fp16_bits_to_fp32(blk.d);
    for (int i = 0; i < 32; ++i) {
        dst[i] = d * static_cast<float>(blk.qs[i]);
    }
}

#if defined(BROTENSOR_HAS_CUDA)
extern "C" unsigned long long brotensor_q8_0_wmma_calls_consume();
#endif

static std::vector<uint16_t> to_fp16_vec(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_q8_0_parity\n");

    constexpr int OUT = 64;
    constexpr int IN  = 256;
    constexpr int BPR = IN / Q8_BLOCK;

    std::mt19937 rng(0xCAFEBABE);
    std::uniform_real_distribution<float> dw(-0.5f, 0.5f);

    std::vector<float>   Wf(static_cast<size_t>(OUT) * IN);
    for (auto& v : Wf) v = dw(rng);
    std::vector<Q8Block> Wq(static_cast<size_t>(OUT) * BPR);
    std::vector<float>   Wd(static_cast<size_t>(OUT) * IN);
    for (int r = 0; r < OUT; ++r) {
        for (int sb = 0; sb < BPR; ++sb) {
            quantize_q8_0_block(&Wf[r * IN + sb * Q8_BLOCK], Wq[r * BPR + sb]);
            dequant_q8_0_block(Wq[r * BPR + sb], &Wd[r * IN + sb * Q8_BLOCK]);
        }
    }
    auto Wd_fp16 = to_fp16_vec(Wd);

    Tensor W_q8_g = Tensor::empty_on(Device::CUDA, OUT, IN, Dtype::Q8_0);
    cudaMemcpy(W_q8_g.data, Wq.data(),
               static_cast<size_t>(OUT) * BPR * Q8_BYTES,
               cudaMemcpyHostToDevice);

    // ─── A: dequant ───────────────────────────────────────────────────────
    {
        Tensor W_fp16_g;
        brotensor::dequant_q8_0_to_fp16(W_q8_g, W_fp16_g);
        CHECK(W_fp16_g.dtype == Dtype::FP16);
        CHECK(W_fp16_g.rows == OUT && W_fp16_g.cols == IN);
        brotensor::sync_all();
        std::vector<uint16_t> got(static_cast<size_t>(OUT) * IN);
        W_fp16_g.copy_to_host_fp16(got.data());

        int mismatched = 0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            const float g = brotensor::fp16_bits_to_fp32(got[i]);
            const float r = brotensor::fp16_bits_to_fp32(Wd_fp16[i]);
            const float e = std::fabs(g - r);
            if (e > max_abs) max_abs = e;
            if (got[i] != Wd_fp16[i]) ++mismatched;
        }
        std::printf("  A: dequant max_abs=%g mismatched_bits=%d/%zu\n",
                    max_abs, mismatched, got.size());
        CHECK(max_abs <= 1e-3f);
    }

    // ─── B: GEMV ──────────────────────────────────────────────────────────
    std::uniform_real_distribution<float> dx(-0.3f, 0.3f);
    std::vector<float> xf(IN);
    for (auto& v : xf) v = dx(rng);
    auto xh = to_fp16_vec(xf);

    std::vector<float> y_ref(OUT, 0.0f);
    for (int r = 0; r < OUT; ++r) {
        float s = 0.0f;
        for (int k = 0; k < IN; ++k) s += Wd[r * IN + k] * xf[k];
        y_ref[r] = s;
    }

    {
        Tensor x_g = Tensor::from_host_fp16_on(Device::CUDA, xh.data(), IN, 1);
        Tensor y_g;
        brotensor::linear_forward_q8_0_fp16(W_q8_g, nullptr, x_g, y_g);
        CHECK(y_g.dtype == Dtype::FP16 && y_g.rows == OUT && y_g.cols == 1);
        brotensor::sync_all();
        std::vector<uint16_t> got(OUT);
        y_g.copy_to_host_fp16(got.data());
        float max_abs = 0.0f;
        for (int r = 0; r < OUT; ++r) {
            const float g = brotensor::fp16_bits_to_fp32(got[r]);
            const float e = std::fabs(g - y_ref[r]);
            if (e > max_abs) max_abs = e;
        }
        std::printf("  B: GEMV max_abs=%g\n", max_abs);
        CHECK(max_abs < 5e-2f);
    }

    // ─── C: GEMV + bias ───────────────────────────────────────────────────
    std::uniform_real_distribution<float> db(-0.1f, 0.1f);
    std::vector<float> bf(OUT);
    for (auto& v : bf) v = db(rng);
    auto bh = to_fp16_vec(bf);
    {
        Tensor x_g    = Tensor::from_host_fp16_on(Device::CUDA, xh.data(), IN, 1);
        Tensor bias_g = Tensor::from_host_fp16_on(Device::CUDA, bh.data(), OUT, 1);
        Tensor y_g;
        brotensor::linear_forward_q8_0_fp16(W_q8_g, &bias_g, x_g, y_g);
        brotensor::sync_all();
        std::vector<uint16_t> got(OUT);
        y_g.copy_to_host_fp16(got.data());
        float max_abs = 0.0f;
        for (int r = 0; r < OUT; ++r) {
            const float g = brotensor::fp16_bits_to_fp32(got[r]);
            const float ref = y_ref[r] + bf[r];
            const float e = std::fabs(g - ref);
            if (e > max_abs) max_abs = e;
        }
        std::printf("  C: GEMV+bias max_abs=%g\n", max_abs);
        CHECK(max_abs < 5e-2f);
    }

    // ─── D: batched (small B exercises GEMV-loop fallback) ────────────────
    constexpr int B = 4;
    std::vector<float> Xf(static_cast<size_t>(B) * IN);
    for (auto& v : Xf) v = dx(rng);
    auto Xh = to_fp16_vec(Xf);
    std::vector<float> Y_ref(static_cast<size_t>(B) * OUT, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int r = 0; r < OUT; ++r) {
            float s = 0.0f;
            for (int k = 0; k < IN; ++k) s += Wd[r * IN + k] * Xf[b * IN + k];
            Y_ref[b * OUT + r] = s;
        }
    }
    {
        Tensor X_g = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), B, IN);
        Tensor Y_g;
        brotensor::linear_forward_batched_q8_0_fp16(W_q8_g, nullptr, X_g, Y_g);
        CHECK(Y_g.dtype == Dtype::FP16 && Y_g.rows == B && Y_g.cols == OUT);
        brotensor::sync_all();
        std::vector<uint16_t> got(static_cast<size_t>(B) * OUT);
        Y_g.copy_to_host_fp16(got.data());
        float max_abs = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            const float g = brotensor::fp16_bits_to_fp32(got[i]);
            const float e = std::fabs(g - Y_ref[i]);
            if (e > max_abs) max_abs = e;
        }
        std::printf("  D: batched max_abs=%g\n", max_abs);
        CHECK(max_abs < 5e-2f);
    }

#if defined(BROTENSOR_HAS_CUDA)
    brotensor::sync_all();
    (void)brotensor_q8_0_wmma_calls_consume();

    auto run_gemm_case = [&](int OUT2, int IN2, int B2, bool expect_wmma,
                             const char* label, float tol) {
        const int BPR2 = IN2 / Q8_BLOCK;
        std::uniform_real_distribution<float> dw2(-0.5f, 0.5f);
        std::uniform_real_distribution<float> dx2(-0.3f, 0.3f);

        std::vector<float>   Wf2(static_cast<size_t>(OUT2) * IN2);
        for (auto& v : Wf2) v = dw2(rng);
        std::vector<Q8Block> Wq2(static_cast<size_t>(OUT2) * BPR2);
        std::vector<float>   Wd2(static_cast<size_t>(OUT2) * IN2);
        for (int r = 0; r < OUT2; ++r) {
            for (int sb = 0; sb < BPR2; ++sb) {
                quantize_q8_0_block(&Wf2[r * IN2 + sb * Q8_BLOCK], Wq2[r * BPR2 + sb]);
                dequant_q8_0_block(Wq2[r * BPR2 + sb], &Wd2[r * IN2 + sb * Q8_BLOCK]);
            }
        }
        Tensor Wg2 = Tensor::empty_on(Device::CUDA, OUT2, IN2, Dtype::Q8_0);
        cudaMemcpy(Wg2.data, Wq2.data(),
                   static_cast<size_t>(OUT2) * BPR2 * Q8_BYTES,
                   cudaMemcpyHostToDevice);

        std::vector<float> Xf2(static_cast<size_t>(B2) * IN2);
        for (auto& v : Xf2) v = dx2(rng);
        auto Xh2 = to_fp16_vec(Xf2);
        std::vector<float> Yref(static_cast<size_t>(B2) * OUT2, 0.0f);
        for (int b = 0; b < B2; ++b) {
            for (int r = 0; r < OUT2; ++r) {
                float s = 0.0f;
                for (int k = 0; k < IN2; ++k) s += Wd2[r * IN2 + k] * Xf2[b * IN2 + k];
                Yref[b * OUT2 + r] = s;
            }
        }

        Tensor Xg2 = Tensor::from_host_fp16_on(Device::CUDA, Xh2.data(), B2, IN2);
        Tensor Yg2;
        brotensor::sync_all();
        (void)brotensor_q8_0_wmma_calls_consume();

        brotensor::linear_forward_batched_q8_0_fp16(Wg2, nullptr, Xg2, Yg2);
        CHECK(Yg2.dtype == Dtype::FP16 && Yg2.rows == B2 && Yg2.cols == OUT2);
        brotensor::sync_all();
        const unsigned long long calls = brotensor_q8_0_wmma_calls_consume();
        std::vector<uint16_t> got(static_cast<size_t>(B2) * OUT2);
        Yg2.copy_to_host_fp16(got.data());

        float max_abs = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            const float g = brotensor::fp16_bits_to_fp32(got[i]);
            const float e = std::fabs(g - Yref[i]);
            if (e > max_abs) max_abs = e;
        }
        std::printf("  %s: B=%d M=%d K=%d wmma_calls=%llu max_abs=%g\n",
                    label, B2, OUT2, IN2, calls, max_abs);
        CHECK(max_abs < tol);
        if (expect_wmma) CHECK(calls > 0);
        else             CHECK(calls == 0);
    };

    run_gemm_case(/*OUT*/64,  /*IN*/512,  /*B*/16, /*wmma*/true,  "E", 5e-2f);
    run_gemm_case(/*OUT*/64,  /*IN*/512,  /*B*/2,  /*wmma*/false, "F", 5e-2f);
    run_gemm_case(/*OUT*/128, /*IN*/1024, /*B*/32, /*wmma*/true,  "G", 1e-1f);
#endif

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
