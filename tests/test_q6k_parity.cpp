// Q6_K dequant + GEMV + GEMM parity vs a host reference. GPU-only (runs on
// whichever GPU is available: CUDA preferred, Metal fallback). The fused
// WMMA call-count assertions only run on CUDA — Metal has no equivalent
// fused GEMM today and uses the per-row GEMV path for every batch size.
// Q6_K layout (210 bytes / 256 elems):
//   ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] int8, fp16 d.
// Per-sub-block scale sc[is] (is in 0..15) for 16 elements; super-block d.
// Element raw value: val6 in [-32, 31]; dequantized = d * sc[is] * val6.
//
// Quantizer: for each super-block, scan all 256 elements grouped into 16
// sub-blocks of 16; for each sub-block, choose scale = max(|x|)/31, then
// quantise to a 6-bit signed integer in [-32, 31] (we keep the int8 storage
// scale of 1 since the encoder picks d to absorb amplitude). Concretely:
// pick d = max-over-subblocks(max|x|/31) and sc[is] = round((max|x| in sb)/d
// / 31 * 127) clamped to int8 — same shape as Q4_K's nested-scale strategy.

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

static constexpr int Q6K_BLOCK = 256;
static constexpr int Q6K_BYTES = 210;

#pragma pack(push, 1)
struct Q6KBlock {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
#pragma pack(pop)
static_assert(sizeof(Q6KBlock) == 210, "Q6KBlock must be 210 bytes");

// Map element index e -> (group, quad, l, sb) per the spec.
static inline void q6k_indices(int e, int& group, int& quad, int& l, int& sb) {
    group = e / 128;
    const int local = e - group * 128;
    quad = local / 32;
    l    = local - quad * 32;
    sb   = group * 8 + quad * 2 + (l / 16);
}

// Pack a 6-bit value `val4to31` (already in [0, 63]) into ql/qh:
//   ql_b = ql[group*64 + (quad%2)*32 + l]
//   qh_b = qh[group*32 + l]
// Low nibble of ql_b = raw4 for quad<2; high nibble for quad>=2.
// 2 bits in qh_b at position (quad*2) = high2.
static inline void q6k_pack(Q6KBlock& out, int e, int raw6_zero_offset) {
    int group, quad, l, sb;
    q6k_indices(e, group, quad, l, sb);
    const int ql_idx = group * 64 + (quad % 2) * 32 + l;
    const int qh_idx = group * 32 + l;
    const int raw4   = raw6_zero_offset & 0x0F;
    const int high2  = (raw6_zero_offset >> 4) & 0x03;

    if (quad < 2) {
        // Low nibble.
        out.ql[ql_idx] = (out.ql[ql_idx] & 0xF0) | static_cast<uint8_t>(raw4);
    } else {
        out.ql[ql_idx] = (out.ql[ql_idx] & 0x0F) | static_cast<uint8_t>(raw4 << 4);
    }
    const int shift = quad * 2;
    out.qh[qh_idx] = static_cast<uint8_t>(
        (out.qh[qh_idx] & ~(0x03 << shift)) | (high2 << shift));
}

static void quantize_q6k_block(const float* src, Q6KBlock& out) {
    std::memset(&out, 0, sizeof(out));
    // Per-sub-block max-abs.
    float amax_sb[16];
    for (int is = 0; is < 16; ++is) {
        float m = 0.0f;
        // Each sub-block is 16 elements with sb=group*8+quad*2+(l/16).
        // Enumerate all 256 elements and bucket by sb.
        amax_sb[is] = 0.0f;
        (void)m;
    }
    for (int e = 0; e < 256; ++e) {
        int g, q, l, sb;
        q6k_indices(e, g, q, l, sb);
        const float a = std::fabs(src[e]);
        if (a > amax_sb[sb]) amax_sb[sb] = a;
    }
    // Pick d so that d * max_sc * max_val6 covers the largest amax.
    // max_val6 = 31, max_sc (signed int8) = 127. Use d = amax_max / (31*127).
    float amax = 0.0f;
    for (int is = 0; is < 16; ++is) if (amax_sb[is] > amax) amax = amax_sb[is];
    const float d = (amax > 0.0f) ? (amax / (31.0f * 127.0f)) : 1.0f;

    int8_t sc[16];
    for (int is = 0; is < 16; ++is) {
        int s = (d > 0.0f) ? static_cast<int>(std::lround(amax_sb[is] / (d * 31.0f))) : 0;
        s = std::clamp(s, 0, 127);
        sc[is] = static_cast<int8_t>(s);
    }

    for (int e = 0; e < 256; ++e) {
        int g, q, l, sb;
        q6k_indices(e, g, q, l, sb);
        const float denom = d * static_cast<float>(sc[sb]);
        int val6 = 0;
        if (denom > 0.0f) {
            val6 = static_cast<int>(std::lround(src[e] / denom));
        }
        val6 = std::clamp(val6, -32, 31);
        const int raw = val6 + 32;   // [0, 63]
        q6k_pack(out, e, raw);
    }

    for (int is = 0; is < 16; ++is) out.scales[is] = sc[is];
    out.d = brotensor::fp32_to_fp16_bits(d);
}

static void dequant_q6k_block(const Q6KBlock& blk, float* dst) {
    const float d = brotensor::fp16_bits_to_fp32(blk.d);
    for (int e = 0; e < 256; ++e) {
        int group, quad, l, sb;
        q6k_indices(e, group, quad, l, sb);
        const uint8_t ql_b = blk.ql[group * 64 + (quad % 2) * 32 + l];
        const uint8_t qh_b = blk.qh[group * 32 + l];
        const int raw4  = (quad < 2) ? (ql_b & 0x0F) : (ql_b >> 4);
        const int high2 = (qh_b >> (quad * 2)) & 0x03;
        const int val6  = static_cast<int>(raw4 | (high2 << 4)) - 32;
        dst[e] = d * static_cast<float>(blk.scales[sb]) * static_cast<float>(val6);
    }
}

#if defined(BROTENSOR_HAS_CUDA)
extern "C" unsigned long long brotensor_q6k_wmma_calls_consume();
#endif

static std::vector<uint16_t> to_fp16_vec(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

int main() {
    brotensor::init();
    Device dev;
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        dev = Device::CUDA;
    } else if (brotensor::is_available(brotensor::Device::Metal)) {
        dev = Device::Metal;
    } else {
        std::printf("no GPU backend available - skipping\n");
        return 0;
    }
    std::printf("test_q6k_parity (device=%s)\n",
                dev == Device::CUDA ? "CUDA" : "Metal");

    constexpr int OUT = 64;
    constexpr int IN  = 256;
    constexpr int BPR = IN / Q6K_BLOCK;

    std::mt19937 rng(0xBADBEEF);
    std::uniform_real_distribution<float> dw(-0.5f, 0.5f);

    std::vector<float>    Wf(static_cast<size_t>(OUT) * IN);
    for (auto& v : Wf) v = dw(rng);
    std::vector<Q6KBlock> Wq(static_cast<size_t>(OUT) * BPR);
    std::vector<float>    Wd(static_cast<size_t>(OUT) * IN);
    for (int r = 0; r < OUT; ++r) {
        for (int sb = 0; sb < BPR; ++sb) {
            quantize_q6k_block(&Wf[r * IN + sb * Q6K_BLOCK], Wq[r * BPR + sb]);
            dequant_q6k_block(Wq[r * BPR + sb], &Wd[r * IN + sb * Q6K_BLOCK]);
        }
    }
    auto Wd_fp16 = to_fp16_vec(Wd);

    Tensor W_q6k_g = Tensor::empty_on(dev, OUT, IN, Dtype::Q6_K);
    cudaMemcpy(W_q6k_g.data, Wq.data(),
               static_cast<size_t>(OUT) * BPR * Q6K_BYTES,
               cudaMemcpyHostToDevice);

    // ─── A: dequant ───────────────────────────────────────────────────────
    {
        Tensor W_fp16_g;
        brotensor::dequant_q6k_to_fp16(W_q6k_g, W_fp16_g);
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
        Tensor x_g = Tensor::from_host_fp16_on(dev, xh.data(), IN, 1);
        Tensor y_g;
        brotensor::linear_forward_q6k_fp16(W_q6k_g, nullptr, x_g, y_g);
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
        Tensor x_g    = Tensor::from_host_fp16_on(dev, xh.data(), IN, 1);
        Tensor bias_g = Tensor::from_host_fp16_on(dev, bh.data(), OUT, 1);
        Tensor y_g;
        brotensor::linear_forward_q6k_fp16(W_q6k_g, &bias_g, x_g, y_g);
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

    // ─── D: batched (small B fallback) ────────────────────────────────────
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
        Tensor X_g = Tensor::from_host_fp16_on(dev, Xh.data(), B, IN);
        Tensor Y_g;
        brotensor::linear_forward_batched_q6k_fp16(W_q6k_g, nullptr, X_g, Y_g);
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
    (void)brotensor_q6k_wmma_calls_consume();

    auto run_gemm_case = [&](int OUT2, int IN2, int B2, bool expect_wmma,
                             const char* label, float tol) {
        const int BPR2 = IN2 / Q6K_BLOCK;
        std::uniform_real_distribution<float> dw2(-0.5f, 0.5f);
        std::uniform_real_distribution<float> dx2(-0.3f, 0.3f);

        std::vector<float>    Wf2(static_cast<size_t>(OUT2) * IN2);
        for (auto& v : Wf2) v = dw2(rng);
        std::vector<Q6KBlock> Wq2(static_cast<size_t>(OUT2) * BPR2);
        std::vector<float>    Wd2(static_cast<size_t>(OUT2) * IN2);
        for (int r = 0; r < OUT2; ++r) {
            for (int sb = 0; sb < BPR2; ++sb) {
                quantize_q6k_block(&Wf2[r * IN2 + sb * Q6K_BLOCK], Wq2[r * BPR2 + sb]);
                dequant_q6k_block(Wq2[r * BPR2 + sb], &Wd2[r * IN2 + sb * Q6K_BLOCK]);
            }
        }
        Tensor Wg2 = Tensor::empty_on(dev, OUT2, IN2, Dtype::Q6_K);
        cudaMemcpy(Wg2.data, Wq2.data(),
                   static_cast<size_t>(OUT2) * BPR2 * Q6K_BYTES,
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

        Tensor Xg2 = Tensor::from_host_fp16_on(dev, Xh2.data(), B2, IN2);
        Tensor Yg2;
        brotensor::sync_all();
        (void)brotensor_q6k_wmma_calls_consume();

        brotensor::linear_forward_batched_q6k_fp16(Wg2, nullptr, Xg2, Yg2);
        CHECK(Yg2.dtype == Dtype::FP16 && Yg2.rows == B2 && Yg2.cols == OUT2);
        brotensor::sync_all();
        const unsigned long long calls = brotensor_q6k_wmma_calls_consume();
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
