// Q4_K dequant + GEMV parity vs a host reference. GPU-only; skips cleanly
// on CPU/Metal hosts. Quantises a small FP32 weight to Q4_K with a simple
// affine min/max-per-32 scheme, packs the 6-bit sub-scales using the inverse
// of get_scale_min_k4, runs the CUDA dequant + GEMV kernels, compares against
// dequant_host -> matmul_host.

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

static constexpr int Q4K_BLOCK = 256;
static constexpr int Q4K_BYTES = 144;

// Host Q4_K block layout, in bytes laid out per the GGUF spec.
struct Q4KBlock {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "Q4KBlock must be 144 bytes");

// Pack 6-bit sc/m pairs into the 12-byte scales array. Inverse of
// get_scale_min_k4 (j < 4 stores low-6 bits directly; j >= 4 splits
// into the low nibble of scales[j+4] + top bits of scales[j-4]/scales[j]).
static void pack_sc_m(uint8_t scales[12], const uint8_t sc[8], const uint8_t m[8]) {
    std::memset(scales, 0, 12);
    for (int j = 0; j < 4; ++j) {
        scales[j]     = sc[j] & 0x3F;          // low 6 bits = sc
        scales[j + 4] = m[j]  & 0x3F;          // low 6 bits = m
    }
    // For j in 4..7: 4-bit lo in scales[j+4] low nibble (sc) / high nibble (m),
    // 2-bit hi piggybacks on the top of scales[j-4] (sc) / scales[j] (m).
    for (int j = 4; j < 8; ++j) {
        const uint8_t sc_lo = sc[j] & 0x0F;
        const uint8_t sc_hi = (sc[j] >> 4) & 0x03;
        const uint8_t m_lo  = m[j]  & 0x0F;
        const uint8_t m_hi  = (m[j]  >> 4) & 0x03;
        scales[j + 4] = static_cast<uint8_t>(sc_lo | (m_lo << 4));
        scales[j - 4] |= static_cast<uint8_t>(sc_hi << 6);
        scales[j]     |= static_cast<uint8_t>(m_hi  << 6);
    }
}

static void unpack_sc_m(const uint8_t scales[12], uint8_t* sc, uint8_t* m) {
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = scales[j]     & 0x3F;
            m [j] = scales[j + 4] & 0x3F;
        } else {
            sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            m [j] = (scales[j + 4] >> 4)   | ((scales[j - 0] >> 6) << 4);
        }
    }
}

// Quantise one 256-element row chunk into a Q4KBlock. Sub-blocks of 32:
//   for each sub-block find (min, max) and pick a per-sub-block scale (sc[is])
//   and offset (m[is]) such that y = d * sc * nibble - dmin * m approximates
//   the original. We use a single super-block d/dmin and per-sub-block sc/m
//   capped at 6 bits.
static void quantize_q4k_block(const float* src, Q4KBlock& out) {
    // Find per-sub-block (min, max).
    float lo[8], hi[8];
    for (int is = 0; is < 8; ++is) {
        lo[is] = src[is * 32];
        hi[is] = src[is * 32];
        for (int l = 1; l < 32; ++l) {
            const float v = src[is * 32 + l];
            if (v < lo[is]) lo[is] = v;
            if (v > hi[is]) hi[is] = v;
        }
    }
    // Pick super-block d such that the largest per-sub-block range / 15
    // fits in a 6-bit sc value. sc[is] * d ≈ (hi - lo) / 15.
    float max_range = 0.0f;
    for (int is = 0; is < 8; ++is) {
        const float r = hi[is] - lo[is];
        if (r > max_range) max_range = r;
    }
    // d * 63 = max_range / 15 (max sc is 63 for the largest sub-block range).
    const float d = (max_range > 0.0f) ? (max_range / (15.0f * 63.0f)) : 1.0f;
    // dmin chosen similarly so dmin * m approximates -lo. Largest -lo:
    float max_neg_lo = 0.0f;
    for (int is = 0; is < 8; ++is) {
        const float v = -lo[is];
        if (v > max_neg_lo) max_neg_lo = v;
    }
    const float dmin = (max_neg_lo > 0.0f) ? (max_neg_lo / 63.0f) : 1.0f;

    uint8_t sc[8], m[8];
    for (int is = 0; is < 8; ++is) {
        const float r = hi[is] - lo[is];
        int sc_i = (d > 0.0f) ? static_cast<int>(std::lround(r / (15.0f * d))) : 0;
        sc_i = std::clamp(sc_i, 0, 63);
        sc[is] = static_cast<uint8_t>(sc_i);

        int m_i = (dmin > 0.0f) ? static_cast<int>(std::lround(-lo[is] / dmin)) : 0;
        m_i = std::clamp(m_i, 0, 63);
        m[is] = static_cast<uint8_t>(m_i);
    }

    // Pack 4-bit quants. Pair layout: for pair p (0..3), nibble-byte offset
    // is p * 32 + l (l in 0..31). Low nibble -> sub-block is = 2*p,
    // high nibble -> sub-block is = 2*p+1.
    std::memset(out.qs, 0, 128);
    for (int p = 0; p < 4; ++p) {
        const int is_lo = 2 * p;
        const int is_hi = 2 * p + 1;
        const float w_lo = static_cast<float>(sc[is_lo]) * d;
        const float w_hi = static_cast<float>(sc[is_hi]) * d;
        const float b_lo = static_cast<float>(m [is_lo]) * dmin;
        const float b_hi = static_cast<float>(m [is_hi]) * dmin;
        for (int l = 0; l < 32; ++l) {
            int n_lo = 0;
            if (w_lo > 0.0f) {
                n_lo = static_cast<int>(std::lround((src[is_lo * 32 + l] + b_lo) / w_lo));
            }
            n_lo = std::clamp(n_lo, 0, 15);

            int n_hi = 0;
            if (w_hi > 0.0f) {
                n_hi = static_cast<int>(std::lround((src[is_hi * 32 + l] + b_hi) / w_hi));
            }
            n_hi = std::clamp(n_hi, 0, 15);

            out.qs[p * 32 + l] = static_cast<uint8_t>((n_lo & 0x0F) | ((n_hi & 0x0F) << 4));
        }
    }

    out.d    = brotensor::fp32_to_fp16_bits(d);
    out.dmin = brotensor::fp32_to_fp16_bits(dmin);
    pack_sc_m(out.scales, sc, m);
}

// Dequantise one Q4KBlock to 256 FP32 values — host reference using the
// same math as the kernel.
static void dequant_q4k_block(const Q4KBlock& blk, float* dst) {
    const float d    = brotensor::fp16_bits_to_fp32(blk.d);
    const float dmin = brotensor::fp16_bits_to_fp32(blk.dmin);
    uint8_t sc[8], m[8];
    unpack_sc_m(blk.scales, sc, m);
    for (int t = 0; t < 256; ++t) {
        const int is   = t >> 5;
        const int l    = t & 31;
        const int pair = is >> 1;
        const uint8_t qb = blk.qs[pair * 32 + l];
        const int nib  = (is & 1) ? (qb >> 4) : (qb & 0x0F);
        dst[t] = d * static_cast<float>(sc[is]) * static_cast<float>(nib)
               - dmin * static_cast<float>(m[is]);
    }
}

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
    std::printf("test_q4k_parity\n");

    constexpr int OUT = 64;
    constexpr int IN  = 256;  // one super-block per row (keeps it tight).
    constexpr int BLOCKS_PER_ROW = IN / Q4K_BLOCK;

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dw(-0.5f, 0.5f);

    // Build FP32 weight, then quantise row-by-row.
    std::vector<float>    Wf(static_cast<size_t>(OUT) * IN);
    for (auto& v : Wf) v = dw(rng);

    std::vector<Q4KBlock> Wq(static_cast<size_t>(OUT) * BLOCKS_PER_ROW);
    std::vector<float>    W_deq(static_cast<size_t>(OUT) * IN);
    for (int r = 0; r < OUT; ++r) {
        for (int sb = 0; sb < BLOCKS_PER_ROW; ++sb) {
            quantize_q4k_block(&Wf[r * IN + sb * Q4K_BLOCK],
                               Wq[r * BLOCKS_PER_ROW + sb]);
            dequant_q4k_block(Wq[r * BLOCKS_PER_ROW + sb],
                              &W_deq[r * IN + sb * Q4K_BLOCK]);
        }
    }
    auto W_deq_fp16 = to_fp16_vec(W_deq);

    // Upload Q4_K weight bytes to the device.
    Tensor W_q4k_g = Tensor::empty_on(Device::CUDA, OUT, IN, Dtype::Q4_K);
    cudaMemcpy(W_q4k_g.data, Wq.data(),
               static_cast<size_t>(OUT) * BLOCKS_PER_ROW * Q4K_BYTES,
               cudaMemcpyHostToDevice);

    // ─── Test A: dequant parity ────────────────────────────────────────────
    {
        Tensor W_fp16_g;
        brotensor::dequant_q4k_to_fp16(W_q4k_g, W_fp16_g);
        CHECK(W_fp16_g.dtype == Dtype::FP16);
        CHECK(W_fp16_g.rows == OUT && W_fp16_g.cols == IN);
        brotensor::sync_all();
        std::vector<uint16_t> got(static_cast<size_t>(OUT) * IN);
        W_fp16_g.copy_to_host_fp16(got.data());

        int mismatched = 0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            const float g = brotensor::fp16_bits_to_fp32(got[i]);
            const float r = brotensor::fp16_bits_to_fp32(W_deq_fp16[i]);
            const float e = std::fabs(g - r);
            if (e > max_abs) max_abs = e;
            if (got[i] != W_deq_fp16[i]) ++mismatched;
        }
        std::printf("  A: dequant max_abs=%g mismatched_bits=%d/%zu\n",
                    max_abs, mismatched, got.size());
        // FP16 deterministic round-trip — allow at most 1 ulp slack.
        CHECK(max_abs <= 1e-3f);
    }

    // ─── Test B: GEMV parity ───────────────────────────────────────────────
    std::uniform_real_distribution<float> dx(-0.3f, 0.3f);
    std::vector<float> xf(IN);
    for (auto& v : xf) v = dx(rng);
    auto xh = to_fp16_vec(xf);

    // Host reference: dequant(W) @ x in FP32.
    std::vector<float> y_ref(OUT, 0.0f);
    for (int r = 0; r < OUT; ++r) {
        float s = 0.0f;
        for (int k = 0; k < IN; ++k) {
            s += W_deq[r * IN + k] * xf[k];
        }
        y_ref[r] = s;
    }

    {
        Tensor x_g = Tensor::from_host_fp16_on(Device::CUDA, xh.data(), IN, 1);
        Tensor y_g;
        brotensor::linear_forward_q4k_fp16(W_q4k_g, nullptr, x_g, y_g);
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

    // ─── Test C: GEMV with bias ────────────────────────────────────────────
    std::uniform_real_distribution<float> db(-0.1f, 0.1f);
    std::vector<float> bf(OUT);
    for (auto& v : bf) v = db(rng);
    auto bh = to_fp16_vec(bf);
    {
        Tensor x_g    = Tensor::from_host_fp16_on(Device::CUDA, xh.data(), IN, 1);
        Tensor bias_g = Tensor::from_host_fp16_on(Device::CUDA, bh.data(), OUT, 1);
        Tensor y_g;
        brotensor::linear_forward_q4k_fp16(W_q4k_g, &bias_g, x_g, y_g);
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

    // ─── Test D: batched ──────────────────────────────────────────────────
    constexpr int B = 4;
    std::vector<float> Xf(static_cast<size_t>(B) * IN);
    for (auto& v : Xf) v = dx(rng);
    auto Xh = to_fp16_vec(Xf);

    std::vector<float> Y_ref(static_cast<size_t>(B) * OUT, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int r = 0; r < OUT; ++r) {
            float s = 0.0f;
            for (int k = 0; k < IN; ++k) {
                s += W_deq[r * IN + k] * Xf[b * IN + k];
            }
            Y_ref[b * OUT + r] = s;
        }
    }

    {
        Tensor X_g = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(), B, IN);
        Tensor Y_g;
        brotensor::linear_forward_batched_q4k_fp16(W_q4k_g, nullptr, X_g, Y_g);
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

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
