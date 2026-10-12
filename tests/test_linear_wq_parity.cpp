// quantize_rows + linear_forward_batched_wq_ex: CPU <-> GPU parity.
//
// The GPU quantiser must produce the CPU's codes and scales exactly (both
// round to nearest even; FP8 E4M3 through the host encoder vs the device
// cvt). The linear then runs on each side from its own quantised weight:
// INT8 and FP8 weights, FP16 and BF16 activations, every epilogue, one to
// eight rows on the warp-per-column GEMV, more rows in slices of eight, and
// the scalar fallback (K % 16 != 0). The quantisation error against the
// unquantised weight is reported per format.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace bt_parity;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

float q16(float v) { return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v)); }
float qbf(float v) { return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v)); }

Tensor rnd(int r, int c, SplitMix64& rng, float scale, Dtype dt) {
    Tensor t = Tensor::mat(r, c);
    for (int i = 0; i < t.size(); ++i) {
        const float x = (rng.next_unit() * 2.0f - 1.0f) * scale;
        t.ptr()[i] = dt == Dtype::FP16 ? q16(x) : qbf(x);
    }
    return t;
}

Tensor up(const Tensor& t, Dtype dt) { return dt == Dtype::FP16 ? to_fp16_gpu(t) : to_bf16_gpu(t); }
Tensor down(const Tensor& g, Dtype dt) {
    const Tensor h = download_to_host(g);
    return dt == Dtype::FP16 ? fp16_host_to_f32(h) : bf16_host_to_f32(h);
}

constexpr int kStore = brotensor::kLinearEpiStore;
constexpr int kAcc = brotensor::kLinearEpiAccumulate;
constexpr int kGeglu = brotensor::kLinearEpiGeglu;
constexpr int kSwiglu = brotensor::kLinearEpiSwiglu;

void check(int M, int N, int K, int act, int epi, bool use_bias, Dtype fmt, Dtype dt, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = rnd(N, K, rng, 0.06f, dt);
    // A few outlier columns, as real weights have.
    for (int r = 0; r < N; r += 7) {
        float& v = W.ptr()[static_cast<std::size_t>(r) * K + (r % K)];
        v = dt == Dtype::FP16 ? q16(v * 9.0f) : qbf(v * 9.0f);
    }
    const Tensor X = rnd(M, K, rng, 1.0f, dt);
    const Tensor b = rnd(N, 1, rng, 0.5f, dt);
    const int out_cols = (epi == kGeglu || epi == kSwiglu) ? N / 2 : N;
    const Tensor Y0 = rnd(M, out_cols, rng, 1.0f, dt);

    Tensor cq, cs;
    brotensor::quantize_rows(W, fmt, cq, cs);
    Tensor gW = up(W, dt), gq, gs;
    brotensor::quantize_rows(gW, fmt, gq, gs);
    const Tensor hq = download_to_host(gq), hs = download_to_host(gs);
    const std::size_t nw = static_cast<std::size_t>(N) * K;
    BT_CHECK(hq.dtype == fmt && hq.rows == N && hq.cols == K);
    BT_CHECK(std::memcmp(hq.data, cq.data, nw) == 0);
    BT_CHECK(std::memcmp(hs.data, cs.data, sizeof(float) * N) == 0);

    Tensor yc = Y0.clone();
    brotensor::linear_forward_batched_wq_ex(cq, cs, use_bias ? &b : nullptr, X, act, epi, yc);
    Tensor gX = up(X, dt), gb = up(b, dt), gY = up(Y0, dt);
    brotensor::linear_forward_batched_wq_ex(gq, gs, use_bias ? &gb : nullptr, gX, act, epi, gY);
    const float tol = dt == Dtype::FP16 ? 1.5e-2f : 6e-2f;
    compare_tensors(yc, down(gY, dt), "linear_wq", tol, tol);
}

// Relative RMS error of the dequantised weight, per format (reported only).
void report_error(Dtype fmt) {
    SplitMix64 rng(0x99);
    Tensor W = Tensor::mat(256, 2048);
    for (int i = 0; i < W.size(); ++i) {
        // Roughly Gaussian (sum of uniforms), the shape of trained weights.
        float s = 0;
        for (int k = 0; k < 4; ++k) s += rng.next_unit() - 0.5f;
        W.ptr()[i] = s * 0.05f;
    }
    Tensor q, s;
    brotensor::quantize_rows(W, fmt, q, s);
    const auto* qb = static_cast<const uint8_t*>(q.data);
    double se = 0, sw = 0;
    for (int r = 0; r < W.rows; ++r)
        for (int k = 0; k < W.cols; ++k) {
            const std::size_t i = static_cast<std::size_t>(r) * W.cols + k;
            const float d = fmt == Dtype::INT8 ? static_cast<float>(static_cast<int8_t>(qb[i]))
                                               : brotensor::fp8_e4m3_bits_to_fp32(qb[i]);
            const double e = d * s.ptr()[r] - W.ptr()[i];
            se += e * e;
            sw += static_cast<double>(W.ptr()[i]) * W.ptr()[i];
        }
    std::printf("        %s per-row weight error: relative rms %.4f\n", fmt == Dtype::INT8 ? "INT8" : "FP8 E4M3",
                std::sqrt(se / sw));
}

}  // namespace

BT_PARITY_TEST(fp8_codec) {
    // Every code decodes and re-encodes to itself; the encoder rounds to the
    // nearer neighbour, ties to even, and saturates.
    for (int c = 0; c < 256; ++c) {
        if ((c & 0x7f) == 0x7f) continue;   // NaN
        const float v = brotensor::fp8_e4m3_bits_to_fp32(static_cast<uint8_t>(c));
        if (c == 0x80) continue;            // -0 encodes as -0 but compares equal to 0
        BT_CHECK(brotensor::fp32_to_fp8_e4m3_bits(v) == c);
    }
    BT_CHECK(brotensor::fp8_e4m3_bits_to_fp32(0x7e) == 448.0f);
    BT_CHECK(brotensor::fp32_to_fp8_e4m3_bits(1000.0f) == 0x7e);
    BT_CHECK(brotensor::fp32_to_fp8_e4m3_bits(-1000.0f) == 0xfe);
    // 1.0625 lies halfway between 1.0 (0x38) and 1.125 (0x39): ties to even.
    BT_CHECK(brotensor::fp32_to_fp8_e4m3_bits(1.0625f) == 0x38);
    BT_CHECK(brotensor::fp32_to_fp8_e4m3_bits(1.1875f) == 0x3a);
    report_error(Dtype::INT8);
    report_error(Dtype::FP8_E4M3);
}

BT_PARITY_TEST(gemv_rows_int8_bf16) {
    for (int m = 1; m <= 8; ++m) check(m, 512, 2048, 0, kStore, false, Dtype::INT8, Dtype::BF16, 0x100 + m);
}
BT_PARITY_TEST(gemv_rows_fp8_bf16) {
    for (int m = 1; m <= 8; ++m) check(m, 512, 2048, 0, kStore, false, Dtype::FP8_E4M3, Dtype::BF16, 0x200 + m);
}
BT_PARITY_TEST(epilogues) {
    for (Dtype fmt : {Dtype::INT8, Dtype::FP8_E4M3}) {
        check(2, 1024, 2048, 0, kAcc, false, fmt, Dtype::BF16, 0x301);
        check(1, 1024, 6144, 0, kAcc, false, fmt, Dtype::BF16, 0x302);
        check(2, 2048, 2048, 0, kSwiglu, false, fmt, Dtype::BF16, 0x303);
        check(3, 512, 1024, 0, kGeglu, false, fmt, Dtype::FP16, 0x304);
        check(4, 333, 512, 3, kStore, true, fmt, Dtype::FP16, 0x305);   // odd N, bias, GELU
        check(1, 256, 512, 4, kStore, true, fmt, Dtype::BF16, 0x306);
    }
}
BT_PARITY_TEST(slices_and_fallback) {
    check(13, 256, 1024, 0, kAcc, false, Dtype::INT8, Dtype::BF16, 0x401);   // 8 + 5
    check(20, 256, 512, 0, kSwiglu, false, Dtype::FP8_E4M3, Dtype::BF16, 0x402);
    check(3, 64, 1000, 0, kStore, true, Dtype::INT8, Dtype::FP16, 0x403);    // K % 16 != 0
    check(2, 64, 200, 0, kSwiglu, false, Dtype::FP8_E4M3, Dtype::BF16, 0x404);
    check(3, 64, 1003, 0, kStore, true, Dtype::INT8, Dtype::FP16, 0x405);    // K % 8 != 0: no 8-wide loads
    check(2, 64, 1003, 0, kSwiglu, false, Dtype::FP8_E4M3, Dtype::FP32, 0x406);
}

// dequantize_rows: the GPU's FP32 output is the CPU's bit for bit (one FP32
// multiply each), its BF16 / FP16 outputs that value rounded once, and a W
// already of the right shape (a view into a scratch) is written in place.
BT_PARITY_TEST(dequantize_rows) {
    for (Dtype fmt : {Dtype::INT8, Dtype::FP8_E4M3}) {
        SplitMix64 rng(0x501);
        const int N = 97, K = 1003;   // n % 4 != 0: the last thread's tail
        const Tensor W = rnd(N, K, rng, 0.08f, Dtype::BF16);
        Tensor cq, cs;
        brotensor::quantize_rows(W, fmt, cq, cs);
        Tensor cw;
        brotensor::dequantize_rows(cq, cs, Dtype::FP32, cw);
        BT_CHECK(cw.rows == N && cw.cols == K && cw.dtype == Dtype::FP32);
        Tensor gW = up(W, Dtype::BF16), gq, gs;
        brotensor::quantize_rows(gW, fmt, gq, gs);
        Tensor g32;
        brotensor::dequantize_rows(gq, gs, Dtype::FP32, g32);
        const Tensor h32 = download_to_host(g32);
        BT_CHECK(std::memcmp(h32.data, cw.data, sizeof(float) * N * K) == 0);
        for (Dtype dt : {Dtype::BF16, Dtype::FP16}) {
            Tensor g16 = up(Tensor::mat(N, K), dt);   // pre-shaped: written in place
            void* before = g16.data;
            brotensor::dequantize_rows(gq, gs, dt, g16);
            BT_CHECK(g16.data == before);
            const Tensor h = down(g16, dt);
            int bad = 0;
            for (int i = 0; i < N * K; ++i) {
                const float want = dt == Dtype::FP16 ? q16(cw.ptr()[i]) : qbf(cw.ptr()[i]);
                if (h.ptr()[i] != want) {
                    if (bad < 3)
                        std::printf("        %s->%s [%d]: %.9g want %.9g (fp32 %.9g)\n",
                                    fmt == Dtype::INT8 ? "int8" : "fp8", dt == Dtype::FP16 ? "fp16" : "bf16", i,
                                    h.ptr()[i], want, cw.ptr()[i]);
                    ++bad;
                }
            }
            BT_CHECK(bad == 0);
        }
    }
}

int main() { return run_all("linear_forward_batched_wq_ex parity"); }
