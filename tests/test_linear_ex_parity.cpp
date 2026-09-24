// linear_forward_batched_ex: fused-epilogue batched linear, CPU<->GPU parity.
//
// Covers every epilogue (store / accumulate / GeGLU) and fused activations on
// FP16 and BF16, over shapes that take each GPU path: large tiles (M fills the
// device), small tiles + split-K (short M, long K, workspace given), the same
// short shape unsplit (no workspace), and the WMMA fallback (K % 8 != 0).
// Split-K must also be deterministic run to run.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
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

void check(int M, int N, int K, int act, int epi, bool use_ws, bool use_bias, Dtype dt, uint64_t seed) {
    SplitMix64 rng(seed);
    const Tensor W = rnd(N, K, rng, 0.06f, dt);
    const Tensor X = rnd(M, K, rng, 1.0f, dt);
    const Tensor b = rnd(N, 1, rng, 0.5f, dt);
    const int out_cols = (epi & 15) == brotensor::kLinearEpiGeglu ? N / 2 : N;
    const Tensor Y0 = rnd(M, out_cols, rng, 1.0f, dt);

    Tensor yc = Y0.clone();
    brotensor::linear_forward_batched_ex(W, use_bias ? &b : nullptr, X, act, epi, nullptr, yc);

    Tensor gW = up(W, dt), gX = up(X, dt), gb = up(b, dt);
    Tensor gY = up(Y0, dt);
    Tensor ws;
    brotensor::linear_forward_batched_ex(gW, use_bias ? &gb : nullptr, gX, act, epi, use_ws ? &ws : nullptr, gY);
    const Tensor got = down(gY, dt);
    const float tol = dt == Dtype::FP16 ? 1.5e-2f : 6e-2f;
    compare_tensors(yc, got, "linear_ex", tol, tol);

    if (use_ws) {  // split-K reduces in a fixed order: bitwise stable.
        Tensor gY2 = up(Y0, dt);
        brotensor::linear_forward_batched_ex(gW, use_bias ? &gb : nullptr, gX, act, epi, &ws, gY2);
        const Tensor again = down(gY2, dt);
        BT_CHECK(std::memcmp(again.ptr(), got.ptr(), sizeof(float) * got.size()) == 0);
    }
}

constexpr int kStore = brotensor::kLinearEpiStore;
constexpr int kAcc = brotensor::kLinearEpiAccumulate;
constexpr int kGeglu = brotensor::kLinearEpiGeglu;

BT_PARITY_TEST(cpu_epilogues) {
    // CPU accumulate and GeGLU against the plain CPU store.
    SplitMix64 rng(0x11);
    const Tensor W = rnd(6, 5, rng, 1.0f, Dtype::FP16), X = rnd(3, 5, rng, 1.0f, Dtype::FP16);
    Tensor r;
    brotensor::linear_forward_batched_ex(W, nullptr, X, 0, kStore, nullptr, r);
    Tensor acc = Tensor::mat(3, 6);
    for (int i = 0; i < acc.size(); ++i) acc.ptr()[i] = 1.0f;
    brotensor::linear_forward_batched_ex(W, nullptr, X, 0, kAcc, nullptr, acc);
    Tensor g;
    brotensor::linear_forward_batched_ex(W, nullptr, X, 0, kGeglu, nullptr, g);
    BT_CHECK(g.rows == 3 && g.cols == 3);
    for (int m = 0; m < 3; ++m) {
        for (int n = 0; n < 6; ++n) BT_CHECK(std::fabs(acc(m, n) - (r(m, n) + 1.0f)) < 1e-6f);
        for (int j = 0; j < 3; ++j) {
            const float v = r(m, 2 * j + 1);
            const float gelu = 0.5f * v * (1.0f + std::erf(v * 0.70710678118f));
            // Relative: |g| reaches ~150 here, where one float ulp is 1.5e-5 and an
            // FMA-contracted difference (arm64 clang) exposes the product's rounding.
            BT_CHECK(std::fabs(g(m, j) - r(m, 2 * j) * gelu) <= 1e-6f * (1.0f + std::fabs(g(m, j))));
        }
    }
}

// Large tiles (M * N fills the device: >= 128 128x128 tiles on a 128-SM part;
// the 960-row cases take the 64 x 64 tile there).
BT_PARITY_TEST(large_store_fp16) {
    check(960, 1024, 1024, 0, kStore, true, true, Dtype::FP16, 0x21);
    check(2100, 1536, 512, 3, kStore, true, true, Dtype::FP16, 0x24);
    check(2100, 1536, 512, 0, kAcc | brotensor::kLinearEpiFastAccum, true, false, Dtype::FP16, 0x25);
    check(2100, 1536, 256, 0, kGeglu, true, false, Dtype::BF16, 0x26);
}
BT_PARITY_TEST(large_acc_fp16) { check(700, 1024, 512, 0, kAcc, true, false, Dtype::FP16, 0x22); }
BT_PARITY_TEST(large_geglu_bf16) { check(600, 1024, 256, 0, kGeglu, true, false, Dtype::BF16, 0x23); }
// Short M: small tiles, split-K with a workspace, unsplit without.
BT_PARITY_TEST(split_store_gelu_fp16) { check(112, 1024, 2624, 3, kStore, true, true, Dtype::FP16, 0x31); }
BT_PARITY_TEST(split_acc_fp16) { check(37, 1024, 1024, 0, kAcc, true, true, Dtype::FP16, 0x32); }
BT_PARITY_TEST(split_geglu_fp16) { check(50, 1536, 1024, 0, kGeglu, true, false, Dtype::FP16, 0x33); }
BT_PARITY_TEST(split_acc_bf16) { check(9, 512, 2048, 0, kAcc, true, true, Dtype::BF16, 0x34); }
BT_PARITY_TEST(nosplit_acc_fp16) { check(37, 1024, 1024, 0, kAcc, false, true, Dtype::FP16, 0x35); }
BT_PARITY_TEST(relu_silu_fp16) {
    check(64, 256, 128, 1, kStore, true, true, Dtype::FP16, 0x36);
    check(64, 256, 128, 4, kStore, false, true, Dtype::FP16, 0x37);
}
// FP16 hybrid accumulation (kLinearEpiFastAccum): 16-term FP16 partial sums
// folded into FP32 — same tolerance must hold at K up to 2624.
constexpr int kFast = brotensor::kLinearEpiFastAccum;
BT_PARITY_TEST(fast_accum_fp16) {
    check(960, 1024, 1024, 0, kStore | kFast, true, true, Dtype::FP16, 0x51);
    check(112, 1024, 2624, 0, kAcc | kFast, true, true, Dtype::FP16, 0x52);
    check(300, 1536, 1024, 0, kGeglu | kFast, true, false, Dtype::FP16, 0x53);
    check(9, 512, 2048, 3, kStore | kFast, true, true, Dtype::FP16, 0x54);
    check(64, 256, 128, 0, kAcc | kFast, true, true, Dtype::BF16, 0x55);  // flag ignored for BF16
}

// K % 8 != 0: WMMA fallback + separate epilogue pass.
BT_PARITY_TEST(fallback_odd_k) {
    check(21, 64, 1028, 3, kStore, true, true, Dtype::FP16, 0x41);
    check(21, 64, 1028, 0, kAcc, true, true, Dtype::FP16, 0x42);
    check(21, 64, 100, 0, kGeglu, false, false, Dtype::FP16, 0x43);
}

}  // namespace

int main() { return run_all("linear_forward_batched_ex (fused epilogue / split-K) CPU<->GPU parity"); }
