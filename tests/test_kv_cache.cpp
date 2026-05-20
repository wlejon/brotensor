// Tests for kv_cache_append + flash_attention_decode.
// Decode is checked against the causal flash_attention_forward.
// CUDA-only — guarded out on a CPU-only build.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;
#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void test_append() {
    std::printf("  kv_cache_append\n");
    const int L_max = 8, D = 4;
    std::mt19937 rng(0x200);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Pre-fill caches with zeros.
    Tensor Kc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);
    Tensor Vc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);

    // Append two chunks: 3 rows then 2 rows.
    std::vector<float> K1f(3 * D), V1f(3 * D), K2f(2 * D), V2f(2 * D);
    for (auto& v : K1f) v = dist(rng);
    for (auto& v : V1f) v = dist(rng);
    for (auto& v : K2f) v = dist(rng);
    for (auto& v : V2f) v = dist(rng);

    auto K1h = to_fp16(K1f), V1h = to_fp16(V1f);
    auto K2h = to_fp16(K2f), V2h = to_fp16(V2f);
    Tensor K1 = Tensor::from_host_fp16_on(Device::CUDA, K1h.data(), 3, D);
    Tensor V1 = Tensor::from_host_fp16_on(Device::CUDA, V1h.data(), 3, D);
    Tensor K2 = Tensor::from_host_fp16_on(Device::CUDA, K2h.data(), 2, D);
    Tensor V2 = Tensor::from_host_fp16_on(Device::CUDA, V2h.data(), 2, D);

    brotensor::kv_cache_append(K1, V1, 0, Kc, Vc);
    brotensor::kv_cache_append(K2, V2, 3, Kc, Vc);
    brotensor::sync(Device::CUDA);

    std::vector<uint16_t> gotK = Kc.to_host_vector_fp16();
    std::vector<uint16_t> gotV = Vc.to_host_vector_fp16();

    // First 3*D entries should match K1f (after fp16 quant), next 2*D match K2f.
    for (int i = 0; i < 3 * D; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(gotK[i]);
        const float r = brotensor::fp16_bits_to_fp32(K1h[i]);
        CHECK(std::fabs(g - r) < 1e-5f);
    }
    for (int i = 0; i < 2 * D; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(gotK[3 * D + i]);
        const float r = brotensor::fp16_bits_to_fp32(K2h[i]);
        CHECK(std::fabs(g - r) < 1e-5f);
    }
    // Remaining tail should be zero.
    for (int i = 5 * D; i < L_max * D; ++i) {
        CHECK(gotK[i] == 0);
        CHECK(gotV[i] == 0);
    }
}

static void test_decode_vs_causal_forward() {
    std::printf("  flash_attention_decode vs causal forward\n");
    // Build a "complete" attention setup with L_total tokens; decode is the
    // tail Lq rows treating the first valid_len cache rows.
    const int L_total = 12, D = 16, nh = 2;
    const int L_max = 16;
    std::mt19937 rng(0x201);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> Qf(L_total * D), Kf(L_total * D), Vf(L_total * D);
    for (auto& v : Qf) v = dist(rng);
    for (auto& v : Kf) v = dist(rng);
    for (auto& v : Vf) v = dist(rng);

    auto Qh = to_fp16(Qf), Kh = to_fp16(Kf), Vh = to_fp16(Vf);

    // Reference: causal full-sequence forward.
    Tensor Qg = Tensor::from_host_fp16_on(Device::CUDA, Qh.data(), L_total, D);
    Tensor Kg = Tensor::from_host_fp16_on(Device::CUDA, Kh.data(), L_total, D);
    Tensor Vg = Tensor::from_host_fp16_on(Device::CUDA, Vh.data(), L_total, D);
    Tensor Oref = Tensor::empty_on(Device::CUDA, L_total, D, Dtype::FP16);
    brotensor::flash_attention_forward(Qg, Kg, Vg, nullptr, nh,
                                       /*causal=*/true, Oref);
    brotensor::sync(Device::CUDA);
    std::vector<uint16_t> ref_h = Oref.to_host_vector_fp16();

    // Decode: fill K/V caches with all L_total rows, query is the last Lq=3
    // rows of Q. valid_len = L_total. seq_offset = L_total - Lq.
    const int Lq = 3;
    Tensor Kc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);
    Tensor Vc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);
    brotensor::kv_cache_append(Kg, Vg, 0, Kc, Vc);

    // Take the last Lq rows of Q via copy_d2d.
    Tensor Qtail = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    brotensor::copy_d2d(Qg, (L_total - Lq) * D, Qtail, 0, Lq * D);

    Tensor Odec = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    brotensor::flash_attention_decode(Qtail, Kc, Vc, L_total, nh, Odec);
    brotensor::sync(Device::CUDA);
    std::vector<uint16_t> dec_h = Odec.to_host_vector_fp16();

    // Compare against the last Lq*D entries of ref_h.
    float me = 0.0f;
    int bad = 0;
    const int off = (L_total - Lq) * D;
    for (int i = 0; i < Lq * D; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(dec_h[i]);
        const float r = brotensor::fp16_bits_to_fp32(ref_h[off + i]);
        const float e = std::fabs(g - r);
        if (e > me) me = e;
        if (e > 5e-3f + 5e-2f * std::fabs(r)) ++bad;
    }
    std::printf("    decode vs causal: max_err=%g bad=%d/%d\n", me, bad, Lq * D);
    CHECK(bad == 0);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_kv_cache\n");
    test_append();
    test_decode_vs_causal_forward();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
