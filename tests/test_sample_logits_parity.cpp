// CPU↔GPU parity tests for sample_logits (brosoundml CHUNK 7, family F).
//
// sample_logits draws a discrete token id per row, so parity means *exact*
// integer agreement. The Metal kernel reduces in FP32 where the CPU op uses
// FP64 accumulators, so a draw whose Philox uniform lands within a few ulp of a
// CDF-bucket boundary could pick a different token. Every case here is built so
// the realised draw is robust to that:
//
//   * temperature == 0          — deterministic argmax, no RNG.
//   * top_k == 1                — the kept set is a single token (argmax).
//   * peaky logits + top_p      — one token dominates; the nucleus is {argmax}.
//
// That still exercises the whole pipeline — temperature scaling, softmax, the
// descending-probability sort, top-k / top-p filtering, renormalisation, the
// Philox RNG and the inverse-CDF draw — on both backends.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstdio>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

// Exact comparison for the (N,1) INT32 sampled-id tensors.
void compare_ids(const Tensor& cpu, const Tensor& gpu, const char* tag) {
    if (cpu.rows != gpu.rows || cpu.cols != gpu.cols) {
        std::printf("    [%s] shape mismatch: cpu (%d,%d) vs gpu (%d,%d)\n",
                    tag, cpu.rows, cpu.cols, gpu.rows, gpu.cols);
        throw 0;
    }
    const auto* a = static_cast<const int32_t*>(cpu.host_raw());
    const auto* b = static_cast<const int32_t*>(gpu.host_raw());
    for (int i = 0; i < cpu.size(); ++i) {
        if (a[i] != b[i]) {
            std::printf("    [%s] mismatch at row=%d  cpu=%d gpu=%d\n",
                        tag, i, a[i], b[i]);
            throw 0;
        }
    }
}

// Run the op on both backends with the given (key, counter) and compare.
void run_and_compare(const Tensor& logits, float temp, int top_k, float top_p,
                     uint64_t key, uint64_t counter, const char* tag) {
    Tensor cpu_idx;
    brotensor::sample_logits(logits, temp, top_k, top_p, key, counter, cpu_idx);

    Tensor g = logits.to(gpu_device());
    Tensor gpu_idx;
    brotensor::sample_logits(g, temp, top_k, top_p, key, counter, gpu_idx);

    compare_ids(cpu_idx, download_to_host(gpu_idx), tag);
}

// Logits with a guaranteed unique maximum per row (so argmax is unambiguous).
Tensor peaky_logits(int N, int V, SplitMix64& rng, float peak = 8.0f) {
    Tensor L = Tensor::mat(N, V);
    fill_random(L, rng, 1.0f);                       // background in [-1, 1]
    for (int n = 0; n < N; ++n) {
        int col = static_cast<int>(rng.next_u64() % static_cast<uint64_t>(V));
        L[n * V + col] = peak + rng.next_f01();      // dominant token
    }
    return L;
}

// ─── temperature == 0 : deterministic argmax ───────────────────────────────
void run_argmax(int N, int V, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor L = peaky_logits(N, V, rng);
    run_and_compare(L, 0.0f, 0, 1.0f, 0x1234ull, 0ull, "argmax");
}

// ─── top_k == 1 : stochastic path, but the kept set is the single argmax ───
void run_topk1(int N, int V, float temp, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor L = peaky_logits(N, V, rng);
    run_and_compare(L, temp, 1, 1.0f, 0xC0FFEEull, 42ull, "topk1");
}

// ─── peaky logits + top_p : the nucleus collapses to {argmax} ──────────────
void run_peaky_topp(int N, int V, float temp, float top_p,
                    uint64_t counter, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor L = peaky_logits(N, V, rng, 9.0f);   // prob(argmax) > 0.999
    run_and_compare(L, temp, 0, top_p, 0x5EEDull, counter, "peaky_topp");
}

// ── sample_logits_into (graph-safe variant) parity ─────────────────────────
//
// The device-counter / caller-scratch variant must draw exactly what the plain
// host-counter sample_logits draws on the CPU at the same base counter, on the
// GPU backend, and must advance the device counter by N.
void run_into_and_compare(const Tensor& logits, float temp, int top_k,
                          float top_p, uint64_t key, uint32_t base,
                          const char* tag) {
    const int N = logits.rows, V = logits.cols;

    // CPU reference via the original host-counter op at the same base.
    Tensor cpu_ref;
    brotensor::sample_logits(logits, temp, top_k, top_p, key, base, cpu_ref);

    // GPU sample_logits_into: device counter, caller scratch, pre-sized indices.
    Tensor g = logits.to(gpu_device());
    Tensor counter_h = Tensor::zeros_on(brotensor::Device::CPU, 1, 1,
                                        brotensor::Dtype::INT32);
    static_cast<int32_t*>(counter_h.host_raw_mut())[0] =
        static_cast<int32_t>(base);
    Tensor counter = counter_h.to(gpu_device());
    Tensor scratch = Tensor::zeros_on(gpu_device(), 1, 3 * N * V,
                                      brotensor::Dtype::FP32);
    Tensor idx = Tensor::zeros_on(gpu_device(), N, 1, brotensor::Dtype::INT32);
    brotensor::sample_logits_into(g, temp, top_k, top_p, key, counter, scratch, idx);

    compare_ids(cpu_ref, download_to_host(idx), tag);

    // Counter advanced by N on-device when sampling; untouched when greedy.
    Tensor cdown = download_to_host(counter);
    const uint32_t got =
        static_cast<uint32_t>(static_cast<const int32_t*>(cdown.host_raw())[0]);
    const uint32_t want = (temp == 0.0f) ? base : base + static_cast<uint32_t>(N);
    if (got != want) {
        std::printf("    [%s] counter advance: got %u want %u\n", tag, got, want);
        throw 0;
    }
}

void run_into_argmax(int N, int V, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor L = peaky_logits(N, V, rng);
    run_into_and_compare(L, 0.0f, 0, 1.0f, 0x1234ull, 0u, "into_argmax");
}
void run_into_topk1(int N, int V, float temp, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor L = peaky_logits(N, V, rng);
    run_into_and_compare(L, temp, 1, 1.0f, 0xC0FFEEull, 42u, "into_topk1");
}
void run_into_peaky_topp(int N, int V, float temp, float top_p,
                         uint32_t base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor L = peaky_logits(N, V, rng, 9.0f);
    run_into_and_compare(L, temp, 0, top_p, 0x5EEDull, base, "into_peaky_topp");
}

} // namespace

// ─── temperature == 0 ──────────────────────────────────────────────────────
BT_PARITY_TEST(sl_argmax_4x32)   { run_argmax(4, 32, 0xC100ull); }
BT_PARITY_TEST(sl_argmax_1x7)    { run_argmax(1, 7, 0xC101ull); }
BT_PARITY_TEST(sl_argmax_8x256)  { run_argmax(8, 256, 0xC102ull); }

// ─── top_k == 1 (stochastic path, deterministic outcome) ───────────────────
BT_PARITY_TEST(sl_topk1_t1)      { run_topk1(6, 64, 1.0f, 0xC110ull); }
BT_PARITY_TEST(sl_topk1_t07)     { run_topk1(4, 48, 0.7f, 0xC111ull); }
BT_PARITY_TEST(sl_topk1_t13)     { run_topk1(8, 100, 1.3f, 0xC112ull); }

// ─── peaky logits + top_p (nucleus = {argmax}) ─────────────────────────────
BT_PARITY_TEST(sl_peaky_p03)     { run_peaky_topp(6, 64, 1.0f, 0.3f, 7ull,  0xC120ull); }
BT_PARITY_TEST(sl_peaky_p05)     { run_peaky_topp(5, 50, 0.8f, 0.5f, 99ull, 0xC121ull); }
BT_PARITY_TEST(sl_peaky_p09)     { run_peaky_topp(8, 80, 1.0f, 0.9f, 256ull,0xC122ull); }
BT_PARITY_TEST(sl_peaky_p095)    { run_peaky_topp(4, 40, 1.0f, 0.95f, 5ull, 0xC123ull); }

// ─── sample_logits_into: same coverage on the graph-safe variant ───────────
BT_PARITY_TEST(sli_argmax_4x32)  { run_into_argmax(4, 32, 0xD100ull); }
BT_PARITY_TEST(sli_argmax_1x7)   { run_into_argmax(1, 7, 0xD101ull); }
BT_PARITY_TEST(sli_topk1_t1)     { run_into_topk1(6, 64, 1.0f, 0xD110ull); }
BT_PARITY_TEST(sli_topk1_t07)    { run_into_topk1(4, 48, 0.7f, 0xD111ull); }
BT_PARITY_TEST(sli_peaky_p03)    { run_into_peaky_topp(6, 64, 1.0f, 0.3f, 7u,  0xD120ull); }
BT_PARITY_TEST(sli_peaky_p09)    { run_into_peaky_topp(8, 80, 1.0f, 0.9f, 256u, 0xD122ull); }

int main() { return run_all("sample_logits cpu/gpu parity"); }
