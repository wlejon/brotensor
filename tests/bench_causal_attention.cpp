// Micro-benchmark: causal vs non-causal prefill attention.
//
// flash_attention_forward routes to a fused FlashAttention-2 WMMA kernel
// (src/cuda/flash_attention_fused.cu) only for the non-causal case — see the
// `if (!causal && flash_fused::supported(head_dim))` gate in
// flash_attention.cu. With causal=true it falls through to a scalar fallback
// that computes each score as a serial per-thread head_dim dot product read
// straight from global memory, with a shared-memory tree reduction per tile.
//
// So every LLM prefill — which is causal by definition — misses the tensor
// cores entirely. This bench times the same shape both ways: the non-causal
// row is the fused kernel, the causal row is the fallback. The ratio is the
// cliff, and it is the number to watch when the causal path is taught to use
// the fused kernel.
//
// Causal does strictly *less* arithmetic (roughly half the score matrix), so
// a correct causal path should end up FASTER than its non-causal twin, not
// slower. Any ratio above 1.0x is pure lost work.
//
// GFLOP/s uses each variant's own flop count: 4*L*L*hd*nh non-causal (two
// GEMMs), half that for causal.
//
// Correctness is verified once at a small shape against a host FP32 reference
// (both variants), then the large shapes are timed — a host reference at
// L=8192 costs minutes and adds nothing.
//
// NOT registered with ctest — invoke manually:
//   ./build-cuda/tests/Release/brotensor_bench_causal_attention

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

int g_failures = 0;

std::vector<float> rand_vec(size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

Tensor upload_fp16(const std::vector<float>& v, int rows, int cols) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return Tensor::from_host_fp16_on(Device::CUDA, h.data(), rows, cols);
}

std::vector<float> download_fp16(const Tensor& t) {
    std::vector<uint16_t> h(t.size());
    t.copy_to_host_fp16(h.data());
    brotensor::sync_all();
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i) v[i] = brotensor::fp16_bits_to_fp32(h[i]);
    return v;
}

float q16(float v) {
    return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

// Host FP32 reference for one (L, D) self-attention, nh heads. O(L^2*hd*nh) —
// only ever called at the small verification shape.
std::vector<float> attention_ref(const std::vector<float>& qv,
                                 const std::vector<float>& kv,
                                 const std::vector<float>& vv,
                                 int L, int D, int nh, bool causal) {
    const int hd = D / nh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> out(static_cast<size_t>(L) * D, 0.0f);
    std::vector<float> scores(static_cast<size_t>(L));

    for (int h = 0; h < nh; ++h) {
        const int off = h * hd;
        for (int i = 0; i < L; ++i) {
            const int jmax = causal ? i : L - 1;
            float m = -1e30f;
            for (int j = 0; j <= jmax; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) {
                    dot += q16(qv[static_cast<size_t>(i) * D + off + d]) *
                           q16(kv[static_cast<size_t>(j) * D + off + d]);
                }
                scores[static_cast<size_t>(j)] = dot * scale;
                if (scores[static_cast<size_t>(j)] > m) m = scores[static_cast<size_t>(j)];
            }
            float sum = 0.0f;
            for (int j = 0; j <= jmax; ++j) {
                scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
                sum += scores[static_cast<size_t>(j)];
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int j = 0; j <= jmax; ++j) {
                    acc += scores[static_cast<size_t>(j)] *
                           q16(vv[static_cast<size_t>(j) * D + off + d]);
                }
                out[static_cast<size_t>(i) * D + off + d] = acc * inv;
            }
        }
    }
    return out;
}

void verify(int L, int nh, int hd) {
    const int D = nh * hd;
    auto qv = rand_vec(static_cast<size_t>(L) * D, 0x11u, 0.5f);
    auto kv = rand_vec(static_cast<size_t>(L) * D, 0x22u, 0.5f);
    auto vv = rand_vec(static_cast<size_t>(L) * D, 0x33u, 0.5f);
    Tensor Q = upload_fp16(qv, L, D);
    Tensor K = upload_fp16(kv, L, D);
    Tensor V = upload_fp16(vv, L, D);

    for (bool causal : {false, true}) {
        Tensor O;
        brotensor::flash_attention_forward(Q, K, V, nullptr, nh, causal, O);
        brotensor::sync_all();
        auto got = download_fp16(O);
        auto ref = attention_ref(qv, kv, vv, L, D, nh, causal);

        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double e = std::fabs(got[i] - ref[i]) / (std::fabs(ref[i]) + 1e-2);
            if (e > worst) worst = e;
        }
        const bool ok = worst < 3e-2;
        if (!ok) ++g_failures;
        std::printf("  verify L=%d nh=%d hd=%d %-11s worst rel err %.3g  %s\n",
                    L, nh, hd, causal ? "causal" : "non-causal", worst,
                    ok ? "ok" : "MISMATCH");
    }
}

void bench(int L, int nh, int hd) {
    const int D = nh * hd;
    auto qv = rand_vec(static_cast<size_t>(L) * D, 0x44u, 0.5f);
    auto kv = rand_vec(static_cast<size_t>(L) * D, 0x55u, 0.5f);
    auto vv = rand_vec(static_cast<size_t>(L) * D, 0x66u, 0.5f);
    Tensor Q = upload_fp16(qv, L, D);
    Tensor K = upload_fp16(kv, L, D);
    Tensor V = upload_fp16(vv, L, D);

    Tensor O;
    // Two GEMMs of (L x L x hd) per head, 2 flops each.
    const double full_flops = 4.0 * static_cast<double>(L) * L * hd * nh;

    brotensor::flash_attention_forward(Q, K, V, nullptr, nh, false, O);
    brotensor::sync_all();
    const float ms_nc = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_forward(Q, K, V, nullptr, nh, false, O);
    });

    brotensor::flash_attention_forward(Q, K, V, nullptr, nh, true, O);
    brotensor::sync_all();
    const float ms_c = bt_bench::time_min_ms([&] {
        brotensor::flash_attention_forward(Q, K, V, nullptr, nh, true, O);
    });

    // Causal skips roughly half the score matrix.
    const double causal_flops = full_flops * 0.5;

    std::printf("  L=%-6d nh=%-3d hd=%-4d  non-causal %8.3f ms (%8.1f GFLOP/s)"
                "   causal %8.3f ms (%8.1f GFLOP/s)   causal/non %5.1fx\n",
                L, nh, hd,
                ms_nc, full_flops / (ms_nc * 1e-3) / 1e9,
                ms_c,  causal_flops / (ms_c * 1e-3) / 1e9,
                ms_c / ms_nc);
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available — skipping causal attention bench\n");
        return 0;
    }
    // Pull the SM clock off its P8 idle floor before any timing.
    bt_bench::spin_up();
    std::printf("brotensor_bench_causal_attention  (warmup %.0f ms/op, best of %d)\n",
                bt_bench::kWarmupMs, bt_bench::kSamples);
    std::printf("non-causal takes the fused WMMA path; causal falls back to "
                "the scalar kernel.\n");
    std::printf("causal does ~half the arithmetic, so a healthy ratio is "
                "BELOW 1.0x.\n\n");

    std::printf("-- correctness (host FP32 reference) --\n");
    verify(256, 8, 64);
    verify(256, 4, 128);

    std::printf("\n-- prefill shapes --\n");
    // head_dim 64 and 128 are both instantiated in flash_fused::supported(),
    // so the non-causal row is the fused kernel in every case below.
    for (int L : {1024, 2048, 4096, 8192}) {
        bench(L, 16, 64);
    }
    std::printf("\n");
    for (int L : {1024, 2048, 4096}) {
        bench(L, 32, 128);   // 8B-class: 32 heads, head_dim 128
    }

    if (g_failures != 0) {
        std::printf("\n%d MISMATCH(es)\n", g_failures);
        return 1;
    }
    std::printf("\nall verification rows passed\n");
    return 0;
}
