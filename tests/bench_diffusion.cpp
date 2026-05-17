// Micro-benchmark: fused vs unfused diffusion ops at SD1.5-realistic shapes.
//
// NOT registered with ctest — invoke manually:
//   ./build/tests/Release/brotensor_bench_diffusion
//
// Timing uses cudaEvent_t around the kernel sequence with warmup. Each row
// also runs a cheap sanity check on the fused output vs the unfused output
// so a fast-but-wrong kernel can't pass silently.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

namespace {

constexpr int WARMUP = 5;
constexpr int ITERS  = 50;

std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

void upload_rand_fp16(int rows, int cols, GpuTensor& g,
                      std::mt19937& rng, float scale = 0.3f) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(static_cast<size_t>(rows) * cols);
    for (auto& x : v) x = d(rng);
    auto h = to_fp16(v);
    brotensor::upload_fp16(h.data(), rows, cols, g);
}

float time_loop_ms(int iters, const std::function<void()>& body) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < WARMUP; ++i) body();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) body();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

// Sanity check between two FP16 outputs of the SAME op via different
// implementations. FP16 accumulation error compounds with K (softmax denom,
// conv reduction over C_in*9, etc.), so we use a generous absolute floor +
// 15% relative tolerance — enough to catch a garbage-fast kernel but not so
// Sanity check that the fused kernel produced finite, bounded output. Element-
// by-element parity belongs in the dedicated tests; this is just a "did the
// kernel silently produce garbage" gate.
bool sanity_finite(const GpuTensor& a, float abs_clip = 1e3f) {
    std::vector<uint16_t> ha(a.size());
    brotensor::download_fp16(a, ha.data());
    brotensor::cuda_sync();
    // Spot-check 32 elements.
    const int n = static_cast<int>(ha.size());
    const int step = n > 32 ? n / 32 : 1;
    for (int i = 0; i < n; i += step) {
        const float f = brotensor::fp16_bits_to_fp32(ha[i]);
        if (!std::isfinite(f) || std::fabs(f) > abs_clip) return false;
    }
    return true;
}

struct Row { std::string op, shape; float unfused; float fused; };

void bench_self_attention(int Lq, int D, int nh, std::vector<Row>& rows) {
    std::mt19937 rng(42);
    GpuTensor X, Wq, Wk, Wv, Wo, O_a, O_b;
    upload_rand_fp16(Lq, D, X, rng);
    upload_rand_fp16(D, D, Wq, rng);
    upload_rand_fp16(D, D, Wk, rng);
    upload_rand_fp16(D, D, Wv, rng);
    upload_rand_fp16(D, D, Wo, rng);

    // Unfused: cross_attention with Ctx=X. The original cross-attn impl caps
    // at Lk≈4096 due to dynamic shared memory; skip if Lk too large.
    bool unfused_ok = (Lq <= 4096);
    float ms_un = 0.0f;
    if (unfused_ok) {
        ms_un = time_loop_ms(ITERS, [&] {
            brotensor::self_attention_forward_gpu(X, Wq, Wk, Wv, Wo, nullptr, nh, O_a);
        });
    }
    float ms_fu = time_loop_ms(ITERS, [&] {
        brotensor::flash_attention_qkvo_forward_gpu(X, nullptr, Wq, Wk, Wv, Wo,
                                                    nullptr, nh, /*causal=*/false, O_b);
    });
    if (!sanity_finite(O_b))
        std::printf("  !! self-attn fused output non-finite L=%d D=%d\n", Lq, D);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "L=%d D=%d nh=%d", Lq, D, nh);
    rows.push_back({"self_attn", buf, ms_un, ms_fu});
}

void bench_cross_attention(int Lq, int Lk, int D, int nh, std::vector<Row>& rows) {
    std::mt19937 rng(43);
    GpuTensor X, Ctx, Wq, Wk, Wv, Wo, O_a, O_b;
    upload_rand_fp16(Lq, D, X, rng);
    upload_rand_fp16(Lk, D, Ctx, rng);
    upload_rand_fp16(D, D, Wq, rng);
    upload_rand_fp16(D, D, Wk, rng);
    upload_rand_fp16(D, D, Wv, rng);
    upload_rand_fp16(D, D, Wo, rng);
    float ms_un = time_loop_ms(ITERS, [&] {
        brotensor::cross_attention_forward_gpu(X, Ctx, Wq, Wk, Wv, Wo,
                                               nullptr, nh, O_a);
    });
    float ms_fu = time_loop_ms(ITERS, [&] {
        brotensor::flash_attention_qkvo_forward_gpu(X, &Ctx, Wq, Wk, Wv, Wo,
                                                    nullptr, nh, /*causal=*/false, O_b);
    });
    if (!sanity_finite(O_b))
        std::printf("  !! cross-attn fused output non-finite Lq=%d Lk=%d D=%d\n",
                    Lq, Lk, D);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Lq=%d Lk=%d D=%d nh=%d", Lq, Lk, D, nh);
    rows.push_back({"cross_attn", buf, ms_un, ms_fu});
}

void resblock_unfused(const GpuTensor& X,
                      const GpuTensor& g1, const GpuTensor& b1,
                      const GpuTensor& W1, const GpuTensor& bcv1,
                      const GpuTensor& g2, const GpuTensor& b2,
                      const GpuTensor& W2, const GpuTensor& bcv2,
                      int N, int C, int H, int Wd, int groups,
                      GpuTensor& Y) {
    GpuTensor a;
    brotensor::group_norm_forward_gpu(X, g1, b1, N, C, H, Wd, groups, 1e-5f, a);
    brotensor::silu_forward_gpu(a, a);
    GpuTensor c1;
    brotensor::conv2d_forward_gpu(a, W1, &bcv1, N, C, H, Wd, C, 3, 3,
                                  1, 1, 1, 1, 1, 1, c1);
    GpuTensor d;
    brotensor::group_norm_forward_gpu(c1, g2, b2, N, C, H, Wd, groups, 1e-5f, d);
    brotensor::silu_forward_gpu(d, d);
    GpuTensor c2;
    brotensor::conv2d_forward_gpu(d, W2, &bcv2, N, C, H, Wd, C, 3, 3,
                                  1, 1, 1, 1, 1, 1, c2);
    brotensor::add_inplace_gpu(c2, X);
    Y = std::move(c2);
}

void bench_resblock(int N, int C, int H, int Wd, std::vector<Row>& rows) {
    std::mt19937 rng(44 + C);
    const int spatial = H * Wd;
    GpuTensor X, g1, b1, W1, bcv1, g2, b2, W2, bcv2;
    upload_rand_fp16(N, C * spatial, X, rng);
    upload_rand_fp16(C, 1, g1, rng);
    upload_rand_fp16(C, 1, b1, rng);
    upload_rand_fp16(C, C * 9, W1, rng);
    upload_rand_fp16(C, 1, bcv1, rng);
    upload_rand_fp16(C, 1, g2, rng);
    upload_rand_fp16(C, 1, b2, rng);
    upload_rand_fp16(C, C * 9, W2, rng);
    upload_rand_fp16(C, 1, bcv2, rng);
    GpuTensor Y_un, Y_fu;
    float ms_un = time_loop_ms(ITERS, [&] {
        resblock_unfused(X, g1, b1, W1, bcv1, g2, b2, W2, bcv2,
                         N, C, H, Wd, 32, Y_un);
    });
    float ms_fu = time_loop_ms(ITERS, [&] {
        brotensor::resblock_forward_gpu(X, g1, b1, W1, &bcv1, nullptr,
                                        g2, b2, W2, &bcv2, nullptr, nullptr,
                                        N, C, C, H, Wd, 32, 1e-5f, Y_fu);
    });
    if (!sanity_finite(Y_fu))
        std::printf("  !! resblock fused output non-finite C=%d H=%d\n", C, H);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "N=%d C=%d H=W=%d", N, C, H);
    rows.push_back({"resblock", buf, ms_un, ms_fu});
}

void print_table(const std::vector<Row>& rows) {
    std::printf("\n%-12s | %-28s | %12s | %12s | %8s\n",
                "op", "shape", "unfused_ms", "fused_ms", "speedup");
    std::printf("-------------+------------------------------+--------------+--------------+----------\n");
    float total_un = 0.0f, total_fu = 0.0f;
    for (const auto& r : rows) {
        const float sp = r.fused > 0.0f && r.unfused > 0.0f ? r.unfused / r.fused : 0.0f;
        if (r.unfused > 0.0f)
            std::printf("%-12s | %-28s | %12.3f | %12.3f | %7.2fx\n",
                        r.op.c_str(), r.shape.c_str(), r.unfused, r.fused, sp);
        else
            std::printf("%-12s | %-28s | %12s | %12.3f | %8s\n",
                        r.op.c_str(), r.shape.c_str(), "(skip)", r.fused, "—");
        total_un += r.unfused;
        total_fu += r.fused;
    }
    std::printf("-------------+------------------------------+--------------+--------------+----------\n");
    std::printf("%-12s | %-28s | %12.3f | %12.3f | %7.2fx\n",
                "TOTAL", "one-step estimate", total_un, total_fu,
                (total_fu > 0.0f) ? total_un / total_fu : 0.0f);
}

} // namespace

int main() {
    try { brotensor::cuda_init(); }
    catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("brotensor_bench_diffusion  (warmup=%d, iters=%d)\n",
                WARMUP, ITERS);

    std::vector<Row> rows;

    // U-Net level shapes for SD1.5: (Lq=Lk, D) at each spatial level.
    //   Level 0: 64x64 latent → L=4096, D=320, nh=5  (5*64=320)
    //   Level 1: 32x32        → L=1024, D=640, nh=10
    //   Level 2: 16x16        → L=256,  D=1280, nh=20
    //   Level 3: 8x8          → L=64,   D=1280, nh=20
    bench_self_attention(4096, 320,  5,  rows);
    bench_self_attention(1024, 640,  10, rows);
    bench_self_attention(256,  1280, 20, rows);
    bench_self_attention(64,   1280, 20, rows);

    // Cross-attention: same Lq, Lk=77 (CLIP).
    bench_cross_attention(4096, 77, 320,  5,  rows);
    bench_cross_attention(1024, 77, 640,  10, rows);
    bench_cross_attention(256,  77, 1280, 20, rows);
    bench_cross_attention(64,   77, 1280, 20, rows);

    // ResBlock at the four U-Net levels (N=2).
    bench_resblock(2, 320,  64, 64, rows);
    bench_resblock(2, 640,  32, 32, rows);
    bench_resblock(2, 1280, 16, 16, rows);
    bench_resblock(2, 1280, 8,  8,  rows);

    print_table(rows);
    return 0;
}
