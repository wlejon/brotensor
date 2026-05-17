// Parity for resblock_forward_gpu against the same block composed from
// existing primitives (group_norm + silu + conv2d + add + skip 1x1 conv).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void download_to_f(const GpuTensor& g, std::vector<float>& out) {
    std::vector<uint16_t> tmp(g.size());
    brotensor::download_fp16(g, tmp.data());
    brotensor::cuda_sync();
    out.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) out[i] = brotensor::fp16_bits_to_fp32(tmp[i]);
}

// Reference path: compose the resblock from existing primitives, all FP16.
static void compose_ref(const GpuTensor& X,
                        const GpuTensor& g1, const GpuTensor& b1_,
                        const GpuTensor& W1, const GpuTensor* bcv1,
                        const GpuTensor* t_emb,
                        const GpuTensor& g2, const GpuTensor& b2_,
                        const GpuTensor& W2, const GpuTensor* bcv2,
                        const GpuTensor* Wskip, const GpuTensor* bskip,
                        int N, int C_in, int C_out, int H, int Wd,
                        int num_groups, float eps,
                        GpuTensor& Y_out) {
    GpuTensor a;
    brotensor::group_norm_forward_gpu(X, g1, b1_, N, C_in, H, Wd, num_groups, eps, a);
    brotensor::silu_forward_gpu(a, a);
    GpuTensor c1;
    brotensor::conv2d_forward_gpu(a, W1, bcv1, N, C_in, H, Wd,
                                  C_out, 3, 3, 1, 1, 1, 1, 1, 1, c1);
    if (t_emb) {
        // Broadcast t_emb (N, C_out) or (C_out,) across HxW. We do this on
        // host because there's no broadcast-add primitive — only need to
        // produce a tensor of the same shape as c1.
        const int total = N * C_out * H * Wd;
        std::vector<uint16_t> emb_h(t_emb->size());
        brotensor::download_fp16(*t_emb, emb_h.data());
        brotensor::cuda_sync();
        std::vector<float> emb_f(emb_h.size());
        for (size_t i = 0; i < emb_h.size(); ++i) emb_f[i] = brotensor::fp16_bits_to_fp32(emb_h[i]);
        const bool has_N = (t_emb->rows == N && t_emb->cols == C_out);
        std::vector<float> broadcast(total);
        for (int n = 0; n < N; ++n)
            for (int oc = 0; oc < C_out; ++oc) {
                const float v = has_N ? emb_f[n*C_out+oc] : emb_f[oc];
                for (int p = 0; p < H*Wd; ++p)
                    broadcast[(n*C_out+oc)*H*Wd + p] = v;
            }
        auto bh = to_fp16(broadcast);
        GpuTensor bg;
        brotensor::upload_fp16(bh.data(), N, C_out*H*Wd, bg);
        brotensor::add_inplace_gpu(c1, bg);
    }
    GpuTensor d;
    brotensor::group_norm_forward_gpu(c1, g2, b2_, N, C_out, H, Wd, num_groups, eps, d);
    brotensor::silu_forward_gpu(d, d);
    GpuTensor c2;
    brotensor::conv2d_forward_gpu(d, W2, bcv2, N, C_out, H, Wd,
                                  C_out, 3, 3, 1, 1, 1, 1, 1, 1, c2);
    // Skip path.
    if (Wskip == nullptr) {
        // c2 += X (same shape since C_in==C_out)
        brotensor::add_inplace_gpu(c2, X);
    } else {
        GpuTensor s;
        brotensor::conv2d_forward_gpu(X, *Wskip, bskip,
                                      N, C_in, H, Wd, C_out, 1, 1,
                                      1, 1, 0, 0, 1, 1, s);
        brotensor::add_inplace_gpu(c2, s);
    }
    // Move c2 into Y_out.
    Y_out = std::move(c2);
}

static void run_one(const char* label,
                    int N, int C_in, int C_out, int H, int Wd, int num_groups,
                    bool with_temb, bool need_skip_conv) {
    std::printf("  %s  N=%d Cin=%d Cout=%d H=%d W=%d groups=%d temb=%d skip_conv=%d\n",
                label, N, C_in, C_out, H, Wd, num_groups, (int)with_temb, (int)need_skip_conv);
    std::mt19937 rng(0xABCD ^ (C_in*131 + C_out));
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    auto rand = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };
    const int spatial = H * Wd;
    auto X = rand(N * C_in * spatial);
    auto g1 = rand(C_in), b1 = rand(C_in);
    auto W1 = rand(C_out * C_in * 9);
    auto bcv1 = rand(C_out);
    auto g2 = rand(C_out), b2 = rand(C_out);
    auto W2 = rand(C_out * C_out * 9);
    auto bcv2 = rand(C_out);
    auto Wsk = need_skip_conv ? rand(C_out * C_in) : std::vector<float>{};
    auto bsk = need_skip_conv ? rand(C_out) : std::vector<float>{};
    auto temb = with_temb ? rand(N * C_out) : std::vector<float>{};

    auto up = [&](const std::vector<float>& v, int r, int c, GpuTensor& g) {
        auto h = to_fp16(v);
        brotensor::upload_fp16(h.data(), r, c, g);
    };
    GpuTensor Xg, g1g, b1g, W1g, bcv1g, g2g, b2g, W2g, bcv2g, Wskg, bskg, tembg;
    up(X, N, C_in*spatial, Xg);
    up(g1, C_in, 1, g1g); up(b1, C_in, 1, b1g);
    up(W1, C_out, C_in*9, W1g); up(bcv1, C_out, 1, bcv1g);
    up(g2, C_out, 1, g2g); up(b2, C_out, 1, b2g);
    up(W2, C_out, C_out*9, W2g); up(bcv2, C_out, 1, bcv2g);
    if (need_skip_conv) { up(Wsk, C_out, C_in, Wskg); up(bsk, C_out, 1, bskg); }
    if (with_temb) up(temb, N, C_out, tembg);

    GpuTensor Y_ref;
    compose_ref(Xg, g1g, b1g, W1g, &bcv1g,
                with_temb ? &tembg : nullptr,
                g2g, b2g, W2g, &bcv2g,
                need_skip_conv ? &Wskg : nullptr,
                need_skip_conv ? &bskg : nullptr,
                N, C_in, C_out, H, Wd, num_groups, 1e-5f, Y_ref);

    GpuTensor Y_fused;
    brotensor::resblock_forward_gpu(Xg, g1g, b1g, W1g, &bcv1g,
                                    with_temb ? &tembg : nullptr,
                                    g2g, b2g, W2g, &bcv2g,
                                    need_skip_conv ? &Wskg : nullptr,
                                    need_skip_conv ? &bskg : nullptr,
                                    N, C_in, C_out, H, Wd, num_groups, 1e-5f, Y_fused);

    CHECK(Y_fused.rows == N && Y_fused.cols == C_out*spatial && Y_fused.dtype == Dtype::FP16);
    std::vector<float> ref, fused;
    download_to_f(Y_ref, ref);
    download_to_f(Y_fused, fused);
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(fused[i] - ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-2f + 1e-2f * std::fabs(ref[i])) {
            if (bad < 3) std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n", i, fused[i], ref[i], e);
            ++bad;
        }
    }
    std::printf("    max_err=%g bad=%d / %zu\n", max_err, bad, ref.size());
    CHECK(bad == 0);
}

int main() {
    try { brotensor::cuda_init(); }
    catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_resblock\n");

    // Cin == Cout (identity skip).
    run_one("same-channels-no-temb",   1, 32, 32, 8,  8,  4, false, false);
    run_one("same-channels-with-temb", 2, 32, 32, 8,  8,  4, true,  false);
    // Cin != Cout (1x1 skip conv).
    run_one("up-channels",             1, 32, 64, 8,  8,  4, true,  true);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll resblock checks passed.\n");
    return 0;
}
