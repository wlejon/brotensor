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

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void download_to_f(const Tensor& g, std::vector<float>& out) {
    std::vector<uint16_t> tmp(g.size());
    g.copy_to_host_fp16(tmp.data());
    brotensor::sync_all();
    out.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) out[i] = brotensor::fp16_bits_to_fp32(tmp[i]);
}

// Reference path: compose the resblock from existing primitives, all FP16.
static void compose_ref(const Tensor& X,
                        const Tensor& g1, const Tensor& b1_,
                        const Tensor& W1, const Tensor* bcv1,
                        const Tensor* t_emb,
                        const Tensor& g2, const Tensor& b2_,
                        const Tensor& W2, const Tensor* bcv2,
                        const Tensor* Wskip, const Tensor* bskip,
                        int N, int C_in, int C_out, int H, int Wd,
                        int num_groups, float eps,
                        Tensor& Y_out) {
    Tensor a;
    brotensor::group_norm_forward(X, g1, b1_, N, C_in, H, Wd, num_groups, eps, a);
    brotensor::silu_forward(a, a);
    Tensor c1;
    brotensor::conv2d_forward(a, W1, bcv1, N, C_in, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, c1);
    if (t_emb) {
        // Broadcast t_emb (N, C_out) or (C_out,) across HxW. We do this on
        // host because there's no broadcast-add primitive — only need to
        // produce a tensor of the same shape as c1.
        const int total = N * C_out * H * Wd;
        std::vector<uint16_t> emb_h = t_emb->to_host_vector_fp16();
        brotensor::sync_all();
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
        Tensor bg =
            Tensor::from_host_fp16_on(Device::CUDA, bh.data(), N, C_out*H*Wd);
        brotensor::add_inplace(c1, bg);
    }
    Tensor d;
    brotensor::group_norm_forward(c1, g2, b2_, N, C_out, H, Wd, num_groups, eps, d);
    brotensor::silu_forward(d, d);
    Tensor c2;
    brotensor::conv2d_forward(d, W2, bcv2, N, C_out, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, c2);
    // Skip path.
    if (Wskip == nullptr) {
        // c2 += X (same shape since C_in==C_out)
        brotensor::add_inplace(c2, X);
    } else {
        Tensor s;
        brotensor::conv2d_forward(X, *Wskip, bskip,
                                  N, C_in, H, Wd, C_out, 1, 1,
                                  1, 1, 0, 0, 1, 1, s);
        brotensor::add_inplace(c2, s);
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

    auto up = [&](const std::vector<float>& v, int r, int c, Tensor& g) {
        auto h = to_fp16(v);
        g = Tensor::from_host_fp16_on(Device::CUDA, h.data(), r, c);
    };
    Tensor Xg, g1g, b1g, W1g, bcv1g, g2g, b2g, W2g, bcv2g, Wskg, bskg, tembg;
    up(X, N, C_in*spatial, Xg);
    up(g1, C_in, 1, g1g); up(b1, C_in, 1, b1g);
    up(W1, C_out, C_in*9, W1g); up(bcv1, C_out, 1, bcv1g);
    up(g2, C_out, 1, g2g); up(b2, C_out, 1, b2g);
    up(W2, C_out, C_out*9, W2g); up(bcv2, C_out, 1, bcv2g);
    if (need_skip_conv) { up(Wsk, C_out, C_in, Wskg); up(bsk, C_out, 1, bskg); }
    if (with_temb) up(temb, N, C_out, tembg);

    Tensor Y_ref;
    compose_ref(Xg, g1g, b1g, W1g, &bcv1g,
                with_temb ? &tembg : nullptr,
                g2g, b2g, W2g, &bcv2g,
                need_skip_conv ? &Wskg : nullptr,
                need_skip_conv ? &bskg : nullptr,
                N, C_in, C_out, H, Wd, num_groups, 1e-5f, Y_ref);

    Tensor Y_fused;
    brotensor::resblock_forward(Xg, g1g, b1g, W1g, &bcv1g,
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

// ─── Backward parity ─────────────────────────────────────────────────────────
//
// Build the reference dX/d{gamma,beta,W,b}{1,2}/dt_emb/d{W,b}skip by
// composing the same forward DAG with public primitives and walking it
// backward step-by-step. Then call resblock_backward_gpu and verify every
// gradient matches within FP16 tolerance.

static void compose_ref_backward(
        const Tensor& X,
        const Tensor& g1, const Tensor& b1_,
        const Tensor& W1, const Tensor* bcv1,
        const Tensor* t_emb,
        const Tensor& g2, const Tensor& b2_,
        const Tensor& W2, const Tensor* bcv2,
        const Tensor* Wskip, const Tensor* bskip,
        int N, int C_in, int C_out, int H, int Wd,
        int num_groups, float eps,
        const Tensor& dY,
        Tensor& dX,
        Tensor& dGamma1, Tensor& dBeta1,
        Tensor& dW1, Tensor* db1,
        Tensor* dt_emb,
        Tensor& dGamma2, Tensor& dBeta2,
        Tensor& dW2, Tensor* db2,
        Tensor* dWskip, Tensor* dbskip) {
    const int spatial = H * Wd;

    // Forward recompute.
    Tensor h1_pre_silu, h1, h2_pre_t, h2, h3_pre_silu, h3;
    brotensor::group_norm_forward(X, g1, b1_, N, C_in, H, Wd, num_groups, eps, h1_pre_silu);
    brotensor::silu_forward(h1_pre_silu, h1);
    brotensor::conv2d_forward(h1, W1, bcv1, N, C_in, H, Wd,
                              C_out, 3, 3, 1, 1, 1, 1, 1, 1, h2_pre_t);
    // h2 = h2_pre_t + broadcast(t_emb)
    h2 = std::move(h2_pre_t);
    if (t_emb) {
        // Broadcast on host (same as forward test path).
        std::vector<uint16_t> emb_h = t_emb->to_host_vector_fp16();
        brotensor::sync_all();
        std::vector<float> emb_f(emb_h.size());
        for (size_t i = 0; i < emb_h.size(); ++i) emb_f[i] = brotensor::fp16_bits_to_fp32(emb_h[i]);
        const bool has_N = (t_emb->rows == N && t_emb->cols == C_out);
        std::vector<float> broadcast(N * C_out * spatial);
        for (int n = 0; n < N; ++n)
            for (int oc = 0; oc < C_out; ++oc) {
                const float v = has_N ? emb_f[n*C_out+oc] : emb_f[oc];
                for (int p = 0; p < spatial; ++p)
                    broadcast[(n*C_out+oc)*spatial + p] = v;
            }
        auto bh = to_fp16(broadcast);
        Tensor bg =
            Tensor::from_host_fp16_on(Device::CUDA, bh.data(), N, C_out*spatial);
        brotensor::add_inplace(h2, bg);
    }
    brotensor::group_norm_forward(h2, g2, b2_, N, C_out, H, Wd, num_groups, eps, h3_pre_silu);
    brotensor::silu_forward(h3_pre_silu, h3);

    // Backward.
    // Conv2.
    Tensor dh3 = Tensor::zeros_on(Device::CUDA, N, C_out * spatial, brotensor::Dtype::FP16);
    brotensor::conv2d_backward_input(W2, dY, N, C_out, H, Wd,
                                     C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh3);
    brotensor::conv2d_backward_weight(h3, dY, N, C_out, H, Wd,
                                      C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW2);
    if (db2) {
        brotensor::conv2d_backward_bias(dY, N, C_out, H, Wd, *db2);
    }
    // SiLU2.
    Tensor dh3_pre;
    brotensor::silu_backward(h3_pre_silu, dh3, dh3_pre);
    // GN2.
    Tensor dh2;
    brotensor::group_norm_backward(h2, g2, dh3_pre, N, C_out, H, Wd,
                                   num_groups, eps, dh2, dGamma2, dBeta2);

    // dt_emb: per-(n,c) HW sum (or per-c NHW sum).
    if (t_emb && dt_emb) {
        std::vector<uint16_t> dh2_h = dh2.to_host_vector_fp16();
        brotensor::sync_all();
        std::vector<float> dh2_f(dh2_h.size());
        for (size_t i = 0; i < dh2_h.size(); ++i) dh2_f[i] = brotensor::fp16_bits_to_fp32(dh2_h[i]);
        const bool has_N = (t_emb->rows == N && t_emb->cols == C_out);
        // Download current dt_emb (it's accumulated into).
        std::vector<uint16_t> cur_h = dt_emb->to_host_vector_fp16();
        brotensor::sync_all();
        std::vector<float> cur_f(cur_h.size());
        for (size_t i = 0; i < cur_h.size(); ++i) cur_f[i] = brotensor::fp16_bits_to_fp32(cur_h[i]);
        if (has_N) {
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C_out; ++c) {
                    float acc = 0.0f;
                    for (int p = 0; p < spatial; ++p) acc += dh2_f[(n*C_out+c)*spatial + p];
                    cur_f[n*C_out + c] += acc;
                }
        } else {
            for (int c = 0; c < C_out; ++c) {
                float acc = 0.0f;
                for (int n = 0; n < N; ++n)
                    for (int p = 0; p < spatial; ++p)
                        acc += dh2_f[(n*C_out+c)*spatial + p];
                cur_f[c] += acc;
            }
        }
        auto out_h = to_fp16(cur_f);
        *dt_emb = Tensor::from_host_fp16_on(Device::CUDA, out_h.data(),
                                            dt_emb->rows, dt_emb->cols);
    }

    // Conv1.
    Tensor dh1 = Tensor::zeros_on(Device::CUDA, N, C_in * spatial, brotensor::Dtype::FP16);
    brotensor::conv2d_backward_input(W1, dh2, N, C_in, H, Wd,
                                     C_out, 3, 3, 1, 1, 1, 1, 1, 1, dh1);
    brotensor::conv2d_backward_weight(h1, dh2, N, C_in, H, Wd,
                                      C_out, 3, 3, 1, 1, 1, 1, 1, 1, dW1);
    if (db1) {
        brotensor::conv2d_backward_bias(dh2, N, C_out, H, Wd, *db1);
    }
    // SiLU1.
    Tensor dh1_pre;
    brotensor::silu_backward(h1_pre_silu, dh1, dh1_pre);
    // GN1 → dX.
    brotensor::group_norm_backward(X, g1, dh1_pre, N, C_in, H, Wd,
                                   num_groups, eps, dX, dGamma1, dBeta1);

    // Skip.
    if (Wskip == nullptr) {
        brotensor::add_inplace(dX, dY);
    } else {
        Tensor dX_skip = Tensor::zeros_on(Device::CUDA, N, C_in * spatial, brotensor::Dtype::FP16);
        brotensor::conv2d_backward_input(*Wskip, dY, N, C_in, H, Wd,
                                         C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX_skip);
        if (dWskip) {
            brotensor::conv2d_backward_weight(X, dY, N, C_in, H, Wd,
                                              C_out, 1, 1, 1, 1, 0, 0, 1, 1, *dWskip);
        }
        if (dbskip) {
            brotensor::conv2d_backward_bias(dY, N, C_out, H, Wd, *dbskip);
        }
        brotensor::add_inplace(dX, dX_skip);
    }
}

static void compare_fp16(const Tensor& got, const Tensor& ref,
                         const char* label, float atol = 1e-2f, float rtol = 1e-2f) {
    CHECK(got.size() == ref.size());
    std::vector<float> g, r;
    download_to_f(got, g);
    download_to_f(ref, r);
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < r.size(); ++i) {
        const float e = std::fabs(g[i] - r[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(r[i])) {
            if (bad < 3) std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                                     label, i, g[i], r[i], e);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, r.size());
    CHECK(bad == 0);
}

static void zero_fp16_like(Tensor& g, int rows, int cols) {
    g = Tensor::zeros_on(Device::CUDA, rows, cols, Dtype::FP16);
}

static void run_backward_one(const char* label,
                             int N, int C_in, int C_out, int H, int Wd,
                             int num_groups,
                             bool with_temb, bool need_skip_conv) {
    std::printf("  %s  N=%d Cin=%d Cout=%d H=%d W=%d groups=%d temb=%d skip_conv=%d\n",
                label, N, C_in, C_out, H, Wd, num_groups, (int)with_temb, (int)need_skip_conv);
    std::mt19937 rng(0xBEEF ^ (C_in*131 + C_out));
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    auto rand = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };
    const int spatial = H * Wd;
    auto X = rand(N * C_in * spatial);
    auto g1 = rand(C_in), b1v = rand(C_in);
    auto W1 = rand(C_out * C_in * 9);
    auto bcv1 = rand(C_out);
    auto g2 = rand(C_out), b2v = rand(C_out);
    auto W2 = rand(C_out * C_out * 9);
    auto bcv2 = rand(C_out);
    auto Wsk = need_skip_conv ? rand(C_out * C_in) : std::vector<float>{};
    auto bsk = need_skip_conv ? rand(C_out) : std::vector<float>{};
    auto temb = with_temb ? rand(N * C_out) : std::vector<float>{};
    auto dY = rand(N * C_out * spatial);

    auto up = [&](const std::vector<float>& v, int r, int c, Tensor& g) {
        auto h = to_fp16(v);
        g = Tensor::from_host_fp16_on(Device::CUDA, h.data(), r, c);
    };
    Tensor Xg, g1g, b1g, W1g, bcv1g, g2g, b2g, W2g, bcv2g, Wskg, bskg, tembg, dYg;
    up(X, N, C_in*spatial, Xg);
    up(g1, C_in, 1, g1g); up(b1v, C_in, 1, b1g);
    up(W1, C_out, C_in*9, W1g); up(bcv1, C_out, 1, bcv1g);
    up(g2, C_out, 1, g2g); up(b2v, C_out, 1, b2g);
    up(W2, C_out, C_out*9, W2g); up(bcv2, C_out, 1, bcv2g);
    if (need_skip_conv) { up(Wsk, C_out, C_in, Wskg); up(bsk, C_out, 1, bskg); }
    if (with_temb) up(temb, N, C_out, tembg);
    up(dY, N, C_out*spatial, dYg);

    // Reference grads (caller-zeroed; backward accumulates).
    Tensor dX_ref, dG1_ref, dB1_ref, dW1_ref, db1_ref, dt_ref;
    Tensor dG2_ref, dB2_ref, dW2_ref, db2_ref, dWsk_ref, dbsk_ref;
    zero_fp16_like(dG1_ref, C_in, 1);   zero_fp16_like(dB1_ref, C_in, 1);
    zero_fp16_like(dW1_ref, C_out, C_in * 9);
    zero_fp16_like(db1_ref, C_out, 1);
    if (with_temb) zero_fp16_like(dt_ref, N, C_out);
    zero_fp16_like(dG2_ref, C_out, 1);  zero_fp16_like(dB2_ref, C_out, 1);
    zero_fp16_like(dW2_ref, C_out, C_out * 9);
    zero_fp16_like(db2_ref, C_out, 1);
    if (need_skip_conv) {
        zero_fp16_like(dWsk_ref, C_out, C_in);
        zero_fp16_like(dbsk_ref, C_out, 1);
    }
    compose_ref_backward(Xg, g1g, b1g, W1g, &bcv1g,
                         with_temb ? &tembg : nullptr,
                         g2g, b2g, W2g, &bcv2g,
                         need_skip_conv ? &Wskg : nullptr,
                         need_skip_conv ? &bskg : nullptr,
                         N, C_in, C_out, H, Wd, num_groups, 1e-5f,
                         dYg,
                         dX_ref,
                         dG1_ref, dB1_ref, dW1_ref, &db1_ref,
                         with_temb ? &dt_ref : nullptr,
                         dG2_ref, dB2_ref, dW2_ref, &db2_ref,
                         need_skip_conv ? &dWsk_ref : nullptr,
                         need_skip_conv ? &dbsk_ref : nullptr);

    // Tested grads.
    Tensor dX_got, dG1_got, dB1_got, dW1_got, db1_got, dt_got;
    Tensor dG2_got, dB2_got, dW2_got, db2_got, dWsk_got, dbsk_got;
    zero_fp16_like(dG1_got, C_in, 1);   zero_fp16_like(dB1_got, C_in, 1);
    zero_fp16_like(dW1_got, C_out, C_in * 9);
    zero_fp16_like(db1_got, C_out, 1);
    if (with_temb) zero_fp16_like(dt_got, N, C_out);
    zero_fp16_like(dG2_got, C_out, 1);  zero_fp16_like(dB2_got, C_out, 1);
    zero_fp16_like(dW2_got, C_out, C_out * 9);
    zero_fp16_like(db2_got, C_out, 1);
    if (need_skip_conv) {
        zero_fp16_like(dWsk_got, C_out, C_in);
        zero_fp16_like(dbsk_got, C_out, 1);
    }
    brotensor::resblock_backward(
        Xg, g1g, b1g, W1g, &bcv1g,
        with_temb ? &tembg : nullptr,
        g2g, b2g, W2g, &bcv2g,
        need_skip_conv ? &Wskg : nullptr,
        need_skip_conv ? &bskg : nullptr,
        N, C_in, C_out, H, Wd, num_groups, 1e-5f,
        dYg,
        dX_got,
        dG1_got, dB1_got, dW1_got, &db1_got,
        with_temb ? &dt_got : nullptr,
        dG2_got, dB2_got, dW2_got, &db2_got,
        need_skip_conv ? &dWsk_got : nullptr,
        need_skip_conv ? &dbsk_got : nullptr);

    compare_fp16(dX_got,  dX_ref,  "dX");
    compare_fp16(dG1_got, dG1_ref, "dGamma1");
    compare_fp16(dB1_got, dB1_ref, "dBeta1");
    compare_fp16(dW1_got, dW1_ref, "dW1");
    compare_fp16(db1_got, db1_ref, "db1");
    if (with_temb)       compare_fp16(dt_got,  dt_ref,  "dt_emb_shift");
    compare_fp16(dG2_got, dG2_ref, "dGamma2");
    compare_fp16(dB2_got, dB2_ref, "dBeta2");
    compare_fp16(dW2_got, dW2_ref, "dW2");
    compare_fp16(db2_got, db2_ref, "db2");
    if (need_skip_conv) {
        compare_fp16(dWsk_got, dWsk_ref, "dWskip");
        compare_fp16(dbsk_got, dbsk_ref, "dbskip");
    }
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_resblock\n");

    // Cin == Cout (identity skip).
    run_one("same-channels-no-temb",   1, 32, 32, 8,  8,  4, false, false);
    run_one("same-channels-with-temb", 2, 32, 32, 8,  8,  4, true,  false);
    // Cin != Cout (1x1 skip conv).
    run_one("up-channels",             1, 32, 64, 8,  8,  4, true,  true);

    std::printf("\nBackward parity:\n");
    run_backward_one("bwd-same-channels", 1, 8, 8, 8, 8, 4, true,  false);
    run_backward_one("bwd-up-channels",   1, 8, 16, 4, 4, 4, true,  true);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll resblock checks passed.\n");
    return 0;
}
