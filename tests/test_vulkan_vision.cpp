// Vulkan parity for the chunk-6 vision and attention ops against the CPU
// backend: SAM's decomposed-rel-pos attention (global and windowed, padded
// windows included, with and without projection biases), deform_conv2d
// (groups, deform groups, stride, dilation, with and without the modulator
// mask and bias), StyleGAN3's modulated_conv2d (with and without
// demodulation) and its backward, conv2d_backward_weight, the single-head attention_forward (with its caches, masked
// and unmasked) and cross_attention_forward_train. FP32 / FP16 / BF16; the
// 16-bit tolerances cover intermediates kept in the activations' dtype.

#include "test_vulkan_common.h"

#include <cmath>
#include <string>
#include <vector>

namespace vkt {

namespace {

const Dtype kFloatTypes[] = {Dtype::FP32, Dtype::FP16, Dtype::BF16};

Tensor cpu_tensor(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

float tol16(Dtype dt, float f32, float f16, float bf16) {
    return dt == Dtype::FP32 ? f32 : dt == Dtype::FP16 ? f16 : bf16;
}

std::string tg(const char* op, Dtype dt, const std::string& extra = "") {
    return std::string(op) + " " + dt_name(dt) + (extra.empty() ? "" : " " + extra);
}

struct Pair {
    Tensor c, g;
};

Pair make(std::size_t n, int rows, int cols, std::uint64_t seed, float lo, float hi, Dtype dt) {
    const auto v = random_values(n, seed, lo, hi, dt);
    return {cpu_tensor(v, rows, cols), upload(v, rows, cols, dt)};
}

void test_sam_attention() {
    std::printf("self_attention_decomposed_rel_pos (global, windowed)\n");
    struct S { int gh, gw, D, H, window; bool bias; };
    const S shapes[] = {{6, 7, 48, 4, 0, true}, {5, 5, 32, 2, 0, false}, {9, 10, 48, 3, 4, true},
                        {8, 8, 64, 4, 4, true}, {7, 9, 32, 4, 5, false}};
    std::uint64_t seed = 100;
    for (Dtype dt : kFloatTypes) {
        for (const S& s : shapes) {
            const int L = s.gh * s.gw, dh = s.D / s.H;
            const int rg_h = s.window ? s.window : s.gh, rg_w = s.window ? s.window : s.gw;
            const float ws = 1.0f / std::sqrt(float(s.D));
            Pair X = make(std::size_t(L) * s.D, L, s.D, seed++, -1, 1, dt);
            Pair Wq = make(std::size_t(s.D) * s.D, s.D, s.D, seed++, -ws, ws, dt);
            Pair Wk = make(std::size_t(s.D) * s.D, s.D, s.D, seed++, -ws, ws, dt);
            Pair Wv = make(std::size_t(s.D) * s.D, s.D, s.D, seed++, -ws, ws, dt);
            Pair Wo = make(std::size_t(s.D) * s.D, s.D, s.D, seed++, -ws, ws, dt);
            Pair bq = make(s.D, s.D, 1, seed++, -0.2f, 0.2f, dt), bk = make(s.D, s.D, 1, seed++, -0.2f, 0.2f, dt);
            Pair bv = make(s.D, s.D, 1, seed++, -0.2f, 0.2f, dt), bo = make(s.D, s.D, 1, seed++, -0.2f, 0.2f, dt);
            Pair rh = make(std::size_t(2 * rg_h - 1) * dh, 2 * rg_h - 1, dh, seed++, -0.5f, 0.5f, dt);
            Pair rw = make(std::size_t(2 * rg_w - 1) * dh, 2 * rg_w - 1, dh, seed++, -0.5f, 0.5f, dt);
            const float scale = 1.0f / std::sqrt(float(dh));
            Tensor oc, og;
            auto b = [&](Pair& p, bool gpu) -> const Tensor* { return s.bias ? (gpu ? &p.g : &p.c) : nullptr; };
            if (s.window) {
                brotensor::self_attention_decomposed_rel_pos_windowed_forward(
                    X.c, Wq.c, b(bq, false), Wk.c, b(bk, false), Wv.c, b(bv, false), Wo.c, b(bo, false), rh.c, rw.c,
                    s.H, s.gh, s.gw, s.window, scale, oc);
                brotensor::self_attention_decomposed_rel_pos_windowed_forward(
                    X.g, Wq.g, b(bq, true), Wk.g, b(bk, true), Wv.g, b(bv, true), Wo.g, b(bo, true), rh.g, rw.g, s.H,
                    s.gh, s.gw, s.window, scale, og);
            } else {
                brotensor::self_attention_decomposed_rel_pos_forward(
                    X.c, Wq.c, b(bq, false), Wk.c, b(bk, false), Wv.c, b(bv, false), Wo.c, b(bo, false), rh.c, rw.c,
                    s.H, s.gh, s.gw, scale, oc);
                brotensor::self_attention_decomposed_rel_pos_forward(X.g, Wq.g, b(bq, true), Wk.g, b(bk, true), Wv.g,
                                                                     b(bv, true), Wo.g, b(bo, true), rh.g, rw.g, s.H,
                                                                     s.gh, s.gw, scale, og);
            }
            VKT_CHECK(og.dtype == dt && og.rows == L && og.cols == s.D);
            const float t = tol16(dt, 2e-5f, 6e-3f, 4e-2f);
            expect_close(download(og), oc.to_host_vector(), t, t,
                         tg("decomposed_rel_pos", dt, std::to_string(s.gh) + "x" + std::to_string(s.gw) +
                                                         (s.window ? " window " + std::to_string(s.window) : " global") +
                                                         (s.bias ? " biases" : "")));
        }
    }
}

void test_deform_conv() {
    std::printf("deform_conv2d_forward\n");
    struct S { int n, cin, h, w, cout, k, stride, pad, dil, groups, dg; bool mask, bias; };
    const S shapes[] = {{2, 8, 9, 10, 12, 3, 1, 1, 1, 1, 1, true, true}, {1, 8, 11, 9, 16, 3, 2, 1, 1, 2, 2, true, false},
                        {1, 6, 8, 8, 6, 3, 1, 2, 2, 1, 3, false, true}, {2, 32, 12, 12, 32, 3, 1, 1, 1, 1, 4, true, true}};
    std::uint64_t seed = 200;
    for (Dtype dt : kFloatTypes) {
        for (const S& s : shapes) {
            const int Ho = (s.h + 2 * s.pad - s.dil * (s.k - 1) - 1) / s.stride + 1;
            const int Wo = (s.w + 2 * s.pad - s.dil * (s.k - 1) - 1) / s.stride + 1;
            const int kk = s.k * s.k, P = Ho * Wo;
            Pair X = make(std::size_t(s.n) * s.cin * s.h * s.w, s.n, s.cin * s.h * s.w, seed++, -1, 1, dt);
            Pair off = make(std::size_t(s.n) * s.dg * 2 * kk * P, s.n, s.dg * 2 * kk * P, seed++, -2.5f, 2.5f, dt);
            Pair msk = make(std::size_t(s.n) * s.dg * kk * P, s.n, s.dg * kk * P, seed++, 0, 1, dt);
            const int K = s.cin / s.groups * kk;
            Pair W = make(std::size_t(s.cout) * K, s.cout, K, seed++, -0.3f, 0.3f, dt);
            Pair B = make(s.cout, s.cout, 1, seed++, -0.5f, 0.5f, dt);
            Tensor yc, yg;
            brotensor::deform_conv2d_forward(X.c, off.c, s.mask ? &msk.c : nullptr, W.c, s.bias ? &B.c : nullptr, s.n,
                                             s.cin, s.h, s.w, s.cout, s.k, s.k, s.stride, s.stride, s.pad, s.pad, s.dil,
                                             s.dil, s.groups, s.dg, yc);
            brotensor::deform_conv2d_forward(X.g, off.g, s.mask ? &msk.g : nullptr, W.g, s.bias ? &B.g : nullptr, s.n,
                                             s.cin, s.h, s.w, s.cout, s.k, s.k, s.stride, s.stride, s.pad, s.pad, s.dil,
                                             s.dil, s.groups, s.dg, yg);
            const float t = tol16(dt, 1e-5f, 8e-3f, 5e-2f);
            expect_close(download(yg), yc.to_host_vector(), t, t,
                         tg("deform_conv2d_forward", dt, "cin " + std::to_string(s.cin) + " g" +
                                                             std::to_string(s.groups) + " dg" + std::to_string(s.dg) +
                                                             " s" + std::to_string(s.stride) + " d" + std::to_string(s.dil)));
        }
    }
}

void test_modulated_conv() {
    std::printf("modulated_conv2d_forward\n");
    struct S { int n, cin, h, w, cout, k, pad; };
    const S shapes[] = {{2, 6, 7, 8, 20, 3, 1}, {3, 32, 10, 9, 32, 1, 0}, {1, 16, 6, 6, 24, 3, 2}};
    std::uint64_t seed = 300;
    for (Dtype dt : kFloatTypes) {
        for (const S& s : shapes) {
            for (bool demod : {true, false}) {
                const int K = s.cin * s.k * s.k;
                Pair X = make(std::size_t(s.n) * s.cin * s.h * s.w, s.n, s.cin * s.h * s.w, seed++, -1, 1, dt);
                Pair W = make(std::size_t(s.cout) * K, s.cout, K, seed++, -0.5f, 0.5f, dt);
                Pair sv = make(std::size_t(s.n) * s.cin, s.n, s.cin, seed++, 0.2f, 1.5f, dt);
                Tensor dc, yc, dg, yg;
                brotensor::modulated_conv2d_forward(X.c, W.c, sv.c, s.n, s.cin, s.h, s.w, s.cout, s.k, s.k, s.pad,
                                                    s.pad, demod, 1e-8f, dc, yc);
                brotensor::modulated_conv2d_forward(X.g, W.g, sv.g, s.n, s.cin, s.h, s.w, s.cout, s.k, s.k, s.pad,
                                                    s.pad, demod, 1e-8f, dg, yg);
                const std::string tag = "cin " + std::to_string(s.cin) + " k" + std::to_string(s.k) +
                                        (demod ? " demod" : "");
                expect_close(dg.to_host_vector(), dc.to_host_vector(), 1e-6f, 1e-5f,
                             tg("modulated_conv2d_forward dcoef", dt, tag));
                const float t = tol16(dt, 2e-5f, 1e-2f, 6e-2f);
                const auto want = yc.to_host_vector();
                double energy = 0.0;
                for (float v : want) energy += double(v) * v;
                VKT_CHECK(energy > 1.0);   // a non-trivial reference
                expect_close(download(yg), want, t, t, tg("modulated_conv2d_forward Y", dt, tag));
            }
        }
    }
}

void test_conv_weight_grads() {
    std::printf("conv2d_backward_weight, modulated_conv2d_backward\n");
    struct S { int n, cin, h, w, cout, k, stride, pad, dil, groups; };
    const S shapes[] = {{2, 8, 9, 10, 12, 3, 1, 1, 1, 1}, {1, 8, 11, 9, 16, 3, 2, 1, 1, 2},
                        {2, 6, 8, 8, 6, 3, 1, 2, 2, 3}, {1, 32, 12, 12, 32, 1, 1, 0, 1, 1}};
    std::uint64_t seed = 500;
    for (Dtype dt : kFloatTypes) {
        for (const S& s : shapes) {
            const int Ho = (s.h + 2 * s.pad - s.dil * (s.k - 1) - 1) / s.stride + 1;
            const int Wo = (s.w + 2 * s.pad - s.dil * (s.k - 1) - 1) / s.stride + 1;
            const int K = s.cin / s.groups * s.k * s.k;
            Pair X = make(std::size_t(s.n) * s.cin * s.h * s.w, s.n, s.cin * s.h * s.w, seed++, -1, 1, dt);
            Pair dY = make(std::size_t(s.n) * s.cout * Ho * Wo, s.n, s.cout * Ho * Wo, seed++, -1, 1, dt);
            Pair W0 = make(std::size_t(s.cout) * K, s.cout, K, seed++, -0.5f, 0.5f, dt);   // accumulated onto
            brotensor::conv2d_backward_weight(X.c, dY.c, s.n, s.cin, s.h, s.w, s.cout, s.k, s.k, s.stride, s.stride,
                                              s.pad, s.pad, s.dil, s.dil, s.groups, W0.c);
            brotensor::conv2d_backward_weight(X.g, dY.g, s.n, s.cin, s.h, s.w, s.cout, s.k, s.k, s.stride, s.stride,
                                              s.pad, s.pad, s.dil, s.dil, s.groups, W0.g);
            const float t = tol16(dt, 2e-5f, 3e-2f, 2e-1f);
            expect_close(download(W0.g), W0.c.to_host_vector(), t, tol16(dt, 1e-5f, 4e-3f, 3e-2f),
                         tg("conv2d_backward_weight", dt, "cin " + std::to_string(s.cin) + " g" +
                                                              std::to_string(s.groups) + " s" + std::to_string(s.stride) +
                                                              " d" + std::to_string(s.dil)));
        }
        struct M { int n, cin, h, w, cout, k, pad; };
        for (const M& m : {M{2, 6, 7, 8, 20, 3, 1}, M{3, 16, 6, 5, 16, 1, 0}}) {
            for (bool demod : {true, false}) {
                const int K = m.cin * m.k * m.k, Ho = m.h + 2 * m.pad - (m.k - 1), Wo = m.w + 2 * m.pad - (m.k - 1);
                Pair X = make(std::size_t(m.n) * m.cin * m.h * m.w, m.n, m.cin * m.h * m.w, seed++, -1, 1, dt);
                Pair W = make(std::size_t(m.cout) * K, m.cout, K, seed++, -0.5f, 0.5f, dt);
                Pair sv = make(std::size_t(m.n) * m.cin, m.n, m.cin, seed++, 0.2f, 1.5f, dt);
                Pair dY = make(std::size_t(m.n) * m.cout * Ho * Wo, m.n, m.cout * Ho * Wo, seed++, -1, 1, dt);
                Tensor dcc, yc, dcg, yg;
                brotensor::modulated_conv2d_forward(X.c, W.c, sv.c, m.n, m.cin, m.h, m.w, m.cout, m.k, m.k, m.pad,
                                                    m.pad, demod, 1e-8f, dcc, yc);
                brotensor::modulated_conv2d_forward(X.g, W.g, sv.g, m.n, m.cin, m.h, m.w, m.cout, m.k, m.k, m.pad,
                                                    m.pad, demod, 1e-8f, dcg, yg);
                Tensor dXc, dsc, dXg, dsg;
                Tensor dWc = Tensor::zeros_on(Device::cpu(), m.cout, K), dWg = Tensor::zeros_on(vk(), m.cout, K, dt);
                brotensor::modulated_conv2d_backward(X.c, W.c, sv.c, dcc, dY.c, m.n, m.cin, m.h, m.w, m.cout, m.k, m.k,
                                                     m.pad, m.pad, demod, 1e-8f, dXc, dWc, dsc);
                brotensor::modulated_conv2d_backward(X.g, W.g, sv.g, dcg, dY.g, m.n, m.cin, m.h, m.w, m.cout, m.k, m.k,
                                                     m.pad, m.pad, demod, 1e-8f, dXg, dWg, dsg);
                const std::string tag = "cin " + std::to_string(m.cin) + " k" + std::to_string(m.k) +
                                        (demod ? " demod" : "");
                const float t = tol16(dt, 5e-5f, 3e-2f, 2e-1f), r = tol16(dt, 1e-4f, 1e-2f, 5e-2f);
                expect_close(download(dXg), dXc.to_host_vector(), t, r, tg("modulated_conv2d_backward dX", dt, tag));
                expect_close(download(dWg), dWc.to_host_vector(), t, r, tg("modulated_conv2d_backward dW", dt, tag));
                expect_close(download(dsg), dsc.to_host_vector(), t, r, tg("modulated_conv2d_backward ds", dt, tag));
                // Frozen weights (GAN inversion): an uncommitted dW is skipped.
                Tensor dXg2, dsg2, none;
                brotensor::modulated_conv2d_backward(X.g, W.g, sv.g, dcg, dY.g, m.n, m.cin, m.h, m.w, m.cout, m.k, m.k,
                                                     m.pad, m.pad, demod, 1e-8f, dXg2, none, dsg2);
                VKT_CHECK(none.data == nullptr);
                expect_close(download(dsg2), download(dsg), 0, 0, tg("modulated_conv2d_backward ds (no dW)", dt, tag));
            }
        }
    }
}

void test_attention_forward() {
    std::printf("attention_forward, cross_attention_forward_train\n");
    const int N = 13, D = 24, Lk = 9, H = 3;
    const std::vector<float> mask = {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1};
    Tensor mg = upload(mask, N, 1, Dtype::FP32);
    const float ws = 1.0f / std::sqrt(float(D));
    std::uint64_t seed = 400;
    for (int m = 0; m < 2; ++m) {
        Pair X = make(std::size_t(N) * D, N, D, seed++, -1, 1, Dtype::FP32);
        Pair Wq = make(std::size_t(D) * D, D, D, seed++, -ws, ws, Dtype::FP32);
        Pair Wk = make(std::size_t(D) * D, D, D, seed++, -ws, ws, Dtype::FP32);
        Pair Wv = make(std::size_t(D) * D, D, D, seed++, -ws, ws, Dtype::FP32);
        Pair Wo = make(std::size_t(D) * D, D, D, seed++, -ws, ws, Dtype::FP32);
        const float* mc = m ? mask.data() : nullptr;
        const float* mgp = m ? static_cast<const float*>(mg.data) : nullptr;
        Tensor c[6], g[6];
        brotensor::attention_forward(X.c, Wq.c, Wk.c, Wv.c, Wo.c, mc, c[0], c[1], c[2], c[3], c[4], c[5]);
        brotensor::attention_forward(X.g, Wq.g, Wk.g, Wv.g, Wo.g, mgp, g[0], g[1], g[2], g[3], g[4], g[5]);
        const char* names[] = {"Q", "K", "V", "Attn", "Y_pre_Wo", "O"};
        for (int i = 0; i < 6; ++i) {
            expect_close(g[i].to_host_vector(), c[i].to_host_vector(), 2e-6f, 1e-5f,
                         std::string("attention_forward ") + names[i] + (m ? " masked" : ""));
        }
        // Cross attention over a context of Lk rows (mask over the keys only
        // when Lk != N, so a Lk-row mask).
        Pair Ctx = make(std::size_t(Lk) * D, Lk, D, seed++, -1, 1, Dtype::FP32);
        Tensor cm = upload(std::vector<float>(mask.begin(), mask.begin() + Lk), Lk, 1, Dtype::FP32);
        const float* cmc = m ? mask.data() : nullptr;
        const float* cmg = m ? static_cast<const float*>(cm.data) : nullptr;
        Tensor cc[6], cg[6];
        brotensor::cross_attention_forward_train(X.c, Ctx.c, Wq.c, Wk.c, Wv.c, Wo.c, cmc, H, cc[0], cc[1], cc[2],
                                                 cc[3], cc[4], cc[5]);
        brotensor::cross_attention_forward_train(X.g, Ctx.g, Wq.g, Wk.g, Wv.g, Wo.g, cmg, H, cg[0], cg[1], cg[2],
                                                 cg[3], cg[4], cg[5]);
        const char* cn[] = {"Qh", "Kh", "Vh", "Attnh", "Yconcat", "O"};
        for (int i = 0; i < 6; ++i) {
            expect_close(cg[i].to_host_vector(), cc[i].to_host_vector(), 2e-6f, 1e-5f,
                         std::string("cross_attention_forward_train ") + cn[i] + (m ? " masked" : ""));
        }
    }
}

}  // namespace

void run_vision_tests() {
    test_sam_attention();
    test_deform_conv();
    test_modulated_conv();
    test_conv_weight_grads();
    test_attention_forward();
}

}  // namespace vkt
