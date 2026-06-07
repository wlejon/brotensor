// CPU↔CUDA parity for the StyleGAN3-R op surface:
//   sin / cos / rsqrt / pixel_norm   (fwd + bwd)
//   bias_act                         (fwd + bwd, dX + dB)
//   upfirdn2d                        (fwd + bwd)
//   modulated_conv2d                 (fwd + bwd, dX + dW + ds + dcoef)
//   filtered_lrelu                   (composite over bias_act + upfirdn2d)
//
// The validated CPU FP32 path is the reference; each op runs on CPU and on
// CUDA over identical inputs and the device result is compared back. Skips
// cleanly when no CUDA backend is present.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static std::vector<float> rnd(int n, uint64_t seed, float lo, float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static Tensor cpu(const std::vector<float>& v, int r, int c) {
    return Tensor::from_host_on(Device::CPU, v.data(), r, c);
}
static Tensor gpu(const std::vector<float>& v, int r, int c) {
    return Tensor::from_host_on(Device::CUDA, v.data(), r, c);
}

// Compare a CUDA-resident tensor against a CPU-resident reference tensor.
static void cmp(const char* label, const Tensor& ref_cpu, const Tensor& got_gpu,
                float atol = 1e-4f, float rtol = 1e-4f) {
    brotensor::sync_all();
    CHECK(ref_cpu.rows == got_gpu.rows);
    CHECK(ref_cpu.cols == got_gpu.cols);
    const int n = ref_cpu.rows * ref_cpu.cols;
    std::vector<float> g(static_cast<size_t>(n), 0.0f);
    const_cast<Tensor&>(got_gpu).copy_to_host(g.data());
    brotensor::sync_all();
    const float* r = ref_cpu.host_f32();
    int bad = 0; float me = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float e = std::fabs(g[i] - r[i]);
        if (e > me) me = e;
        if (e > atol + rtol * std::fabs(r[i])) {
            if (bad < 5)
                std::printf("    %s mismatch i=%d gpu=%g cpu=%g err=%g\n",
                            label, i, g[i], r[i], e);
            ++bad;
        }
    }
    std::printf("    %-24s max_err=%g bad=%d / %d\n", label, me, bad, n);
    CHECK(bad == 0);
}

// ─── elementwise: sin / cos / rsqrt ─────────────────────────────────────────
static void test_elementwise() {
    const int R = 5, C = 7, n = R * C;
    auto x  = rnd(n, 1, -3.0f, 3.0f);
    auto xp = rnd(n, 2, 0.3f, 4.0f);       // strictly positive for rsqrt
    auto w  = rnd(n, 3, -1.0f, 1.0f);

    Tensor Xc = cpu(x, R, C), Xg = gpu(x, R, C);
    Tensor Yc, Yg;

    brotensor::sin_forward(Xc, Yc); brotensor::sin_forward(Xg, Yg);
    cmp("sin_forward", Yc, Yg);
    { Tensor dYc = cpu(w, R, C), dYg = gpu(w, R, C), dXc, dXg;
      brotensor::sin_backward(Xc, dYc, dXc); brotensor::sin_backward(Xg, dYg, dXg);
      cmp("sin_backward", dXc, dXg); }

    brotensor::cos_forward(Xc, Yc); brotensor::cos_forward(Xg, Yg);
    cmp("cos_forward", Yc, Yg);
    { Tensor dYc = cpu(w, R, C), dYg = gpu(w, R, C), dXc, dXg;
      brotensor::cos_backward(Xc, dYc, dXc); brotensor::cos_backward(Xg, dYg, dXg);
      cmp("cos_backward", dXc, dXg); }

    Tensor Xpc = cpu(xp, R, C), Xpg = gpu(xp, R, C);
    Tensor Ypc, Ypg;
    brotensor::rsqrt_forward(Xpc, Ypc); brotensor::rsqrt_forward(Xpg, Ypg);
    cmp("rsqrt_forward", Ypc, Ypg);
    { Tensor dYc = cpu(w, R, C), dYg = gpu(w, R, C), dXc, dXg;
      // rsqrt backward reads the OUTPUT y as primal.
      brotensor::rsqrt_backward(Ypc, dYc, dXc); brotensor::rsqrt_backward(Ypg, dYg, dXg);
      cmp("rsqrt_backward", dXc, dXg); }
}

// ─── pixel_norm ─────────────────────────────────────────────────────────────
static void test_pixel_norm() {
    const int R = 6, C = 257, n = R * C;   // odd C to exercise the reduction tail
    const float eps = 1e-8f;
    auto x = rnd(n, 11, -2.0f, 2.0f);
    auto w = rnd(n, 12, -1.0f, 1.0f);
    Tensor Xc = cpu(x, R, C), Xg = gpu(x, R, C), Yc, Yg;
    brotensor::pixel_norm_forward(Xc, eps, Yc);
    brotensor::pixel_norm_forward(Xg, eps, Yg);
    cmp("pixel_norm_forward", Yc, Yg);
    Tensor dYc = cpu(w, R, C), dYg = gpu(w, R, C), dXc, dXg;
    brotensor::pixel_norm_backward(Xc, dYc, eps, dXc);
    brotensor::pixel_norm_backward(Xg, dYg, eps, dXg);
    cmp("pixel_norm_backward", dXc, dXg);
}

// ─── bias_act ────────────────────────────────────────────────────────────────
static void test_bias_act() {
    const int N = 3, C = 4, HW = 5, cols = C * HW;
    const float alpha = 0.2f;
    for (int act = 0; act <= 1; ++act)
        for (int hb = 0; hb <= 1; ++hb)
            for (float clamp : {-1.0f, 0.8f}) {
                const float gain = (act == 1) ? std::sqrt(2.0f) : 1.3f;
                auto x = rnd(N * cols, 200 + act * 31 + hb * 7 + int(clamp * 13), -2.0f, 2.0f);
                auto b = rnd(C, 300 + act, -1.0f, 1.0f);
                auto w = rnd(N * cols, 400 + hb, -1.0f, 1.0f);
                Tensor Xc = cpu(x, N, cols), Xg = gpu(x, N, cols);
                Tensor Bc = cpu(b, C, 1), Bg = gpu(b, C, 1);
                const Tensor* bpc = hb ? &Bc : nullptr;
                const Tensor* bpg = hb ? &Bg : nullptr;
                Tensor Yc, Yg;
                brotensor::bias_act_forward(Xc, bpc, N, C, HW, act, alpha, gain, clamp, Yc);
                brotensor::bias_act_forward(Xg, bpg, N, C, HW, act, alpha, gain, clamp, Yg);
                cmp("bias_act_forward", Yc, Yg);

                Tensor dYc = cpu(w, N, cols), dYg = gpu(w, N, cols), dXc, dXg;
                Tensor dBc = Tensor::zeros_on(Device::CPU,  C, 1);
                Tensor dBg = Tensor::zeros_on(Device::CUDA, C, 1);
                brotensor::bias_act_backward(dYc, Xc, bpc, N, C, HW, act, alpha, gain, clamp,
                                             dXc, hb ? &dBc : nullptr);
                brotensor::bias_act_backward(dYg, Xg, bpg, N, C, HW, act, alpha, gain, clamp,
                                             dXg, hb ? &dBg : nullptr);
                cmp("bias_act_backward dX", dXc, dXg);
                if (hb) cmp("bias_act_backward dB", dBc, dBg);
            }
}

// ─── upfirdn2d ───────────────────────────────────────────────────────────────
static void test_upfirdn2d() {
    const int N = 2, C = 3, H = 8, Wd = 8, fH = 4, fW = 4;
    const int up_x = 2, up_y = 2, down_x = 1, down_y = 1;
    const int px0 = 1, px1 = 1, py0 = 1, py1 = 1;
    const float gain = float(up_x * up_y);
    auto x = rnd(N * C * H * Wd, 21, -1.0f, 1.0f);
    auto f = rnd(fH * fW, 22, -0.5f, 0.5f);
    Tensor Xc = cpu(x, N, C * H * Wd), Xg = gpu(x, N, C * H * Wd);
    Tensor Fc = cpu(f, fH, fW), Fg = gpu(f, fH, fW);
    Tensor Yc, Yg;
    brotensor::upfirdn2d_forward(Xc, Fc, N, C, H, Wd, fH, fW,
                                 up_x, up_y, down_x, down_y, px0, px1, py0, py1,
                                 false, gain, Yc);
    brotensor::upfirdn2d_forward(Xg, Fg, N, C, H, Wd, fH, fW,
                                 up_x, up_y, down_x, down_y, px0, px1, py0, py1,
                                 false, gain, Yg);
    cmp("upfirdn2d_forward", Yc, Yg);

    // Backward: dY has the forward-output shape.
    const int Hout = (H * up_y + py0 + py1 - fH) / down_y + 1;
    const int Wout = (Wd * up_x + px0 + px1 - fW) / down_x + 1;
    auto w = rnd(N * C * Hout * Wout, 23, -1.0f, 1.0f);
    Tensor dYc = cpu(w, N, C * Hout * Wout), dYg = gpu(w, N, C * Hout * Wout), dXc, dXg;
    brotensor::upfirdn2d_backward(dYc, Fc, N, C, H, Wd, fH, fW,
                                  up_x, up_y, down_x, down_y, px0, px1, py0, py1,
                                  false, gain, dXc);
    brotensor::upfirdn2d_backward(dYg, Fg, N, C, H, Wd, fH, fW,
                                  up_x, up_y, down_x, down_y, px0, px1, py0, py1,
                                  false, gain, dXg);
    cmp("upfirdn2d_backward", dXc, dXg);
}

// ─── modulated_conv2d ────────────────────────────────────────────────────────
static void test_modulated_conv2d() {
    const int N = 2, C_in = 3, H = 6, Wd = 6, C_out = 4, kH = 3, kW = 3;
    const int pad_h = 1, pad_w = 1;
    const float eps = 1e-8f;
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    const int wk = C_in * kH * kW;
    auto x = rnd(N * C_in * H * Wd, 31, -1.0f, 1.0f);
    auto W = rnd(C_out * wk, 32, -0.5f, 0.5f);
    auto s = rnd(N * C_in, 33, 0.2f, 1.5f);

    for (bool demod : {true, false}) {
        Tensor Xc = cpu(x, N, C_in * H * Wd), Xg = gpu(x, N, C_in * H * Wd);
        Tensor Wc = cpu(W, C_out, wk),        Wg = gpu(W, C_out, wk);
        Tensor Sc = cpu(s, N, C_in),          Sg = gpu(s, N, C_in);
        Tensor dcc, dcg, Yc, Yg;
        brotensor::modulated_conv2d_forward(Xc, Wc, Sc, N, C_in, H, Wd, C_out, kH, kW,
                                            pad_h, pad_w, demod, eps, dcc, Yc);
        brotensor::modulated_conv2d_forward(Xg, Wg, Sg, N, C_in, H, Wd, C_out, kH, kW,
                                            pad_h, pad_w, demod, eps, dcg, Yg);
        cmp(demod ? "modconv_fwd Y (demod)" : "modconv_fwd Y (plain)", Yc, Yg);
        cmp(demod ? "modconv_fwd dcoef(demod)" : "modconv_fwd dcoef(plain)", dcc, dcg);

        auto g = rnd(N * C_out * H_out * W_out, 34, -1.0f, 1.0f);
        Tensor dYc = cpu(g, N, C_out * H_out * W_out), dYg = gpu(g, N, C_out * H_out * W_out);
        Tensor dXc, dXg, dsc, dsg;
        Tensor dWc = Tensor::zeros_on(Device::CPU,  C_out, wk);
        Tensor dWg = Tensor::zeros_on(Device::CUDA, C_out, wk);
        brotensor::modulated_conv2d_backward(Xc, Wc, Sc, dcc, dYc, N, C_in, H, Wd,
                                             C_out, kH, kW, pad_h, pad_w, demod, eps,
                                             dXc, dWc, dsc);
        brotensor::modulated_conv2d_backward(Xg, Wg, Sg, dcg, dYg, N, C_in, H, Wd,
                                             C_out, kH, kW, pad_h, pad_w, demod, eps,
                                             dXg, dWg, dsg);
        cmp(demod ? "modconv_bwd dX (demod)" : "modconv_bwd dX (plain)", dXc, dXg, 1e-3f, 1e-4f);
        cmp(demod ? "modconv_bwd dW (demod)" : "modconv_bwd dW (plain)", dWc, dWg, 1e-3f, 1e-4f);
        cmp(demod ? "modconv_bwd ds (demod)" : "modconv_bwd ds (plain)", dsc, dsg, 1e-3f, 1e-4f);
    }
}

// ─── filtered_lrelu (device-agnostic composite over bias_act + upfirdn2d) ────
static void test_filtered_lrelu() {
    const int N = 2, C = 2, H = 6, Wd = 6;
    const int up = 2, down = 2;
    const int px0 = 1, px1 = 2, py0 = 1, py1 = 2;
    const int fuH = 4, fuW = 4, fdH = 4, fdW = 4;
    const float gain = std::sqrt(2.0f), slope = 0.2f, clamp = 0.8f;
    auto x  = rnd(N * C * H * Wd, 41, -1.2f, 1.2f);
    auto b  = rnd(C, 42, -0.5f, 0.5f);
    auto fu = rnd(fuH * fuW, 43, -0.3f, 0.3f);
    auto fd = rnd(fdH * fdW, 44, -0.3f, 0.3f);

    Tensor Xc = cpu(x, N, C * H * Wd), Xg = gpu(x, N, C * H * Wd);
    Tensor Bc = cpu(b, C, 1),         Bg = gpu(b, C, 1);
    Tensor Fuc = cpu(fu, fuH, fuW),   Fug = gpu(fu, fuH, fuW);
    Tensor Fdc = cpu(fd, fdH, fdW),   Fdg = gpu(fd, fdH, fdW);
    Tensor up_bufc, act_bufc, Yc, up_bufg, act_bufg, Yg;
    brotensor::filtered_lrelu_forward(Xc, Fuc, Fdc, &Bc, N, C, H, Wd,
                                      up, down, px0, px1, py0, py1,
                                      gain, slope, clamp, up_bufc, act_bufc, Yc);
    brotensor::filtered_lrelu_forward(Xg, Fug, Fdg, &Bg, N, C, H, Wd,
                                      up, down, px0, px1, py0, py1,
                                      gain, slope, clamp, up_bufg, act_bufg, Yg);
    cmp("filtered_lrelu_forward", Yc, Yg, 1e-3f, 1e-4f);

    // Backward: loss L = sum(g*Y), dY = g; dB accumulates (caller zeros).
    auto g = rnd(Yc.rows * Yc.cols, 45, -1.0f, 1.0f);
    Tensor dYc = cpu(g, Yc.rows, Yc.cols), dYg = gpu(g, Yg.rows, Yg.cols);
    Tensor dXc, dXg;
    Tensor dBc = Tensor::zeros_on(Device::CPU,  C, 1);
    Tensor dBg = Tensor::zeros_on(Device::CUDA, C, 1);
    brotensor::filtered_lrelu_backward(dYc, Xc, Fuc, Fdc, &Bc, N, C, H, Wd,
                                       up, down, px0, px1, py0, py1,
                                       gain, slope, clamp, up_bufc, dXc, &dBc);
    brotensor::filtered_lrelu_backward(dYg, Xg, Fug, Fdg, &Bg, N, C, H, Wd,
                                       up, down, px0, px1, py0, py1,
                                       gain, slope, clamp, up_bufg, dXg, &dBg);
    cmp("filtered_lrelu_backward dX", dXc, dXg, 1e-3f, 1e-4f);
    cmp("filtered_lrelu_backward dB", dBc, dBg, 1e-3f, 1e-4f);
}

int main() {
    brotensor::init();
    std::printf("test_stylegan_cuda_parity\n");
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("  CUDA backend unavailable — skipping.\n");
        return 0;
    }
    test_elementwise();
    test_pixel_norm();
    test_bias_act();
    test_upfirdn2d();
    test_modulated_conv2d();
    test_filtered_lrelu();
    if (g_failures) {
        std::printf("\nstylegan_cuda_parity: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nstylegan_cuda_parity: all passed\n");
    return 0;
}
