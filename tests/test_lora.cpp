#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

// Standalone CPU coverage for the LoRA adapter primitive (ops/lora.h).
//
// Verifies:
//   * Forward matches an independent hand-rolled reference
//     y = W x + b + scale * B (g (.) A x).
//   * lora_backward is exact: every trainable gradient (dA, dB, dG, dX)
//     matches a central finite-difference estimate of a scalar MSE loss.
//   * Both the gated (g != null) and ungated (g == null) paths are covered.
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include <brotensor/ops/lora.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

static void expect_near(double a, double b, double abs_eps, double rel_eps,
                        const std::string& ctx) {
    const double d = std::fabs(a - b);
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    if (d <= abs_eps || d <= rel_eps * m) return;
    std::printf("  FAIL %s: %.8g vs %.8g (|d|=%.3g)\n", ctx.c_str(), a, b, d);
    ++g_failures;
}

// out != in != r so a transpose bug can't hide.
static const int kOut = 6;   // = 2*C with C=3 (mirrors an AdaIN (gamma,beta))
static const int kIn  = 8;   // style dim
static const int kR   = 4;   // LoRA rank
static const float kScale = 0.5f;

// Pin to CPU: this host-side test perturbs tensors through operator[] and the
// default device is the GPU in a CUDA build (from_host() would land there).
static Tensor make_random(int r, int c, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> v(static_cast<std::size_t>(r) * c);
    for (auto& x : v) x = u(rng);
    return Tensor::from_host_on(Device::CPU, v.data(), r, c);
}

// Independent reference forward (no shared code with ops/lora.h).
static std::vector<float> reference_forward(const Tensor& W, const Tensor& b,
                                            const Tensor& x, const Tensor& A,
                                            const Tensor& B, const Tensor& g,
                                            bool use_g) {
    const int out = W.rows, in = W.cols, r = A.rows;
    std::vector<float> h(r);
    for (int k = 0; k < r; ++k) {
        float acc = 0.0f;
        for (int j = 0; j < in; ++j) acc += A.at(k, j) * x.at(j, 0);
        h[k] = acc;
    }
    std::vector<float> hg(r);
    for (int k = 0; k < r; ++k) hg[k] = use_g ? h[k] * g.at(k, 0) : h[k];
    std::vector<float> y(out);
    for (int o = 0; o < out; ++o) {
        float acc = b.at(o, 0);
        for (int j = 0; j < in; ++j) acc += W.at(o, j) * x.at(j, 0);
        float d = 0.0f;
        for (int k = 0; k < r; ++k) d += B.at(o, k) * hg[k];
        y[o] = acc + kScale * d;
    }
    return y;
}

static void run_case(bool use_g, std::mt19937& rng) {
    const std::string tag = use_g ? "[gated] " : "[plain] ";

    Tensor W = make_random(kOut, kIn, rng, 0.5f);
    Tensor b = make_random(kOut, 1,  rng, 0.3f);
    Tensor x = make_random(kIn, 1,   rng, 0.7f);
    Tensor A = make_random(kR, kIn,  rng, 0.5f);
    Tensor B = make_random(kOut, kR, rng, 0.5f);
    Tensor g = make_random(kR, 1,    rng, 0.8f);
    Tensor target = make_random(kOut, 1, rng, 0.5f);

    const Tensor* gp = use_g ? &g : nullptr;

    // ── forward vs reference ────────────────────────────────────────────────
    Tensor y, h, hg;
    brotensor::lora_forward(W, b, x, A, B, kScale, gp, y, h, hg);
    std::vector<float> ref = reference_forward(W, b, x, A, B, g, use_g);
    for (int o = 0; o < kOut; ++o)
        expect_near(y[o], ref[o], 1e-5, 1e-4,
                    tag + "forward[" + std::to_string(o) + "]");

    // loss(y) = 0.5 * sum (y - target)^2 ;  dL/dy = y - target
    auto loss_of = [&](const Tensor& yv) {
        double s = 0.0;
        for (int i = 0; i < yv.size(); ++i) { double d = yv[i] - target[i]; s += 0.5 * d * d; }
        return s;
    };
    Tensor dY = Tensor::mat(kOut, 1);
    for (int i = 0; i < dY.size(); ++i) dY[i] = y[i] - target[i];

    // ── analytic gradients ──────────────────────────────────────────────────
    Tensor dA = Tensor::mat(kR, kIn), dB = Tensor::mat(kOut, kR),
           dG = Tensor::mat(kR, 1), dX = Tensor::mat(kIn, 1);
    brotensor::lora_backward(W, x, A, B, kScale, gp, h, hg, dY,
                             dA, dB, use_g ? &dG : nullptr, &dX);

    // ── central finite differences ──────────────────────────────────────────
    const float eps = 1e-3f;
    auto fd_check = [&](Tensor& P, const Tensor& analytic, const char* name) {
        for (int i = 0; i < P.size(); ++i) {
            float saved = P[i];
            P[i] = saved + eps;
            Tensor yp, hp, hgp; brotensor::lora_forward(W, b, x, A, B, kScale, gp, yp, hp, hgp);
            double lp = loss_of(yp);
            P[i] = saved - eps;
            Tensor ym, hm, hgm; brotensor::lora_forward(W, b, x, A, B, kScale, gp, ym, hm, hgm);
            double lm = loss_of(ym);
            P[i] = saved;
            double fd = (lp - lm) / (2.0 * eps);
            expect_near(analytic[i], fd, 2e-3, 3e-3,
                        tag + name + "[" + std::to_string(i) + "]");
        }
    };
    fd_check(A, dA, "dA");
    fd_check(B, dB, "dB");
    if (use_g) fd_check(g, dG, "dG");
    fd_check(x, dX, "dX");
}

int main() {
    brotensor::init();
    std::mt19937 rng(20260606u);
    run_case(/*use_g=*/true, rng);
    run_case(/*use_g=*/false, rng);

    if (g_failures) { std::printf("test_lora: %d failure(s)\n", g_failures); return 1; }
    std::printf("test_lora: OK\n");
    return 0;
}
