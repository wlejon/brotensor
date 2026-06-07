// CPU FP32 tests for modulated_conv2d (StyleGAN3 synthesis core).
// Forward + dcoef vs an independent per-sample modulated-conv reference;
// dX / dW / ds vs finite difference (double-accumulated loss). Covers
// demodulate on/off and 1x1 / 3x3 kernels.

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool close(float a, float b, float tol = 2e-3f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

static Tensor cpu_vec(const std::vector<float>& v, int r, int c) {
    return Tensor::from_host_on(brotensor::Device::CPU, v.data(), r, c);
}

struct Dims { int N, Cin, H, W, Cout, kH, kW, pad; };

// Independent reference. X (N,Cin*H*W), W (Cout,Cin*kH*kW), s (N,Cin).
static std::vector<float> ref_modconv(const std::vector<float>& X,
                                      const std::vector<float>& Wt,
                                      const std::vector<float>& s,
                                      const Dims& d, bool demod, float eps,
                                      int& Hout, int& Wout) {
    Hout = d.H + 2 * d.pad - (d.kH - 1);
    Wout = d.W + 2 * d.pad - (d.kW - 1);
    const int wk = d.Cin * d.kH * d.kW;
    std::vector<float> Y(static_cast<size_t>(d.N) * d.Cout * Hout * Wout, 0.0f);
    std::vector<float> wpp(wk);
    for (int n = 0; n < d.N; ++n) {
        for (int o = 0; o < d.Cout; ++o) {
            double ss = 0.0;
            for (int i = 0; i < d.Cin; ++i)
                for (int t = 0; t < d.kH * d.kW; ++t) {
                    const float v = Wt[(o * d.Cin + i) * d.kH * d.kW + t] *
                                    s[n * d.Cin + i];
                    wpp[i * d.kH * d.kW + t] = v;
                    ss += static_cast<double>(v) * v;
                }
            const float dc = demod ? 1.0f / std::sqrt(static_cast<float>(ss) + eps) : 1.0f;
            if (demod) for (auto& v : wpp) v *= dc;
            for (int oh = 0; oh < Hout; ++oh)
                for (int ow = 0; ow < Wout; ++ow) {
                    double acc = 0.0;
                    for (int i = 0; i < d.Cin; ++i)
                        for (int kh = 0; kh < d.kH; ++kh) {
                            const int ih = oh - d.pad + kh;
                            if (ih < 0 || ih >= d.H) continue;
                            for (int kw = 0; kw < d.kW; ++kw) {
                                const int iw = ow - d.pad + kw;
                                if (iw < 0 || iw >= d.W) continue;
                                acc += wpp[(i * d.kH + kh) * d.kW + kw] *
                                       X[((static_cast<size_t>(n) * d.Cin + i) * d.H + ih) * d.W + iw];
                            }
                        }
                    Y[((static_cast<size_t>(n) * d.Cout + o) * Hout + oh) * Wout + ow] =
                        static_cast<float>(acc);
                }
        }
    }
    return Y;
}

static void run_case(const Dims& d, bool demod) {
    std::mt19937 rng(static_cast<uint32_t>(d.kH * 31 + d.Cout + (demod ? 500 : 0)));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const float eps = 1e-8f;
    const int wk = d.Cin * d.kH * d.kW;
    const int xn = d.N * d.Cin * d.H * d.W;
    const int wn = d.Cout * wk;
    const int sn = d.N * d.Cin;

    std::vector<float> x(xn), w(wn), s(sn);
    for (auto& v : x) v = dist(rng);
    for (auto& v : w) v = dist(rng);
    for (auto& v : s) v = dist(rng) + 1.2f;  // keep style away from 0

    int Hout = 0, Wout = 0;
    std::vector<float> Yref = ref_modconv(x, w, s, d, demod, eps, Hout, Wout);

    Tensor X = cpu_vec(x, d.N, d.Cin * d.H * d.W);
    Tensor W = cpu_vec(w, d.Cout, wk);
    Tensor S = cpu_vec(s, d.N, d.Cin);
    Tensor dcoef, Y;
    brotensor::modulated_conv2d_forward(X, W, S, d.N, d.Cin, d.H, d.W,
                                        d.Cout, d.kH, d.kW, d.pad, d.pad,
                                        demod, eps, dcoef, Y);
    CHECK(Y.rows == d.N && Y.cols == d.Cout * Hout * Wout);
    for (size_t i = 0; i < Yref.size(); ++i) CHECK(close(Y[i], Yref[i]));

    // Backward. Loss L = sum(g * Y), dY = g.
    std::vector<float> g(Yref.size());
    for (auto& v : g) v = dist(rng);
    Tensor dY = cpu_vec(g, d.N, d.Cout * Hout * Wout);
    Tensor dX;
    Tensor dW = Tensor::zeros_on(brotensor::Device::CPU, d.Cout, wk);  // caller zeros
    Tensor ds;
    brotensor::modulated_conv2d_backward(X, W, S, dcoef, dY, d.N, d.Cin, d.H, d.W,
                                         d.Cout, d.kH, d.kW, d.pad, d.pad,
                                         demod, eps, dX, dW, ds);
    CHECK(dX.rows == d.N && dX.cols == d.Cin * d.H * d.W);
    CHECK(ds.rows == d.N && ds.cols == d.Cin);

    const float h = 1e-3f;
    auto loss = [&](const std::vector<float>& xx, const std::vector<float>& ww,
                    const std::vector<float>& ss) {
        int ho, wo;
        std::vector<float> Yp = ref_modconv(xx, ww, ss, d, demod, eps, ho, wo);
        double L = 0.0;
        for (size_t k = 0; k < g.size(); ++k) L += static_cast<double>(g[k]) * Yp[k];
        return L;
    };
    // dX
    for (int i = 0; i < xn; ++i) {
        std::vector<float> xp = x, xm = x; xp[i] += h; xm[i] -= h;
        const float fd = static_cast<float>((loss(xp, w, s) - loss(xm, w, s)) / (2.0 * h));
        CHECK(close(dX[i], fd, 1e-2f));
    }
    // dW
    for (int j = 0; j < wn; ++j) {
        std::vector<float> wp = w, wm = w; wp[j] += h; wm[j] -= h;
        const float fd = static_cast<float>((loss(x, wp, s) - loss(x, wm, s)) / (2.0 * h));
        CHECK(close(dW[j], fd, 1e-2f));
    }
    // ds
    for (int k = 0; k < sn; ++k) {
        std::vector<float> sp = s, sm = s; sp[k] += h; sm[k] -= h;
        const float fd = static_cast<float>((loss(x, w, sp) - loss(x, w, sm)) / (2.0 * h));
        CHECK(close(ds[k], fd, 1e-2f));
    }
}

int main() {
    // 1x1 (config-R) and 3x3, demod on/off.
    for (bool demod : {true, false}) {
        run_case({2, 3, 4, 5, 4, 1, 1, 0}, demod);
        run_case({2, 3, 4, 5, 4, 3, 3, 1}, demod);
    }
    if (g_failures) {
        std::printf("modulated_conv2d: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("modulated_conv2d: all passed\n");
    return 0;
}
