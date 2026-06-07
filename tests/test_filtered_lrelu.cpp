// CPU FP32 tests for filtered_lrelu (StyleGAN3 alias-free nonlinearity).
//
// The composite is checked against an INDEPENDENT reference that re-implements
// _filtered_lrelu_ref from scratch (literal upfirdn + bias + lrelu/clamp), so
// the test validates the composition order/params (bias-before-upsample, the
// up^2 gain, padding routing) rather than re-using the library sub-ops.
// Backward (dX, dB) is checked vs finite difference of that reference.

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

// Literal upfirdn2d (flip_filter=false) for one (N,C,H,W) tensor.
static std::vector<float> ref_upfirdn(const std::vector<float>& X,
                                      int N, int C, int H, int W,
                                      const std::vector<float>& f, int fH, int fW,
                                      int up_x, int up_y, int down_x, int down_y,
                                      int px0, int px1, int py0, int py1,
                                      float gain, int& Hout, int& Wout) {
    const int Hu = H * up_y, Wu = W * up_x;
    const int Hp = Hu + py0 + py1, Wp = Wu + px0 + px1;
    Hout = (Hp - fH) / down_y + 1;
    Wout = (Wp - fW) / down_x + 1;
    std::vector<float> fe(fH * fW);
    for (int kh = 0; kh < fH; ++kh)
        for (int kw = 0; kw < fW; ++kw)
            fe[kh * fW + kw] = f[(fH - 1 - kh) * fW + (fW - 1 - kw)] * gain;
    std::vector<float> Y(static_cast<size_t>(N) * C * Hout * Wout, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            std::vector<float> P(static_cast<size_t>(Hp) * Wp, 0.0f);
            for (int iy = 0; iy < H; ++iy)
                for (int ix = 0; ix < W; ++ix) {
                    const int ty = iy * up_y + py0, tx = ix * up_x + px0;
                    if (ty < 0 || ty >= Hp || tx < 0 || tx >= Wp) continue;
                    P[ty * static_cast<size_t>(Wp) + tx] =
                        X[((static_cast<size_t>(n) * C + c) * H + iy) * W + ix];
                }
            for (int oy = 0; oy < Hout; ++oy)
                for (int ox = 0; ox < Wout; ++ox) {
                    double acc = 0.0;
                    for (int kh = 0; kh < fH; ++kh)
                        for (int kw = 0; kw < fW; ++kw)
                            acc += P[(oy * down_y + kh) * static_cast<size_t>(Wp) +
                                     (ox * down_x + kw)] * fe[kh * fW + kw];
                    Y[((static_cast<size_t>(n) * C + c) * Hout + oy) * Wout + ox] =
                        static_cast<float>(acc);
                }
        }
    return Y;
}

struct Cfg { int N, C, H, W, up, down, px0, px1, py0, py1, fuH, fuW, fdH, fdW;
             float gain, slope, clamp; bool has_b; };

static std::vector<float> ref_flrelu(const std::vector<float>& X,
                                     const std::vector<float>& fu,
                                     const std::vector<float>& fd,
                                     const std::vector<float>& b,
                                     const Cfg& c, int& Hout, int& Wout) {
    // 1. bias.
    std::vector<float> pre(X.size());
    for (int n = 0; n < c.N; ++n)
        for (int ch = 0; ch < c.C; ++ch)
            for (int k = 0; k < c.H * c.W; ++k) {
                const size_t idx = (static_cast<size_t>(n) * c.C + ch) * c.H * c.W + k;
                pre[idx] = X[idx] + (c.has_b ? b[ch] : 0.0f);
            }
    // 2. upsample (gain up^2).
    int Huo, Wuo;
    std::vector<float> up = ref_upfirdn(pre, c.N, c.C, c.H, c.W, fu, c.fuH, c.fuW,
                                        c.up, c.up, 1, 1, c.px0, c.px1, c.py0, c.py1,
                                        static_cast<float>(c.up * c.up), Huo, Wuo);
    // 3. lrelu + gain + clamp.
    for (auto& v : up) {
        float y = (v > 0 ? v : c.slope * v) * c.gain;
        if (c.clamp >= 0.0f) y = std::fmax(-c.clamp, std::fmin(c.clamp, y));
        v = y;
    }
    // 4. downsample.
    return ref_upfirdn(up, c.N, c.C, Huo, Wuo, fd, c.fdH, c.fdW,
                       1, 1, c.down, c.down, 0, 0, 0, 0, 1.0f, Hout, Wout);
}

static void run_case(const Cfg& c) {
    std::mt19937 rng(static_cast<uint32_t>(c.up * 71 + c.down * 13 + c.H +
                                           (c.clamp >= 0 ? 99 : 0) + (c.has_b ? 7 : 0)));
    std::uniform_real_distribution<float> d(-1.2f, 1.2f);
    const int xn = c.N * c.C * c.H * c.W;
    std::vector<float> x(xn), fu(c.fuH * c.fuW), fd(c.fdH * c.fdW), b(c.C, 0.0f);
    for (auto& v : x) v = d(rng);
    for (auto& v : fu) v = d(rng);
    for (auto& v : fd) v = d(rng);
    if (c.has_b) for (auto& v : b) v = d(rng);

    int Hout, Wout;
    std::vector<float> Yref = ref_flrelu(x, fu, fd, b, c, Hout, Wout);

    Tensor X = cpu_vec(x, c.N, c.C * c.H * c.W);
    Tensor FU = cpu_vec(fu, c.fuH, c.fuW), FD = cpu_vec(fd, c.fdH, c.fdW);
    Tensor B = cpu_vec(b, c.C, 1);
    const Tensor* bp = c.has_b ? &B : nullptr;
    Tensor up_buf, act_buf, Y;
    brotensor::filtered_lrelu_forward(X, FU, FD, bp, c.N, c.C, c.H, c.W,
                                      c.up, c.down, c.px0, c.px1, c.py0, c.py1,
                                      c.gain, c.slope, c.clamp, up_buf, act_buf, Y);
    CHECK(Y.rows == c.N && Y.cols == c.C * Hout * Wout);
    for (size_t i = 0; i < Yref.size(); ++i) CHECK(close(Y[i], Yref[i]));

    // Backward. Loss L = sum(g*Y), dY = g.
    std::vector<float> g(Yref.size());
    for (auto& v : g) v = d(rng);
    Tensor dY = cpu_vec(g, c.N, c.C * Hout * Wout);
    Tensor dX;
    Tensor dB = Tensor::zeros_on(brotensor::Device::CPU, c.C, 1);  // caller zeros
    brotensor::filtered_lrelu_backward(dY, X, FU, FD, bp, c.N, c.C, c.H, c.W,
                                       c.up, c.down, c.px0, c.px1, c.py0, c.py1,
                                       c.gain, c.slope, c.clamp, up_buf,
                                       dX, c.has_b ? &dB : nullptr);
    CHECK(dX.rows == c.N && dX.cols == c.C * c.H * c.W);

    auto loss = [&](const std::vector<float>& xx, const std::vector<float>& bb) {
        int ho, wo;
        std::vector<float> Yp = ref_flrelu(xx, fu, fd, bb, c, ho, wo);
        double L = 0.0;
        for (size_t k = 0; k < g.size(); ++k) L += static_cast<double>(g[k]) * Yp[k];
        return L;
    };
    // filtered_lrelu is piecewise-linear (lrelu + clamp + linear maps): away
    // from a breakpoint the central difference is exact; AT a breakpoint the
    // analytic gradient equals one of the one-sided slopes (a valid
    // subgradient). Accept a match to the central difference or to either side.
    const float h = 7e-4f;
    const double L0 = loss(x, b);
    auto pw_ok = [&](float a, double Lp, double Lm) {
        const double fdp = (Lp - L0) / h, fdm = (L0 - Lm) / h;
        const double central = (Lp - Lm) / (2.0 * h);
        return close(a, static_cast<float>(central), 2e-2f) ||
               close(a, static_cast<float>(fdp), 3e-2f) ||
               close(a, static_cast<float>(fdm), 3e-2f);
    };
    for (int i = 0; i < xn; ++i) {
        std::vector<float> xp = x, xm = x; xp[i] += h; xm[i] -= h;
        CHECK(pw_ok(dX[i], loss(xp, b), loss(xm, b)));
    }
    if (c.has_b) {
        for (int ch = 0; ch < c.C; ++ch) {
            std::vector<float> bp2 = b, bm2 = b; bp2[ch] += h; bm2[ch] -= h;
            CHECK(pw_ok(dB[ch], loss(x, bp2), loss(x, bm2)));
        }
    }
}

int main() {
    // up=2/down=2 with bias + clamp; up=2/down=2 no bias, no clamp;
    // up=1/down=1 small filters (pure FIR + lrelu).
    run_case({2, 2, 6, 6, 2, 2, 1, 2, 1, 2, 4, 4, 4, 4, std::sqrt(2.0f), 0.2f, 0.8f, true});
    run_case({2, 2, 6, 6, 2, 2, 2, 1, 2, 1, 4, 4, 4, 4, std::sqrt(2.0f), 0.2f, -1.0f, false});
    run_case({1, 3, 7, 7, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1.3f, 0.2f, -1.0f, true});
    if (g_failures) {
        std::printf("filtered_lrelu: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("filtered_lrelu: all passed\n");
    return 0;
}
