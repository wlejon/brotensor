// CPU FP32 tests for upfirdn2d (StyleGAN3-R).
//
// Forward is checked against an independent, literal port of NVlabs
// `_upfirdn2d_ref` (explicit zero-insert → pad/crop → flip → valid correlate →
// stride-down). Backward is checked against finite difference of a scalar loss
// (upfirdn2d is linear, so dX = upfirdn2d_backward(dY) must equal d(sum w*Y)/dX).

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

static bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

static Tensor cpu_vec(const std::vector<float>& v, int r, int c) {
    return Tensor::from_host_on(brotensor::Device::CPU, v.data(), r, c);
}

struct Cfg {
    int up_x, up_y, down_x, down_y;
    int px0, px1, py0, py1;
    int fH, fW;
    bool flip;
    float gain;
};

// Literal port of _upfirdn2d_ref for one (N,C,H,W) FP32 tensor.
static std::vector<float> ref_upfirdn(const std::vector<float>& X,
                                      int N, int C, int H, int W,
                                      const std::vector<float>& f,
                                      const Cfg& cfg, int& Hout, int& Wout) {
    const int Hu = H * cfg.up_y, Wu = W * cfg.up_x;
    const int Hp = Hu + cfg.py0 + cfg.py1, Wp = Wu + cfg.px0 + cfg.px1;
    const int Hc = Hp - cfg.fH + 1, Wc = Wp - cfg.fW + 1;
    Hout = (Hc - 1) / cfg.down_y + 1;
    Wout = (Wc - 1) / cfg.down_x + 1;

    // Effective filter: scale by gain (2D => gain^1), flip unless flip_filter.
    std::vector<float> fe(cfg.fH * cfg.fW);
    for (int kh = 0; kh < cfg.fH; ++kh)
        for (int kw = 0; kw < cfg.fW; ++kw) {
            const int sh = cfg.flip ? kh : (cfg.fH - 1 - kh);
            const int sw = cfg.flip ? kw : (cfg.fW - 1 - kw);
            fe[kh * cfg.fW + kw] = f[sh * cfg.fW + sw] * cfg.gain;
        }

    std::vector<float> Y(static_cast<size_t>(N) * C * Hout * Wout, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            // Upsample.
            std::vector<float> U(static_cast<size_t>(Hu) * Wu, 0.0f);
            for (int iy = 0; iy < H; ++iy)
                for (int ix = 0; ix < W; ++ix)
                    U[(iy * cfg.up_y) * static_cast<size_t>(Wu) + ix * cfg.up_x] =
                        X[((static_cast<size_t>(n) * C + c) * H + iy) * W + ix];
            // Pad / crop.
            std::vector<float> P(static_cast<size_t>(Hp) * Wp, 0.0f);
            for (int uy = 0; uy < Hu; ++uy) {
                const int ty = uy + cfg.py0;
                if (ty < 0 || ty >= Hp) continue;
                for (int ux = 0; ux < Wu; ++ux) {
                    const int tx = ux + cfg.px0;
                    if (tx < 0 || tx >= Wp) continue;
                    P[ty * static_cast<size_t>(Wp) + tx] = U[uy * static_cast<size_t>(Wu) + ux];
                }
            }
            // Valid correlate + stride-down.
            for (int oy = 0; oy < Hout; ++oy)
                for (int ox = 0; ox < Wout; ++ox) {
                    const int cy = oy * cfg.down_y, cx = ox * cfg.down_x;
                    float acc = 0.0f;
                    for (int kh = 0; kh < cfg.fH; ++kh)
                        for (int kw = 0; kw < cfg.fW; ++kw)
                            acc += P[(cy + kh) * static_cast<size_t>(Wp) + (cx + kw)] *
                                   fe[kh * cfg.fW + kw];
                    Y[((static_cast<size_t>(n) * C + c) * Hout + oy) * Wout + ox] = acc;
                }
        }
    return Y;
}

static void run_case(const Cfg& cfg, int N, int C, int H, int W) {
    std::mt19937 rng(static_cast<uint32_t>(
        cfg.up_x * 131 + cfg.down_y * 17 + cfg.fH * 7 + cfg.fW + (cfg.flip ? 1000 : 0)));
    std::uniform_real_distribution<float> d(-1.5f, 1.5f);

    std::vector<float> x(static_cast<size_t>(N) * C * H * W), f(static_cast<size_t>(cfg.fH) * cfg.fW);
    for (auto& v : x) v = d(rng);
    for (auto& v : f) v = d(rng);

    int Hout = 0, Wout = 0;
    std::vector<float> Yref = ref_upfirdn(x, N, C, H, W, f, cfg, Hout, Wout);

    Tensor X = cpu_vec(x, N, C * H * W);
    Tensor F = cpu_vec(f, cfg.fH, cfg.fW);
    Tensor Y;
    brotensor::upfirdn2d_forward(X, F, N, C, H, W, cfg.fH, cfg.fW,
                                 cfg.up_x, cfg.up_y, cfg.down_x, cfg.down_y,
                                 cfg.px0, cfg.px1, cfg.py0, cfg.py1,
                                 cfg.flip, cfg.gain, Y);
    CHECK(Y.rows == N);
    CHECK(Y.cols == C * Hout * Wout);
    for (size_t i = 0; i < Yref.size() && i < (size_t)Y.size(); ++i)
        CHECK(close(Y[i], Yref[i]));

    // Backward vs finite difference. Loss = sum(w * Y), dY = w.
    std::vector<float> w(Yref.size());
    for (auto& v : w) v = d(rng);
    Tensor dY = cpu_vec(w, N, C * Hout * Wout), dX;
    brotensor::upfirdn2d_backward(dY, F, N, C, H, W, cfg.fH, cfg.fW,
                                  cfg.up_x, cfg.up_y, cfg.down_x, cfg.down_y,
                                  cfg.px0, cfg.px1, cfg.py0, cfg.py1,
                                  cfg.flip, cfg.gain, dX);
    CHECK(dX.rows == N);
    CHECK(dX.cols == C * H * W);

    const float h = 1e-3f;
    const int total = N * C * H * W;
    for (int i = 0; i < total; ++i) {
        std::vector<float> xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        int h1, w1, h2, w2;
        std::vector<float> Yp = ref_upfirdn(xp, N, C, H, W, f, cfg, h1, w1);
        std::vector<float> Ym = ref_upfirdn(xm, N, C, H, W, f, cfg, h2, w2);
        // Accumulate in double: the loss sums hundreds of O(1) terms and we
        // then take a small (~2h·deriv) difference — float would lose it to
        // catastrophic cancellation even though the op itself is exact.
        double lp = 0.0, lm = 0.0;
        for (size_t k = 0; k < w.size(); ++k) {
            lp += static_cast<double>(w[k]) * Yp[k];
            lm += static_cast<double>(w[k]) * Ym[k];
        }
        const float fd = static_cast<float>((lp - lm) / (2.0 * h));
        CHECK(close(dX[i], fd, 2e-2f));
    }
}

int main() {
    // up-only (2x), 4-tap filter, padding to keep the FIR centered.
    run_case({2, 2, 1, 1, 2, 1, 2, 1, 4, 4, false, 4.0f}, 2, 3, 5, 6);
    // down-only (2x), 4-tap.
    run_case({1, 1, 2, 2, 1, 2, 1, 2, 4, 4, false, 1.0f}, 1, 2, 8, 8);
    // up=2 with flip_filter=true.
    run_case({2, 2, 1, 1, 2, 1, 2, 1, 4, 4, true, 1.0f}, 1, 2, 4, 4);
    // asymmetric per-axis: up_x=2, up_y=1; non-square filter fH=1,fW=3.
    run_case({2, 1, 1, 1, 1, 1, 0, 0, 1, 3, false, 2.0f}, 2, 2, 4, 5);
    // combined up=2 down=2 (net identity scale), 3x3 filter.
    run_case({2, 2, 2, 2, 1, 1, 1, 1, 3, 3, false, 1.0f}, 1, 3, 6, 6);
    if (g_failures) {
        std::printf("upfirdn2d: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("upfirdn2d: all passed\n");
    return 0;
}
