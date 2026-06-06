// Standalone CPU coverage for the trainable LSTM cell (ops/lstm.h).
//
// Verifies:
//   * Forward matches a hand-rolled reference LSTM (PyTorch nn.LSTM math,
//     gate order i|f|g|o) over a multi-step, multi-batch sequence.
//   * lstm_backward is exact: every analytic gradient (dW_ih, dW_hh, db_ih,
//     db_hh, dX, dh0, dc0) matches a central finite-difference estimate of a
//     scalar MSE loss.
//   * The gradients are usable end-to-end: plain SGD over the LSTM params
//     drives the loss down on a fixed regression target.
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
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

// ── problem dimensions (H != I so a transpose bug can't hide) ────────────────
static const int T = 4;
static const int B = 2;
static const int I = 3;
static const int H = 5;
static const int G = 4 * H;

static Tensor make_random(int r, int c, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> v(static_cast<std::size_t>(r) * c);
    for (auto& x : v) x = u(rng);
    return Tensor::from_host(v.data(), r, c);
}

// Hand-rolled reference forward — independent of the kernel under test.
static std::vector<float> reference_forward(const Tensor& X, const Tensor& Wih,
                                            const Tensor& Whh, const Tensor& bih,
                                            const Tensor& bhh, const Tensor& h0,
                                            const Tensor& c0) {
    auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    std::vector<float> Y(static_cast<std::size_t>(T) * B * H);
    std::vector<float> Cs(static_cast<std::size_t>(T) * B * H);
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < B; ++b) {
            for (int n = 0; n < H; ++n) {
                float z[4];
                for (int gate = 0; gate < 4; ++gate) {
                    int roww = gate * H + n;
                    float acc = bih.at(roww, 0) + bhh.at(roww, 0);
                    for (int j = 0; j < I; ++j) acc += Wih.at(roww, j) * X.at(t * B + b, j);
                    for (int m = 0; m < H; ++m) {
                        float hp = (t == 0) ? h0.at(b, m) : Y[(static_cast<std::size_t>(t - 1) * B + b) * H + m];
                        acc += Whh.at(roww, m) * hp;
                    }
                    z[gate] = acc;
                }
                float ig = sig(z[0]), fg = sig(z[1]), gg = std::tanh(z[2]), og = sig(z[3]);
                float cprev = (t == 0) ? c0.at(b, n) : Cs[(static_cast<std::size_t>(t - 1) * B + b) * H + n];
                float cn = fg * cprev + ig * gg;
                Cs[(static_cast<std::size_t>(t) * B + b) * H + n] = cn;
                Y[(static_cast<std::size_t>(t) * B + b) * H + n] = og * std::tanh(cn);
            }
        }
    }
    return Y;
}

int main() {
    brotensor::init();
    std::mt19937 rng(20260605u);

    Tensor X   = make_random(T * B, I, rng, 0.6f);
    Tensor Wih = make_random(G, I, rng, 0.5f);
    Tensor Whh = make_random(G, H, rng, 0.5f);
    Tensor bih = make_random(G, 1, rng, 0.3f);
    Tensor bhh = make_random(G, 1, rng, 0.3f);
    Tensor h0  = make_random(B, H, rng, 0.4f);
    Tensor c0  = make_random(B, H, rng, 0.4f);
    Tensor target = make_random(T * B, H, rng, 0.5f);

    // ── forward vs reference ────────────────────────────────────────────────
    Tensor Y, gates, C;
    brotensor::lstm_forward_train(X, Wih, Whh, &bih, &bhh, &h0, &c0, T, B,
                                  Y, gates, C, nullptr, nullptr);
    std::vector<float> ref = reference_forward(X, Wih, Whh, bih, bhh, h0, c0);
    for (std::size_t i = 0; i < ref.size(); ++i)
        expect_near(Y[static_cast<int>(i)], ref[i], 1e-5, 1e-4,
                    "forward[" + std::to_string(i) + "]");

    // loss(Y) = 0.5 * sum (Y - target)^2 ;  dL/dY = Y - target
    auto loss_of = [&](const Tensor& Yv) {
        double s = 0.0;
        for (int i = 0; i < Yv.size(); ++i) { double d = Yv[i] - target[i]; s += 0.5 * d * d; }
        return s;
    };
    Tensor dY = Tensor::mat(T * B, H);
    for (int i = 0; i < dY.size(); ++i) dY[i] = Y[i] - target[i];

    // ── analytic gradients ──────────────────────────────────────────────────
    Tensor dX, dWih = Tensor::mat(G, I), dWhh = Tensor::mat(G, H),
               dbih = Tensor::mat(G, 1), dbhh = Tensor::mat(G, 1),
               dh0 = Tensor::mat(B, H), dc0 = Tensor::mat(B, H);
    brotensor::lstm_backward(X, Wih, Whh, &h0, &c0, Y, gates, C, dY, T, B,
                             dX, dWih, dWhh, &dbih, &dbhh, &dh0, &dc0);

    // ── finite-difference check ─────────────────────────────────────────────
    const float eps = 1e-3f;
    auto fd_check = [&](Tensor& P, const Tensor& analytic, const char* name) {
        for (int i = 0; i < P.size(); ++i) {
            float saved = P[i];
            P[i] = saved + eps;
            Tensor Yp, gp, cp; brotensor::lstm_forward_train(X, Wih, Whh, &bih, &bhh, &h0, &c0, T, B, Yp, gp, cp, nullptr, nullptr);
            double lp = loss_of(Yp);
            P[i] = saved - eps;
            Tensor Ym, gm, cm; brotensor::lstm_forward_train(X, Wih, Whh, &bih, &bhh, &h0, &c0, T, B, Ym, gm, cm, nullptr, nullptr);
            double lm = loss_of(Ym);
            P[i] = saved;
            double fd = (lp - lm) / (2.0 * eps);
            expect_near(analytic[i], fd, 2e-3, 3e-3,
                        std::string(name) + "[" + std::to_string(i) + "]");
        }
    };
    fd_check(Wih, dWih, "dW_ih");
    fd_check(Whh, dWhh, "dW_hh");
    fd_check(bih, dbih, "db_ih");
    fd_check(bhh, dbhh, "db_hh");
    fd_check(X,   dX,   "dX");
    fd_check(h0,  dh0,  "dh0");
    fd_check(c0,  dc0,  "dc0");

    // db_ih and db_hh receive the same gradient (mirrors nn.LSTM's two biases).
    for (int i = 0; i < dbih.size(); ++i)
        expect_near(dbih[i], dbhh[i], 1e-6, 1e-6, "db_ih==db_hh[" + std::to_string(i) + "]");

    // ── trainability: SGD drives the loss down ──────────────────────────────
    Tensor Wih2 = Wih.clone(), Whh2 = Whh.clone(), bih2 = bih.clone(), bhh2 = bhh.clone();
    double loss0 = 0.0, lossN = 0.0;
    const float lr = 0.05f;
    for (int step = 0; step < 400; ++step) {
        Tensor Ys, gs, cs;
        brotensor::lstm_forward_train(X, Wih2, Whh2, &bih2, &bhh2, &h0, &c0, T, B, Ys, gs, cs, nullptr, nullptr);
        double L = loss_of(Ys);
        if (step == 0) loss0 = L;
        lossN = L;
        Tensor dYs = Tensor::mat(T * B, H);
        for (int i = 0; i < dYs.size(); ++i) dYs[i] = Ys[i] - target[i];
        Tensor dXs, gWih = Tensor::mat(G, I), gWhh = Tensor::mat(G, H),
                    gbih = Tensor::mat(G, 1), gbhh = Tensor::mat(G, 1);
        brotensor::lstm_backward(X, Wih2, Whh2, &h0, &c0, Ys, gs, cs, dYs, T, B,
                                 dXs, gWih, gWhh, &gbih, &gbhh, nullptr, nullptr);
        for (int i = 0; i < Wih2.size(); ++i) Wih2[i] -= lr * gWih[i];
        for (int i = 0; i < Whh2.size(); ++i) Whh2[i] -= lr * gWhh[i];
        for (int i = 0; i < bih2.size(); ++i) bih2[i] -= lr * gbih[i];
        for (int i = 0; i < bhh2.size(); ++i) bhh2[i] -= lr * gbhh[i];
    }
    std::printf("  SGD loss %.5f -> %.5f over 400 steps\n", loss0, lossN);
    if (!(lossN < 0.25 * loss0)) {
        std::printf("  FAIL: SGD did not reduce loss enough (%.5f -> %.5f)\n", loss0, lossN);
        ++g_failures;
    }

    if (g_failures) { std::printf("test_lstm: %d failure(s)\n", g_failures); return 1; }
    std::printf("test_lstm: OK\n");
    return 0;
}
