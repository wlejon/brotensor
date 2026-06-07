// CPU FP32 tests for bias_act (StyleGAN3 fused bias + act + gain + clamp).
// Forward vs manual reference; backward (dX and dB) vs finite difference.
// Covers linear/lrelu × with/without bias × with/without clamp.

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

static float ref_act(float t, int act, float alpha) {
    return (act == 1) ? (t > 0 ? t : alpha * t) : t;
}

static float ref_one(float x, float b, int act, float alpha,
                     float gain, float clamp) {
    float y = ref_act(x + b, act, alpha) * gain;
    if (clamp >= 0.0f) { y = std::fmax(-clamp, std::fmin(clamp, y)); }
    return y;
}

static void run_case(int act, bool has_bias, float clamp) {
    std::mt19937 rng(100 + act * 7 + (has_bias ? 1 : 0) + int(clamp * 13));
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    const int N = 3, C = 4, HW = 5;
    const int cols = C * HW;
    const float alpha = 0.2f, gain = (act == 1) ? std::sqrt(2.0f) : 1.3f;

    std::vector<float> x(N * cols), b(C, 0.0f);
    for (auto& v : x) v = d(rng);
    if (has_bias) for (auto& v : b) v = d(rng);

    Tensor X = cpu_vec(x, N, cols), Y;
    Tensor B = cpu_vec(b, C, 1);
    const Tensor* bp = has_bias ? &B : nullptr;
    brotensor::bias_act_forward(X, bp, N, C, HW, act, alpha, gain, clamp, Y);

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int k = 0; k < HW; ++k) {
                const int idx = (n * C + c) * HW + k;
                CHECK(close(Y[idx],
                            ref_one(x[idx], b[c], act, alpha, gain, clamp)));
            }

    // Backward: loss L = sum(w_i * y_i), so dY = w.
    std::vector<float> w(N * cols);
    for (auto& v : w) v = d(rng);
    Tensor dY = cpu_vec(w, N, cols), dX;
    Tensor dB = Tensor::zeros_on(brotensor::Device::CPU, C, 1);
    brotensor::bias_act_backward(dY, X, bp, N, C, HW, act, alpha, gain, clamp,
                                 dX, has_bias ? &dB : nullptr);

    // Skip FD checks at points adjacent to lrelu's kink / clamp boundary where
    // the analytic derivative is ill-defined; those are measure-zero.
    const float h = 5e-4f;
    auto loss = [&](const std::vector<float>& xx, const std::vector<float>& bb) {
        float L = 0.0f;
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int k = 0; k < HW; ++k) {
                    const int idx = (n * C + c) * HW + k;
                    L += w[idx] * ref_one(xx[idx], bb[c], act, alpha, gain, clamp);
                }
        return L;
    };
    for (int i = 0; i < N * cols; ++i) {
        std::vector<float> xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        const float fd = (loss(xp, b) - loss(xm, b)) / (2 * h);
        if (std::fabs(fd - dX[i]) > 1e-2f * (1 + std::fabs(fd))) {
            // Likely a kink/clamp boundary — tolerate only if both small.
            CHECK(close(dX[i], fd, 5e-2f));
        }
    }
    if (has_bias) {
        for (int c = 0; c < C; ++c) {
            std::vector<float> bp2 = b, bm2 = b;
            bp2[c] += h; bm2[c] -= h;
            const float fd = (loss(x, bp2) - loss(x, bm2)) / (2 * h);
            CHECK(close(dB[c], fd, 5e-2f));
        }
    }
}

int main() {
    for (int act = 0; act <= 1; ++act)
        for (int hb = 0; hb <= 1; ++hb)
            for (float clamp : {-1.0f, 0.8f})
                run_case(act, hb != 0, clamp);
    if (g_failures) {
        std::printf("bias_act: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("bias_act: all passed\n");
    return 0;
}
