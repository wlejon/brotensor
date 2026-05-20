// Standalone CPU op coverage. Validates the CPU backend's public op surface
// via hand-computed forward expectations and finite-difference gradient
// checks for ops with a backward.
//
// All tensors here are CPU-resident — this test is in the always-built group
// and must compile/pass on a CPU-only build. Ops are the device-neutral
// names from <brotensor/ops.h>; dispatch resolves to the CPU backend.
//
// Convention matches the rest of tests/: plain executable, exits non-zero on
// failure, no test framework.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

static bool near_(float a, float b, float abs_eps, float rel_eps) {
    const float d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const float m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

#define EXPECT_NEAR(actual, expected, abs_eps, rel_eps, ctx)                       \
    do {                                                                           \
        const float _a = (actual);                                                 \
        const float _e = (expected);                                               \
        if (!near_(_a, _e, (abs_eps), (rel_eps))) {                                \
            const float _d  = std::fabs(_a - _e);                                  \
            const float _m  = std::fmax(std::fabs(_a), std::fabs(_e));             \
            const float _rd = _m > 0.0f ? _d / _m : 0.0f;                          \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.9g expected=%.9g "          \
                        "abs=%.3g rel=%.3g\n",                                     \
                        __FILE__, __LINE__, (ctx), _a, _e, _d, _rd);               \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

#define EXPECT_TRUE(cond, ctx)                                                     \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",              \
                        __FILE__, __LINE__, (ctx), #cond);                         \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

// Helper: a zeroed CPU tensor of shape (r, c).
static Tensor cpu_zeros(int r, int c = 1) {
    return Tensor::zeros_on(Device::CPU, r, c);
}

// ---- Generic finite-difference gradient checker -----------------------------
//
// f: maps the contents of `in` (a flat float buffer view of length `n`) into a
//    scalar loss. The check perturbs each in[i] by ±h, evaluates the loss
//    twice, and compares (loss+ - loss-)/(2h) against analytic_grad[i].
//
// `in` points directly at a host tensor's FP32 buffer; the loss closures
// re-run the op on that same tensor object, so perturbing in place works.

template <typename Fn>
static void fd_check(const std::string& name,
                     float* in, int n,
                     const std::vector<float>& analytic_grad,
                     Fn&& loss_at,
                     float h = 1e-3f,
                     float abs_eps = 1e-3f,
                     float rel_eps = 1e-2f) {
    for (int i = 0; i < n; ++i) {
        const float saved = in[i];
        in[i] = saved + h;
        const float lp = loss_at();
        in[i] = saved - h;
        const float lm = loss_at();
        in[i] = saved;
        const float num = (lp - lm) / (2.0f * h);
        if (!near_(analytic_grad[i], num, abs_eps, rel_eps)) {
            const float d  = std::fabs(analytic_grad[i] - num);
            const float m  = std::fmax(std::fabs(analytic_grad[i]), std::fabs(num));
            const float rd = m > 0.0f ? d / m : 0.0f;
            std::printf("  FAIL  fd-grad %s  i=%d  analytic=%.6g numeric=%.6g "
                        "abs=%.3g rel=%.3g\n",
                        name.c_str(), i, analytic_grad[i], num, d, rd);
            ++g_failures;
        }
    }
}

// ---- linear -----------------------------------------------------------------

static void test_linear() {
    std::printf("linear_forward/backward\n");

    // Forward: W is 3x2, b len 3, x len 2.
    Tensor W = cpu_zeros(3, 2), b = cpu_zeros(3), x = cpu_zeros(2), y = cpu_zeros(3);
    W.at(0,0)=1; W.at(0,1)=2;
    W.at(1,0)=-1; W.at(1,1)=0.5f;
    W.at(2,0)=0; W.at(2,1)=3;
    b.host_f32_mut()[0] = 0.1f; b.host_f32_mut()[1] = -0.2f; b.host_f32_mut()[2] = 0.0f;
    x.host_f32_mut()[0] = 4.0f; x.host_f32_mut()[1] = -1.0f;
    brotensor::linear_forward(W, b, x, y);
    // y0 = 1*4 + 2*(-1) + 0.1 = 2.1
    // y1 = -1*4 + 0.5*(-1) -0.2 = -4.7
    // y2 = 0*4 + 3*(-1) + 0 = -3
    EXPECT_NEAR(y.host_f32()[0],  2.1f, 1e-6f, 1e-6f, "linear y0");
    EXPECT_NEAR(y.host_f32()[1], -4.7f, 1e-6f, 1e-6f, "linear y1");
    EXPECT_NEAR(y.host_f32()[2], -3.0f, 1e-6f, 1e-6f, "linear y2");

    // Backward (rectangular + square cases). For each, choose a dY vector and
    // do FD against an L(x,W,b) = dot(dY, y) scalar — under that loss,
    //   dx = W^T dY ; dW = dY x^T ; db = dY (then accumulated).
    auto run_one = [&](int out, int in, const char* tag) {
        Tensor Wt = cpu_zeros(out, in), bt = cpu_zeros(out), xt = cpu_zeros(in);
        for (int i = 0; i < out * in; ++i) Wt.host_f32_mut()[i] = 0.1f * static_cast<float>(i) - 0.3f;
        for (int i = 0; i < out; ++i)      bt.host_f32_mut()[i] = 0.05f * static_cast<float>(i) + 0.01f;
        for (int i = 0; i < in;  ++i)      xt.host_f32_mut()[i] = 0.2f * static_cast<float>(i) - 0.4f;

        std::vector<float> dY(out);
        for (int i = 0; i < out; ++i) dY[i] = 0.3f * static_cast<float>(i + 1) - 0.1f;

        Tensor yt = cpu_zeros(out);
        Tensor dXt = cpu_zeros(in);
        Tensor dWt = cpu_zeros(out, in);  // zeroed
        Tensor dBt = cpu_zeros(out);
        Tensor dYt = cpu_zeros(out);
        for (int i = 0; i < out; ++i) dYt.host_f32_mut()[i] = dY[i];

        brotensor::linear_forward(Wt, bt, xt, yt);
        brotensor::linear_backward(Wt, xt, dYt, dXt, dWt, dBt);

        auto loss_at = [&]() {
            Tensor yloc = cpu_zeros(out);
            brotensor::linear_forward(Wt, bt, xt, yloc);
            float s = 0.0f;
            for (int i = 0; i < out; ++i) s += dY[i] * yloc.host_f32()[i];
            return s;
        };

        // dX check.
        std::vector<float> g_dX(in);
        for (int i = 0; i < in; ++i) g_dX[i] = dXt.host_f32()[i];
        fd_check(std::string("linear/") + tag + "/dX", xt.host_f32_mut(), xt.size(),
                 g_dX, loss_at);

        // dW check.
        std::vector<float> g_dW(out * in);
        for (int i = 0; i < out * in; ++i) g_dW[i] = dWt.host_f32()[i];
        fd_check(std::string("linear/") + tag + "/dW", Wt.host_f32_mut(), Wt.size(),
                 g_dW, loss_at);

        // dB check.
        std::vector<float> g_dB(out);
        for (int i = 0; i < out; ++i) g_dB[i] = dBt.host_f32()[i];
        fd_check(std::string("linear/") + tag + "/dB", bt.host_f32_mut(), bt.size(),
                 g_dB, loss_at);

        // Verify accumulation: re-call backward, dW/dB should double.
        brotensor::linear_backward(Wt, xt, dYt, dXt, dWt, dBt);
        for (int i = 0; i < out * in; ++i) {
            EXPECT_NEAR(dWt.host_f32()[i], 2.0f * g_dW[i], 1e-5f, 1e-5f, "linear dW accum");
        }
        for (int i = 0; i < out; ++i) {
            EXPECT_NEAR(dBt.host_f32()[i], 2.0f * g_dB[i], 1e-5f, 1e-5f, "linear dB accum");
        }
    };

    run_one(3, 2, "rect3x2");
    run_one(4, 4, "sq4x4");
}

// ---- Elementwise activations ------------------------------------------------

static void test_relu() {
    std::printf("relu_forward/backward\n");
    Tensor x = cpu_zeros(4), y = cpu_zeros(4);
    x.host_f32_mut()[0] = -1.5f; x.host_f32_mut()[1] = 0.0f;
    x.host_f32_mut()[2] = 0.5f;  x.host_f32_mut()[3] = 2.0f;
    brotensor::relu_forward(x, y);
    EXPECT_NEAR(y.host_f32()[0], 0.0f, 1e-7f, 1e-7f, "relu y0");
    EXPECT_NEAR(y.host_f32()[1], 0.0f, 1e-7f, 1e-7f, "relu y1");
    EXPECT_NEAR(y.host_f32()[2], 0.5f, 1e-7f, 1e-7f, "relu y2");
    EXPECT_NEAR(y.host_f32()[3], 2.0f, 1e-7f, 1e-7f, "relu y3");

    // FD gradient. Use strictly nonzero x to avoid the kink at 0.
    Tensor xb = cpu_zeros(5), dY = cpu_zeros(5), dX = cpu_zeros(5);
    float* xbp = xb.host_f32_mut();
    xbp[0] = -2.3f; xbp[1] = -0.1f; xbp[2] = 0.4f; xbp[3] = 1.7f; xbp[4] = -0.8f;
    float* dYp = dY.host_f32_mut();
    dYp[0] = 0.5f; dYp[1] = -0.3f; dYp[2] = 1.0f; dYp[3] = 0.7f; dYp[4] = -0.4f;
    brotensor::relu_backward(xb, dY, dX);
    std::vector<float> g(5); for (int i = 0; i < 5; ++i) g[i] = dX.host_f32()[i];

    auto loss = [&]() {
        Tensor yloc = cpu_zeros(5);
        brotensor::relu_forward(xb, yloc);
        float s = 0.0f;
        for (int i = 0; i < 5; ++i) s += dY.host_f32()[i] * yloc.host_f32()[i];
        return s;
    };
    fd_check("relu/dX", xb.host_f32_mut(), xb.size(), g, loss);
}

static void test_tanh() {
    std::printf("tanh_forward/backward\n");
    Tensor x = cpu_zeros(4), y = cpu_zeros(4);
    float* xp = x.host_f32_mut();
    xp[0] = -1.0f; xp[1] = 0.0f; xp[2] = 0.5f; xp[3] = 2.0f;
    brotensor::tanh_forward(x, y);
    EXPECT_NEAR(y.host_f32()[0], std::tanh(-1.0f), 1e-6f, 1e-6f, "tanh y0");
    EXPECT_NEAR(y.host_f32()[1], 0.0f,             1e-7f, 1e-7f, "tanh y1");
    EXPECT_NEAR(y.host_f32()[2], std::tanh(0.5f),  1e-6f, 1e-6f, "tanh y2");
    EXPECT_NEAR(y.host_f32()[3], std::tanh(2.0f),  1e-6f, 1e-6f, "tanh y3");

    Tensor xb = cpu_zeros(6), yb = cpu_zeros(6), dY = cpu_zeros(6), dX = cpu_zeros(6);
    for (int i = 0; i < 6; ++i) {
        xb.host_f32_mut()[i] = 0.4f * i - 1.2f;
        dY.host_f32_mut()[i] = 0.1f * i - 0.25f;
    }
    brotensor::tanh_forward(xb, yb);
    brotensor::tanh_backward(yb, dY, dX);
    std::vector<float> g(6); for (int i = 0; i < 6; ++i) g[i] = dX.host_f32()[i];
    auto loss = [&]() {
        Tensor yloc = cpu_zeros(6);
        brotensor::tanh_forward(xb, yloc);
        float s = 0.0f;
        for (int i = 0; i < 6; ++i) s += dY.host_f32()[i] * yloc.host_f32()[i];
        return s;
    };
    fd_check("tanh/dX", xb.host_f32_mut(), xb.size(), g, loss);
}

static void test_sigmoid() {
    std::printf("sigmoid_forward/backward\n");
    Tensor x = cpu_zeros(4), y = cpu_zeros(4);
    float* xp = x.host_f32_mut();
    xp[0] = -2.0f; xp[1] = 0.0f; xp[2] = 1.0f; xp[3] = 3.0f;
    brotensor::sigmoid_forward(x, y);
    for (int i = 0; i < 4; ++i) {
        const float r = 1.0f / (1.0f + std::exp(-x.host_f32()[i]));
        EXPECT_NEAR(y.host_f32()[i], r, 1e-6f, 1e-6f, "sigmoid forward");
    }
    EXPECT_NEAR(y.host_f32()[1], 0.5f, 1e-7f, 1e-7f, "sigmoid y1==0.5");

    Tensor xb = cpu_zeros(6), yb = cpu_zeros(6), dY = cpu_zeros(6), dX = cpu_zeros(6);
    for (int i = 0; i < 6; ++i) {
        xb.host_f32_mut()[i] = 0.5f * i - 1.5f;
        dY.host_f32_mut()[i] = 0.2f * i - 0.5f;
    }
    brotensor::sigmoid_forward(xb, yb);
    brotensor::sigmoid_backward(yb, dY, dX);
    std::vector<float> g(6); for (int i = 0; i < 6; ++i) g[i] = dX.host_f32()[i];
    auto loss = [&]() {
        Tensor yloc = cpu_zeros(6);
        brotensor::sigmoid_forward(xb, yloc);
        float s = 0.0f;
        for (int i = 0; i < 6; ++i) s += dY.host_f32()[i] * yloc.host_f32()[i];
        return s;
    };
    fd_check("sigmoid/dX", xb.host_f32_mut(), xb.size(), g, loss);
}

// ---- Softmax ----------------------------------------------------------------

static void test_softmax() {
    std::printf("softmax_forward/backward\n");

    // Forward, unmasked, against hand-computed result.
    Tensor lg = cpu_zeros(4), pr = cpu_zeros(4);
    float* lgp = lg.host_f32_mut();
    lgp[0] = 1.0f; lgp[1] = 2.0f; lgp[2] = 3.0f; lgp[3] = 4.0f;
    brotensor::softmax_forward(lg, pr);
    float ex[4]; float Z = 0.0f;
    const float m = 4.0f;
    for (int i = 0; i < 4; ++i) { ex[i] = std::exp(lg.host_f32()[i] - m); Z += ex[i]; }
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(pr.host_f32()[i], ex[i] / Z, 1e-6f, 1e-6f, "softmax unmasked");
    }
    float s = 0.0f; for (int i = 0; i < 4; ++i) s += pr.host_f32()[i];
    EXPECT_NEAR(s, 1.0f, 1e-6f, 1e-6f, "softmax sum");

    // Forward, partial mask (mask out index 1). Mask is a host buffer — same
    // device (CPU) as the tensor operands.
    Tensor lg2 = cpu_zeros(4), pr2 = cpu_zeros(4);
    float* lg2p = lg2.host_f32_mut();
    lg2p[0] = 1.0f; lg2p[1] = 2.0f; lg2p[2] = 3.0f; lg2p[3] = 4.0f;
    std::vector<float> mask = {1.0f, 0.0f, 1.0f, 1.0f};
    brotensor::softmax_forward(lg2, pr2, mask.data());
    EXPECT_NEAR(pr2.host_f32()[1], 0.0f, 1e-7f, 1e-7f, "softmax masked entry zero");
    float Z2 = std::exp(1.0f - 4.0f) + std::exp(3.0f - 4.0f) + std::exp(4.0f - 4.0f);
    EXPECT_NEAR(pr2.host_f32()[0], std::exp(1.0f - 4.0f) / Z2, 1e-6f, 1e-6f, "softmax masked p0");
    EXPECT_NEAR(pr2.host_f32()[2], std::exp(3.0f - 4.0f) / Z2, 1e-6f, 1e-6f, "softmax masked p2");
    EXPECT_NEAR(pr2.host_f32()[3], std::exp(4.0f - 4.0f) / Z2, 1e-6f, 1e-6f, "softmax masked p3");
    float s2 = pr2.host_f32()[0] + pr2.host_f32()[1] + pr2.host_f32()[2] + pr2.host_f32()[3];
    EXPECT_NEAR(s2, 1.0f, 1e-6f, 1e-6f, "softmax masked sum");

    // Backward FD against scalar loss dot(dProbs, probs).
    Tensor lgb = cpu_zeros(5), prb = cpu_zeros(5), dP = cpu_zeros(5), dZ = cpu_zeros(5);
    for (int i = 0; i < 5; ++i) {
        lgb.host_f32_mut()[i] = 0.4f * i - 0.7f;
        dP.host_f32_mut()[i]  = 0.3f * i - 0.4f;
    }
    brotensor::softmax_forward(lgb, prb);
    brotensor::softmax_backward(prb, dP, dZ);
    std::vector<float> g(5); for (int i = 0; i < 5; ++i) g[i] = dZ.host_f32()[i];
    auto loss = [&]() {
        Tensor pl = cpu_zeros(5);
        brotensor::softmax_forward(lgb, pl);
        float t = 0.0f;
        for (int i = 0; i < 5; ++i) t += dP.host_f32()[i] * pl.host_f32()[i];
        return t;
    };
    fd_check("softmax/dLogits", lgb.host_f32_mut(), lgb.size(), g, loss);
}

static void test_softmax_xent() {
    std::printf("softmax_xent / softmax_xent_segment\n");

    // Hand-computed: logits = [0,0,0], target=[1,0,0] one-hot → p = [1/3]*3,
    // loss = -log(1/3) = log(3).
    {
        Tensor lg = cpu_zeros(3), tg = cpu_zeros(3), pr = cpu_zeros(3), dz = cpu_zeros(3);
        lg.host_f32_mut()[0]=0; lg.host_f32_mut()[1]=0; lg.host_f32_mut()[2]=0;
        tg.host_f32_mut()[0]=1; tg.host_f32_mut()[1]=0; tg.host_f32_mut()[2]=0;
        const float L = brotensor::softmax_xent(lg, tg, pr, dz);
        EXPECT_NEAR(L, std::log(3.0f), 1e-6f, 1e-6f, "xent loss log3");
        EXPECT_NEAR(pr.host_f32()[0], 1.0f / 3.0f, 1e-6f, 1e-6f, "xent p0");
        // dLogits == p - target.
        EXPECT_NEAR(dz.host_f32()[0], 1.0f / 3.0f - 1.0f, 1e-6f, 1e-6f, "xent dz0");
        EXPECT_NEAR(dz.host_f32()[1], 1.0f / 3.0f, 1e-6f, 1e-6f, "xent dz1");
        EXPECT_NEAR(dz.host_f32()[2], 1.0f / 3.0f, 1e-6f, 1e-6f, "xent dz2");
    }

    // Masked: mask out index 1; target only on legal entries.
    {
        Tensor lg = cpu_zeros(4), tg = cpu_zeros(4), pr = cpu_zeros(4), dz = cpu_zeros(4);
        float* lgp = lg.host_f32_mut();
        lgp[0]=0.5f; lgp[1]=2.0f; lgp[2]=-0.3f; lgp[3]=1.2f;
        float* tgp = tg.host_f32_mut();
        tgp[0]=0.7f; tgp[1]=0.0f; tgp[2]=0.3f;  tgp[3]=0.0f;
        std::vector<float> mask = {1, 0, 1, 1};
        const float L = brotensor::softmax_xent(lg, tg, pr, dz, mask.data());

        // Recompute expected p.
        const float mx = std::fmax(std::fmax(lgp[0], lgp[2]), lgp[3]);
        float e0 = std::exp(lgp[0]-mx), e2 = std::exp(lgp[2]-mx), e3 = std::exp(lgp[3]-mx);
        const float Z = e0 + e2 + e3;
        const float p0 = e0/Z, p2 = e2/Z, p3 = e3/Z;
        EXPECT_NEAR(pr.host_f32()[0], p0, 1e-6f, 1e-6f, "xent masked p0");
        EXPECT_NEAR(pr.host_f32()[1], 0.0f, 1e-7f, 1e-7f, "xent masked p1");
        EXPECT_NEAR(pr.host_f32()[2], p2, 1e-6f, 1e-6f, "xent masked p2");
        EXPECT_NEAR(pr.host_f32()[3], p3, 1e-6f, 1e-6f, "xent masked p3");

        const float Lexp = -(0.7f * std::log(p0) + 0.3f * std::log(p2));
        EXPECT_NEAR(L, Lexp, 1e-5f, 1e-5f, "xent masked loss");
        EXPECT_NEAR(dz.host_f32()[1], 0.0f, 1e-7f, 1e-7f, "xent masked dz1");
        EXPECT_NEAR(dz.host_f32()[0], p0 - 0.7f, 1e-6f, 1e-6f, "xent masked dz0");
    }

    // Segment form should match Tensor form on the same data.
    {
        std::vector<float> lg = {0.1f, -0.4f, 0.9f, 0.0f};
        std::vector<float> tg = {0.0f, 0.0f, 1.0f, 0.0f};
        std::vector<float> p2(4), dz2(4);

        Tensor LG = cpu_zeros(4), TG = cpu_zeros(4), P = cpu_zeros(4), DZ = cpu_zeros(4);
        for (int i = 0; i < 4; ++i) {
            LG.host_f32_mut()[i] = lg[i];
            TG.host_f32_mut()[i] = tg[i];
        }
        const float La = brotensor::softmax_xent(LG, TG, P, DZ);
        const float Lb = brotensor::softmax_xent_segment(lg.data(), tg.data(),
                                                         p2.data(), dz2.data(), 4);
        EXPECT_NEAR(La, Lb, 1e-7f, 1e-7f, "xent segment vs tensor loss");
        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR(P.host_f32()[i],  p2[i],  1e-7f, 1e-7f, "xent segment probs");
            EXPECT_NEAR(DZ.host_f32()[i], dz2[i], 1e-7f, 1e-7f, "xent segment dz");
        }
    }
}

// ---- MSE --------------------------------------------------------------------

static void test_mse() {
    std::printf("mse_scalar\n");
    float dp = 0.0f;
    const float L = brotensor::mse_scalar(2.5f, 1.5f, dp);
    EXPECT_NEAR(L, 0.5f, 1e-7f, 1e-7f, "mse loss 0.5*(1)^2");
    EXPECT_NEAR(dp, 1.0f, 1e-7f, 1e-7f, "mse dp");

    // FD: d/dpred (0.5 (pred-t)^2) = pred-t.
    const float h = 1e-3f;
    float dl;
    const float lp = brotensor::mse_scalar(2.5f + h, 1.5f, dl);
    const float lm = brotensor::mse_scalar(2.5f - h, 1.5f, dl);
    const float num = (lp - lm) / (2.0f * h);
    EXPECT_NEAR(num, dp, 1e-3f, 1e-3f, "mse fd");
}

// ---- add_inplace / add_scalar_inplace ---------------------------------------

static void test_add() {
    std::printf("add_inplace / add_scalar_inplace\n");
    Tensor y = cpu_zeros(4), x = cpu_zeros(4);
    float* yp = y.host_f32_mut();
    yp[0]=1; yp[1]=2; yp[2]=-3; yp[3]=0.5f;
    float* xp = x.host_f32_mut();
    xp[0]=0.1f; xp[1]=-2; xp[2]=0.5f; xp[3]=10;
    brotensor::add_inplace(y, x);
    EXPECT_NEAR(y.host_f32()[0], 1.1f, 1e-6f, 1e-6f, "add y0");
    EXPECT_NEAR(y.host_f32()[1], 0.0f, 1e-6f, 1e-6f, "add y1");
    EXPECT_NEAR(y.host_f32()[2], -2.5f, 1e-6f, 1e-6f, "add y2");
    EXPECT_NEAR(y.host_f32()[3], 10.5f, 1e-6f, 1e-6f, "add y3");

    brotensor::add_scalar_inplace(y, 0.5f);
    EXPECT_NEAR(y.host_f32()[0], 1.6f, 1e-6f, 1e-6f, "addS y0");
    EXPECT_NEAR(y.host_f32()[1], 0.5f, 1e-6f, 1e-6f, "addS y1");
    EXPECT_NEAR(y.host_f32()[2], -2.0f, 1e-6f, 1e-6f, "addS y2");
    EXPECT_NEAR(y.host_f32()[3], 11.0f, 1e-6f, 1e-6f, "addS y3");
}

// ---- Xavier init ------------------------------------------------------------

static void test_xavier() {
    std::printf("xavier_init\n");
    const int rows = 5, cols = 7;

    Tensor A = cpu_zeros(rows, cols), B = cpu_zeros(rows, cols);
    EXPECT_TRUE(A.size() == rows * cols, "xavier size");

    uint64_t sa = 0xDEADBEEFCAFEBABEULL, sb = 0xDEADBEEFCAFEBABEULL;
    brotensor::xavier_init(A, sa);
    brotensor::xavier_init(B, sb);
    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(A.host_f32()[i], B.host_f32()[i], 0.0f, 0.0f, "xavier determinism");
    }
    EXPECT_TRUE(sa == sb, "xavier rng_state advanced identically");

    const float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    for (int i = 0; i < rows * cols; ++i) {
        const float v = A.host_f32()[i];
        EXPECT_TRUE(v >= -limit && v <= limit, "xavier bound");
    }

    // Different seed => different output (cannot reasonably collide on
    // 5*7=35 floats with splitmix64).
    Tensor C = cpu_zeros(rows, cols);
    uint64_t sc = 0x1234567890ABCDEFULL;
    brotensor::xavier_init(C, sc);
    bool any_diff = false;
    for (int i = 0; i < rows * cols; ++i)
        if (A.host_f32()[i] != C.host_f32()[i]) { any_diff = true; break; }
    EXPECT_TRUE(any_diff, "xavier seed-dependent");
}

int main() {
    brotensor::init();
    std::printf("test_cpu_ops\n");

    test_linear();
    test_relu();
    test_tanh();
    test_sigmoid();
    test_softmax();
    test_softmax_xent();
    test_mse();
    test_add();
    test_xavier();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll CPU op checks passed.\n");
    return 0;
}
