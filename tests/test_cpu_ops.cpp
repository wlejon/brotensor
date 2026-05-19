// Standalone CPU op coverage. Validates every public symbol in
// <brotensor/ops_cpu.h> via hand-computed forward expectations and
// finite-difference gradient checks for ops with a backward.
//
// Convention matches the rest of tests/: plain executable, exits non-zero on
// failure, no test framework.

#include <brotensor/ops_cpu.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

// ---- Generic finite-difference gradient checker -----------------------------
//
// f: maps the contents of `in` (a flat float buffer view) into a scalar loss
//    and writes the analytic dIn into `dIn`. The check perturbs each in[i]
//    by ±h, evaluates the loss twice, and compares (loss+ - loss-)/(2h)
//    against dIn[i].
//
// Used per-op by wrapping the op's forward + a closed-form scalar loss (sum
// or dot-with-upstream) into the lambda.

template <typename Fn>
static void fd_check(const std::string& name,
                     std::vector<float>& in,
                     const std::vector<float>& analytic_grad,
                     Fn&& loss_at,
                     float h = 1e-3f,
                     float abs_eps = 1e-3f,
                     float rel_eps = 1e-2f) {
    const int n = static_cast<int>(in.size());
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
    std::printf("linear_forward/backward_cpu\n");

    // Forward: W is 3x2, b len 3, x len 2.
    Tensor W(3, 2), b = Tensor::vec(3), x = Tensor::vec(2), y = Tensor::vec(3);
    W(0,0)=1; W(0,1)=2;
    W(1,0)=-1; W(1,1)=0.5f;
    W(2,0)=0; W(2,1)=3;
    b[0] = 0.1f; b[1] = -0.2f; b[2] = 0.0f;
    x[0] = 4.0f; x[1] = -1.0f;
    brotensor::linear_forward_cpu(W, b, x, y);
    // y0 = 1*4 + 2*(-1) + 0.1 = 2.1
    // y1 = -1*4 + 0.5*(-1) -0.2 = -4.7
    // y2 = 0*4 + 3*(-1) + 0 = -3
    EXPECT_NEAR(y[0],  2.1f, 1e-6f, 1e-6f, "linear y0");
    EXPECT_NEAR(y[1], -4.7f, 1e-6f, 1e-6f, "linear y1");
    EXPECT_NEAR(y[2], -3.0f, 1e-6f, 1e-6f, "linear y2");

    // Backward (rectangular + square cases). For each, choose a dY vector and
    // do FD against an L(x,W,b) = dot(dY, y) scalar — under that loss,
    //   dx = W^T dY ; dW = dY x^T ; db = dY (then accumulated).
    auto run_one = [&](int out, int in, const char* tag) {
        Tensor Wt(out, in), bt = Tensor::vec(out), xt = Tensor::vec(in);
        for (int i = 0; i < out * in; ++i) Wt[i] = 0.1f * static_cast<float>(i) - 0.3f;
        for (int i = 0; i < out; ++i)      bt[i] = 0.05f * static_cast<float>(i) + 0.01f;
        for (int i = 0; i < in;  ++i)      xt[i] = 0.2f * static_cast<float>(i) - 0.4f;

        std::vector<float> dY(out);
        for (int i = 0; i < out; ++i) dY[i] = 0.3f * static_cast<float>(i + 1) - 0.1f;

        Tensor yt = Tensor::vec(out);
        Tensor dXt = Tensor::vec(in);
        Tensor dWt(out, in);  // zeroed by ctor
        Tensor dBt = Tensor::vec(out);
        Tensor dYt = Tensor::vec(out);
        for (int i = 0; i < out; ++i) dYt[i] = dY[i];

        brotensor::linear_forward_cpu(Wt, bt, xt, yt);
        brotensor::linear_backward_cpu(Wt, xt, dYt, dXt, dWt, dBt);

        auto loss_at = [&]() {
            Tensor yloc = Tensor::vec(out);
            brotensor::linear_forward_cpu(Wt, bt, xt, yloc);
            float s = 0.0f;
            for (int i = 0; i < out; ++i) s += dY[i] * yloc[i];
            return s;
        };

        // dX check.
        std::vector<float> g_dX(in);
        for (int i = 0; i < in; ++i) g_dX[i] = dXt[i];
        fd_check(std::string("linear/") + tag + "/dX", xt.data, g_dX, loss_at);

        // dW check.
        std::vector<float> g_dW(out * in);
        for (int i = 0; i < out * in; ++i) g_dW[i] = dWt[i];
        fd_check(std::string("linear/") + tag + "/dW", Wt.data, g_dW, loss_at);

        // dB check.
        std::vector<float> g_dB(out);
        for (int i = 0; i < out; ++i) g_dB[i] = dBt[i];
        fd_check(std::string("linear/") + tag + "/dB", bt.data, g_dB, loss_at);

        // Verify accumulation: re-call backward, dW/dB should double.
        brotensor::linear_backward_cpu(Wt, xt, dYt, dXt, dWt, dBt);
        for (int i = 0; i < out * in; ++i) {
            EXPECT_NEAR(dWt[i], 2.0f * g_dW[i], 1e-5f, 1e-5f, "linear dW accum");
        }
        for (int i = 0; i < out; ++i) {
            EXPECT_NEAR(dBt[i], 2.0f * g_dB[i], 1e-5f, 1e-5f, "linear dB accum");
        }
    };

    run_one(3, 2, "rect3x2");
    run_one(4, 4, "sq4x4");
}

// ---- Elementwise activations ------------------------------------------------

static void test_relu() {
    std::printf("relu_forward/backward_cpu\n");
    Tensor x = Tensor::vec(4), y = Tensor::vec(4);
    x[0] = -1.5f; x[1] = 0.0f; x[2] = 0.5f; x[3] = 2.0f;
    brotensor::relu_forward_cpu(x, y);
    EXPECT_NEAR(y[0], 0.0f, 1e-7f, 1e-7f, "relu y0");
    EXPECT_NEAR(y[1], 0.0f, 1e-7f, 1e-7f, "relu y1");
    EXPECT_NEAR(y[2], 0.5f, 1e-7f, 1e-7f, "relu y2");
    EXPECT_NEAR(y[3], 2.0f, 1e-7f, 1e-7f, "relu y3");

    // FD gradient. Use strictly nonzero x to avoid the kink at 0.
    Tensor xb = Tensor::vec(5), dY = Tensor::vec(5), dX = Tensor::vec(5);
    xb[0] = -2.3f; xb[1] = -0.1f; xb[2] = 0.4f; xb[3] = 1.7f; xb[4] = -0.8f;
    dY[0] = 0.5f; dY[1] = -0.3f; dY[2] = 1.0f; dY[3] = 0.7f; dY[4] = -0.4f;
    brotensor::relu_backward_cpu(xb, dY, dX);
    std::vector<float> g(5); for (int i = 0; i < 5; ++i) g[i] = dX[i];

    auto loss = [&]() {
        Tensor yloc = Tensor::vec(5);
        brotensor::relu_forward_cpu(xb, yloc);
        float s = 0.0f;
        for (int i = 0; i < 5; ++i) s += dY[i] * yloc[i];
        return s;
    };
    fd_check("relu/dX", xb.data, g, loss);
}

static void test_tanh() {
    std::printf("tanh_forward/backward_cpu\n");
    Tensor x = Tensor::vec(4), y = Tensor::vec(4);
    x[0] = -1.0f; x[1] = 0.0f; x[2] = 0.5f; x[3] = 2.0f;
    brotensor::tanh_forward_cpu(x, y);
    EXPECT_NEAR(y[0], std::tanh(-1.0f), 1e-6f, 1e-6f, "tanh y0");
    EXPECT_NEAR(y[1], 0.0f,              1e-7f, 1e-7f, "tanh y1");
    EXPECT_NEAR(y[2], std::tanh(0.5f),   1e-6f, 1e-6f, "tanh y2");
    EXPECT_NEAR(y[3], std::tanh(2.0f),   1e-6f, 1e-6f, "tanh y3");

    Tensor xb = Tensor::vec(6), yb = Tensor::vec(6), dY = Tensor::vec(6), dX = Tensor::vec(6);
    for (int i = 0; i < 6; ++i) { xb[i] = 0.4f * i - 1.2f; dY[i] = 0.1f * i - 0.25f; }
    brotensor::tanh_forward_cpu(xb, yb);
    brotensor::tanh_backward_cpu(yb, dY, dX);
    std::vector<float> g(6); for (int i = 0; i < 6; ++i) g[i] = dX[i];
    auto loss = [&]() {
        Tensor yloc = Tensor::vec(6);
        brotensor::tanh_forward_cpu(xb, yloc);
        float s = 0.0f;
        for (int i = 0; i < 6; ++i) s += dY[i] * yloc[i];
        return s;
    };
    fd_check("tanh/dX", xb.data, g, loss);
}

static void test_sigmoid() {
    std::printf("sigmoid_forward/backward_cpu\n");
    Tensor x = Tensor::vec(4), y = Tensor::vec(4);
    x[0] = -2.0f; x[1] = 0.0f; x[2] = 1.0f; x[3] = 3.0f;
    brotensor::sigmoid_forward_cpu(x, y);
    for (int i = 0; i < 4; ++i) {
        const float r = 1.0f / (1.0f + std::exp(-x[i]));
        EXPECT_NEAR(y[i], r, 1e-6f, 1e-6f, "sigmoid forward");
    }
    EXPECT_NEAR(y[1], 0.5f, 1e-7f, 1e-7f, "sigmoid y1==0.5");

    Tensor xb = Tensor::vec(6), yb = Tensor::vec(6), dY = Tensor::vec(6), dX = Tensor::vec(6);
    for (int i = 0; i < 6; ++i) { xb[i] = 0.5f * i - 1.5f; dY[i] = 0.2f * i - 0.5f; }
    brotensor::sigmoid_forward_cpu(xb, yb);
    brotensor::sigmoid_backward_cpu(yb, dY, dX);
    std::vector<float> g(6); for (int i = 0; i < 6; ++i) g[i] = dX[i];
    auto loss = [&]() {
        Tensor yloc = Tensor::vec(6);
        brotensor::sigmoid_forward_cpu(xb, yloc);
        float s = 0.0f;
        for (int i = 0; i < 6; ++i) s += dY[i] * yloc[i];
        return s;
    };
    fd_check("sigmoid/dX", xb.data, g, loss);
}

// ---- Softmax ----------------------------------------------------------------

static void test_softmax() {
    std::printf("softmax_forward/backward_cpu\n");

    // Forward, unmasked, against hand-computed result.
    Tensor lg = Tensor::vec(4), pr = Tensor::vec(4);
    lg[0] = 1.0f; lg[1] = 2.0f; lg[2] = 3.0f; lg[3] = 4.0f;
    brotensor::softmax_forward_cpu(lg, pr);
    float ex[4]; float Z = 0.0f;
    const float m = 4.0f;
    for (int i = 0; i < 4; ++i) { ex[i] = std::exp(lg[i] - m); Z += ex[i]; }
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(pr[i], ex[i] / Z, 1e-6f, 1e-6f, "softmax unmasked");
    }
    float s = 0.0f; for (int i = 0; i < 4; ++i) s += pr[i];
    EXPECT_NEAR(s, 1.0f, 1e-6f, 1e-6f, "softmax sum");

    // Forward, partial mask (mask out index 1).
    Tensor lg2 = Tensor::vec(4), pr2 = Tensor::vec(4);
    lg2[0] = 1.0f; lg2[1] = 2.0f; lg2[2] = 3.0f; lg2[3] = 4.0f;
    const float mask[4] = {1.0f, 0.0f, 1.0f, 1.0f};
    brotensor::softmax_forward_cpu(lg2, pr2, mask);
    EXPECT_NEAR(pr2[1], 0.0f, 1e-7f, 1e-7f, "softmax masked entry zero");
    float Z2 = std::exp(1.0f - 4.0f) + std::exp(3.0f - 4.0f) + std::exp(4.0f - 4.0f);
    EXPECT_NEAR(pr2[0], std::exp(1.0f - 4.0f) / Z2, 1e-6f, 1e-6f, "softmax masked p0");
    EXPECT_NEAR(pr2[2], std::exp(3.0f - 4.0f) / Z2, 1e-6f, 1e-6f, "softmax masked p2");
    EXPECT_NEAR(pr2[3], std::exp(4.0f - 4.0f) / Z2, 1e-6f, 1e-6f, "softmax masked p3");
    float s2 = pr2[0] + pr2[1] + pr2[2] + pr2[3];
    EXPECT_NEAR(s2, 1.0f, 1e-6f, 1e-6f, "softmax masked sum");

    // Backward FD against scalar loss dot(dProbs, probs).
    Tensor lgb = Tensor::vec(5), prb = Tensor::vec(5), dP = Tensor::vec(5), dZ = Tensor::vec(5);
    for (int i = 0; i < 5; ++i) { lgb[i] = 0.4f * i - 0.7f; dP[i] = 0.3f * i - 0.4f; }
    brotensor::softmax_forward_cpu(lgb, prb);
    brotensor::softmax_backward_cpu(prb, dP, dZ);
    std::vector<float> g(5); for (int i = 0; i < 5; ++i) g[i] = dZ[i];
    auto loss = [&]() {
        Tensor pl = Tensor::vec(5);
        brotensor::softmax_forward_cpu(lgb, pl);
        float t = 0.0f;
        for (int i = 0; i < 5; ++i) t += dP[i] * pl[i];
        return t;
    };
    fd_check("softmax/dLogits", lgb.data, g, loss);
}

static void test_softmax_xent() {
    std::printf("softmax_xent_cpu / softmax_xent_segment_cpu\n");

    // Hand-computed: logits = [0,0,0], target=[1,0,0] one-hot → p = [1/3]*3,
    // loss = -log(1/3) = log(3).
    {
        Tensor lg = Tensor::vec(3), tg = Tensor::vec(3), pr = Tensor::vec(3), dz = Tensor::vec(3);
        lg[0]=0; lg[1]=0; lg[2]=0;
        tg[0]=1; tg[1]=0; tg[2]=0;
        const float L = brotensor::softmax_xent_cpu(lg, tg, pr, dz);
        EXPECT_NEAR(L, std::log(3.0f), 1e-6f, 1e-6f, "xent loss log3");
        EXPECT_NEAR(pr[0], 1.0f / 3.0f, 1e-6f, 1e-6f, "xent p0");
        // dLogits == p - target.
        EXPECT_NEAR(dz[0], 1.0f / 3.0f - 1.0f, 1e-6f, 1e-6f, "xent dz0");
        EXPECT_NEAR(dz[1], 1.0f / 3.0f, 1e-6f, 1e-6f, "xent dz1");
        EXPECT_NEAR(dz[2], 1.0f / 3.0f, 1e-6f, 1e-6f, "xent dz2");
    }

    // Masked: mask out index 1; target only on legal entries.
    {
        Tensor lg = Tensor::vec(4), tg = Tensor::vec(4), pr = Tensor::vec(4), dz = Tensor::vec(4);
        lg[0]=0.5f; lg[1]=2.0f; lg[2]=-0.3f; lg[3]=1.2f;
        tg[0]=0.7f; tg[1]=0.0f; tg[2]=0.3f;  tg[3]=0.0f;
        const float mask[4] = {1, 0, 1, 1};
        const float L = brotensor::softmax_xent_cpu(lg, tg, pr, dz, mask);

        // Recompute expected p.
        const float mx = std::fmax(std::fmax(lg[0], lg[2]), lg[3]);
        float e0 = std::exp(lg[0]-mx), e2 = std::exp(lg[2]-mx), e3 = std::exp(lg[3]-mx);
        const float Z = e0 + e2 + e3;
        const float p0 = e0/Z, p2 = e2/Z, p3 = e3/Z;
        EXPECT_NEAR(pr[0], p0, 1e-6f, 1e-6f, "xent masked p0");
        EXPECT_NEAR(pr[1], 0.0f, 1e-7f, 1e-7f, "xent masked p1");
        EXPECT_NEAR(pr[2], p2, 1e-6f, 1e-6f, "xent masked p2");
        EXPECT_NEAR(pr[3], p3, 1e-6f, 1e-6f, "xent masked p3");

        const float Lexp = -(0.7f * std::log(p0) + 0.3f * std::log(p2));
        EXPECT_NEAR(L, Lexp, 1e-5f, 1e-5f, "xent masked loss");
        EXPECT_NEAR(dz[1], 0.0f, 1e-7f, 1e-7f, "xent masked dz1");
        EXPECT_NEAR(dz[0], p0 - 0.7f, 1e-6f, 1e-6f, "xent masked dz0");
    }

    // Segment form should match Tensor form on the same data.
    {
        std::vector<float> lg = {0.1f, -0.4f, 0.9f, 0.0f};
        std::vector<float> tg = {0.0f, 0.0f, 1.0f, 0.0f};
        std::vector<float> p1(4), p2(4), dz1(4), dz2(4);

        Tensor LG = Tensor::vec(4), TG = Tensor::vec(4), P = Tensor::vec(4), DZ = Tensor::vec(4);
        for (int i = 0; i < 4; ++i) { LG[i] = lg[i]; TG[i] = tg[i]; }
        const float La = brotensor::softmax_xent_cpu(LG, TG, P, DZ);
        const float Lb = brotensor::softmax_xent_segment_cpu(lg.data(), tg.data(),
                                                              p2.data(), dz2.data(), 4);
        EXPECT_NEAR(La, Lb, 1e-7f, 1e-7f, "xent segment vs tensor loss");
        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR(P[i],  p2[i],  1e-7f, 1e-7f, "xent segment probs");
            EXPECT_NEAR(DZ[i], dz2[i], 1e-7f, 1e-7f, "xent segment dz");
        }
    }
}

// ---- MSE --------------------------------------------------------------------

static void test_mse() {
    std::printf("mse_scalar_cpu\n");
    float dp = 0.0f;
    const float L = brotensor::mse_scalar_cpu(2.5f, 1.5f, dp);
    EXPECT_NEAR(L, 0.5f, 1e-7f, 1e-7f, "mse loss 0.5*(1)^2");
    EXPECT_NEAR(dp, 1.0f, 1e-7f, 1e-7f, "mse dp");

    // FD: d/dpred (0.5 (pred-t)^2) = pred-t.
    const float h = 1e-3f;
    float dl;
    const float lp = brotensor::mse_scalar_cpu(2.5f + h, 1.5f, dl);
    const float lm = brotensor::mse_scalar_cpu(2.5f - h, 1.5f, dl);
    const float num = (lp - lm) / (2.0f * h);
    EXPECT_NEAR(num, dp, 1e-3f, 1e-3f, "mse fd");
}

// ---- add_inplace / add_scalar_inplace ---------------------------------------

static void test_add() {
    std::printf("add_inplace_cpu / add_scalar_inplace_cpu\n");
    Tensor y = Tensor::vec(4), x = Tensor::vec(4);
    y[0]=1; y[1]=2; y[2]=-3; y[3]=0.5f;
    x[0]=0.1f; x[1]=-2; x[2]=0.5f; x[3]=10;
    brotensor::add_inplace_cpu(y, x);
    EXPECT_NEAR(y[0], 1.1f, 1e-6f, 1e-6f, "add y0");
    EXPECT_NEAR(y[1], 0.0f, 1e-6f, 1e-6f, "add y1");
    EXPECT_NEAR(y[2], -2.5f, 1e-6f, 1e-6f, "add y2");
    EXPECT_NEAR(y[3], 10.5f, 1e-6f, 1e-6f, "add y3");

    brotensor::add_scalar_inplace_cpu(y, 0.5f);
    EXPECT_NEAR(y[0], 1.6f, 1e-6f, 1e-6f, "addS y0");
    EXPECT_NEAR(y[1], 0.5f, 1e-6f, 1e-6f, "addS y1");
    EXPECT_NEAR(y[2], -2.0f, 1e-6f, 1e-6f, "addS y2");
    EXPECT_NEAR(y[3], 11.0f, 1e-6f, 1e-6f, "addS y3");
}

// ---- Xavier init ------------------------------------------------------------

static void test_xavier() {
    std::printf("xavier_init_cpu\n");
    const int rows = 5, cols = 7;

    Tensor A(rows, cols), B(rows, cols);
    EXPECT_TRUE(static_cast<int>(A.data.size()) == rows * cols, "xavier size");

    uint64_t sa = 0xDEADBEEFCAFEBABEULL, sb = 0xDEADBEEFCAFEBABEULL;
    brotensor::xavier_init_cpu(A, sa);
    brotensor::xavier_init_cpu(B, sb);
    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(A[i], B[i], 0.0f, 0.0f, "xavier determinism");
    }
    EXPECT_TRUE(sa == sb, "xavier rng_state advanced identically");

    const float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_TRUE(A[i] >= -limit && A[i] <= limit, "xavier bound");
    }

    // Different seed => different output (cannot reasonably collide on
    // 5*7=35 floats with splitmix64).
    Tensor C(rows, cols);
    uint64_t sc = 0x1234567890ABCDEFULL;
    brotensor::xavier_init_cpu(C, sc);
    bool any_diff = false;
    for (int i = 0; i < rows * cols; ++i) if (A[i] != C[i]) { any_diff = true; break; }
    EXPECT_TRUE(any_diff, "xavier seed-dependent");
}

int main() {
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
