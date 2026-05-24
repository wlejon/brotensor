// ─── CPU-only test for conv_transpose2d_* ───────────────────────────────────
//
// Coverage:
//   1. Output shape matches torch ConvTranspose2d formula across a few
//      stride / padding / dilation / output_padding combos.
//   2. 1x1 kernel, stride 1, no pad reduces to a per-pixel matmul over
//      channels — verified against an explicit matmul reference.
//   3. Bias is added uniformly: zero-weight forward == broadcasted bias.
//   4. backward_input gradcheck (finite differences against forward).
//   5. backward_weight gradcheck.
//   6. backward_bias gradcheck (per-output-channel sum).
//   7. Grouped conv-transpose (groups > 1) shape + forward identity.
//   8. Forward / backward_input adjoint inner-product identity
//      (the cheapest correctness check for a transpose-conv pair).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
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

static bool approx(float a, float b, float tol = 1e-4f) {
    const float d = std::fabs(a - b);
    return d <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

static Tensor make_f32(int r, int c) {
    Tensor t;
    t.resize(r, c, Dtype::FP32);
    return t;
}

static void fill_random(Tensor& t, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = d(rng);
}

static void zero(Tensor& t) {
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = 0.0f;
}

static int convt_out(int L, int s, int pad, int op, int d, int k) {
    return (L - 1) * s - 2 * pad + d * (k - 1) + op + 1;
}

// ── 1. output-shape coverage ──────────────────────────────────────────────
static void test_output_shapes() {
    struct C { int N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw; };
    std::vector<C> cases = {
        {1, 2, 3, 4, 3, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1},
        {1, 2, 3, 4, 3, 3, 3, 2, 2, 0, 0, 0, 0, 1, 1},  // 2x upsample
        {1, 2, 3, 4, 3, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1},  // pad
        {2, 2, 4, 4, 2, 3, 3, 2, 2, 0, 0, 1, 0, 1, 1},  // output_padding
        {1, 1, 5, 5, 1, 3, 3, 1, 1, 0, 0, 0, 0, 2, 2},  // dilated
    };
    for (const auto& c : cases) {
        Tensor X = make_f32(c.N, c.Cin * c.H * c.W);
        Tensor Wt = make_f32(c.Cin, c.Cout * c.kH * c.kW);
        Tensor Y;
        brotensor::conv_transpose2d_forward(X, Wt, /*bias=*/nullptr,
            c.N, c.Cin, c.H, c.W, c.Cout, c.kH, c.kW,
            c.sh, c.sw, c.ph, c.pw, c.oph, c.opw, c.dh, c.dw, 1, Y);
        const int Ho = convt_out(c.H, c.sh, c.ph, c.oph, c.dh, c.kH);
        const int Wo = convt_out(c.W, c.sw, c.pw, c.opw, c.dw, c.kW);
        CHECK(Y.rows == c.N);
        CHECK(Y.cols == c.Cout * Ho * Wo);
    }
}

// ── 2. 1x1 kernel reduces to per-pixel matmul ─────────────────────────────
static void test_1x1_matches_matmul() {
    const int N = 2, Cin = 3, Cout = 4, H = 2, W = 3;
    Tensor X = make_f32(N, Cin * H * W);
    fill_random(X, 0x101);
    Tensor Wt = make_f32(Cin, Cout * 1 * 1);
    fill_random(Wt, 0x102);

    Tensor Y;
    brotensor::conv_transpose2d_forward(X, Wt, /*bias=*/nullptr,
        N, Cin, H, W, Cout, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, Y);
    CHECK(Y.cols == Cout * H * W);
    const float* x = X.host_f32();
    const float* wp = Wt.host_f32();
    const float* y = Y.host_f32();
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < Cout; ++oc)
            for (int p = 0; p < H * W; ++p) {
                double acc = 0.0;
                for (int c_in = 0; c_in < Cin; ++c_in)
                    acc += x[(n * Cin + c_in) * H * W + p]
                         * wp[c_in * Cout + oc];
                CHECK(approx(y[(n * Cout + oc) * H * W + p],
                             static_cast<float>(acc)));
            }
}

// ── 3. zero weights => output is just broadcast bias ──────────────────────
static void test_bias_only() {
    const int N = 1, Cin = 2, Cout = 3, H = 3, W = 3;
    Tensor X = make_f32(N, Cin * H * W);
    fill_random(X, 0x201);
    Tensor Wt = make_f32(Cin, Cout * 2 * 2);
    zero(Wt);
    Tensor B = make_f32(Cout, 1);
    float* bp = B.host_f32_mut();
    bp[0] = 1.0f; bp[1] = -2.5f; bp[2] = 0.25f;

    Tensor Y;
    brotensor::conv_transpose2d_forward(X, Wt, &B,
        N, Cin, H, W, Cout, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, Y);
    const int Ho = convt_out(H, 1, 0, 0, 1, 2);  // 4
    const int Wo = convt_out(W, 1, 0, 0, 1, 2);  // 4
    CHECK(Y.cols == Cout * Ho * Wo);
    const float* y = Y.host_f32();
    for (int oc = 0; oc < Cout; ++oc)
        for (int p = 0; p < Ho * Wo; ++p)
            CHECK(y[oc * Ho * Wo + p] == bp[oc]);
}

// ── 4-6. gradchecks on a small case ───────────────────────────────────────
static double loss_inner(const Tensor& Y, const Tensor& V) {
    const float* y = Y.host_f32();
    const float* v = V.host_f32();
    double s = 0.0;
    for (int i = 0; i < Y.rows * Y.cols; ++i) s += static_cast<double>(y[i]) * v[i];
    return s;
}

static void test_gradchecks() {
    const int N = 1, Cin = 2, Cout = 2, H = 3, W = 3, kH = 3, kW = 3;
    const int sh = 2, sw = 2, ph = 1, pw = 1, oph = 0, opw = 0, dh = 1, dw = 1;
    const int Ho = convt_out(H, sh, ph, oph, dh, kH);  // 5
    const int Wo = convt_out(W, sw, pw, opw, dw, kW);  // 5

    Tensor X = make_f32(N, Cin * H * W);
    Tensor Wt = make_f32(Cin, Cout * kH * kW);
    Tensor B = make_f32(Cout, 1);
    fill_random(X, 0x301);
    fill_random(Wt, 0x302);
    fill_random(B, 0x303);

    Tensor V = make_f32(N, Cout * Ho * Wo);
    fill_random(V, 0x304);  // dL/dY = V

    Tensor Y, dX, dWt, dB;
    dWt = make_f32(Cin, Cout * kH * kW);  zero(dWt);
    dB  = make_f32(Cout, 1);             zero(dB);

    brotensor::conv_transpose2d_forward(X, Wt, &B,
        N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, Y);
    brotensor::conv_transpose2d_backward_input(Wt, V,
        N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, dX);
    brotensor::conv_transpose2d_backward_weight(X, V,
        N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, dWt);
    brotensor::conv_transpose2d_backward_bias(V, N, Cout, Ho, Wo, dB);

    const float* dxp = dX.host_f32();
    const float* dwp = dWt.host_f32();
    const float* dbp = dB.host_f32();
    const float h = 1e-3f;

    // input gradcheck
    {
        bool ok = true;
        float* xp = X.host_f32_mut();
        std::mt19937_64 rng(0x401);
        std::uniform_int_distribution<int> pick(0, X.cols - 1);
        for (int t = 0; t < 12; ++t) {
            const int i = pick(rng);
            const float orig = xp[i];
            xp[i] = orig + h;
            Tensor Yt; brotensor::conv_transpose2d_forward(X, Wt, &B,
                N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, Yt);
            const double Lp = loss_inner(Yt, V);
            xp[i] = orig - h;
            brotensor::conv_transpose2d_forward(X, Wt, &B,
                N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, Yt);
            const double Lm = loss_inner(Yt, V);
            xp[i] = orig;
            const float num = static_cast<float>((Lp - Lm) / (2.0 * h));
            if (!approx(num, dxp[i], 5e-3f)) { ok = false; break; }
        }
        CHECK(ok);
    }
    // weight gradcheck
    {
        bool ok = true;
        float* wp = Wt.host_f32_mut();
        std::mt19937_64 rng(0x402);
        std::uniform_int_distribution<int> pick(0, Wt.cols * Wt.rows - 1);
        for (int t = 0; t < 12; ++t) {
            const int i = pick(rng);
            const float orig = wp[i];
            wp[i] = orig + h;
            Tensor Yt; brotensor::conv_transpose2d_forward(X, Wt, &B,
                N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, Yt);
            const double Lp = loss_inner(Yt, V);
            wp[i] = orig - h;
            brotensor::conv_transpose2d_forward(X, Wt, &B,
                N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, Yt);
            const double Lm = loss_inner(Yt, V);
            wp[i] = orig;
            const float num = static_cast<float>((Lp - Lm) / (2.0 * h));
            if (!approx(num, dwp[i], 5e-3f)) { ok = false; break; }
        }
        CHECK(ok);
    }
    // bias gradcheck — analytical dB should equal sum(V) per output channel.
    for (int oc = 0; oc < Cout; ++oc) {
        double s = 0.0;
        const float* vp = V.host_f32();
        for (int n = 0; n < N; ++n)
            for (int i = 0; i < Ho * Wo; ++i)
                s += vp[(n * Cout + oc) * Ho * Wo + i];
        CHECK(approx(dbp[oc], static_cast<float>(s)));
    }
}

// ── 7. grouped: forward shape matches and depthwise == per-channel ────────
static void test_grouped_forward_depthwise() {
    const int N = 1, Cin = 4, Cout = 4, H = 3, W = 3, kH = 2, kW = 2;
    const int groups = 4;  // depthwise (1 in -> 1 out per group)
    Tensor X = make_f32(N, Cin * H * W);
    fill_random(X, 0x501);
    // Cg_out = Cout/groups = 1, so Wt cols = 1*kH*kW = 4.
    Tensor Wt = make_f32(Cin, 1 * kH * kW);
    fill_random(Wt, 0x502);
    Tensor Y;
    brotensor::conv_transpose2d_forward(X, Wt, nullptr,
        N, Cin, H, W, Cout, kH, kW, 1, 1, 0, 0, 0, 0, 1, 1, groups, Y);
    const int Ho = convt_out(H, 1, 0, 0, 1, kH);
    const int Wo = convt_out(W, 1, 0, 0, 1, kW);
    CHECK(Y.rows == N);
    CHECK(Y.cols == Cout * Ho * Wo);
    // Each output channel oc depends only on input channel oc (depthwise).
    // Verify by zeroing X's channel oc and confirming output channel oc is zero.
    for (int target = 0; target < Cin; ++target) {
        Tensor Xc = make_f32(N, Cin * H * W);
        const float* x_src = X.host_f32();
        float* xcp = Xc.host_f32_mut();
        for (int i = 0; i < Xc.cols; ++i) xcp[i] = x_src[i];
        for (int p = 0; p < H * W; ++p)
            xcp[target * H * W + p] = 0.0f;
        Tensor Yc;
        brotensor::conv_transpose2d_forward(Xc, Wt, nullptr,
            N, Cin, H, W, Cout, kH, kW, 1, 1, 0, 0, 0, 0, 1, 1, groups, Yc);
        const float* yp = Yc.host_f32();
        // Output channel `target` should be all zero.
        for (int p = 0; p < Ho * Wo; ++p)
            CHECK(yp[target * Ho * Wo + p] == 0.0f);
    }
}

// ── 8. adjoint inner-product identity: <fwd(X), V> == <X, bwd_input(V)> ──
static void test_adjoint_identity() {
    const int N = 1, Cin = 2, Cout = 3, H = 3, W = 4, kH = 3, kW = 3;
    const int sh = 2, sw = 1, ph = 1, pw = 1, oph = 0, opw = 0, dh = 1, dw = 1;
    const int Ho = convt_out(H, sh, ph, oph, dh, kH);
    const int Wo = convt_out(W, sw, pw, opw, dw, kW);
    Tensor X = make_f32(N, Cin * H * W);
    Tensor Wt = make_f32(Cin, Cout * kH * kW);
    Tensor V = make_f32(N, Cout * Ho * Wo);
    fill_random(X, 0x601); fill_random(Wt, 0x602); fill_random(V, 0x603);
    Tensor Y, dX;
    brotensor::conv_transpose2d_forward(X, Wt, nullptr,
        N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, Y);
    brotensor::conv_transpose2d_backward_input(Wt, V,
        N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, 1, dX);
    const double lhs = loss_inner(Y, V);
    const float* xp = X.host_f32();
    const float* dxp = dX.host_f32();
    double rhs = 0.0;
    for (int i = 0; i < X.cols; ++i) rhs += static_cast<double>(xp[i]) * dxp[i];
    CHECK(approx(static_cast<float>(lhs), static_cast<float>(rhs), 1e-4f));
}

int main() {
    brotensor::init();
    std::printf("test_conv_transpose2d (CPU FP32):\n");
    test_output_shapes();
    test_1x1_matches_matmul();
    test_bias_only();
    test_gradchecks();
    test_grouped_forward_depthwise();
    test_adjoint_identity();
    if (g_failures == 0) {
        std::printf("  OK  all conv_transpose2d CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
