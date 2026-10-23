// Issue #138: Adjoint Index Transpose Duality & OOB Bounds Proofs for
// Conv2D, ModulatedConv2D, UpFirDn2D, Resample, RoPE & FlashAttention Backward.
//
// Exercises:
//   1. BVSolver formal proofs of forward/backward index transpose duality and
//      [0, numel) memory bounds across all spatial & sequence backward passes.
//   2. BVSolver formal refutation of the pre-995ac3c UpFirDn2D asymmetric-padding
//      off-by-one bug (confirming the solver catches the historical defect).
//   3. Numerical inner-product adjoint duality <F(X), dY> == <X, B(dY)> on
//      non-trivial shapes (asymmetric padding, stride != 1, dilation != 1,
//      groups > 1, up != down).

#include <brotensor/backward_verify.h>
#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <vector>

using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static double dot_f32(const Tensor& a, const Tensor& b) {
    CHECK(a.size() == b.size());
    double s = 0.0;
    const float* pa = a.host_f32();
    const float* pb = b.host_f32();
    for (int i = 0; i < a.size(); ++i) {
        s += static_cast<double>(pa[i]) * static_cast<double>(pb[i]);
    }
    return s;
}

static void fill_deterministic(Tensor& t, float scale, int seed_offset) {
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.size(); ++i) {
        int v = ((i + seed_offset) * 1103515245 + 12345) & 0x7fffffff;
        p[i] = scale * (static_cast<float>(v % 2001) / 1000.0f - 1.0f);
    }
}

static void test_bv_proof_suite() {
    auto report = brotensor::verify::run_all_backward_index_proofs();
    CHECK(!report.entries.empty());
    for (const auto& entry : report.entries) {
        std::printf("  [BV] %-64s : %s (%s)\n",
                    entry.name.c_str(),
                    entry.proven ? "PROVEN" : "FAILED",
                    entry.detail.c_str());
        CHECK(entry.proven);
    }
    CHECK(report.all_passed());
}

static void test_conv2d_adjoint_dot_product() {
    // Grouped dilated strided Conv2D:
    // N=2, C_in=4, H=7, W=8, C_out=6, kH=3, kW=3, stride=(2,2), pad=(1,2), dil=(2,1), groups=2
    const int N = 2, C_in = 4, H = 7, W = 8;
    const int C_out = 6, kH = 3, kW = 3;
    const int sh = 2, sw = 2, ph = 1, pw = 2, dh = 2, dw = 1, groups = 2;
    const int H_out = (H + 2 * ph - dh * (kH - 1) - 1) / sh + 1;
    const int W_out = (W + 2 * pw - dw * (kW - 1) - 1) / sw + 1;

    Tensor X = Tensor::mat(N, C_in * H * W);
    Tensor Wt = Tensor::mat(C_out, (C_in / groups) * kH * kW);
    Tensor dY = Tensor::mat(N, C_out * H_out * W_out);
    fill_deterministic(X, 0.5f, 11);
    fill_deterministic(Wt, 0.3f, 29);
    fill_deterministic(dY, 0.4f, 47);

    Tensor Y;
    brotensor::conv2d_forward(X, Wt, nullptr, N, C_in, H, W, C_out, kH, kW,
                              sh, sw, ph, pw, dh, dw, groups, Y);

    Tensor dX;
    brotensor::conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                                     sh, sw, ph, pw, dh, dw, groups, dX);

    Tensor dWt = Tensor::mat(C_out, (C_in / groups) * kH * kW);
    dWt.zero();
    brotensor::conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                                      sh, sw, ph, pw, dh, dw, groups, dWt);

    // Bilinear operator adjoint identity:
    // <conv(X, Wt), dY> == <X, dX> == <Wt, dWt>
    const double lhs = dot_f32(Y, dY);
    const double rhs_x = dot_f32(X, dX);
    const double rhs_w = dot_f32(Wt, dWt);
    CHECK(std::fabs(lhs - rhs_x) <= 1e-4 * (1.0 + std::fabs(lhs)));
    CHECK(std::fabs(lhs - rhs_w) <= 1e-4 * (1.0 + std::fabs(lhs)));
}

static void test_conv_transpose2d_adjoint_dot_product() {
    const int N = 2, C_in = 4, H = 5, W = 6;
    const int C_out = 4, kH = 3, kW = 3;
    const int sh = 2, sw = 2, ph = 1, pw = 1, oph = 1, opw = 0, dh = 1, dw = 1, groups = 2;
    const int H_out = (H - 1) * sh - 2 * ph + dh * (kH - 1) + oph + 1;
    const int W_out = (W - 1) * sw - 2 * pw + dw * (kW - 1) + opw + 1;

    Tensor X = Tensor::mat(N, C_in * H * W);
    Tensor Wt = Tensor::mat(C_in, (C_out / groups) * kH * kW);
    Tensor dY = Tensor::mat(N, C_out * H_out * W_out);
    fill_deterministic(X, 0.5f, 61);
    fill_deterministic(Wt, 0.3f, 73);
    fill_deterministic(dY, 0.4f, 89);

    Tensor Y;
    brotensor::conv_transpose2d_forward(X, Wt, nullptr, N, C_in, H, W, C_out, kH, kW,
                                        sh, sw, ph, pw, oph, opw, dh, dw, groups, Y);
    Tensor dX;
    brotensor::conv_transpose2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                                               sh, sw, ph, pw, oph, opw, dh, dw, groups, dX);
    const double lhs = dot_f32(Y, dY);
    const double rhs_x = dot_f32(X, dX);
    CHECK(std::fabs(lhs - rhs_x) <= 1e-4 * (1.0 + std::fabs(lhs)));
}

static void test_modulated_conv2d_adjoint_dot_product() {
    const int N = 2, C_in = 3, H = 5, Wd = 5, C_out = 4, kH = 3, kW = 3, ph = 1, pw = 1;
    const int H_out = H + 2 * ph - (kH - 1);
    const int W_out = Wd + 2 * pw - (kW - 1);

    Tensor X = Tensor::mat(N, C_in * H * Wd);
    Tensor W = Tensor::mat(C_out, C_in * kH * kW);
    Tensor s = Tensor::mat(N, C_in);
    Tensor dY = Tensor::mat(N, C_out * H_out * W_out);
    fill_deterministic(X, 0.5f, 101);
    fill_deterministic(W, 0.4f, 107);
    fill_deterministic(s, 0.6f, 113);
    for (int i = 0; i < s.size(); ++i) s[i] += 1.0f; // avoid near-zero style
    fill_deterministic(dY, 0.3f, 127);

    Tensor dcoef, Y;
    brotensor::modulated_conv2d_forward(X, W, s, N, C_in, H, Wd, C_out, kH, kW,
                                        ph, pw, /*demodulate=*/true, 1e-8f, dcoef, Y);

    Tensor dX, dW = Tensor::mat(C_out, C_in * kH * kW), ds;
    dW.zero();
    brotensor::modulated_conv2d_backward(X, W, s, dcoef, dY, N, C_in, H, Wd, C_out, kH, kW,
                                         ph, pw, /*demodulate=*/true, 1e-8f, dX, dW, ds);

    // Since modulated_conv2d is linear in X for fixed (W, s), <Y, dY> == <X, dX>
    const double lhs = dot_f32(Y, dY);
    const double rhs_x = dot_f32(X, dX);
    CHECK(std::fabs(lhs - rhs_x) <= 1e-4 * (1.0 + std::fabs(lhs)));
}

static void test_upfirdn2d_asymmetric_adjoint_dot_product() {
    // Asymmetric padding and distinct up/down factors along X and Y:
    const int N = 2, C = 2, H = 6, Wd = 7, fH = 4, fW = 4;
    const int up_x = 2, up_y = 3, down_x = 3, down_y = 2;
    const int px0 = 2, px1 = 1, py0 = 1, py1 = 3;
    const int H_out = (H * up_y + py0 + py1 - fH) / down_y + 1;
    const int W_out = (Wd * up_x + px0 + px1 - fW) / down_x + 1;

    Tensor X = Tensor::mat(N, C * H * Wd);
    Tensor f = Tensor::mat(fH, fW);
    Tensor dY = Tensor::mat(N, C * H_out * W_out);
    fill_deterministic(X, 0.5f, 131);
    fill_deterministic(f, 0.4f, 137);
    fill_deterministic(dY, 0.5f, 149);

    for (bool flip : {false, true}) {
        Tensor Y, dX;
        brotensor::upfirdn2d_forward(X, f, N, C, H, Wd, fH, fW,
                                     up_x, up_y, down_x, down_y,
                                     px0, px1, py0, py1, flip, 1.25f, Y);
        brotensor::upfirdn2d_backward(dY, f, N, C, H, Wd, fH, fW,
                                      up_x, up_y, down_x, down_y,
                                      px0, px1, py0, py1, flip, 1.25f, dX);
        const double lhs = dot_f32(Y, dY);
        const double rhs = dot_f32(X, dX);
        CHECK(std::fabs(lhs - rhs) <= 1e-4 * (1.0 + std::fabs(lhs)));
    }
}

static void test_resample_and_rope_adjoint_dot_product() {
    // 1. upsample_bilinear_2x & interp2d (nearest + bilinear)
    const int N = 2, C = 3, H = 5, W = 6;
    Tensor X = Tensor::mat(N, C * H * W);
    Tensor dY_up = Tensor::mat(N, C * (2 * H) * (2 * W));
    fill_deterministic(X, 0.5f, 163);
    fill_deterministic(dY_up, 0.4f, 173);

    Tensor Y_up, dX_up;
    brotensor::upsample_bilinear_2x(X, N, C, H, W, Y_up);
    brotensor::upsample_bilinear_2x_backward(dY_up, N, C, H, W, dX_up);
    CHECK(std::fabs(dot_f32(Y_up, dY_up) - dot_f32(X, dX_up)) <= 1e-4);

    // Non-integer scale interp2d (5x6 -> 8x9)
    const int H_out = 8, W_out = 9;
    Tensor dY_interp = Tensor::mat(N, C * H_out * W_out);
    fill_deterministic(dY_interp, 0.4f, 181);
    for (int mode : {0, 1}) {
        Tensor Y_i, dX_i;
        brotensor::interp2d_forward(X, N, C, H, W, H_out, W_out, mode, Y_i);
        brotensor::interp2d_backward(dY_interp, N, C, H, W, H_out, W_out, mode, dX_i);
        CHECK(std::fabs(dot_f32(Y_i, dY_interp) - dot_f32(X, dX_i)) <= 1e-4);
    }

    // 2. RoPE forward/backward orthogonal adjoint: <R(X), dY> == <X, R^T(dY)>
    //    and R^T(R(X)) == X.
    const int L = 7, num_heads = 4, head_dim = 16;
    Tensor X_rope = Tensor::mat(L, num_heads * head_dim);
    Tensor dY_rope = Tensor::mat(L, num_heads * head_dim);
    fill_deterministic(X_rope, 0.7f, 191);
    fill_deterministic(dY_rope, 0.6f, 197);

    Tensor Y_rope, dX_rope, X_roundtrip;
    brotensor::rope_forward(X_rope, head_dim, num_heads, /*seq_offset=*/3, 10000.0f, Y_rope);
    brotensor::rope_backward(dY_rope, head_dim, num_heads, /*seq_offset=*/3, 10000.0f, dX_rope);
    brotensor::rope_backward(Y_rope, head_dim, num_heads, /*seq_offset=*/3, 10000.0f, X_roundtrip);

    CHECK(std::fabs(dot_f32(Y_rope, dY_rope) - dot_f32(X_rope, dX_rope)) <= 1e-4);
    for (int i = 0; i < X_rope.size(); ++i) {
        CHECK(std::fabs(X_rope[i] - X_roundtrip[i]) <= 1e-5f);
    }
}

int main() {
    std::printf("=== Issue #138: Backward Index Transpose Duality & Bounds Proofs ===\n");
    test_bv_proof_suite();
    test_conv2d_adjoint_dot_product();
    test_conv_transpose2d_adjoint_dot_product();
    test_modulated_conv2d_adjoint_dot_product();
    test_upfirdn2d_asymmetric_adjoint_dot_product();
    test_resample_and_rope_adjoint_dot_product();

    if (g_failures == 0) {
        std::printf("ALL PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
