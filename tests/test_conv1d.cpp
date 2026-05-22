// Standalone CPU coverage for the brosoundml 1D-convolution family (CHUNK 3).
//
// Verifies:
//   * conv1d header wrapper == the equivalent conv2d call (H=1 collapsed),
//     groups=1 and depthwise; conv1d backward wrappers match too.
//   * conv_transpose1d forward against a hand-computed reference for stride>1
//     and groups>1, plus finite-difference gradient checks for all three
//     backward halves (input / weight / bias).
//   * causal_conv1d_update streamed step-by-step == one full causal_conv1d
//     over the concatenated sequence (zero-initialised state).
//   * pad1d zero / reflect / replicate modes against numpy semantics, plus
//     FD gradient checks on pad1d_backward for all three modes.
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

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
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

static bool near_(double a, double b, double abs_eps, double rel_eps) {
    const double d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

#define EXPECT_NEAR(actual, expected, abs_eps, rel_eps, ctx)                    \
    do {                                                                       \
        const double _a = (actual);                                            \
        const double _e = (expected);                                          \
        if (!near_(_a, _e, (abs_eps), (rel_eps))) {                            \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.9g expected=%.9g\n",     \
                        __FILE__, __LINE__, (ctx), _a, _e);                     \
            ++g_failures;                                                       \
        }                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",          \
                        __FILE__, __LINE__, (ctx), #cond);                      \
            ++g_failures;                                                       \
        }                                                                      \
    } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    float next() {  // uniform in [-1, 1)
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 11) /
                                  static_cast<double>(1ULL << 53)) * 2.0f - 1.0f;
    }
};

static Tensor cpu_rand(Rng& rng, int r, int c) {
    Tensor t = Tensor::zeros_on(Device::CPU, r, c);
    for (int i = 0; i < r * c; ++i) t.host_f32_mut()[i] = rng.next();
    return t;
}
static Tensor cpu_zeros(int r, int c) {
    return Tensor::zeros_on(Device::CPU, r, c);
}

// ─── conv1d wrapper == conv2d ───────────────────────────────────────────────
static void test_conv1d_matches_conv2d(int groups) {
    char ctx[96];
    std::snprintf(ctx, sizeof ctx, "conv1d==conv2d groups=%d", groups);
    Rng rng(0xC0FFEE ^ static_cast<uint64_t>(groups));

    // C_out divisible by 4 so groups ∈ {1,2,4} all evenly divide it.
    const int N = 2, C_in = 4, C_out = 8, L = 17, kL = 5;
    const int stride = 2, padding = 2, dilation = 2;
    const int Cg_in = C_in / groups;

    Tensor X  = cpu_rand(rng, N, C_in * L);
    Tensor Wt = cpu_rand(rng, C_out, Cg_in * kL);
    Tensor B  = cpu_rand(rng, C_out, 1);

    Tensor Y1, Y2;
    brotensor::conv1d(X, Wt, &B, N, C_in, L, C_out, kL, stride, padding,
                      dilation, groups, Y1);
    // Equivalent conv2d call: H=1, kH=1, stride_h=1, pad_h=0, dil_h=1.
    brotensor::conv2d_forward(X, Wt, &B, N, C_in, /*H=*/1, /*W=*/L, C_out,
                              /*kH=*/1, /*kW=*/kL, /*stride_h=*/1,
                              /*stride_w=*/stride, /*pad_h=*/0,
                              /*pad_w=*/padding, /*dil_h=*/1,
                              /*dil_w=*/dilation, groups, Y2);
    EXPECT_TRUE(Y1.rows == Y2.rows && Y1.cols == Y2.cols, ctx);
    for (int i = 0; i < Y1.rows * Y1.cols; ++i) {
        EXPECT_NEAR(Y1.host_f32()[i], Y2.host_f32()[i], 1e-6, 1e-5, ctx);
    }
    const int L_out = Y1.cols / C_out;

    // Backward wrappers vs conv2d backward.
    Tensor dY = cpu_rand(rng, N, C_out * L_out);

    Tensor dX1, dX2;
    brotensor::conv1d_backward_input(Wt, dY, N, C_in, L, C_out, kL, stride,
                                     padding, dilation, groups, dX1);
    brotensor::conv2d_backward_input(Wt, dY, N, C_in, 1, L, C_out, 1, kL, 1,
                                     stride, 0, padding, 1, dilation, groups,
                                     dX2);
    for (int i = 0; i < dX1.rows * dX1.cols; ++i) {
        EXPECT_NEAR(dX1.host_f32()[i], dX2.host_f32()[i], 1e-6, 1e-5, ctx);
    }

    Tensor dW1 = cpu_zeros(C_out, Cg_in * kL);
    Tensor dW2 = cpu_zeros(C_out, Cg_in * kL);
    brotensor::conv1d_backward_weight(X, dY, N, C_in, L, C_out, kL, stride,
                                      padding, dilation, groups, dW1);
    brotensor::conv2d_backward_weight(X, dY, N, C_in, 1, L, C_out, 1, kL, 1,
                                      stride, 0, padding, 1, dilation, groups,
                                      dW2);
    for (int i = 0; i < dW1.rows * dW1.cols; ++i) {
        EXPECT_NEAR(dW1.host_f32()[i], dW2.host_f32()[i], 1e-5, 1e-4, ctx);
    }

    Tensor dB1 = cpu_zeros(C_out, 1);
    Tensor dB2 = cpu_zeros(C_out, 1);
    brotensor::conv1d_backward_bias(dY, N, C_out, L_out, dB1);
    brotensor::conv2d_backward_bias(dY, N, C_out, 1, L_out, dB2);
    for (int i = 0; i < C_out; ++i) {
        EXPECT_NEAR(dB1.host_f32()[i], dB2.host_f32()[i], 1e-5, 1e-4, ctx);
    }
}

// ─── conv_transpose1d hand-computed reference ───────────────────────────────
// Naive scatter reference, FP64 accumulation.
static std::vector<double> convt1d_ref(const std::vector<float>& X,
                                       const std::vector<float>& Wt,
                                       const std::vector<float>& bias,
                                       bool has_bias,
                                       int N, int C_in, int L, int C_out,
                                       int kL, int stride, int padding,
                                       int output_padding, int dilation,
                                       int groups, int& L_out_out) {
    const int Cg_in = C_in / groups, Cg_out = C_out / groups;
    const int L_out = (L - 1) * stride - 2 * padding + dilation * (kL - 1)
                      + output_padding + 1;
    L_out_out = L_out;
    std::vector<double> Y(static_cast<size_t>(N) * C_out * L_out, 0.0);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            double bv = has_bias ? bias[oc] : 0.0;
            for (int lo = 0; lo < L_out; ++lo)
                Y[(static_cast<size_t>(n) * C_out + oc) * L_out + lo] = bv;
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int g = c_in / Cg_in;
            for (int l = 0; l < L; ++l) {
                const float xv = X[(static_cast<size_t>(n) * C_in + c_in) * L + l];
                for (int kl = 0; kl < kL; ++kl) {
                    const int lo = l * stride - padding + kl * dilation;
                    if (lo < 0 || lo >= L_out) continue;
                    for (int ocl = 0; ocl < Cg_out; ++ocl) {
                        const int oc = g * Cg_out + ocl;
                        const float w = Wt[(c_in * Cg_out + ocl) * kL + kl];
                        Y[(static_cast<size_t>(n) * C_out + oc) * L_out + lo]
                            += static_cast<double>(xv) * w;
                    }
                }
            }
        }
    }
    return Y;
}

static void test_convt1d_forward(int groups, int stride, int output_padding) {
    char ctx[112];
    std::snprintf(ctx, sizeof ctx,
                  "conv_transpose1d fwd groups=%d stride=%d outpad=%d",
                  groups, stride, output_padding);
    Rng rng(0xDEC0DE ^ (static_cast<uint64_t>(groups) << 8)
            ^ (static_cast<uint64_t>(stride) << 4)
            ^ static_cast<uint64_t>(output_padding));

    // C_out divisible by 4 so groups ∈ {1,2,4} all evenly divide it.
    const int N = 2, C_in = 4, C_out = 8, L = 9, kL = 3;
    const int padding = 1, dilation = 1;
    const int Cg_out = C_out / groups;

    Tensor X  = cpu_rand(rng, N, C_in * L);
    Tensor Wt = cpu_rand(rng, C_in, Cg_out * kL);
    Tensor B  = cpu_rand(rng, C_out, 1);

    Tensor Y;
    brotensor::conv_transpose1d_forward(X, Wt, &B, N, C_in, L, C_out, kL,
                                        stride, padding, output_padding,
                                        dilation, groups, Y);

    std::vector<float> Xv(X.host_f32(), X.host_f32() + N * C_in * L);
    std::vector<float> Wv(Wt.host_f32(), Wt.host_f32() + C_in * Cg_out * kL);
    std::vector<float> Bv(B.host_f32(), B.host_f32() + C_out);
    int L_out = 0;
    std::vector<double> ref = convt1d_ref(Xv, Wv, Bv, true, N, C_in, L,
                                          C_out, kL, stride, padding,
                                          output_padding, dilation, groups,
                                          L_out);
    EXPECT_TRUE(Y.rows == N && Y.cols == C_out * L_out, ctx);
    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(Y.host_f32()[i], ref[i], 1e-5, 1e-5, ctx);
    }
}

// FD gradient checks for the three conv_transpose1d backward halves.
static void test_convt1d_backward_fd(int groups) {
    char ctx[80];
    std::snprintf(ctx, sizeof ctx, "conv_transpose1d FD grads groups=%d",
                  groups);
    Rng rng(0xBADF00D ^ static_cast<uint64_t>(groups));

    const int N = 2, C_in = 4, C_out = 6, L = 7, kL = 3;
    const int stride = 2, padding = 1, output_padding = 1, dilation = 1;
    const int Cg_out = C_out / groups;

    Tensor X  = cpu_rand(rng, N, C_in * L);
    Tensor Wt = cpu_rand(rng, C_in, Cg_out * kL);
    Tensor B  = cpu_rand(rng, C_out, 1);

    Tensor Y;
    brotensor::conv_transpose1d_forward(X, Wt, &B, N, C_in, L, C_out, kL,
                                        stride, padding, output_padding,
                                        dilation, groups, Y);
    const int L_out = Y.cols / C_out;
    // Random upstream gradient; scalar loss = sum(dY * Y).
    Tensor dY = cpu_rand(rng, N, C_out * L_out);

    auto forward_loss = [&](const Tensor& Xin, const Tensor& Win,
                            const Tensor& Bin) -> double {
        Tensor Yt;
        brotensor::conv_transpose1d_forward(Xin, Win, &Bin, N, C_in, L, C_out,
                                            kL, stride, padding,
                                            output_padding, dilation, groups,
                                            Yt);
        double s = 0.0;
        for (int i = 0; i < Yt.rows * Yt.cols; ++i)
            s += static_cast<double>(Yt.host_f32()[i]) * dY.host_f32()[i];
        return s;
    };

    // Analytic gradients.
    Tensor dX;
    brotensor::conv_transpose1d_backward_input(Wt, dY, N, C_in, L, C_out, kL,
                                               stride, padding, output_padding,
                                               dilation, groups, dX);
    Tensor dW = cpu_zeros(C_in, Cg_out * kL);
    brotensor::conv_transpose1d_backward_weight(X, dY, N, C_in, L, C_out, kL,
                                                stride, padding,
                                                output_padding, dilation,
                                                groups, dW);
    Tensor dB = cpu_zeros(C_out, 1);
    brotensor::conv_transpose1d_backward_bias(dY, N, C_out, L_out, dB);

    const double h = 1e-3;
    // dX
    for (int i = 0; i < N * C_in * L; i += 5) {
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (forward_loss(Xp, Wt, B) - forward_loss(Xm, Wt, B))
                          / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 2e-2, 2e-2, ctx);
    }
    // dW
    for (int i = 0; i < C_in * Cg_out * kL; i += 4) {
        Tensor Wp = Wt.clone(), Wm = Wt.clone();
        Wp.host_f32_mut()[i] += static_cast<float>(h);
        Wm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (forward_loss(X, Wp, B) - forward_loss(X, Wm, B))
                          / (2.0 * h);
        EXPECT_NEAR(dW.host_f32()[i], fd, 2e-2, 2e-2, ctx);
    }
    // dB
    for (int i = 0; i < C_out; ++i) {
        Tensor Bp = B.clone(), Bm = B.clone();
        Bp.host_f32_mut()[i] += static_cast<float>(h);
        Bm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (forward_loss(X, Wt, Bp) - forward_loss(X, Wt, Bm))
                          / (2.0 * h);
        EXPECT_NEAR(dB.host_f32()[i], fd, 2e-2, 2e-2, ctx);
    }
}

// ─── causal_conv1d_update streaming == one full causal_conv1d ────────────────
static void test_causal_conv1d_update(int kL, int dilation,
                                      const std::vector<int>& chunks) {
    char ctx[96];
    std::snprintf(ctx, sizeof ctx, "causal_conv1d_update kL=%d dil=%d",
                  kL, dilation);
    Rng rng(0x5712EAA ^ (static_cast<uint64_t>(kL) << 8)
            ^ static_cast<uint64_t>(dilation));

    const int N = 2, C = 5;
    int L_total = 0;
    for (int c : chunks) L_total += c;

    Tensor X  = cpu_rand(rng, N, C * L_total);   // full sequence
    Tensor Wt = cpu_rand(rng, C, kL);            // depthwise filter / channel
    Tensor B  = cpu_rand(rng, C, 1);

    // Reference: one full causal_conv1d (depthwise: groups = C, C_in=C_out=C,
    // each filter (1*kL)). causal_conv1d expects weights (C_out, Cg_in*kL) =
    // (C, 1*kL) which is exactly (C, kL).
    Tensor scratch, Yfull;
    brotensor::causal_conv1d(X, Wt, &B, N, C, L_total, C, kL, /*stride=*/1,
                             dilation, /*groups=*/C, scratch, Yfull);
    EXPECT_TRUE(Yfull.rows == N && Yfull.cols == C * L_total, ctx);

    // Streaming: zero-initialised state, feed chunk by chunk.
    const int hist = (kL - 1) * dilation;
    Tensor state = cpu_zeros(N, C * hist);
    int offset = 0;
    for (int csz : chunks) {
        // Slice X columns [offset, offset+csz) per (n, c) into a step tensor.
        Tensor step = cpu_zeros(N, C * csz);
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < csz; ++t) {
                    step.host_f32_mut()[(n * C + c) * csz + t] =
                        X.host_f32()[(n * C + c) * L_total + offset + t];
                }
            }
        }
        Tensor Ystep;
        brotensor::causal_conv1d_update(step, Wt, &B, N, C, csz, kL, dilation,
                                        state, Ystep);
        EXPECT_TRUE(Ystep.rows == N && Ystep.cols == C * csz, ctx);
        // Compare against the corresponding slice of the full causal conv.
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < csz; ++t) {
                    const double got =
                        Ystep.host_f32()[(n * C + c) * csz + t];
                    const double exp_ =
                        Yfull.host_f32()[(n * C + c) * L_total + offset + t];
                    EXPECT_NEAR(got, exp_, 1e-5, 1e-5, ctx);
                }
            }
        }
        offset += csz;
    }
}

// ─── pad1d modes ────────────────────────────────────────────────────────────
static int ref_reflect(int q, int L) {
    if (L == 1) return 0;
    const int period = 2 * (L - 1);
    int m = q % period;
    if (m < 0) m += period;
    return (m < L) ? m : period - m;
}

static void test_pad1d_modes() {
    const char* ctx = "pad1d modes";
    Rng rng(0x9AD1D0ULL);
    const int N = 2, C = 3, L = 8, pl = 3, pr = 2;
    Tensor X = cpu_rand(rng, N, C * L);

    for (int mode = 0; mode <= 2; ++mode) {
        Tensor Y;
        brotensor::pad1d_forward(X, N, C, L, pl, pr, mode, Y);
        const int L_pad = L + pl + pr;
        EXPECT_TRUE(Y.rows == N && Y.cols == C * L_pad, ctx);
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                const float* xr = X.host_f32() + (n * C + c) * L;
                const float* yr = Y.host_f32() + (n * C + c) * L_pad;
                for (int p = 0; p < L_pad; ++p) {
                    const int rel = p - pl;
                    double want;
                    if (rel >= 0 && rel < L) {
                        want = xr[rel];
                    } else if (mode == 0) {
                        want = 0.0;
                    } else if (mode == 2) {
                        want = rel < 0 ? xr[0] : xr[L - 1];
                    } else {
                        want = xr[ref_reflect(rel, L)];
                    }
                    EXPECT_NEAR(yr[p], want, 1e-6, 1e-6, ctx);
                }
            }
        }
    }
}

// FD gradient check on pad1d_backward for all three modes.
static void test_pad1d_backward_fd() {
    const char* ctx = "pad1d backward FD";
    Rng rng(0x9AD9AD);
    const int N = 2, C = 2, L = 6, pl = 2, pr = 2;
    const int L_pad = L + pl + pr;
    Tensor X = cpu_rand(rng, N, C * L);

    for (int mode = 0; mode <= 2; ++mode) {
        Tensor dY = cpu_rand(rng, N, C * L_pad);
        Tensor dX;
        brotensor::pad1d_backward(dY, N, C, L, pl, pr, mode, dX);

        auto loss = [&](const Tensor& Xin) -> double {
            Tensor Y;
            brotensor::pad1d_forward(Xin, N, C, L, pl, pr, mode, Y);
            double s = 0.0;
            for (int i = 0; i < Y.rows * Y.cols; ++i)
                s += static_cast<double>(Y.host_f32()[i]) * dY.host_f32()[i];
            return s;
        };
        const double h = 1e-3;
        for (int i = 0; i < N * C * L; ++i) {
            Tensor Xp = X.clone(), Xm = X.clone();
            Xp.host_f32_mut()[i] += static_cast<float>(h);
            Xm.host_f32_mut()[i] -= static_cast<float>(h);
            const double fd = (loss(Xp) - loss(Xm)) / (2.0 * h);
            EXPECT_NEAR(dX.host_f32()[i], fd, 5e-3, 5e-3, ctx);
        }
    }
}

int main() {
    brotensor::init();
  try {
    test_conv1d_matches_conv2d(/*groups=*/1);
    test_conv1d_matches_conv2d(/*groups=*/4);   // grouped, C_in == groups
    test_conv1d_matches_conv2d(/*groups=*/2);

    test_convt1d_forward(/*groups=*/1, /*stride=*/1, /*outpad=*/0);
    test_convt1d_forward(/*groups=*/1, /*stride=*/3, /*outpad=*/2);
    test_convt1d_forward(/*groups=*/2, /*stride=*/2, /*outpad=*/1);
    test_convt1d_forward(/*groups=*/4, /*stride=*/2, /*outpad=*/0);  // depthwise

    test_convt1d_backward_fd(/*groups=*/1);
    test_convt1d_backward_fd(/*groups=*/2);

    test_causal_conv1d_update(/*kL=*/3, /*dilation=*/1, {1, 1, 1, 1, 1, 1, 1});
    test_causal_conv1d_update(/*kL=*/4, /*dilation=*/1, {3, 2, 5, 1, 4});
    test_causal_conv1d_update(/*kL=*/3, /*dilation=*/2, {2, 4, 3, 6});
    test_causal_conv1d_update(/*kL=*/5, /*dilation=*/3, {7, 8});

    test_pad1d_modes();
    test_pad1d_backward_fd();
  } catch (const std::exception& e) {
    std::printf("test_conv1d: uncaught exception: %s\n", e.what());
    return 2;
  }

    if (g_failures == 0) {
        std::printf("test_conv1d: all checks passed\n");
        return 0;
    }
    std::printf("test_conv1d: %d FAILURE(S)\n", g_failures);
    return 1;
}
