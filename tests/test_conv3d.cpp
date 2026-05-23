// CPU smoke test for conv3d_forward (FP32).
//
// Validates the NCTHW direct-conv math against a small hand-checked case and
// covers groups / dilation / stride / bias paths. CPU-only — the brotensor
// CPU backend is FP32-only by convention. The CPU↔GPU parity test
// (test_conv3d_parity.cpp) covers FP16 / BF16 / FP32 cross-backend agreement.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
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

namespace {

// Naive reference (FP32) — same indexing/contract as the kernels.
void conv3d_ref(const std::vector<float>& X,
                const std::vector<float>& Wt,
                const std::vector<float>& bias, bool has_bias,
                int N, int C_in, int T, int H, int W,
                int C_out, int kT, int kH, int kW,
                int stride_t, int stride_h, int stride_w,
                int pad_t, int pad_h, int pad_w,
                int dil_t, int dil_h, int dil_w,
                int groups,
                int T_out, int H_out, int W_out,
                std::vector<float>& Y) {
    Y.assign(static_cast<size_t>(N) * C_out * T_out * H_out * W_out, 0.0f);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            const int g = oc / Cg_out;
            const int ic_base = g * Cg_in;
            for (int ot = 0; ot < T_out; ++ot) {
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        float acc = has_bias ? bias[oc] : 0.0f;
                        for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                            const int ic = ic_base + ic_local;
                            for (int kt = 0; kt < kT; ++kt) {
                                const int in_t =
                                    ot * stride_t - pad_t + kt * dil_t;
                                if (in_t < 0 || in_t >= T) continue;
                                for (int kh = 0; kh < kH; ++kh) {
                                    const int in_h =
                                        oh * stride_h - pad_h + kh * dil_h;
                                    if (in_h < 0 || in_h >= H) continue;
                                    for (int kw = 0; kw < kW; ++kw) {
                                        const int in_w =
                                            ow * stride_w - pad_w + kw * dil_w;
                                        if (in_w < 0 || in_w >= W) continue;
                                        const int x_idx =
                                            (((n * C_in + ic) * T + in_t) * H +
                                             in_h) * W + in_w;
                                        const int w_idx =
                                            (((oc * Cg_in + ic_local) * kT +
                                              kt) * kH + kh) * kW + kw;
                                        acc += X[x_idx] * Wt[w_idx];
                                    }
                                }
                            }
                        }
                        const int y_idx =
                            (((n * C_out + oc) * T_out + ot) * H_out + oh) *
                                W_out + ow;
                        Y[y_idx] = acc;
                    }
                }
            }
        }
    }
}

void run_one(const char* label,
             int N, int C_in, int T, int H, int W,
             int C_out, int kT, int kH, int kW,
             int stride_t, int stride_h, int stride_w,
             int pad_t, int pad_h, int pad_w,
             int dil_t, int dil_h, int dil_w,
             int groups, bool has_bias, uint64_t seed) {
    std::printf("  %s\n", label);
    const int Cg_in = C_in / groups;
    const int T_out = (T + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    CHECK(T_out > 0 && H_out > 0 && W_out > 0);

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int x_n = N * C_in * T * H * W;
    const int w_n = C_out * Cg_in * kT * kH * kW;
    std::vector<float> X(x_n), Wt(w_n), bias(has_bias ? C_out : 0);
    for (auto& v : X)    v = dist(rng);
    for (auto& v : Wt)   v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    std::vector<float> Y_ref;
    conv3d_ref(X, Wt, bias, has_bias,
               N, C_in, T, H, W, C_out, kT, kH, kW,
               stride_t, stride_h, stride_w,
               pad_t, pad_h, pad_w,
               dil_t, dil_h, dil_w, groups,
               T_out, H_out, W_out, Y_ref);

    Tensor Xt = Tensor::from_host_on(brotensor::Device::CPU,
                                     X.data(), N, C_in * T * H * W);
    Tensor Wtt = Tensor::from_host_on(brotensor::Device::CPU,
                                      Wt.data(), C_out, Cg_in * kT * kH * kW);
    Tensor Bt;
    Tensor* Bp = nullptr;
    if (has_bias) {
        Bt = Tensor::from_host_on(brotensor::Device::CPU,
                                  bias.data(), C_out, 1);
        Bp = &Bt;
    }
    Tensor Y;
    brotensor::conv3d_forward(Xt, Wtt, Bp,
                              N, C_in, T, H, W, C_out, kT, kH, kW,
                              stride_t, stride_h, stride_w,
                              pad_t, pad_h, pad_w,
                              dil_t, dil_h, dil_w, groups, Y);
    CHECK(Y.rows == N);
    CHECK(Y.cols == C_out * T_out * H_out * W_out);
    CHECK(Y.dtype == Dtype::FP32);

    const float* Yp = Y.host_f32();
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < Y_ref.size(); ++i) {
        const float err = std::fabs(Yp[i] - Y_ref[i]);
        if (err > max_err) max_err = err;
        const float tol = 1e-5f + 1e-5f * std::fabs(Y_ref[i]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, Yp[i], Y_ref[i], err);
            }
            ++bad;
        }
    }
    std::printf("    max_err=%g  bad=%d / %zu\n",
                max_err, bad, Y_ref.size());
    CHECK(bad == 0);
}

void test_non_positive_output() {
    // kT larger than T_padded → non-positive T_out.
    Tensor X = Tensor::zeros_on(brotensor::Device::CPU, 1, 1 * 1 * 4 * 4);
    Tensor Wt = Tensor::zeros_on(brotensor::Device::CPU, 1, 1 * 5 * 3 * 3);
    Tensor Y;
    bool threw = false;
    try {
        brotensor::conv3d_forward(X, Wt, nullptr,
                                  1, 1, 1, 4, 4, 1, 5, 3, 3,
                                  1, 1, 1, 0, 1, 1, 1, 1, 1, 1, Y);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_int8_not_implemented_on_cpu() {
    // conv3d_int8w_fp16_forward has no CPU registration; the dispatcher must
    // throw "not implemented on CPU" cleanly. We don't need real FP16/INT8
    // tensors — the dispatcher fires before the kernel.
    Tensor X = Tensor::zeros_on(brotensor::Device::CPU, 1, 1 * 1 * 2 * 2,
                                Dtype::FP16);
    Tensor W = Tensor::zeros_on(brotensor::Device::CPU, 1, 1 * 1 * 1 * 1,
                                Dtype::INT8);
    Tensor S = Tensor::zeros_on(brotensor::Device::CPU, 1, 1, Dtype::FP32);
    Tensor Y;
    bool threw = false;
    try {
        brotensor::conv3d_int8w_fp16_forward(X, W, S, nullptr,
                                             1, 1, 1, 2, 2, 1, 1, 1, 1,
                                             1, 1, 1, 0, 0, 0, 1, 1, 1, 1, Y);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    brotensor::init();
    std::printf("test_conv3d\n===========\n");

    // Tiny full-conv, no bias, no padding.
    run_one("tiny 1x1x1 stride1 nobias",
            /*N*/1, /*C_in*/2, /*T*/2, /*H*/3, /*W*/3,
            /*C_out*/2, /*kT*/1, /*kH*/1, /*kW*/1,
            1, 1, 1,  0, 0, 0,  1, 1, 1,  1, false, 0xA001);

    // 2x2x2 same-pad, with bias.
    run_one("k2x2x2 pad-half bias",
            1, 2, 4, 4, 4,  3, 2, 2, 2,
            1, 1, 1,  0, 0, 0,  1, 1, 1,  1, true,  0xA002);

    // 3x3x3 same-pad.
    run_one("k3x3x3 same-pad bias",
            2, 3, 4, 5, 5,  4, 3, 3, 3,
            1, 1, 1,  1, 1, 1,  1, 1, 1,  1, true,  0xA003);

    // Strided 2x along H/W.
    run_one("k3x3x3 stride T=1 H=2 W=2",
            1, 2, 3, 6, 6,  4, 3, 3, 3,
            1, 2, 2,  1, 1, 1,  1, 1, 1,  1, false, 0xA004);

    // Dilation per-axis.
    run_one("k3x3x3 dilation T=2",
            1, 2, 5, 5, 5,  2, 3, 3, 3,
            1, 1, 1,  2, 2, 2,  2, 1, 1,  1, true,  0xA005);

    // Groups = C_in (depthwise-style in 3D).
    run_one("depthwise-3d groups=Cin",
            1, 4, 3, 5, 5,  4, 3, 3, 3,
            1, 1, 1,  1, 1, 1,  1, 1, 1,  4, true,  0xA006);

    // Generic grouped conv (groups divides Cin and Cout).
    run_one("grouped groups=2",
            1, 4, 3, 4, 4,  6, 3, 3, 3,
            1, 1, 1,  1, 1, 1,  1, 1, 1,  2, false, 0xA007);

    test_non_positive_output();
    test_int8_not_implemented_on_cpu();

    if (g_failures != 0) {
        std::printf("\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall passed\n");
    return 0;
}
