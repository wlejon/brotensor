// NCHW ↔ sequence transpose parity (FP32 + FP16, round-trip + hand-checked).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        out[i] = brotensor::fp32_to_fp16_bits(src[i]);
    return out;
}

// Reference: NCHW row-major X[n,c,h,w] -> sequence Y[n*HW + p, c]
// with p = h*W + w.
static void nchw_to_seq_cpu(const std::vector<float>& X,
                            int N, int C, int H, int W,
                            std::vector<float>& Y) {
    const int HW = H * W;
    Y.assign(static_cast<size_t>(N) * HW * C, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int p = 0; p < HW; ++p)
                Y[(static_cast<size_t>(n) * HW + p) * C + c] =
                    X[(static_cast<size_t>(n) * C + c) * HW + p];
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_transpose\n");

    // Cover a few shapes including the VAE mid-block (N=1, C=512, H=W=64
    // means HW=4096 — we use smaller proxy shapes to keep the test fast).
    struct Shape { int N, C, H, W; };
    const Shape shapes[] = {
        {1, 4, 2, 3},     // tiny hand-checkable
        {1, 8, 4, 4},
        {2, 6, 3, 5},     // batched
        {1, 32, 8, 8},    // bigger
    };

    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const auto& s : shapes) {
        const int N = s.N, C = s.C, H = s.H, W = s.W, HW = H * W;
        const int total = N * C * HW;
        std::vector<float> X(total);
        for (auto& v : X) v = dist(rng);

        std::vector<float> Y_ref;
        nchw_to_seq_cpu(X, N, C, H, W, Y_ref);

        // ─── FP32 path ─────────────────────────────────────────────────
        Tensor Yg, Rg;
        Tensor Xg = Tensor::from_host_on(Device::CUDA, X.data(), N, C * HW);
        brotensor::nchw_to_sequence(Xg, N, C, H, W, Yg);
        CHECK(Yg.rows == N * HW);
        CHECK(Yg.cols == C);
        CHECK(Yg.dtype == Dtype::FP32);
        std::vector<float> Y_h(static_cast<size_t>(Yg.size()), 0.0f);
        brotensor::sync_all();
        Yg.copy_to_host(Y_h.data());
        for (size_t i = 0; i < Y_ref.size(); ++i) {
            if (Y_h[i] != Y_ref[i]) {
                std::printf("  FP32 mismatch at %zu shape=(%d,%d,%d,%d): got %g want %g\n",
                            i, N, C, H, W, Y_h[i], Y_ref[i]);
                ++g_failures;
                break;
            }
        }

        // Round-trip: sequence_to_nchw should recover X exactly.
        brotensor::sequence_to_nchw(Yg, N, C, H, W, Rg);
        CHECK(Rg.rows == N);
        CHECK(Rg.cols == C * HW);
        std::vector<float> R_h(static_cast<size_t>(Rg.size()), 0.0f);
        brotensor::sync_all();
        Rg.copy_to_host(R_h.data());
        for (size_t i = 0; i < X.size(); ++i) {
            if (R_h[i] != X[i]) {
                std::printf("  FP32 round-trip mismatch at %zu shape=(%d,%d,%d,%d)\n",
                            i, N, C, H, W);
                ++g_failures;
                break;
            }
        }

        // ─── FP16 path ─────────────────────────────────────────────────
        auto X_h16 = to_fp16(X);
        Tensor Yg16, Rg16;
        Tensor Xg16 = Tensor::from_host_fp16_on(Device::CUDA, X_h16.data(), N, C * HW);
        brotensor::nchw_to_sequence(Xg16, N, C, H, W, Yg16);
        CHECK(Yg16.rows == N * HW);
        CHECK(Yg16.cols == C);
        CHECK(Yg16.dtype == Dtype::FP16);
        std::vector<uint16_t> Y_h16(static_cast<size_t>(Yg16.size()), 0);
        brotensor::sync_all();
        Yg16.copy_to_host_fp16(Y_h16.data());
        // Compare bit-for-bit against the FP16-of-Y_ref (transpose is a pure
        // gather — no rounding introduced).
        auto Y_ref16 = to_fp16(Y_ref);
        for (size_t i = 0; i < Y_ref16.size(); ++i) {
            if (Y_h16[i] != Y_ref16[i]) {
                std::printf("  FP16 mismatch at %zu shape=(%d,%d,%d,%d): bits got=0x%04x want=0x%04x\n",
                            i, N, C, H, W, Y_h16[i], Y_ref16[i]);
                ++g_failures;
                break;
            }
        }

        brotensor::sequence_to_nchw(Yg16, N, C, H, W, Rg16);
        std::vector<uint16_t> R_h16(static_cast<size_t>(Rg16.size()), 0);
        brotensor::sync_all();
        Rg16.copy_to_host_fp16(R_h16.data());
        for (size_t i = 0; i < X_h16.size(); ++i) {
            if (R_h16[i] != X_h16[i]) {
                std::printf("  FP16 round-trip mismatch at %zu shape=(%d,%d,%d,%d)\n",
                            i, N, C, H, W);
                ++g_failures;
                break;
            }
        }
    }

    // Hand-checked permutation: N=1, C=2, H=2, W=2.
    // X[c,h,w] flattened c-major:
    //   c=0: [a,b,c,d]   (positions p=0,1,2,3)
    //   c=1: [e,f,g,h]
    // Sequence layout Y has 4 rows × 2 cols:
    //   p=0: (a,e), p=1: (b,f), p=2: (c,g), p=3: (d,h)
    {
        const int N = 1, C = 2, H = 2, W = 2;
        std::vector<float> X = {1, 2, 3, 4,   5, 6, 7, 8};
        const std::vector<float> expected = {1, 5,  2, 6,  3, 7,  4, 8};

        Tensor Yg;
        Tensor Xg = Tensor::from_host_on(Device::CUDA, X.data(), N, C * H * W);
        brotensor::nchw_to_sequence(Xg, N, C, H, W, Yg);
        std::vector<float> Y_h(static_cast<size_t>(Yg.size()), 0.0f);
        brotensor::sync_all();
        Yg.copy_to_host(Y_h.data());
        for (size_t i = 0; i < expected.size(); ++i) {
            if (Y_h[i] != expected[i]) {
                std::printf("  hand-check fail at %zu: got %g want %g\n",
                            i, Y_h[i], expected[i]);
                ++g_failures;
            }
        }
    }

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll transpose checks passed.\n");
    return 0;
}
