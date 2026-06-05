// CPU smoke test for spatial_merge_2x2_forward (FP32).
//
// Hand-checked tiny cases plus a few larger shapes, for both channel orderings:
//   channel_major=false: c_out = (dh*2 + dw)*C + c_in  (Qwen-VL, block-major)
//   channel_major=true:  c_out = c_in*4 + (dh*2 + dw)  (torch pixel_unshuffle)
// over the 2x2 block (dh, dw) at each output spatial location.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
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

void ref_spatial_merge(const std::vector<float>& X,
                       int N, int C, int H, int W, bool channel_major,
                       std::vector<float>& Y) {
    const int H_out = H / 2;
    const int W_out = W / 2;
    const int C_out = 4 * C;
    Y.assign((size_t)N * C_out * H_out * W_out, 0.0f);
    const int HW = H * W;
    const int HW_out = H_out * W_out;
    for (int n = 0; n < N; ++n) {
        for (int dh = 0; dh < 2; ++dh) {
            for (int dw = 0; dw < 2; ++dw) {
                const int block = dh * 2 + dw;
                for (int c_in = 0; c_in < C; ++c_in) {
                    const int c_out =
                        channel_major ? c_in * 4 + block : block * C + c_in;
                    for (int h_out = 0; h_out < H_out; ++h_out) {
                        for (int w_out = 0; w_out < W_out; ++w_out) {
                            const int h_in = 2 * h_out + dh;
                            const int w_in = 2 * w_out + dw;
                            const int x_idx = (n * C + c_in) * HW
                                            + h_in * W + w_in;
                            const int y_idx = (n * C_out + c_out) * HW_out
                                            + h_out * W_out + w_out;
                            Y[y_idx] = X[x_idx];
                        }
                    }
                }
            }
        }
    }
}

void test_hand_checked() {
    std::printf("  hand-checked 1x1x4x4 (block at (0,0))\n");
    // X is a 4x4 image of distinct integers. After merge we expect:
    //   Y has 4 channels of shape 2x2.
    //   For output (h_out=0, w_out=0), input block is the 2x2 at rows 0..1, cols 0..1:
    //     X[0,0]=0  X[0,1]=1
    //     X[1,0]=4  X[1,1]=5
    //   So Y[c_out=0] (dh=0, dw=0) at (0,0) is X[0,0]=0,
    //      Y[c_out=1] (dh=0, dw=1) at (0,0) is X[0,1]=1,
    //      Y[c_out=2] (dh=1, dw=0) at (0,0) is X[1,0]=4,
    //      Y[c_out=3] (dh=1, dw=1) at (0,0) is X[1,1]=5.
    std::vector<float> X(16);
    for (int i = 0; i < 16; ++i) X[i] = (float)i;
    Tensor Xt = Tensor::from_host_on(brotensor::Device::CPU,
                                     X.data(), 1, 16);
    Tensor Y;
    brotensor::spatial_merge_2x2_forward(Xt, 1, 1, 4, 4, /*channel_major=*/false, Y);
    CHECK(Y.rows == 1);
    CHECK(Y.cols == 4 * 2 * 2);
    const float* Yp = Y.host_f32();
    // (h_out=0, w_out=0): channels 0..3.
    CHECK(Yp[0 * 4 + 0] == 0.0f);  // c_out=0
    CHECK(Yp[1 * 4 + 0] == 1.0f);  // c_out=1
    CHECK(Yp[2 * 4 + 0] == 4.0f);  // c_out=2
    CHECK(Yp[3 * 4 + 0] == 5.0f);  // c_out=3
    // (h_out=1, w_out=1) => input block at rows 2..3, cols 2..3:
    //   X[2,2]=10 X[2,3]=11 X[3,2]=14 X[3,3]=15.
    CHECK(Yp[0 * 4 + 3] == 10.0f);
    CHECK(Yp[1 * 4 + 3] == 11.0f);
    CHECK(Yp[2 * 4 + 3] == 14.0f);
    CHECK(Yp[3 * 4 + 3] == 15.0f);

    // With C=1 the two orderings coincide (c_in*4+block == block when c_in=0),
    // so use C=2 to actually distinguish block-major from channel-major. Check
    // both against the explicit reference.
    std::vector<float> X2(2 * 16);
    for (int i = 0; i < 2 * 16; ++i) X2[i] = (float)i;
    Tensor X2t = Tensor::from_host_on(brotensor::Device::CPU, X2.data(), 1, 32);
    for (bool cm : {false, true}) {
        Tensor Y2;
        brotensor::spatial_merge_2x2_forward(X2t, 1, 2, 4, 4, cm, Y2);
        std::vector<float> want;
        ref_spatial_merge(X2, 1, 2, 4, 4, cm, want);
        const float* y2 = Y2.host_f32();
        int bad = 0;
        for (size_t i = 0; i < want.size(); ++i) if (y2[i] != want[i]) ++bad;
        std::printf("    C=2 channel_major=%d : %s\n", cm ? 1 : 0,
                    bad == 0 ? "ok" : "MISMATCH");
        CHECK(bad == 0);
    }
}

void run_shape(const char* label, int N, int C, int H, int W, uint64_t seed) {
    std::mt19937 rng((uint32_t)seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int x_n = N * C * H * W;
    std::vector<float> X(x_n);
    for (auto& v : X) v = dist(rng);

    Tensor Xt = Tensor::from_host_on(brotensor::Device::CPU,
                                     X.data(), N, C * H * W);
    for (bool cm : {false, true}) {
        std::printf("  %s  (N=%d, C=%d, H=%d, W=%d, channel_major=%d)\n",
                    label, N, C, H, W, cm ? 1 : 0);
        std::vector<float> Y_ref;
        ref_spatial_merge(X, N, C, H, W, cm, Y_ref);
        Tensor Y;
        brotensor::spatial_merge_2x2_forward(Xt, N, C, H, W, cm, Y);
        CHECK(Y.rows == N);
        CHECK(Y.cols == 4 * C * (H/2) * (W/2));
        CHECK(Y.dtype == Dtype::FP32);
        const float* Yp = Y.host_f32();
        int bad = 0;
        for (size_t i = 0; i < Y_ref.size(); ++i) {
            if (Yp[i] != Y_ref[i]) {
                if (bad < 5) std::printf("    mismatch i=%zu got=%g ref=%g\n",
                                         i, Yp[i], Y_ref[i]);
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
}

void test_odd_dim_rejected() {
    Tensor X = Tensor::zeros_on(brotensor::Device::CPU, 1, 1 * 3 * 4);
    Tensor Y;
    bool threw = false;
    try {
        brotensor::spatial_merge_2x2_forward(X, 1, 1, 3, 4, /*channel_major=*/false, Y);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    brotensor::init();
    std::printf("test_spatial_merge\n==================\n");
    test_hand_checked();
    run_shape("small",       1, 2, 4, 4, 0xB001);
    run_shape("rect",        2, 3, 4, 6, 0xB002);
    run_shape("more chan",   1, 5, 6, 6, 0xB003);
    run_shape("batched",     3, 4, 8, 8, 0xB004);
    test_odd_dim_rejected();
    if (g_failures != 0) {
        std::printf("\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall passed\n");
    return 0;
}
