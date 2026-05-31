// CPU↔GPU parity tests for brotensor::l2_normalize_nchw_forward.
//
// Per-pixel channel-axis L2 normalize, NCHW:
//   Y[n,c,h,w] = X[n,c,h,w] / max(sqrt(sum_c X^2), eps)
// FP32 on both sides. Beyond CPU↔GPU parity, one case checks the CPU result
// against a hand-rolled host reference so correctness is anchored, including a
// near-zero-vector pixel that exercises the eps floor.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

void run(int N, int C, int H, int W, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::l2_normalize_nchw_forward(X, N, C, H, W, eps, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::l2_normalize_nchw_forward(gX, N, C, H, W, eps, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "l2norm_nchw", 1e-5f, 1e-4f);

    // Independent host reference.
    const int HW = H * W;
    std::vector<float> ref(static_cast<size_t>(N) * C * HW);
    for (int n = 0; n < N; ++n)
      for (int p = 0; p < HW; ++p) {
        double ss = 0.0;
        for (int c = 0; c < C; ++c) {
            const double v = X[(size_t)n * C * HW + (size_t)c * HW + p];
            ss += v * v;
        }
        const double inv = 1.0 / std::max(std::sqrt(ss), (double)eps);
        for (int c = 0; c < C; ++c) {
            const size_t off = (size_t)n * C * HW + (size_t)c * HW + p;
            ref[off] = static_cast<float>(X[off] * inv);
        }
      }
    for (int i = 0; i < cpu_Y.size(); ++i)
        BT_CHECK(std::fabs(cpu_Y[i] - ref[i]) <= 1e-6f + 1e-5f * std::fabs(ref[i]));
}

// Zero-vector pixel: every channel 0 at one position → eps floor keeps output
// finite (and zero). Verifies the divisor floor on both backends.
void run_zero_pixel(float eps, uint64_t seed) {
    const int N = 1, C = 3, H = 2, W = 2;
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng);
    const int HW = H * W;
    for (int c = 0; c < C; ++c) X.ptr()[(size_t)c * HW + 0] = 0.0f;  // pixel 0 = 0

    Tensor cpu_Y, gpu_Y;
    brotensor::l2_normalize_nchw_forward(X, N, C, H, W, eps, cpu_Y);
    Tensor gX = X.to(gpu_device());
    brotensor::l2_normalize_nchw_forward(gX, N, C, H, W, eps, gpu_Y);
    compare_tensors(cpu_Y, download_to_host(gpu_Y), "l2norm_zero", 1e-6f, 1e-5f);
    for (int c = 0; c < C; ++c) BT_CHECK(cpu_Y[(size_t)c * HW + 0] == 0.0f);
}

} // namespace

BT_PARITY_TEST(l2norm_nchw_3ch)   { run(1, 3, 8, 8,  1e-12f, 0x9200ull); }
BT_PARITY_TEST(l2norm_nchw_64ch)  { run(2, 64, 5, 7, 1e-12f, 0x9201ull); }
BT_PARITY_TEST(l2norm_nchw_1px)   { run(1, 16, 1, 1, 1e-12f, 0x9202ull); }
BT_PARITY_TEST(l2norm_nchw_eps)   { run(1, 8, 4, 4,  1e-2f,  0x9203ull); }
BT_PARITY_TEST(l2norm_zero_pixel) { run_zero_pixel(1e-12f, 0x9204ull); }

int main() { return run_all("l2_normalize parity"); }
