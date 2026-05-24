// CPU↔GPU parity tests for slice2d_forward / slice2d_backward.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-4f;

void run_fwd(int N, int C, int H, int W,
             int h0, int w0, int Ho, int Wo, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::slice2d_forward(X, N, C, H, W, h0, w0, Ho, Wo, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::slice2d_forward(gX, N, C, H, W, h0, w0, Ho, Wo, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "slice2d_fwd", kAtol, kRtol);
}

void run_bwd(int N, int C, int H, int W,
             int h0, int w0, int Ho, int Wo, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(N, C * Ho * Wo);
    fill_random(dY, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::slice2d_backward(dY, N, C, H, W, h0, w0, Ho, Wo, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::slice2d_backward(gdY, N, C, H, W, h0, w0, Ho, Wo, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "slice2d_bwd",
                    kAtol, kRtol);
}

} // namespace

BT_PARITY_TEST(slice2d_fwd_basic)  { run_fwd(2, 3, 8, 10, 2, 3, 4, 5, 0xC200ull); }
BT_PARITY_TEST(slice2d_fwd_full)   { run_fwd(1, 4, 6, 7, 0, 0, 6, 7, 0xC201ull); }
BT_PARITY_TEST(slice2d_fwd_edge)   { run_fwd(2, 2, 5, 9, 3, 5, 2, 4, 0xC202ull); }
BT_PARITY_TEST(slice2d_bwd_basic)  { run_bwd(2, 3, 8, 10, 2, 3, 4, 5, 0xC210ull); }
BT_PARITY_TEST(slice2d_bwd_edge)   { run_bwd(2, 2, 5, 9, 3, 5, 2, 4, 0xC211ull); }

int main() { return run_all("slice2d cpu/gpu parity"); }
