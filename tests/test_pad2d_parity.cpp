// CPU↔GPU parity tests for pad2d_forward / pad2d_backward.
//
//   pad2d_forward  — Y  OVERWRITTEN.
//   pad2d_backward — dX OVERWRITTEN. Scatter adjoint (replicate/reflect can
//                    fold multiple output pixels onto the same input).
//
// FP32-only. NCHW flat layout. Modes: 0 zero, 1 reflect, 2 replicate.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-4f;

void run_fwd(int N, int C, int H, int W,
             int pt, int pb, int pl, int pr, int mode, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr, mode, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::pad2d_forward(gX, N, C, H, W, pt, pb, pl, pr, mode, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "pad2d_fwd", kAtol, kRtol);
}

void run_bwd(int N, int C, int H, int W,
             int pt, int pb, int pl, int pr, int mode, uint64_t seed) {
    SplitMix64 rng(seed);
    const int H_pad = H + pt + pb;
    const int W_pad = W + pl + pr;
    Tensor dY = Tensor::mat(N, C * H_pad * W_pad);
    fill_random(dY, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::pad2d_backward(dY, N, C, H, W, pt, pb, pl, pr, mode, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::pad2d_backward(gdY, N, C, H, W, pt, pb, pl, pr, mode, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "pad2d_bwd",
                    kAtol, kRtol);
}

} // namespace

// zero pad
BT_PARITY_TEST(pad2d_fwd_zero_sym)   { run_fwd(2, 3, 5, 7, 1, 1, 2, 2, 0, 0xC100ull); }
BT_PARITY_TEST(pad2d_fwd_zero_asym)  { run_fwd(2, 3, 5, 7, 1, 3, 0, 2, 0, 0xC101ull); }
BT_PARITY_TEST(pad2d_bwd_zero_sym)   { run_bwd(2, 3, 5, 7, 1, 1, 2, 2, 0, 0xC102ull); }

// reflect
BT_PARITY_TEST(pad2d_fwd_reflect)    { run_fwd(2, 3, 5, 7, 2, 2, 3, 3, 1, 0xC110ull); }
BT_PARITY_TEST(pad2d_fwd_reflect_a)  { run_fwd(1, 4, 6, 8, 0, 1, 1, 0, 1, 0xC111ull); }
BT_PARITY_TEST(pad2d_bwd_reflect)    { run_bwd(2, 3, 5, 7, 2, 2, 3, 3, 1, 0xC112ull); }

// replicate
BT_PARITY_TEST(pad2d_fwd_replicate)  { run_fwd(2, 3, 4, 5, 3, 1, 4, 2, 2, 0xC120ull); }
BT_PARITY_TEST(pad2d_bwd_replicate)  { run_bwd(2, 3, 4, 5, 3, 1, 4, 2, 2, 0xC121ull); }
BT_PARITY_TEST(pad2d_bwd_replicate2) { run_bwd(1, 2, 6, 8, 5, 0, 0, 7, 2, 0xC122ull); }

int main() { return run_all("pad2d cpu/gpu parity"); }
