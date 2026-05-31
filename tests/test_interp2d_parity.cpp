// CPU↔GPU parity tests for the 2D arbitrary-scale resample ops.
//
//   interp2d_forward  — Y  OVERWRITTEN. (N, C, H_in, W_in) -> (N, C, H_out, W_out).
//   interp2d_backward — dX OVERWRITTEN. Exact adjoint of the forward.
//
// mode 0 = nearest (round-half-to-even), mode 1 = bilinear (align_corners=False),
// mode 2 = bicubic Keys/Catmull-Rom (a = -0.5). Bicubic backward is not
// implemented (forward-only — checked elsewhere).
//
// FP32-only here (CPU is FP32-only). NCHW flat layout.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-3f;

void run_fwd(int N, int C, int H_in, int W_in,
             int H_out, int W_out, int mode, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H_in * W_in);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::interp2d_forward(X, N, C, H_in, W_in, H_out, W_out, mode, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::interp2d_forward(gX, N, C, H_in, W_in, H_out, W_out, mode, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "interp2d_fwd",
                    kAtol, kRtol);
}

void run_bwd(int N, int C, int H_in, int W_in,
             int H_out, int W_out, int mode, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(N, C * H_out * W_out);
    fill_random(dY, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::interp2d_backward(dY, N, C, H_in, W_in, H_out, W_out, mode, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::interp2d_backward(gdY, N, C, H_in, W_in, H_out, W_out, mode, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "interp2d_bwd",
                    kAtol, kRtol);
}

void run_fwd_ac(int N, int C, int H_in, int W_in,
                int H_out, int W_out, int mode, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H_in * W_in);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::interp2d_align_corners_forward(X, N, C, H_in, W_in,
                                              H_out, W_out, mode, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::interp2d_align_corners_forward(gX, N, C, H_in, W_in,
                                              H_out, W_out, mode, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "interp2d_ac_fwd",
                    kAtol, kRtol);
}

} // namespace

// ─── nearest (mode 0) ──────────────────────────────────────────────────────
BT_PARITY_TEST(i2d_fwd_near_up)     { run_fwd(2, 3, 4, 5, 8, 10, 0, 0xB200ull); }
BT_PARITY_TEST(i2d_fwd_near_down)   { run_fwd(2, 3, 16, 16, 5, 5, 0, 0xB201ull); }
BT_PARITY_TEST(i2d_fwd_near_2x)     { run_fwd(1, 4, 7, 9, 14, 18, 0, 0xB202ull); }
BT_PARITY_TEST(i2d_fwd_near_same)   { run_fwd(1, 2, 6, 7, 6, 7, 0, 0xB203ull); }
BT_PARITY_TEST(i2d_bwd_near_up)     { run_bwd(2, 3, 4, 5, 8, 10, 0, 0xB204ull); }
BT_PARITY_TEST(i2d_bwd_near_down)   { run_bwd(2, 3, 16, 16, 5, 5, 0, 0xB205ull); }

// ─── bilinear (mode 1) ─────────────────────────────────────────────────────
BT_PARITY_TEST(i2d_fwd_bil_up)      { run_fwd(2, 3, 4, 5, 11, 13, 1, 0xB210ull); }
BT_PARITY_TEST(i2d_fwd_bil_down)    { run_fwd(2, 3, 20, 24, 7, 9, 1, 0xB211ull); }
BT_PARITY_TEST(i2d_fwd_bil_2x)      { run_fwd(1, 4, 5, 6, 10, 12, 1, 0xB212ull); }
BT_PARITY_TEST(i2d_fwd_bil_wide)    { run_fwd(4, 6, 9, 11, 23, 17, 1, 0xB213ull); }
BT_PARITY_TEST(i2d_bwd_bil_up)      { run_bwd(2, 3, 4, 5, 11, 13, 1, 0xB214ull); }
BT_PARITY_TEST(i2d_bwd_bil_down)    { run_bwd(2, 3, 20, 24, 7, 9, 1, 0xB215ull); }
BT_PARITY_TEST(i2d_bwd_bil_wide)    { run_bwd(4, 6, 9, 11, 23, 17, 1, 0xB216ull); }

// ─── bicubic (mode 2, forward only) ────────────────────────────────────────
BT_PARITY_TEST(i2d_fwd_bic_up)      { run_fwd(2, 3, 5, 6, 12, 14, 2, 0xB220ull); }
BT_PARITY_TEST(i2d_fwd_bic_down)    { run_fwd(2, 3, 24, 24, 8, 8, 2, 0xB221ull); }
BT_PARITY_TEST(i2d_fwd_bic_2x)      { run_fwd(1, 4, 6, 7, 12, 14, 2, 0xB222ull); }

// ─── align_corners=True forward (nearest / bilinear / bicubic) ──────────────
BT_PARITY_TEST(i2d_ac_fwd_near_up)  { run_fwd_ac(2, 3, 4, 5, 9, 11, 0, 0xB230ull); }
BT_PARITY_TEST(i2d_ac_fwd_bil_up)   { run_fwd_ac(2, 3, 4, 5, 11, 13, 1, 0xB231ull); }
BT_PARITY_TEST(i2d_ac_fwd_bil_down) { run_fwd_ac(2, 3, 20, 24, 7, 9, 1, 0xB232ull); }
BT_PARITY_TEST(i2d_ac_fwd_bil_2x)   { run_fwd_ac(1, 4, 5, 6, 10, 12, 1, 0xB233ull); }
BT_PARITY_TEST(i2d_ac_fwd_bic_up)   { run_fwd_ac(2, 3, 5, 6, 12, 14, 2, 0xB234ull); }

int main() { return run_all("interp2d cpu/gpu parity"); }
