// CPU↔GPU parity tests for the 1D resampling ops (brosoundml CHUNK 6, family E).
//
//   resample1d_forward  — Y  OVERWRITTEN. (N, C, L_in) -> (N, C, L_out).
//   resample1d_backward — dX OVERWRITTEN. Exact adjoint of the forward.
//
// mode 0 = nearest (round-half-to-even), mode 1 = linear (align_corners=False).
// FP32-only on every backend. NCL flat layout: (n, c, l) at (n*C + c)*L + l.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-3f;

void run_fwd(int N, int C, int L_in, int L_out, int mode, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * L_in);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::resample1d_forward(X, N, C, L_in, L_out, mode, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::resample1d_forward(gX, N, C, L_in, L_out, mode, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "resample1d_fwd",
                    kAtol, kRtol);
}

void run_bwd(int N, int C, int L_in, int L_out, int mode, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(N, C * L_out);
    fill_random(dY, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::resample1d_backward(dY, N, C, L_in, L_out, mode, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::resample1d_backward(gdY, N, C, L_in, L_out, mode, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "resample1d_bwd",
                    kAtol, kRtol);
}

} // namespace

// ─── nearest (mode 0) ──────────────────────────────────────────────────────
BT_PARITY_TEST(r1d_fwd_near_up)    { run_fwd(2, 3, 8, 16, 0, 0xA100ull); }
BT_PARITY_TEST(r1d_fwd_near_down)  { run_fwd(2, 3, 16, 8, 0, 0xA101ull); }
BT_PARITY_TEST(r1d_fwd_near_same)  { run_fwd(1, 4, 12, 12, 0, 0xA102ull); }
BT_PARITY_TEST(r1d_bwd_near_up)    { run_bwd(2, 3, 8, 16, 0, 0xA103ull); }
BT_PARITY_TEST(r1d_bwd_near_down)  { run_bwd(2, 3, 16, 8, 0, 0xA104ull); }

// ─── linear (mode 1) ───────────────────────────────────────────────────────
BT_PARITY_TEST(r1d_fwd_lin_up)     { run_fwd(2, 3, 8, 21, 1, 0xA110ull); }
BT_PARITY_TEST(r1d_fwd_lin_down)   { run_fwd(2, 3, 20, 7, 1, 0xA111ull); }
BT_PARITY_TEST(r1d_fwd_lin_wide)   { run_fwd(4, 6, 17, 40, 1, 0xA112ull); }
BT_PARITY_TEST(r1d_bwd_lin_up)     { run_bwd(2, 3, 8, 21, 1, 0xA113ull); }
BT_PARITY_TEST(r1d_bwd_lin_down)   { run_bwd(2, 3, 20, 7, 1, 0xA114ull); }
BT_PARITY_TEST(r1d_bwd_lin_wide)   { run_bwd(4, 6, 17, 40, 1, 0xA115ull); }

int main() { return run_all("resample1d cpu/gpu parity"); }
