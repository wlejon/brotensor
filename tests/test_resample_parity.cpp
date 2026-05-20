// CPU↔GPU parity tests for the resample op family (CHUNK 4).
//
//   upsample_nearest_2x            — Y  OVERWRITTEN.
//   upsample_bilinear_2x          — Y  OVERWRITTEN.
//   downsample_avg_2x             — Y  OVERWRITTEN.
//   upsample_nearest_2x_backward  — dX OVERWRITTEN.
//   upsample_bilinear_2x_backward — dX OVERWRITTEN (GPU memsets then atomic-
//                                   scatters; net effect is overwrite).
//   downsample_avg_2x_backward    — dX OVERWRITTEN.
//
// All tensors NCHW; 2x spatial scale. Bilinear uses the half-pixel sampling
// convention src = (out + 0.5) * 0.5 - 0.5 with border-clamped neighbours.
//
// Both CPU and GPU run the FP32 path here, so tolerances are tight. The
// bilinear backward gets a slightly looser tol because the GPU sums via
// atomicAdd in nondeterministic order.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

struct Shape { int N, C, H, W; };

// ─── forward ───────────────────────────────────────────────────────────────
void run_up_nearest(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(s.N, s.C * s.H * s.W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::upsample_nearest_2x(X, s.N, s.C, s.H, s.W, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::upsample_nearest_2x(gX, s.N, s.C, s.H, s.W, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "up_nearest", 1e-5f, 1e-4f);
}

void run_up_bilinear(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(s.N, s.C * s.H * s.W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::upsample_bilinear_2x(X, s.N, s.C, s.H, s.W, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::upsample_bilinear_2x(gX, s.N, s.C, s.H, s.W, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "up_bilinear", 1e-5f, 1e-4f);
}

void run_down_avg(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(s.N, s.C * s.H * s.W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::downsample_avg_2x(X, s.N, s.C, s.H, s.W, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::downsample_avg_2x(gX, s.N, s.C, s.H, s.W, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "down_avg", 1e-5f, 1e-4f);
}

// ─── backward (dX overwritten — no baseline pre-fill needed) ────────────────
void run_up_nearest_bwd(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(s.N, s.C * (2 * s.H) * (2 * s.W));
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::upsample_nearest_2x_backward(dY, s.N, s.C, s.H, s.W, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::upsample_nearest_2x_backward(gdY, s.N, s.C, s.H, s.W, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "up_nearest_bwd",
                    1e-5f, 1e-4f);
}

void run_up_bilinear_bwd(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(s.N, s.C * (2 * s.H) * (2 * s.W));
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::upsample_bilinear_2x_backward(dY, s.N, s.C, s.H, s.W, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::upsample_bilinear_2x_backward(gdY, s.N, s.C, s.H, s.W, gpu_dX);

    // Looser tol: GPU accumulates via atomicAdd in nondeterministic order.
    compare_tensors(cpu_dX, download_to_host(gpu_dX), "up_bilinear_bwd",
                    1e-4f, 1e-3f);
}

void run_down_avg_bwd(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(s.N, s.C * (s.H / 2) * (s.W / 2));
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::downsample_avg_2x_backward(dY, s.N, s.C, s.H, s.W, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::downsample_avg_2x_backward(gdY, s.N, s.C, s.H, s.W, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "down_avg_bwd",
                    1e-5f, 1e-4f);
}

// Shape bank — downsample needs even H, W.
const Shape kTiny  {1, 3, 4, 4};
const Shape kBatch {2, 4, 6, 6};
const Shape kWide  {1, 8, 4, 8};
const Shape kBig   {2, 6, 8, 10};

} // namespace

// ─── forward ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(resample_up_nearest_tiny)  { run_up_nearest(kTiny,  0x7000ull); }
BT_PARITY_TEST(resample_up_nearest_batch) { run_up_nearest(kBatch, 0x7001ull); }
BT_PARITY_TEST(resample_up_nearest_wide)  { run_up_nearest(kWide,  0x7002ull); }
BT_PARITY_TEST(resample_up_nearest_big)   { run_up_nearest(kBig,   0x7003ull); }

BT_PARITY_TEST(resample_up_bilinear_tiny) { run_up_bilinear(kTiny,  0x7010ull); }
BT_PARITY_TEST(resample_up_bilinear_batch){ run_up_bilinear(kBatch, 0x7011ull); }
BT_PARITY_TEST(resample_up_bilinear_wide) { run_up_bilinear(kWide,  0x7012ull); }
BT_PARITY_TEST(resample_up_bilinear_big)  { run_up_bilinear(kBig,   0x7013ull); }

BT_PARITY_TEST(resample_down_avg_tiny)  { run_down_avg(kTiny,  0x7020ull); }
BT_PARITY_TEST(resample_down_avg_batch) { run_down_avg(kBatch, 0x7021ull); }
BT_PARITY_TEST(resample_down_avg_wide)  { run_down_avg(kWide,  0x7022ull); }
BT_PARITY_TEST(resample_down_avg_big)   { run_down_avg(kBig,   0x7023ull); }

// ─── backward ──────────────────────────────────────────────────────────────
BT_PARITY_TEST(resample_up_nearest_bwd_tiny)  { run_up_nearest_bwd(kTiny,  0x7030ull); }
BT_PARITY_TEST(resample_up_nearest_bwd_batch) { run_up_nearest_bwd(kBatch, 0x7031ull); }
BT_PARITY_TEST(resample_up_nearest_bwd_big)   { run_up_nearest_bwd(kBig,   0x7032ull); }

BT_PARITY_TEST(resample_up_bilinear_bwd_tiny)  { run_up_bilinear_bwd(kTiny,  0x7040ull); }
BT_PARITY_TEST(resample_up_bilinear_bwd_batch) { run_up_bilinear_bwd(kBatch, 0x7041ull); }
BT_PARITY_TEST(resample_up_bilinear_bwd_big)   { run_up_bilinear_bwd(kBig,   0x7042ull); }

BT_PARITY_TEST(resample_down_avg_bwd_tiny)  { run_down_avg_bwd(kTiny,  0x7050ull); }
BT_PARITY_TEST(resample_down_avg_bwd_batch) { run_down_avg_bwd(kBatch, 0x7051ull); }
BT_PARITY_TEST(resample_down_avg_bwd_big)   { run_down_avg_bwd(kBig,   0x7052ull); }

int main() { return run_all("resample cpu/gpu parity"); }
