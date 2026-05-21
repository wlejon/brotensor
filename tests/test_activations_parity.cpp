// CPU↔GPU parity tests for the brotensor activation ops:
//   silu, gelu (tanh-approx), gelu_exact (erf-based), quick_gelu — fwd + bwd.
//
// CHUNK 2.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

// Forward driver: op(x) -> y, compared CPU vs GPU.
template <typename Fwd>
void run_fwd(Fwd fwd, const char* tag, int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::mat(r, c);
    fill_random(x, rng, 3.0f);  // wide range exercises saturating tails

    Tensor cpu_y;
    fwd(x, cpu_y);

    Tensor gx = x.to(gpu_device());
    Tensor gpu_y;
    fwd(gx, gpu_y);

    compare_tensors(cpu_y, download_to_host(gpu_y), tag);
}

// Backward driver: op(x, dY) -> dX, compared CPU vs GPU.
template <typename Bwd>
void run_bwd(Bwd bwd, const char* tag, int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x  = Tensor::mat(r, c);
    Tensor dY = Tensor::mat(r, c);
    fill_random(x, rng, 3.0f);
    fill_random(dY, rng);

    Tensor cpu_dX;
    bwd(x, dY, cpu_dX);

    Tensor gx  = x.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    bwd(gx, gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), tag);
}

// BF16 forward driver: rounds input to BF16, runs op on CUDA, widens result.
template <typename Fwd>
void run_fwd_bf16(Fwd fwd, const char* tag, int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::mat(r, c);
    fill_random(x, rng, 3.0f);

    Tensor cpu_y;
    fwd(x, cpu_y);  // FP32 CPU reference

    Tensor gx = to_bf16_gpu(x);
    Tensor gpu_y;
    fwd(gx, gpu_y);

    compare_tensors(cpu_y, bf16_host_to_f32(download_to_host(gpu_y)), tag, 2e-2f, 2e-2f);
}

// BF16 backward driver.
template <typename Bwd>
void run_bwd_bf16(Bwd bwd, const char* tag, int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x  = Tensor::mat(r, c);
    Tensor dY = Tensor::mat(r, c);
    fill_random(x, rng, 3.0f);
    fill_random(dY, rng);

    Tensor cpu_dX;
    bwd(x, dY, cpu_dX);  // FP32 CPU reference

    Tensor gx  = to_bf16_gpu(x);
    Tensor gdY = to_bf16_gpu(dY);
    Tensor gpu_dX;
    bwd(gx, gdY, gpu_dX);

    compare_tensors(cpu_dX, bf16_host_to_f32(download_to_host(gpu_dX)), tag, 2e-2f, 2e-2f);
}

} // namespace

// ─── silu ──────────────────────────────────────────────────────────────────
BT_PARITY_TEST(silu_fwd_1x1)   { run_fwd(brotensor::silu_forward, "silu_fwd", 1, 1, 0x2000ull); }
BT_PARITY_TEST(silu_fwd_8x32)  { run_fwd(brotensor::silu_forward, "silu_fwd", 8, 32, 0x2001ull); }
BT_PARITY_TEST(silu_fwd_vec)   { run_fwd(brotensor::silu_forward, "silu_fwd", 64, 1, 0x2002ull); }
BT_PARITY_TEST(silu_bwd_1x1)   { run_bwd(brotensor::silu_backward, "silu_bwd", 1, 1, 0x2003ull); }
BT_PARITY_TEST(silu_bwd_8x32)  { run_bwd(brotensor::silu_backward, "silu_bwd", 8, 32, 0x2004ull); }

// ─── gelu (tanh-approx) ────────────────────────────────────────────────────
BT_PARITY_TEST(gelu_fwd_1x1)   { run_fwd(brotensor::gelu_forward, "gelu_fwd", 1, 1, 0x2010ull); }
BT_PARITY_TEST(gelu_fwd_8x32)  { run_fwd(brotensor::gelu_forward, "gelu_fwd", 8, 32, 0x2011ull); }
BT_PARITY_TEST(gelu_bwd_1x1)   { run_bwd(brotensor::gelu_backward, "gelu_bwd", 1, 1, 0x2012ull); }
BT_PARITY_TEST(gelu_bwd_8x32)  { run_bwd(brotensor::gelu_backward, "gelu_bwd", 8, 32, 0x2013ull); }

// ─── gelu_exact (erf-based) ────────────────────────────────────────────────
BT_PARITY_TEST(gelu_exact_fwd_1x1)  { run_fwd(brotensor::gelu_exact_forward, "gelu_exact_fwd", 1, 1, 0x2020ull); }
BT_PARITY_TEST(gelu_exact_fwd_8x32) { run_fwd(brotensor::gelu_exact_forward, "gelu_exact_fwd", 8, 32, 0x2021ull); }
BT_PARITY_TEST(gelu_exact_bwd_1x1)  { run_bwd(brotensor::gelu_exact_backward, "gelu_exact_bwd", 1, 1, 0x2022ull); }
BT_PARITY_TEST(gelu_exact_bwd_8x32) { run_bwd(brotensor::gelu_exact_backward, "gelu_exact_bwd", 8, 32, 0x2023ull); }

// ─── quick_gelu ────────────────────────────────────────────────────────────
BT_PARITY_TEST(quick_gelu_fwd_1x1)  { run_fwd(brotensor::quick_gelu_forward, "quick_gelu_fwd", 1, 1, 0x2030ull); }
BT_PARITY_TEST(quick_gelu_fwd_8x32) { run_fwd(brotensor::quick_gelu_forward, "quick_gelu_fwd", 8, 32, 0x2031ull); }
BT_PARITY_TEST(quick_gelu_bwd_1x1)  { run_bwd(brotensor::quick_gelu_backward, "quick_gelu_bwd", 1, 1, 0x2032ull); }
BT_PARITY_TEST(quick_gelu_bwd_8x32) { run_bwd(brotensor::quick_gelu_backward, "quick_gelu_bwd", 8, 32, 0x2033ull); }

// ─── BF16 parity tests ─────────────────────────────────────────────────────
BT_PARITY_TEST(silu_fwd_bf16_8x32)  { run_fwd_bf16(brotensor::silu_forward, "silu_fwd_bf16", 8, 32, 0x2100ull); }
BT_PARITY_TEST(silu_fwd_bf16_64x1)  { run_fwd_bf16(brotensor::silu_forward, "silu_fwd_bf16", 64, 1, 0x2101ull); }
BT_PARITY_TEST(silu_bwd_bf16_8x32)  { run_bwd_bf16(brotensor::silu_backward, "silu_bwd_bf16", 8, 32, 0x2102ull); }

BT_PARITY_TEST(gelu_fwd_bf16_8x32)  { run_fwd_bf16(brotensor::gelu_forward, "gelu_fwd_bf16", 8, 32, 0x2110ull); }
BT_PARITY_TEST(gelu_fwd_bf16_64x1)  { run_fwd_bf16(brotensor::gelu_forward, "gelu_fwd_bf16", 64, 1, 0x2111ull); }
BT_PARITY_TEST(gelu_bwd_bf16_8x32)  { run_bwd_bf16(brotensor::gelu_backward, "gelu_bwd_bf16", 8, 32, 0x2112ull); }

BT_PARITY_TEST(gelu_exact_fwd_bf16_8x32)  { run_fwd_bf16(brotensor::gelu_exact_forward, "gelu_exact_fwd_bf16", 8, 32, 0x2120ull); }
BT_PARITY_TEST(gelu_exact_fwd_bf16_64x1)  { run_fwd_bf16(brotensor::gelu_exact_forward, "gelu_exact_fwd_bf16", 64, 1, 0x2121ull); }
BT_PARITY_TEST(gelu_exact_bwd_bf16_8x32)  { run_bwd_bf16(brotensor::gelu_exact_backward, "gelu_exact_bwd_bf16", 8, 32, 0x2122ull); }

BT_PARITY_TEST(quick_gelu_fwd_bf16_8x32)  { run_fwd_bf16(brotensor::quick_gelu_forward, "quick_gelu_fwd_bf16", 8, 32, 0x2130ull); }
BT_PARITY_TEST(quick_gelu_fwd_bf16_64x1)  { run_fwd_bf16(brotensor::quick_gelu_forward, "quick_gelu_fwd_bf16", 64, 1, 0x2131ull); }
BT_PARITY_TEST(quick_gelu_bwd_bf16_8x32)  { run_bwd_bf16(brotensor::quick_gelu_backward, "quick_gelu_bwd_bf16", 8, 32, 0x2132ull); }

int main() { return run_all("silu/gelu/gelu_exact/quick_gelu cpu/gpu parity"); }
