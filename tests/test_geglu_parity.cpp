// CPU↔GPU parity tests for the brotensor GEGLU ops:
//   geglu / geglu_exact — forward + backward.
//
// CHUNK 2. X is (B, 2D); first half is the value, second half is the gate.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

template <typename Fwd>
void run_fwd(Fwd fwd, const char* tag, int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(B, 2 * D);
    fill_random(X, rng, 3.0f);

    Tensor cpu_Y;
    fwd(X, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    fwd(gX, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), tag);
}

template <typename Bwd>
void run_bwd(Bwd bwd, const char* tag, int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(B, 2 * D);
    Tensor dY = Tensor::mat(B, D);
    fill_random(X, rng, 3.0f);
    fill_random(dY, rng);

    Tensor cpu_dX;
    bwd(X, dY, cpu_dX);

    Tensor gX  = X.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    bwd(gX, gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), tag);
}

template <typename Fwd>
void run_fwd_bf16(Fwd fwd, const char* tag, int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(B, 2 * D);
    fill_random(X, rng, 3.0f);

    Tensor cpu_Y;
    fwd(X, cpu_Y);  // FP32 CPU reference

    Tensor gX = to_bf16_cuda(X);
    Tensor gpu_Y;
    fwd(gX, gpu_Y);

    compare_tensors(cpu_Y, bf16_host_to_f32(download_to_host(gpu_Y)), tag, 2e-2f, 2e-2f);
}

template <typename Bwd>
void run_bwd_bf16(Bwd bwd, const char* tag, int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(B, 2 * D);
    Tensor dY = Tensor::mat(B, D);
    fill_random(X, rng, 3.0f);
    fill_random(dY, rng);

    Tensor cpu_dX;
    bwd(X, dY, cpu_dX);  // FP32 CPU reference

    Tensor gX  = to_bf16_cuda(X);
    Tensor gdY = to_bf16_cuda(dY);
    Tensor gpu_dX;
    bwd(gX, gdY, gpu_dX);

    compare_tensors(cpu_dX, bf16_host_to_f32(download_to_host(gpu_dX)), tag, 2e-2f, 2e-2f);
}

} // namespace

// ─── geglu (tanh-approx) ───────────────────────────────────────────────────
BT_PARITY_TEST(geglu_fwd_1x2)   { run_fwd(brotensor::geglu_forward, "geglu_fwd", 1, 1, 0x3000ull); }
BT_PARITY_TEST(geglu_fwd_8x64)  { run_fwd(brotensor::geglu_forward, "geglu_fwd", 8, 32, 0x3001ull); }
BT_PARITY_TEST(geglu_fwd_5x14)  { run_fwd(brotensor::geglu_forward, "geglu_fwd", 5, 7, 0x3002ull); }
BT_PARITY_TEST(geglu_bwd_1x2)   { run_bwd(brotensor::geglu_backward, "geglu_bwd", 1, 1, 0x3003ull); }
BT_PARITY_TEST(geglu_bwd_8x64)  { run_bwd(brotensor::geglu_backward, "geglu_bwd", 8, 32, 0x3004ull); }

// ─── geglu_exact (erf-based) ───────────────────────────────────────────────
BT_PARITY_TEST(geglu_exact_fwd_1x2)  { run_fwd(brotensor::geglu_exact_forward, "geglu_exact_fwd", 1, 1, 0x3010ull); }
BT_PARITY_TEST(geglu_exact_fwd_8x64) { run_fwd(brotensor::geglu_exact_forward, "geglu_exact_fwd", 8, 32, 0x3011ull); }
BT_PARITY_TEST(geglu_exact_bwd_1x2)  { run_bwd(brotensor::geglu_exact_backward, "geglu_exact_bwd", 1, 1, 0x3012ull); }
BT_PARITY_TEST(geglu_exact_bwd_8x64) { run_bwd(brotensor::geglu_exact_backward, "geglu_exact_bwd", 8, 32, 0x3013ull); }

// ─── BF16 parity tests ─────────────────────────────────────────────────────
BT_PARITY_TEST(geglu_fwd_bf16_8x64)       { run_fwd_bf16(brotensor::geglu_forward, "geglu_fwd_bf16", 8, 32, 0x3100ull); }
BT_PARITY_TEST(geglu_fwd_bf16_5x14)       { run_fwd_bf16(brotensor::geglu_forward, "geglu_fwd_bf16", 5, 7, 0x3101ull); }
BT_PARITY_TEST(geglu_bwd_bf16_8x64)       { run_bwd_bf16(brotensor::geglu_backward, "geglu_bwd_bf16", 8, 32, 0x3102ull); }

BT_PARITY_TEST(geglu_exact_fwd_bf16_8x64) { run_fwd_bf16(brotensor::geglu_exact_forward, "geglu_exact_fwd_bf16", 8, 32, 0x3110ull); }
BT_PARITY_TEST(geglu_exact_fwd_bf16_5x14) { run_fwd_bf16(brotensor::geglu_exact_forward, "geglu_exact_fwd_bf16", 5, 7, 0x3111ull); }
BT_PARITY_TEST(geglu_exact_bwd_bf16_8x64) { run_bwd_bf16(brotensor::geglu_exact_backward, "geglu_exact_bwd_bf16", 8, 32, 0x3112ull); }

int main() { return run_all("geglu/geglu_exact cpu/gpu parity"); }
