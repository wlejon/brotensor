// CPU↔GPU parity tests for the brosoundml log / exp / round elementwise ops
// (CHUNK 6, family G).
//
//   log_forward / log_backward   y = log(x);   dX = dY / x   (x > 0 required)
//   exp_forward / exp_backward   y = exp(x);   dX = dY * exp(x)
//   round_forward                y = round-half-to-even(x)
//   round_backward               straight-through estimator: dX = dY
//
// All backwards OVERWRITE dX (no learnable params). FP32-only on every backend.
// round_forward intentionally avoids exact half-integers — round-half-to-even
// is bit-identical CPU↔GPU there, but a fuzz seed could otherwise land on a
// .5 and surface harmless near-tie noise from upstream FP ops; the inputs are
// scaled so ties are vanishingly unlikely.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-3f;

// Strictly-positive input for the log family: |uniform| * scale + 0.05.
Tensor positive_mat(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) {
        t.ptr()[i] = std::fabs(rng.next_unit()) * scale + 0.05f;
    }
    return t;
}

// ─── log ────────────────────────────────────────────────────────────────────
void run_log_fwd(int rows, int cols, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = positive_mat(rows, cols, rng, 4.0f);

    Tensor cpu_Y;
    brotensor::log_forward(X, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::log_forward(gX, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "log_fwd", kAtol, kRtol);
}

void run_log_bwd(int rows, int cols, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = positive_mat(rows, cols, rng, 4.0f);
    Tensor dY = Tensor::mat(rows, cols);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::log_backward(X, dY, cpu_dX);

    Tensor gX = X.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::log_backward(gX, gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "log_bwd", kAtol, kRtol);
}

// ─── exp ────────────────────────────────────────────────────────────────────
void run_exp_fwd(int rows, int cols, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, 3.0f);   // exp over [-3, 3]

    Tensor cpu_Y;
    brotensor::exp_forward(X, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::exp_forward(gX, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "exp_fwd", kAtol, kRtol);
}

void run_exp_bwd(int rows, int cols, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    Tensor dY = Tensor::mat(rows, cols);
    fill_random(X, rng, 3.0f);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::exp_backward(X, dY, cpu_dX);

    Tensor gX = X.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::exp_backward(gX, gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "exp_bwd", kAtol, kRtol);
}

// ─── round ──────────────────────────────────────────────────────────────────
void run_round_fwd(int rows, int cols, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, 8.0f);   // span several integers, both signs

    Tensor cpu_Y;
    brotensor::round_forward(X, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::round_forward(gX, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "round_fwd", kAtol, kRtol);
}

void run_round_bwd(int rows, int cols, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(rows, cols);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::round_backward(dY, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::round_backward(gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "round_bwd", kAtol, kRtol);
}

} // namespace

// ─── log ────────────────────────────────────────────────────────────────────
BT_PARITY_TEST(log_fwd_8x32)   { run_log_fwd(8, 32, 0x9000ull); }
BT_PARITY_TEST(log_fwd_5x7)    { run_log_fwd(5, 7,  0x9001ull); }
BT_PARITY_TEST(log_fwd_wide)   { run_log_fwd(3, 1153, 0x9002ull); }
BT_PARITY_TEST(log_bwd_8x32)   { run_log_bwd(8, 32, 0x9003ull); }
BT_PARITY_TEST(log_bwd_5x7)    { run_log_bwd(5, 7,  0x9004ull); }

// ─── exp ────────────────────────────────────────────────────────────────────
BT_PARITY_TEST(exp_fwd_8x32)   { run_exp_fwd(8, 32, 0x9010ull); }
BT_PARITY_TEST(exp_fwd_5x7)    { run_exp_fwd(5, 7,  0x9011ull); }
BT_PARITY_TEST(exp_fwd_wide)   { run_exp_fwd(3, 1153, 0x9012ull); }
BT_PARITY_TEST(exp_bwd_8x32)   { run_exp_bwd(8, 32, 0x9013ull); }
BT_PARITY_TEST(exp_bwd_5x7)    { run_exp_bwd(5, 7,  0x9014ull); }

// ─── round ──────────────────────────────────────────────────────────────────
BT_PARITY_TEST(round_fwd_8x32) { run_round_fwd(8, 32, 0x9020ull); }
BT_PARITY_TEST(round_fwd_5x7)  { run_round_fwd(5, 7,  0x9021ull); }
BT_PARITY_TEST(round_fwd_wide) { run_round_fwd(3, 1153, 0x9022ull); }
BT_PARITY_TEST(round_bwd_8x32) { run_round_bwd(8, 32, 0x9023ull); }

int main() { return run_all("log/exp/round cpu/gpu parity"); }
