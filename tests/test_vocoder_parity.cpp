// CPU↔GPU parity tests for the brosoundml vocoder/codec activations
// (CHUNK 4, family C).
//
//   snake_forward  — Y OVERWRITTEN. Per-channel alpha/beta over an NCL tensor;
//                    covers both plain snake (beta == null) and snakebeta.
//   snake_backward — dX OVERWRITTEN; dAlpha / dBeta ACCUMULATE (+=). The tests
//                    pre-fill dAlpha / dBeta with a non-zero baseline to verify
//                    the contract.
//   elu / leaky_relu forward + backward — plain elementwise, no params.
//
// FP32-only on every backend — there is no BF16 path for these audio ops.
// snake involves sin/cos directly, so its tolerance is looser than the plain
// elementwise activations'.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-3f;
constexpr float kSnakeAtol = 2e-3f;   // sin/cos direct — looser
constexpr float kSnakeRtol = 2e-3f;

// Per-channel alpha/beta away from zero, so snake's 1/denom guard never fires
// and CPU and GPU stay on the same arithmetic path. Range [0.5, 1.5].
Tensor channel_param(int C, SplitMix64& rng) {
    Tensor t = Tensor::vec(C);
    for (int c = 0; c < C; ++c) t.ptr()[c] = rng.next_unit() * 0.5f + 1.0f;
    return t;
}

// ─── snake_forward ─────────────────────────────────────────────────────────
void run_snake_fwd(int N, int C, int L, bool has_beta, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * L);
    fill_random(X, rng, 0.8f);
    Tensor alpha = channel_param(C, rng);
    Tensor beta;
    Tensor* betap = nullptr;
    if (has_beta) { beta = channel_param(C, rng); betap = &beta; }

    Tensor cpu_Y;
    brotensor::snake_forward(X, alpha, betap, N, C, L, cpu_Y);

    Tensor gX = X.to(gpu_device()), gA = alpha.to(gpu_device());
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_beta) { gB = beta.to(gpu_device()); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::snake_forward(gX, gA, gBp, N, C, L, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "snake_fwd",
                    kSnakeAtol, kSnakeRtol);
}

// ─── snake_backward — dX overwrite; dAlpha / dBeta accumulate ──────────────
void run_snake_bwd(int N, int C, int L, bool has_beta, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * L);
    Tensor dY = Tensor::mat(N, C * L);
    fill_random(X, rng, 0.8f);
    fill_random(dY, rng, 0.5f);
    Tensor alpha = channel_param(C, rng);
    Tensor beta;
    Tensor* betap = nullptr;
    if (has_beta) { beta = channel_param(C, rng); betap = &beta; }

    // Non-zero baselines to verify the += accumulation contract.
    Tensor dA0 = channel_param(C, rng);
    Tensor dB0 = channel_param(C, rng);

    Tensor cpu_dX, cpu_dA = dA0, cpu_dB = dB0;   // deep copies
    Tensor* cpu_dBp = has_beta ? &cpu_dB : nullptr;
    brotensor::snake_backward(X, alpha, betap, dY, N, C, L,
                              cpu_dX, cpu_dA, cpu_dBp);

    Tensor gX = X.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gA = alpha.to(gpu_device());
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_beta) { gB = beta.to(gpu_device()); gBp = &gB; }
    Tensor gpu_dX, gpu_dA = dA0.to(gpu_device()), gpu_dB = dB0.to(gpu_device());
    Tensor* gpu_dBp = has_beta ? &gpu_dB : nullptr;
    brotensor::snake_backward(gX, gA, gBp, gdY, N, C, L,
                              gpu_dX, gpu_dA, gpu_dBp);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "snake_bwd_dX",
                    kSnakeAtol, kSnakeRtol);
    compare_tensors(cpu_dA, download_to_host(gpu_dA), "snake_bwd_dAlpha",
                    kSnakeAtol, kSnakeRtol);
    if (has_beta) {
        compare_tensors(cpu_dB, download_to_host(gpu_dB), "snake_bwd_dBeta",
                        kSnakeAtol, kSnakeRtol);
    }
}

// ─── elu ───────────────────────────────────────────────────────────────────
void run_elu_fwd(int rows, int cols, float alpha, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, 2.0f);   // span both signs comfortably

    Tensor cpu_Y;
    brotensor::elu_forward(X, alpha, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::elu_forward(gX, alpha, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "elu_fwd", kAtol, kRtol);
}

void run_elu_bwd(int rows, int cols, float alpha, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    Tensor dY = Tensor::mat(rows, cols);
    fill_random(X, rng, 2.0f);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::elu_backward(X, dY, alpha, cpu_dX);

    Tensor gX = X.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::elu_backward(gX, gdY, alpha, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "elu_bwd", kAtol, kRtol);
}

// ─── leaky_relu ────────────────────────────────────────────────────────────
void run_leaky_fwd(int rows, int cols, float slope, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, 2.0f);

    Tensor cpu_Y;
    brotensor::leaky_relu_forward(X, slope, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::leaky_relu_forward(gX, slope, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "leaky_relu_fwd",
                    kAtol, kRtol);
}

// FP16 / BF16 forward (CUDA-only path; FP32 CPU reference over inputs
// rounded through the 16-bit type — leaky_relu keeps positive values
// bit-exactly, only the slope*x branch re-rounds on store).
void run_leaky_fwd_16(int rows, int cols, float slope, bool bf16,
                      uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, 2.0f);
    Tensor Xq = bf16 ? bf16_host_to_f32(to_bf16_host(X))
                     : fp16_host_to_f32(to_fp16_host(X));

    Tensor cpu_Y;
    brotensor::leaky_relu_forward(Xq, slope, cpu_Y);

    Tensor gX = bf16 ? to_bf16_gpu(X) : to_fp16_gpu(X);
    Tensor gpu_Y;
    brotensor::leaky_relu_forward(gX, slope, gpu_Y);
    BT_CHECK(gpu_Y.dtype == gX.dtype);

    Tensor host = download_to_host(gpu_Y);
    Tensor wide = bf16 ? bf16_host_to_f32(host) : fp16_host_to_f32(host);
    compare_tensors(cpu_Y, wide, bf16 ? "leaky_fwd_bf16" : "leaky_fwd_fp16",
                    bf16 ? 2e-2f : 2e-3f, bf16 ? 2e-2f : 2e-3f);
}

void run_leaky_bwd(int rows, int cols, float slope, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    Tensor dY = Tensor::mat(rows, cols);
    fill_random(X, rng, 2.0f);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::leaky_relu_backward(X, dY, slope, cpu_dX);

    Tensor gX = X.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::leaky_relu_backward(gX, gdY, slope, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "leaky_relu_bwd",
                    kAtol, kRtol);
}

} // namespace

// ─── snake_forward (plain + snakebeta) ─────────────────────────────────────
BT_PARITY_TEST(snake_fwd_plain_2x3x5)  { run_snake_fwd(2, 3, 5, false, 0x8000ull); }
BT_PARITY_TEST(snake_fwd_beta_2x3x5)   { run_snake_fwd(2, 3, 5, true,  0x8001ull); }
BT_PARITY_TEST(snake_fwd_plain_1x8x1)  { run_snake_fwd(1, 8, 1, false, 0x8002ull); }
BT_PARITY_TEST(snake_fwd_beta_4x6x17)  { run_snake_fwd(4, 6, 17, true, 0x8003ull); }

// ─── snake_backward (plain + snakebeta) ────────────────────────────────────
BT_PARITY_TEST(snake_bwd_plain_2x3x5)  { run_snake_bwd(2, 3, 5, false, 0x8010ull); }
BT_PARITY_TEST(snake_bwd_beta_2x3x5)   { run_snake_bwd(2, 3, 5, true,  0x8011ull); }
BT_PARITY_TEST(snake_bwd_beta_4x6x17)  { run_snake_bwd(4, 6, 17, true, 0x8012ull); }

// ─── elu ───────────────────────────────────────────────────────────────────
BT_PARITY_TEST(elu_fwd_a1)     { run_elu_fwd(8, 32, 1.0f, 0x8020ull); }
BT_PARITY_TEST(elu_fwd_a05)    { run_elu_fwd(5, 7,  0.5f, 0x8021ull); }
BT_PARITY_TEST(elu_fwd_wide)   { run_elu_fwd(3, 1153, 1.3f, 0x8022ull); }
BT_PARITY_TEST(elu_bwd_a1)     { run_elu_bwd(8, 32, 1.0f, 0x8023ull); }
BT_PARITY_TEST(elu_bwd_a05)    { run_elu_bwd(5, 7,  0.5f, 0x8024ull); }

// ─── leaky_relu ────────────────────────────────────────────────────────────
BT_PARITY_TEST(leaky_fwd_01)   { run_leaky_fwd(8, 32, 0.1f,  0x8030ull); }
BT_PARITY_TEST(leaky_fwd_001)  { run_leaky_fwd(5, 7,  0.01f, 0x8031ull); }
BT_PARITY_TEST(leaky_fwd_wide) { run_leaky_fwd(3, 1153, 0.2f, 0x8032ull); }
BT_PARITY_TEST(leaky_bwd_01)   { run_leaky_bwd(8, 32, 0.1f,  0x8033ull); }
BT_PARITY_TEST(leaky_bwd_001)  { run_leaky_bwd(5, 7,  0.01f, 0x8034ull); }
BT_PARITY_TEST(leaky_fwd_fp16)      { run_leaky_fwd_16(8, 32, 0.1f,  false, 0x8035ull); }
BT_PARITY_TEST(leaky_fwd_fp16_wide) { run_leaky_fwd_16(3, 1153, 0.2f, false, 0x8036ull); }
BT_PARITY_TEST(leaky_fwd_bf16)      { run_leaky_fwd_16(8, 32, 0.1f,  true,  0x8037ull); }

int main() { return run_all("vocoder cpu/gpu parity"); }
