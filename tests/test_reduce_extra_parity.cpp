// CPU↔GPU parity tests for the newly-CPU-ported public reductions:
//   sum_rows / sum_cols / argmax_rows.
//
// CHUNK 1. test_reduce_parity.cpp covers masked_mean_pool; this file covers
// the public reduction ops.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

void run_sum_rows(int M, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(M, N);
    fill_random(X, rng);

    Tensor y_cpu;
    brotensor::sum_rows(X, y_cpu);

    Tensor gX = X.to(gpu_device());
    Tensor gy;
    brotensor::sum_rows(gX, gy);
    Tensor y_gpu = download_to_host(gy);
    compare_tensors(y_cpu, y_gpu, "sum_rows");
}

void run_sum_cols(int M, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(M, N);
    fill_random(X, rng);

    Tensor y_cpu;
    brotensor::sum_cols(X, y_cpu);

    Tensor gX = X.to(gpu_device());
    Tensor gy;
    brotensor::sum_cols(gX, gy);
    Tensor y_gpu = download_to_host(gy);
    compare_tensors(y_cpu, y_gpu, "sum_cols");
}

void run_argmax(int M, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(M, N);
    fill_random(X, rng);

    Tensor idx_cpu;
    brotensor::argmax_rows(X, idx_cpu);

    Tensor gX = X.to(gpu_device());
    Tensor gidx;
    brotensor::argmax_rows(gX, gidx);
    Tensor idx_gpu = download_to_host(gidx);
    // Indices are exact integers stored as FP32 — require bit-exact match.
    compare_tensors(idx_cpu, idx_gpu, "argmax_rows", 0.0f, 0.0f);
}

} // namespace

BT_PARITY_TEST(sum_rows_1x1)    { run_sum_rows(1, 1, 0xB00ull); }
BT_PARITY_TEST(sum_rows_8x32)   { run_sum_rows(8, 32, 0xB01ull); }
BT_PARITY_TEST(sum_rows_3x257)  { run_sum_rows(3, 257, 0xB02ull); }

BT_PARITY_TEST(sum_cols_1x1)    { run_sum_cols(1, 1, 0xB10ull); }
BT_PARITY_TEST(sum_cols_8x32)   { run_sum_cols(8, 32, 0xB11ull); }
BT_PARITY_TEST(sum_cols_257x3)  { run_sum_cols(257, 3, 0xB12ull); }

BT_PARITY_TEST(argmax_1x1)      { run_argmax(1, 1, 0xB20ull); }
BT_PARITY_TEST(argmax_8x32)     { run_argmax(8, 32, 0xB21ull); }
BT_PARITY_TEST(argmax_5x300)    { run_argmax(5, 300, 0xB22ull); }

// ─── BF16 parity: BF16-on-CUDA vs FP32 CPU reference ─────────────────────

namespace {

void run_sum_rows_bf16(int M, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_f32 = Tensor::mat(M, N);
    fill_random(X_f32, rng);

    // FP32 CPU reference.
    Tensor y_cpu;
    brotensor::sum_rows(X_f32, y_cpu);

    // BF16 GPU.
    Tensor gX = to_bf16_gpu(X_f32);
    Tensor gy;
    brotensor::sum_rows(gX, gy);
    BT_CHECK(gy.dtype == Dtype::BF16);
    Tensor y_gpu = bf16_host_to_f32(download_to_host(gy));
    // Long rows accumulate many BF16 additions — use looser tol for N>32.
    const float tol = (N > 32) ? 6e-2f : 3e-2f;
    compare_tensors(y_cpu, y_gpu, "sum_rows_bf16", tol, tol);
}

void run_sum_cols_bf16(int M, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_f32 = Tensor::mat(M, N);
    fill_random(X_f32, rng);

    Tensor y_cpu;
    brotensor::sum_cols(X_f32, y_cpu);

    Tensor gX = to_bf16_gpu(X_f32);
    Tensor gy;
    brotensor::sum_cols(gX, gy);
    BT_CHECK(gy.dtype == Dtype::BF16);
    Tensor y_gpu = bf16_host_to_f32(download_to_host(gy));
    const float tol = (M > 32) ? 6e-2f : 3e-2f;
    compare_tensors(y_cpu, y_gpu, "sum_cols_bf16", tol, tol);
}

void run_argmax_bf16(int M, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_f32 = Tensor::mat(M, N);
    fill_random(X_f32, rng);
    // Force unique max per row to avoid ties broken differently by BF16 rounding.
    for (int m = 0; m < M; ++m)
        X_f32[static_cast<size_t>(m) * N + (m % N)] = 5.0f + static_cast<float>(m);

    Tensor idx_cpu;
    brotensor::argmax_rows(X_f32, idx_cpu);

    Tensor gX = to_bf16_gpu(X_f32);
    Tensor gidx;
    brotensor::argmax_rows(gX, gidx);
    BT_CHECK(gidx.dtype == Dtype::FP32);
    Tensor idx_gpu = download_to_host(gidx);
    // Argmax indices are exact integers stored as FP32.
    compare_tensors(idx_cpu, idx_gpu, "argmax_rows_bf16", 0.0f, 0.0f);
}

} // namespace (inner)

BT_PARITY_TEST(sum_rows_bf16_1x1)    { run_sum_rows_bf16(1, 1,   0xBF10ull); }
BT_PARITY_TEST(sum_rows_bf16_8x32)   { run_sum_rows_bf16(8, 32,  0xBF11ull); }
BT_PARITY_TEST(sum_rows_bf16_3x257)  { run_sum_rows_bf16(3, 257, 0xBF12ull); }

BT_PARITY_TEST(sum_cols_bf16_1x1)    { run_sum_cols_bf16(1,  1,   0xBF20ull); }
BT_PARITY_TEST(sum_cols_bf16_8x32)   { run_sum_cols_bf16(8,  32,  0xBF21ull); }
BT_PARITY_TEST(sum_cols_bf16_257x3)  { run_sum_cols_bf16(257, 3,  0xBF22ull); }

BT_PARITY_TEST(argmax_bf16_1x1)      { run_argmax_bf16(1, 1,   0xBF30ull); }
BT_PARITY_TEST(argmax_bf16_8x32)     { run_argmax_bf16(8, 32,  0xBF31ull); }
BT_PARITY_TEST(argmax_bf16_5x300)    { run_argmax_bf16(5, 300, 0xBF32ull); }

int main() { return run_all("reduce-extra cpu/gpu parity"); }
