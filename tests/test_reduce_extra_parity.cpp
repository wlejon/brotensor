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

int main() { return run_all("reduce-extra cpu/gpu parity"); }
