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

// ─── rows_count_above ───────────────────────────────────────────────────────

namespace {

void check_counts_equal(const Tensor& a, const Tensor& b, const char* tag) {
    BT_CHECK(a.rows == b.rows && a.cols == b.cols);
    BT_CHECK(a.dtype == Dtype::INT32 && b.dtype == Dtype::INT32);
    const int32_t* ap = static_cast<const int32_t*>(a.data);
    const int32_t* bp = static_cast<const int32_t*>(b.data);
    for (int i = 0; i < a.size(); ++i) {
        if (ap[i] != bp[i]) {
            std::printf("    [%s] count mismatch at i=%d: %d vs %d\n",
                        tag, i, ap[i], bp[i]);
            throw 0;
        }
    }
}

// Random matrix: CPU and CUDA must agree exactly (integer counts).
void run_rows_count_above(int R, int C, float t_lo, float t_hi, bool fp16,
                          uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(R, C);
    fill_random(X, rng, 1.0f);

    Tensor cpu_counts;
    Tensor gX;
    if (fp16) {
        // Round the CPU input through FP16 so both sides see the same values.
        Tensor Xq = fp16_host_to_f32(to_fp16_host(X));
        brotensor::rows_count_above(Xq, t_lo, t_hi, cpu_counts);
        gX = to_fp16_gpu(X);
    } else {
        brotensor::rows_count_above(X, t_lo, t_hi, cpu_counts);
        gX = X.to(gpu_device());
    }
    BT_CHECK(cpu_counts.rows == R && cpu_counts.cols == 2);

    Tensor gpu_counts;
    brotensor::rows_count_above(gX, t_lo, t_hi, gpu_counts);
    check_counts_equal(cpu_counts, download_to_host(gpu_counts),
                       "rows_count_above");
}

// Handwritten matrix with values exactly at both thresholds: strict > means
// the ties are NOT counted. Expected counts computed by hand.
void run_rows_count_above_handwritten() {
    const float t_lo = 0.0f, t_hi = 0.5f;
    // Row 0: {-1, 0, 0.25, 0.5, 0.75}  -> n_lo = 3 (0 is a tie), n_hi = 1
    //                                     (0.5 is a tie).
    // Row 1: {0, 0, 0, 0, 0}           -> n_lo = 0, n_hi = 0.
    // Row 2: {0.5, 0.5, 1, 2, -0.5}    -> n_lo = 4, n_hi = 2.
    const float vals[3][5] = {
        {-1.0f, 0.0f, 0.25f, 0.5f, 0.75f},
        { 0.0f, 0.0f, 0.0f,  0.0f, 0.0f},
        { 0.5f, 0.5f, 1.0f,  2.0f, -0.5f},
    };
    const int32_t expect[3][2] = {{3, 1}, {0, 0}, {4, 2}};

    Tensor X = Tensor::mat(3, 5);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 5; ++c) X.host_f32_mut()[r * 5 + c] = vals[r][c];

    // CPU.
    Tensor cpu_counts;
    brotensor::rows_count_above(X, t_lo, t_hi, cpu_counts);
    const int32_t* cp = static_cast<const int32_t*>(cpu_counts.data);
    for (int r = 0; r < 3; ++r) {
        BT_CHECK(cp[2 * r + 0] == expect[r][0]);
        BT_CHECK(cp[2 * r + 1] == expect[r][1]);
    }

    // GPU — FP32 and FP16 (the handwritten values are FP16-exact).
    for (int pass = 0; pass < 2; ++pass) {
        Tensor gX = (pass == 0) ? X.to(gpu_device()) : to_fp16_gpu(X);
        Tensor gpu_counts;
        brotensor::rows_count_above(gX, t_lo, t_hi, gpu_counts);
        Tensor host = download_to_host(gpu_counts);
        const int32_t* gp = static_cast<const int32_t*>(host.data);
        for (int r = 0; r < 3; ++r) {
            BT_CHECK(gp[2 * r + 0] == expect[r][0]);
            BT_CHECK(gp[2 * r + 1] == expect[r][1]);
        }
    }
}

} // namespace (rows_count_above)

BT_PARITY_TEST(rca_small)        { run_rows_count_above(4, 7, -0.2f, 0.3f, false, 0xCA00ull); }
BT_PARITY_TEST(rca_wide)         { run_rows_count_above(3, 1031, -0.5f, 0.5f, false, 0xCA01ull); }
BT_PARITY_TEST(rca_one_row)      { run_rows_count_above(1, 300, 0.0f, 0.9f, false, 0xCA02ull); }
BT_PARITY_TEST(rca_fp16)         { run_rows_count_above(5, 513, -0.1f, 0.4f, true, 0xCA03ull); }
BT_PARITY_TEST(rca_handwritten)  { run_rows_count_above_handwritten(); }

int main() { return run_all("reduce-extra cpu/gpu parity"); }
