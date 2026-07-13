// CPU↔GPU parity tests for brotensor::softmax_rows_forward.
//
// Row-wise softmax over a flat buffer of `rows` independent rows of `cols`
// each — one launch covers every row. The GPU kernel is a block-per-row
// reduction (shared-memory max, then sum), so the shapes below deliberately
// straddle the block width: fewer columns than threads, exactly one block's
// worth, and several strided passes per thread.
//
// FP32 on both sides; the GPU additionally carries FP16/BF16 storage paths,
// compared against the FP32 CPU reference with the usual widened tolerance.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

void run_fp32(int rows, int cols, uint64_t seed, float scale = 3.0f) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, scale);

    Tensor cpu_Y;
    brotensor::softmax_rows_forward(X, cpu_Y, rows, cols);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::softmax_rows_forward(gX, gpu_Y, rows, cols);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "softmax_rows_fp32");
}

void run_fp16(int rows, int cols, uint64_t seed, float scale = 3.0f) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, scale);

    Tensor cpu_Y;
    brotensor::softmax_rows_forward(X, cpu_Y, rows, cols);

    Tensor gX = to_fp16_gpu(X);
    Tensor gpu_Y;
    brotensor::softmax_rows_forward(gX, gpu_Y, rows, cols);

    Tensor widened = fp16_host_to_f32(download_to_host(gpu_Y));
    // Probabilities live in [0,1]; FP16's ~11-bit mantissa over an exp + a
    // length-`cols` sum reduction lands well inside 1e-2.
    compare_tensors(cpu_Y, widened, "softmax_rows_fp16", 1e-2f, 1e-2f);
}

void run_bf16(int rows, int cols, uint64_t seed, float scale = 3.0f) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, scale);

    Tensor cpu_Y;
    brotensor::softmax_rows_forward(X, cpu_Y, rows, cols);

    Tensor gX = to_bf16_gpu(X);
    Tensor gpu_Y;
    brotensor::softmax_rows_forward(gX, gpu_Y, rows, cols);

    Tensor widened = bf16_host_to_f32(download_to_host(gpu_Y));
    // BF16 keeps only 8 mantissa bits — rounding the *logits* moves the
    // softmax itself, so this needs the looser reduction tolerance.
    compare_tensors(cpu_Y, widened, "softmax_rows_bf16", 3e-2f, 3e-2f);
}

// In-place (Y aliases X) must agree with the out-of-place GPU result — the
// header permits Y == X and the kernel reads the whole row before writing it.
void run_in_place(int rows, int cols, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(rows, cols);
    fill_random(X, rng, 3.0f);

    Tensor cpu_Y;
    brotensor::softmax_rows_forward(X, cpu_Y, rows, cols);

    Tensor gX = X.to(gpu_device());
    brotensor::softmax_rows_forward(gX, gX, rows, cols);   // in-place on the GPU

    compare_tensors(cpu_Y, download_to_host(gX), "softmax_rows_inplace");
}

} // namespace

// ── FP32 ── shapes straddle the kernel's block width (256 threads/row).
BT_PARITY_TEST(softmax_rows_fp32_1x1)        { run_fp32(1,   1,    0x9101ull); }
BT_PARITY_TEST(softmax_rows_fp32_1x16)       { run_fp32(1,   16,   0x9102ull); }
BT_PARITY_TEST(softmax_rows_fp32_8x13)       { run_fp32(8,   13,   0x9103ull); }
BT_PARITY_TEST(softmax_rows_fp32_32x64)      { run_fp32(32,  64,   0x9104ull); }
BT_PARITY_TEST(softmax_rows_fp32_4x256)      { run_fp32(4,   256,  0x9105ull); }  // exactly one block
BT_PARITY_TEST(softmax_rows_fp32_4x257)      { run_fp32(4,   257,  0x9106ull); }  // one past
BT_PARITY_TEST(softmax_rows_fp32_3x1024)     { run_fp32(3,   1024, 0x9107ull); }  // strided passes
BT_PARITY_TEST(softmax_rows_fp32_128x33)     { run_fp32(128, 33,   0x9108ull); }  // many rows
BT_PARITY_TEST(softmax_rows_fp32_wide_range) { run_fp32(6,   48,   0x9109ull, 20.0f); }  // big logits

// ── in-place ──
BT_PARITY_TEST(softmax_rows_inplace_8x64)    { run_in_place(8, 64,  0x9110ull); }
BT_PARITY_TEST(softmax_rows_inplace_2x300)   { run_in_place(2, 300, 0x9111ull); }

// ── FP16 ──
BT_PARITY_TEST(softmax_rows_fp16_8x13)       { run_fp16(8,  13,   0x9120ull); }
BT_PARITY_TEST(softmax_rows_fp16_4x256)      { run_fp16(4,  256,  0x9121ull); }
BT_PARITY_TEST(softmax_rows_fp16_3x1024)     { run_fp16(3,  1024, 0x9122ull); }

// ── BF16 ──
BT_PARITY_TEST(softmax_rows_bf16_8x13)       { run_bf16(8,  13,   0x9130ull); }
BT_PARITY_TEST(softmax_rows_bf16_4x256)      { run_bf16(4,  256,  0x9131ull); }
BT_PARITY_TEST(softmax_rows_bf16_3x1024)     { run_bf16(3,  1024, 0x9132ull); }

int main() { return run_all("softmax_rows cpu/gpu parity"); }
