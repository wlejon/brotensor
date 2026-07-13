// CPU↔GPU parity tests for brotensor::pixel_shuffle_upsample_2x_forward.
//
// DC-AE up-shortcut: channel repeat_interleave fused with a 2x pixel-shuffle,
// as a single gather:
//     Y[n, c_out, 2h+i, 2w+j] = X[n, (4*c_out + 2i + j) / repeats, h, w]
//     repeats = 4*C_out / C_in
// The three regimes worth covering are repeats == 1 (plain pixel-shuffle),
// repeats == 4 (degenerates to a 2x nearest upsample), and the general case in
// between (repeats == 2).
//
// It is a pure gather, so FP16/BF16 parity is exact once widened — no
// arithmetic happens, only a copy. The reduced-dtype cases still get a small
// tolerance rather than an exact compare, since the *inputs* are rounded on
// the way in.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

void run_fp32(int N, int C_in, int H, int W, int C_out, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C_in * H * W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::pixel_shuffle_upsample_2x_forward(gX, N, C_in, H, W, C_out, gpu_Y);

    // Pure gather — the GPU must reproduce the CPU result bit-for-bit.
    compare_tensors(cpu_Y, download_to_host(gpu_Y), "pixel_shuffle_fp32",
                    0.0f, 0.0f);
}

void run_fp16(int N, int C_in, int H, int W, int C_out, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C_in * H * W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, cpu_Y);

    Tensor gX = to_fp16_gpu(X);
    Tensor gpu_Y;
    brotensor::pixel_shuffle_upsample_2x_forward(gX, N, C_in, H, W, C_out, gpu_Y);

    Tensor widened = fp16_host_to_f32(download_to_host(gpu_Y));
    compare_tensors(cpu_Y, widened, "pixel_shuffle_fp16", 1e-2f, 1e-2f);
}

void run_bf16(int N, int C_in, int H, int W, int C_out, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C_in * H * W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, cpu_Y);

    Tensor gX = to_bf16_gpu(X);
    Tensor gpu_Y;
    brotensor::pixel_shuffle_upsample_2x_forward(gX, N, C_in, H, W, C_out, gpu_Y);

    Tensor widened = bf16_host_to_f32(download_to_host(gpu_Y));
    compare_tensors(cpu_Y, widened, "pixel_shuffle_bf16", 3e-2f, 3e-2f);
}

} // namespace

// ── FP32, repeats == 2 (the general case: C_in = 2*C_out) ──
BT_PARITY_TEST(pixel_shuffle_fp32_r2_small)   { run_fp32(1,  8,  3,  5,  4,  0x9301ull); }
BT_PARITY_TEST(pixel_shuffle_fp32_r2_batched) { run_fp32(3,  16, 8,  8,  8,  0x9302ull); }
BT_PARITY_TEST(pixel_shuffle_fp32_r2_wide)    { run_fp32(1,  64, 16, 16, 32, 0x9303ull); }

// ── FP32, repeats == 1 (C_in == 4*C_out: plain pixel-shuffle) ──
BT_PARITY_TEST(pixel_shuffle_fp32_r1_small)   { run_fp32(1,  8,  2,  2,  2,  0x9310ull); }
BT_PARITY_TEST(pixel_shuffle_fp32_r1_batched) { run_fp32(2,  32, 7,  5,  8,  0x9311ull); }
BT_PARITY_TEST(pixel_shuffle_fp32_r1_wide)    { run_fp32(1, 128, 16, 16, 32, 0x9312ull); }

// ── FP32, repeats == 4 (C_in == C_out: 2x nearest upsample) ──
BT_PARITY_TEST(pixel_shuffle_fp32_r4_small)   { run_fp32(2,  3,  4,  3,  3,  0x9320ull); }
BT_PARITY_TEST(pixel_shuffle_fp32_r4_batched) { run_fp32(4,  16, 8,  8,  16, 0x9321ull); }
BT_PARITY_TEST(pixel_shuffle_fp32_r4_wide)    { run_fp32(1,  64, 32, 32, 64, 0x9322ull); }

// ── FP32 edge shapes ──
BT_PARITY_TEST(pixel_shuffle_fp32_1x1_spatial){ run_fp32(1,  4,  1,  1,  1,  0x9330ull); }  // H=W=1
BT_PARITY_TEST(pixel_shuffle_fp32_tall)       { run_fp32(1,  4,  17, 1,  1,  0x9331ull); }  // W=1
BT_PARITY_TEST(pixel_shuffle_fp32_wide_hw)    { run_fp32(1,  4,  1,  17, 1,  0x9332ull); }  // H=1

// ── FP16 ── one shape per repeats regime.
BT_PARITY_TEST(pixel_shuffle_fp16_r2)         { run_fp16(2,  16, 8,  8,  8,  0x9340ull); }
BT_PARITY_TEST(pixel_shuffle_fp16_r1)         { run_fp16(2,  32, 7,  5,  8,  0x9341ull); }
BT_PARITY_TEST(pixel_shuffle_fp16_r4)         { run_fp16(2,  16, 8,  8,  16, 0x9342ull); }

// ── BF16 ── one shape per repeats regime.
BT_PARITY_TEST(pixel_shuffle_bf16_r2)         { run_bf16(2,  16, 8,  8,  8,  0x9350ull); }
BT_PARITY_TEST(pixel_shuffle_bf16_r1)         { run_bf16(2,  32, 7,  5,  8,  0x9351ull); }
BT_PARITY_TEST(pixel_shuffle_bf16_r4)         { run_bf16(2,  16, 8,  8,  16, 0x9352ull); }

int main() { return run_all("pixel_shuffle_upsample cpu/gpu parity"); }
