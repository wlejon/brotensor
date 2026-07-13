// CPU↔GPU parity tests for brotensor::add_channel_bias_inplace.
//
// Channel-broadcast bias add over a channel-major (C, L) buffer:
//     y[c*L + i] += bias[c]
// In-place on y, so each case snapshots the original buffer and feeds the same
// starting values to both backends.
//
// FP32 on both sides; the GPU additionally carries FP16/BF16 storage paths
// (both y and bias in the reduced dtype), compared against the FP32 CPU
// reference with the usual widened tolerance.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

void run_fp32(int C, int L, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::mat(1, C * L);
    fill_random(y, rng);
    Tensor bias = Tensor::mat(1, C);
    fill_random(bias, rng);

    // Snapshot before the CPU op mutates y — the GPU side needs the same input.
    Tensor y0 = y.clone();

    brotensor::add_channel_bias_inplace(y, bias, C, L);       // CPU, in-place

    Tensor gy = y0.to(gpu_device());
    Tensor gbias = bias.to(gpu_device());
    brotensor::add_channel_bias_inplace(gy, gbias, C, L);     // GPU, in-place

    compare_tensors(y, download_to_host(gy), "add_channel_bias_fp32");
}

void run_fp16(int C, int L, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::mat(1, C * L);
    fill_random(y, rng);
    Tensor bias = Tensor::mat(1, C);
    fill_random(bias, rng);

    Tensor y0 = y.clone();
    brotensor::add_channel_bias_inplace(y, bias, C, L);       // FP32 CPU reference

    Tensor gy = to_fp16_gpu(y0);
    Tensor gbias = to_fp16_gpu(bias);
    brotensor::add_channel_bias_inplace(gy, gbias, C, L);

    Tensor widened = fp16_host_to_f32(download_to_host(gy));
    // One add of two FP16-rounded operands — error is dominated by the input
    // rounding, not the arithmetic.
    compare_tensors(y, widened, "add_channel_bias_fp16", 1e-2f, 1e-2f);
}

void run_bf16(int C, int L, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::mat(1, C * L);
    fill_random(y, rng);
    Tensor bias = Tensor::mat(1, C);
    fill_random(bias, rng);

    Tensor y0 = y.clone();
    brotensor::add_channel_bias_inplace(y, bias, C, L);       // FP32 CPU reference

    Tensor gy = to_bf16_gpu(y0);
    Tensor gbias = to_bf16_gpu(bias);
    brotensor::add_channel_bias_inplace(gy, gbias, C, L);

    Tensor widened = bf16_host_to_f32(download_to_host(gy));
    // BF16 keeps 8 mantissa bits; the result is stored back in BF16 too.
    compare_tensors(y, widened, "add_channel_bias_bf16", 3e-2f, 3e-2f);
}

} // namespace

// ── FP32 ── C/L combinations straddling the grid-stride loop's block width.
BT_PARITY_TEST(add_channel_bias_fp32_1x1)      { run_fp32(1,    1,    0x9201ull); }
BT_PARITY_TEST(add_channel_bias_fp32_1x64)     { run_fp32(1,    64,   0x9202ull); }  // C == 1: one scalar
BT_PARITY_TEST(add_channel_bias_fp32_64x1)     { run_fp32(64,   1,    0x9203ull); }  // L == 1: plain vector add
BT_PARITY_TEST(add_channel_bias_fp32_4x6)      { run_fp32(4,    6,    0x9204ull); }
BT_PARITY_TEST(add_channel_bias_fp32_16x256)   { run_fp32(16,   256,  0x9205ull); }
BT_PARITY_TEST(add_channel_bias_fp32_3x257)    { run_fp32(3,    257,  0x9206ull); }  // L not a block multiple
BT_PARITY_TEST(add_channel_bias_fp32_128x64)   { run_fp32(128,  64,   0x9207ull); }
BT_PARITY_TEST(add_channel_bias_fp32_512x37)   { run_fp32(512,  37,   0x9208ull); }  // many channels, odd L

// ── FP16 ──
BT_PARITY_TEST(add_channel_bias_fp16_4x6)      { run_fp16(4,    6,    0x9210ull); }
BT_PARITY_TEST(add_channel_bias_fp16_16x256)   { run_fp16(16,   256,  0x9211ull); }
BT_PARITY_TEST(add_channel_bias_fp16_128x64)   { run_fp16(128,  64,   0x9212ull); }

// ── BF16 ──
BT_PARITY_TEST(add_channel_bias_bf16_4x6)      { run_bf16(4,    6,    0x9220ull); }
BT_PARITY_TEST(add_channel_bias_bf16_16x256)   { run_bf16(16,   256,  0x9221ull); }
BT_PARITY_TEST(add_channel_bias_bf16_128x64)   { run_bf16(128,  64,   0x9222ull); }

int main() { return run_all("add_channel_bias cpu/gpu parity"); }
