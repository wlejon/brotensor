// CPU↔GPU parity tests for the codec quantization ops
// (brosoundml CHUNK 5, family D).
//
//   vq_encode_forward     — (N,1) INT32 indices + (N,D) FP32 quantized.
//   vq_encode_backward    — dX OVERWRITTEN (straight-through identity).
//   fsq_quantize_forward  — (N,D) FP32 quantized + (N,1) INT32 packed indices.
//   fsq_quantize_backward — dX OVERWRITTEN (straight-through identity).
//
// The INT32 outputs (codeword index / mixed-radix packed code) must match the
// CPU op *exactly* — both backends do the argmin / round in FP32 with the same
// operation order, so the discrete result is bit-identical, not just close.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstdio>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-4f;

// Exact element-wise comparison for INT32 index tensors. Both arguments must be
// host-resident (download_to_host the GPU side first).
void compare_int_tensors(const Tensor& cpu, const Tensor& gpu,
                         const char* tag) {
    if (cpu.rows != gpu.rows || cpu.cols != gpu.cols) {
        std::printf("    [%s] shape mismatch: cpu (%d,%d) vs gpu (%d,%d)\n",
                    tag, cpu.rows, cpu.cols, gpu.rows, gpu.cols);
        throw 0;
    }
    const auto* a = static_cast<const int32_t*>(cpu.host_raw());
    const auto* b = static_cast<const int32_t*>(gpu.host_raw());
    for (int i = 0; i < cpu.size(); ++i) {
        if (a[i] != b[i]) {
            std::printf("    [%s] mismatch at i=%d  cpu=%d gpu=%d\n",
                        tag, i, a[i], b[i]);
            throw 0;
        }
    }
}

// ─── vq_encode ───────────────────────────────────────────────────────────────
void run_vq_fwd(int N, int D, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::mat(N, D);
    Tensor codebook = Tensor::mat(K, D);
    fill_random(x, rng, 1.0f);
    fill_random(codebook, rng, 1.0f);

    Tensor cpu_idx, cpu_q;
    brotensor::vq_encode_forward(x, codebook, cpu_idx, cpu_q);

    Tensor gx = x.to(gpu_device()), gcb = codebook.to(gpu_device());
    Tensor gpu_idx, gpu_q;
    brotensor::vq_encode_forward(gx, gcb, gpu_idx, gpu_q);

    compare_int_tensors(cpu_idx, download_to_host(gpu_idx), "vq_fwd_indices");
    compare_tensors(cpu_q, download_to_host(gpu_q), "vq_fwd_quantized",
                    kAtol, kRtol);
}

void run_vq_bwd(int N, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dQ = Tensor::mat(N, D);
    fill_random(dQ, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::vq_encode_backward(dQ, cpu_dX);

    Tensor gdQ = dQ.to(gpu_device());
    Tensor gpu_dX;
    brotensor::vq_encode_backward(gdQ, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "vq_bwd", kAtol, kRtol);
}

// ─── fsq_quantize ────────────────────────────────────────────────────────────
// Per-coordinate level counts in [2, 9). dimension 0 is the least-significant
// mixed-radix digit.
Tensor make_levels(int D, SplitMix64& rng) {
    Tensor lv = Tensor::zeros_on(Device::CPU, D, 1, Dtype::INT32);
    auto* p = static_cast<int32_t*>(lv.host_raw_mut());
    for (int d = 0; d < D; ++d) {
        p[d] = 2 + static_cast<int>(rng.next_u64() % 7);   // [2, 8]
    }
    return lv;
}

void run_fsq_fwd(int N, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::mat(N, D);
    fill_random(x, rng, 1.2f);          // spans past [-1,1] so the clamp fires
    Tensor levels = make_levels(D, rng);

    Tensor cpu_q, cpu_packed;
    brotensor::fsq_quantize_forward(x, levels, cpu_q, cpu_packed);

    Tensor gx = x.to(gpu_device()), glv = levels.to(gpu_device());
    Tensor gpu_q, gpu_packed;
    brotensor::fsq_quantize_forward(gx, glv, gpu_q, gpu_packed);

    compare_tensors(cpu_q, download_to_host(gpu_q), "fsq_fwd_quantized",
                    kAtol, kRtol);
    compare_int_tensors(cpu_packed, download_to_host(gpu_packed),
                        "fsq_fwd_packed");
}

void run_fsq_bwd(int N, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dQ = Tensor::mat(N, D);
    fill_random(dQ, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::fsq_quantize_backward(dQ, cpu_dX);

    Tensor gdQ = dQ.to(gpu_device());
    Tensor gpu_dX;
    brotensor::fsq_quantize_backward(gdQ, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "fsq_bwd", kAtol, kRtol);
}

} // namespace

// ─── vq_encode ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(vq_fwd_8x4x16)   { run_vq_fwd(8, 4, 16, 0xB100ull); }
BT_PARITY_TEST(vq_fwd_5x8x3)    { run_vq_fwd(5, 8, 3, 0xB101ull); }
BT_PARITY_TEST(vq_fwd_wide)     { run_vq_fwd(33, 16, 256, 0xB102ull); }
BT_PARITY_TEST(vq_bwd_8x4)      { run_vq_bwd(8, 4, 0xB103ull); }
BT_PARITY_TEST(vq_bwd_5x17)     { run_vq_bwd(5, 17, 0xB104ull); }

// ─── fsq_quantize ────────────────────────────────────────────────────────────
BT_PARITY_TEST(fsq_fwd_8x4)     { run_fsq_fwd(8, 4, 0xB110ull); }
BT_PARITY_TEST(fsq_fwd_5x6)     { run_fsq_fwd(5, 6, 0xB111ull); }
BT_PARITY_TEST(fsq_fwd_wide)    { run_fsq_fwd(40, 9, 0xB112ull); }
BT_PARITY_TEST(fsq_bwd_8x4)     { run_fsq_bwd(8, 4, 0xB113ull); }
BT_PARITY_TEST(fsq_bwd_5x17)    { run_fsq_bwd(5, 17, 0xB114ull); }

int main() { return run_all("codec quantization cpu/gpu parity"); }
