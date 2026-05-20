// CPU<->GPU parity for attention_token_moments (CHUNK 5).
//
// DTYPE NOTE: the CUDA kernel reads an FP16 attention map and writes FP32
// mass/centroid. The CPU backend is FP32-only and reads an FP32 attention
// map. The parity harness:
//   * quantises the attention map through FP16 so CPU (FP32) and GPU (FP16)
//     start from the exact same input bit patterns,
//   * feeds an FP16 tensor to the GPU and an FP32 tensor to the CPU,
//   * compares the FP32 mass/centroid outputs with a loose tolerance — the
//     GPU's long FP16 reduction over up to ~1024 spatial tokens accumulates
//     noise relative to the CPU's FP32 accumulation.
//
// mass and centroid are OVERWRITTEN by the op (no accumulation).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

inline float q16(float v) {
    return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

// Build a CPU FP32 attention map (Lq, Lk) whose rows are softmax-like
// distributions (positive, row-normalised), FP16-quantised.
Tensor make_attn_cpu(int h_lat, int w_lat, int Lk, SplitMix64& rng) {
    const int Lq = h_lat * w_lat;
    Tensor t = Tensor::mat(Lq, Lk);
    for (int q = 0; q < Lq; ++q) {
        float s = 0.0f;
        for (int k = 0; k < Lk; ++k) {
            const float v = rng.next_f01() + 1e-3f;  // strictly positive
            t.ptr()[q * Lk + k] = v;
            s += v;
        }
        const float inv = 1.0f / s;
        for (int k = 0; k < Lk; ++k)
            t.ptr()[q * Lk + k] = q16(t.ptr()[q * Lk + k] * inv);
    }
    return t;
}

// Upload a CPU FP32 tensor as an FP16 CUDA tensor.
Tensor to_fp16_cuda(const Tensor& cpu) {
    const int n = cpu.size();
    std::vector<uint16_t> h(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h[i] = brotensor::fp32_to_fp16_bits(cpu[i]);
    return Tensor::from_host_fp16_on(gpu_device(), h.data(),
                                     cpu.rows, cpu.cols);
}

void run_moments(int h_lat, int w_lat, int Lk, uint64_t seed,
                 float mass_atol, float cent_atol) {
    SplitMix64 rng(seed);
    Tensor Attn = make_attn_cpu(h_lat, w_lat, Lk, rng);

    // CPU FP32 path.
    Tensor mass_c, centroid_c;
    brotensor::attention_token_moments(Attn, h_lat, w_lat, mass_c, centroid_c);

    // GPU FP16 path.
    Tensor gAttn = to_fp16_cuda(Attn);
    Tensor mass_g, centroid_g;
    brotensor::attention_token_moments(gAttn, h_lat, w_lat, mass_g, centroid_g);

    compare_tensors(mass_c, download_to_host(mass_g),
                    "moments.mass", mass_atol, 1e-2f);
    compare_tensors(centroid_c, download_to_host(centroid_g),
                    "moments.centroid", cent_atol, 1e-2f);
}

} // namespace

BT_PARITY_TEST(moments_4x4_Lk3)   { run_moments(4,  4,  3,  0x700ull, 1e-2f, 1e-2f); }
BT_PARITY_TEST(moments_8x8_Lk4)   { run_moments(8,  8,  4,  0x701ull, 2e-2f, 2e-2f); }
BT_PARITY_TEST(moments_8x16_Lk16) { run_moments(8,  16, 16, 0x702ull, 3e-2f, 3e-2f); }
BT_PARITY_TEST(moments_16x16_Lk8) { run_moments(16, 16, 8,  0x703ull, 5e-2f, 5e-2f); }
// SD-realistic: 32x32 latent, 77 text tokens. Long FP16 reductions.
BT_PARITY_TEST(moments_32x32_Lk77) {
    run_moments(32, 32, 77, 0x704ull, 2e-1f, 2e-1f);
}

int main() { return run_all("attention_moments cpu/gpu parity"); }
