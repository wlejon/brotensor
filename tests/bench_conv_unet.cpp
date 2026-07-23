// Micro-benchmark: conv2d forward at UNet-ladder shapes, against the GEMM
// roofline for the same arithmetic.
//
// NOT registered with ctest — invoke manually:
//   ./build/tests/Release/brotensor_bench_conv_unet
//
// An implicit-GEMM convolution is a matrix product of shape
//   M = N*H_out*W_out,  N_gemm = C_out,  K = C_in*kH*kW
// so every row here also times `matmul` at exactly that M/N/K. The two do the
// same number of FLOPs; the ratio is what the convolution's addressing,
// tiling, and occupancy cost over a plain product. That ratio, not the
// absolute TFLOP/s, is the number to optimise against — it is invariant to
// clock state, and it says how much is left on the table.
//
// Shapes are the diffusion-UNet ladders that actually run: terrain-diffusion's
// decoder (batch 1, 512^2 down to 64^2) and latent base net (batch 16, 64^2
// down to 8^2), plus a couple of SD1.5 rungs so a regression there shows up
// here too.
//
// Each row is also correctness-gated: the FP16 result is compared against the
// FP32 path (which the dispatcher routes to the naive direct-conv kernel), so
// a fast-but-wrong kernel cannot post a good number.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

std::vector<float> rand_vec(std::size_t n, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

Tensor to_fp16(const std::vector<float>& h, int rows, int cols, Device dev) {
    std::vector<std::uint16_t> bits(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) bits[i] = brotensor::fp32_to_fp16_bits(h[i]);
    return Tensor::from_host_fp16_on(dev, bits.data(), rows, cols);
}

struct Shape {
    const char* name;
    int N, C_in, H, W, C_out, k;
};

// Relative error of the FP16 conv against the FP32 conv, normalised by the
// FP32 result's RMS. FP16 accumulation into FP32 over K up to ~7k gives a few
// times 1e-3 relative; the gate is loose enough for that and tight enough to
// catch an indexing bug.
double rel_error(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(b[i]) * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

void run(const Shape& s, std::mt19937& rng, Device dev, int& failures) {
    const int pad   = s.k / 2;
    const int H_out = s.H + 2 * pad - s.k + 1;
    const int W_out = s.W + 2 * pad - s.k + 1;

    const std::size_t nx = static_cast<std::size_t>(s.N) * s.C_in * s.H * s.W;
    const std::size_t nw = static_cast<std::size_t>(s.C_out) * s.C_in * s.k * s.k;
    const std::vector<float> hx = rand_vec(nx, rng, 1.0f);
    const std::vector<float> hw = rand_vec(nw, rng, 0.1f);

    Tensor X16 = to_fp16(hx, s.N, s.C_in * s.H * s.W, dev);
    Tensor W16 = to_fp16(hw, s.C_out, s.C_in * s.k * s.k, dev);
    Tensor X32 = Tensor::from_host_on(dev, hx.data(), s.N, s.C_in * s.H * s.W);
    Tensor W32 = Tensor::from_host_on(dev, hw.data(), s.C_out, s.C_in * s.k * s.k);

    Tensor Y16, Y32;
    auto conv16 = [&] {
        brotensor::conv2d_forward(X16, W16, nullptr, s.N, s.C_in, s.H, s.W,
                                  s.C_out, s.k, s.k, 1, 1, pad, pad, 1, 1, 1, Y16);
    };
    conv16();
    brotensor::conv2d_forward(X32, W32, nullptr, s.N, s.C_in, s.H, s.W,
                              s.C_out, s.k, s.k, 1, 1, pad, pad, 1, 1, 1, Y32);
    brotensor::sync_all();
    std::vector<float> y16f;
    {
        const std::vector<std::uint16_t> bits = Y16.to_host_vector_fp16();
        y16f.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            y16f[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
    }
    const double err = rel_error(y16f, Y32.to_host_vector());

    // The equivalent GEMM: (M, K) @ (K, N_gemm).
    const int M = s.N * H_out * W_out;
    const int K = s.C_in * s.k * s.k;
    Tensor A = to_fp16(rand_vec(static_cast<std::size_t>(M) * K, rng, 0.1f), M, K, dev);
    Tensor B = to_fp16(rand_vec(static_cast<std::size_t>(K) * s.C_out, rng, 0.1f), K, s.C_out, dev);
    Tensor C;

    const double gflop = 2.0 * M * static_cast<double>(s.C_out) * K / 1e9;
    const float conv_ms = bt_bench::time_min_ms(conv16);
    const float gemm_ms = bt_bench::time_min_ms([&] { brotensor::matmul(A, B, C); });

    const bool bad = !(err < 5e-3);
    if (bad) ++failures;
    // GFLOP per millisecond is TFLOP/s exactly (1e9 flop / 1e-3 s = 1e12 flop/s).
    std::printf("  %-18s %3d %5d %4d %5d %2d | %8d %6d %6d | %8.3f %7.1f | "
                "%8.3f %7.1f | %5.2fx  %8.1e%s\n",
                s.name, s.N, s.C_in, s.H, s.C_out, s.k,
                M, s.C_out, K,
                conv_ms, gflop / conv_ms,
                gemm_ms, gflop / gemm_ms,
                conv_ms / gemm_ms, err, bad ? "  <-- FAIL" : "");
}

}  // namespace

int main() {
    brotensor::init();
    const Device dev = brotensor::default_device();
    if (dev == Device::CPU) {
        std::printf("bench_conv_unet: no GPU backend, nothing to measure\n");
        return 0;
    }
    bt_bench::spin_up();

    const Shape shapes[] = {
        // terrain-diffusion decoder — batch 1, 64ch at 512^2 down to 256ch at 64^2.
        {"terrain dec L0",   1,  64, 512, 512,  64, 3},
        {"terrain dec L1",   1, 128, 256, 256, 128, 3},
        {"terrain dec L2",   1, 192, 128, 128, 192, 3},
        {"terrain dec L3",   1, 256,  64,  64, 256, 3},
        // terrain-diffusion latent base net — batch 16, 192ch at 64^2 to 768ch at 8^2.
        {"terrain base L0", 16, 192,  64,  64, 192, 3},
        {"terrain base L1", 16, 384,  32,  32, 384, 3},
        {"terrain base L2", 16, 576,  16,  16, 576, 3},
        {"terrain base L3", 16, 768,   8,   8, 768, 3},
        // 1x1 projections (attention qkv / skip convs).
        {"terrain dec skip", 1, 128, 256, 256, 128, 1},
        {"terrain base qkv",16, 768,   8,   8, 768, 1},
        // SD1.5 UNet rungs, so a regression there is visible here.
        {"sd15 320@64",      2, 320,  64,  64, 320, 3},
        {"sd15 640@32",      2, 640,  32,  32, 640, 3},
        {"sd15 1280@16",     2,1280,  16,  16,1280, 3},
    };

    std::printf("conv2d forward vs the GEMM of identical M/N/K (FP16)\n");
    std::printf("  %-18s %3s %5s %4s %5s %2s | %8s %6s %6s | %8s %7s | %8s %7s | %6s %9s\n",
                "shape", "N", "C_in", "H", "C_out", "k",
                "M", "N_gemm", "K", "conv ms", "TFLOP/s", "gemm ms", "TFLOP/s",
                "ratio", "rel err");

    std::mt19937 rng(20260723);
    int failures = 0;
    for (const Shape& s : shapes) run(s, rng, dev, failures);

    if (failures) {
        std::printf("\n%d shape(s) exceeded the correctness bar\n", failures);
        return 1;
    }
    std::printf("\nall shapes within the correctness bar\n");
    return 0;
}
