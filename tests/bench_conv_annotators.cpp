// Micro-benchmark: conv2d / conv_transpose2d at the brovisionml annotator
// shapes (HED / lineart / openpose / DSINE) — FP32 vs FP16 per shape.
//
// NOT registered with ctest — invoke manually:
//   ./build/tests/Release/brotensor_bench_conv_annotators
//
// Each row times the public op (so it measures whatever the dispatcher picks:
// naive direct conv for FP32, WMMA implicit-GEMM for FP16 when the shape is
// on the fast path). Timing uses cudaEvent_t around the full op with warmup.
// Each row also runs a finite/bounded spot check on the output so a
// fast-but-wrong kernel can't pass silently.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

constexpr int WARMUP = 3;
constexpr int ITERS  = 20;

std::vector<float> rand_vec(size_t n, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

Tensor upload(const std::vector<float>& v, int rows, int cols, Dtype dt) {
    if (dt == Dtype::FP32) {
        return Tensor::from_host_on(Device::CUDA, v.data(), rows, cols);
    }
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return Tensor::from_host_fp16_on(Device::CUDA, h.data(), rows, cols);
}

float time_loop_ms(const std::function<void()>& body) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < WARMUP; ++i) body();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < ITERS; ++i) body();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / ITERS;
}

// Spot check that the op produced finite, bounded output. Element-by-element
// parity belongs in the dedicated tests; this is the garbage gate.
bool sanity_finite(const Tensor& g, float abs_clip = 1e4f) {
    Tensor h = g.to(Device::CPU);
    const int n = h.size();
    const int step = n > 64 ? n / 64 : 1;
    for (int i = 0; i < n; i += step) {
        float f;
        if (h.dtype == Dtype::FP16) {
            f = brotensor::fp16_bits_to_fp32(h.host_fp16()[i]);
        } else {
            f = h.host_f32()[i];
        }
        if (!std::isfinite(f) || std::fabs(f) > abs_clip) return false;
    }
    return true;
}

void bench_conv2d(const char* tag,
                  int N, int C_in, int H, int W,
                  int C_out, int kH, int kW,
                  int stride, int pad, Dtype dt) {
    std::mt19937 rng(42);
    Tensor X  = upload(rand_vec(static_cast<size_t>(N) * C_in * H * W, rng, 0.5f),
                       N, C_in * H * W, dt);
    Tensor Wt = upload(rand_vec(static_cast<size_t>(C_out) * C_in * kH * kW, rng, 0.05f),
                       C_out, C_in * kH * kW, dt);
    Tensor B  = upload(rand_vec(static_cast<size_t>(C_out), rng, 0.1f), C_out, 1, dt);
    Tensor Y;

    auto body = [&] {
        brotensor::conv2d_forward(X, Wt, &B, N, C_in, H, W, C_out, kH, kW,
                                  stride, stride, pad, pad, 1, 1, Y);
    };
    const float ms = time_loop_ms(body);
    const bool ok = sanity_finite(Y);
    std::printf("%-52s %s %8.3f ms   %s\n", tag,
                dt == Dtype::FP32 ? "FP32" : "FP16", ms, ok ? "ok" : "NOT FINITE");
}

void bench_convt2d(const char* tag,
                   int N, int C_in, int H, int W,
                   int C_out, int k, int stride, int pad, int out_pad,
                   Dtype dt) {
    std::mt19937 rng(42);
    Tensor X  = upload(rand_vec(static_cast<size_t>(N) * C_in * H * W, rng, 0.5f),
                       N, C_in * H * W, dt);
    Tensor Wt = upload(rand_vec(static_cast<size_t>(C_in) * C_out * k * k, rng, 0.05f),
                       C_in, C_out * k * k, dt);
    Tensor B  = upload(rand_vec(static_cast<size_t>(C_out), rng, 0.1f), C_out, 1, dt);
    Tensor Y;

    try {
        auto body = [&] {
            brotensor::conv_transpose2d_forward(X, Wt, &B, N, C_in, H, W,
                                                C_out, k, k, stride, stride,
                                                pad, pad, out_pad, out_pad,
                                                1, 1, 1, Y);
        };
        const float ms = time_loop_ms(body);
        const bool ok = sanity_finite(Y);
        std::printf("%-52s %s %8.3f ms   %s\n", tag,
                    dt == Dtype::FP32 ? "FP32" : "FP16", ms,
                    ok ? "ok" : "NOT FINITE");
    } catch (const std::exception& e) {
        std::printf("%-52s %s   unsupported (%s)\n", tag,
                    dt == Dtype::FP32 ? "FP32" : "FP16", e.what());
    }
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("annotator conv bench, avg of %d iters\n\n", ITERS);

    for (Dtype dt : {Dtype::FP32, Dtype::FP16}) {
        // 3x3 pad 1 stride 1 — HED / DSINE / generic VGG-ish backbone conv.
        bench_conv2d("conv 3x3 p1 s1  64->64    256x256 (HED)",
                     1, 64, 256, 256, 64, 3, 3, 1, 1, dt);
        // 3x3 pad 0 stride 1 — lineart res blocks (input reflect-pre-padded).
        bench_conv2d("conv 3x3 p0 s1  256->256  130x130 (lineart res)",
                     1, 256, 130, 130, 256, 3, 3, 1, 0, dt);
        // 7x7 pad 3 stride 1 — openpose CPM stages.
        bench_conv2d("conv 7x7 p3 s1  185->128  46x46   (openpose CPM)",
                     1, 185, 46, 46, 128, 7, 7, 1, 3, dt);
        bench_conv2d("conv 7x7 p3 s1  128->128  92x92   (openpose CPM hi)",
                     1, 128, 92, 92, 128, 7, 7, 1, 3, dt);
        // 7x7 pad 0 stride 1 — lineart head/tail (pre-padded input).
        bench_conv2d("conv 7x7 p0 s1  3->64     262x262 (lineart head)",
                     1, 3, 262, 262, 64, 7, 7, 1, 0, dt);
        bench_conv2d("conv 7x7 p0 s1  64->1     262x262 (lineart tail)",
                     1, 64, 262, 262, 1, 7, 7, 1, 0, dt);
        // 5x5 pad 2 stride 1 — general coverage.
        bench_conv2d("conv 5x5 p2 s1  96->96    128x128",
                     1, 96, 128, 128, 96, 5, 5, 1, 2, dt);
        // conv_transpose2d — lineart upsamplers + SAM/DPT 2x2 upscalers.
        bench_convt2d("convT 3x3 s2 p1 op1 256->128 128x128 (lineart up1)",
                      1, 256, 128, 128, 128, 3, 2, 1, 1, dt);
        bench_convt2d("convT 3x3 s2 p1 op1 128->64  256x256 (lineart up2)",
                      1, 128, 256, 256, 64, 3, 2, 1, 1, dt);
        bench_convt2d("convT 2x2 s2 p0     256->64  64x64   (SAM upscale)",
                      1, 256, 64, 64, 64, 2, 2, 0, 0, dt);
        std::printf("\n");
    }
    return 0;
}
