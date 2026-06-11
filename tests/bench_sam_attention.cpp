// Micro-benchmark: SAM / ViTDet encoder attention at ViT-B shapes — the
// decomposed rel-pos op (global 64x64 block) and its windowed variant
// (64x64 grid, window 14), FP32 and FP16.
//
// NOT registered with ctest — invoke manually:
//   ./build/tests/Release/brotensor_bench_sam_attention
//
// Timing uses cudaEvent_t around the full op with warmup. Each row also runs
// a finite/bounded spot check on the output so a fast-but-wrong kernel can't
// pass silently.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

constexpr int WARMUP = 2;
constexpr int ITERS  = 10;

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
bool sanity_finite(const Tensor& g, float abs_clip = 1e3f) {
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

void bench(const char* tag, bool windowed, int gh, int gw, int window,
           int D, int H, Dtype dt) {
    std::mt19937 rng(42);
    const int L  = gh * gw;
    const int dh = D / H;
    const int rel_rows = windowed ? 2 * window - 1 : 0;
    const int rh_rows = windowed ? rel_rows : 2 * gh - 1;
    const int rw_rows = windowed ? rel_rows : 2 * gw - 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    Tensor X  = upload(rand_vec(static_cast<size_t>(L) * D, rng, 0.5f), L, D, dt);
    Tensor Wq = upload(rand_vec(static_cast<size_t>(D) * D, rng, 0.05f), D, D, dt);
    Tensor Wk = upload(rand_vec(static_cast<size_t>(D) * D, rng, 0.05f), D, D, dt);
    Tensor Wv = upload(rand_vec(static_cast<size_t>(D) * D, rng, 0.05f), D, D, dt);
    Tensor Wo = upload(rand_vec(static_cast<size_t>(D) * D, rng, 0.05f), D, D, dt);
    Tensor bq = upload(rand_vec(D, rng, 0.1f), D, 1, dt);
    Tensor bv = upload(rand_vec(D, rng, 0.1f), D, 1, dt);
    Tensor bo = upload(rand_vec(D, rng, 0.1f), D, 1, dt);
    Tensor rh = upload(rand_vec(static_cast<size_t>(rh_rows) * dh, rng, 0.2f),
                       rh_rows, dh, dt);
    Tensor rw = upload(rand_vec(static_cast<size_t>(rw_rows) * dh, rng, 0.2f),
                       rw_rows, dh, dt);
    Tensor O;

    auto body = [&] {
        if (windowed) {
            brotensor::self_attention_decomposed_rel_pos_windowed_forward(
                X, Wq, &bq, Wk, nullptr, Wv, &bv, Wo, &bo, rh, rw,
                H, gh, gw, window, scale, O);
        } else {
            brotensor::self_attention_decomposed_rel_pos_forward(
                X, Wq, &bq, Wk, nullptr, Wv, &bv, Wo, &bo, rh, rw,
                H, gh, gw, scale, O);
        }
    };
    const float ms = time_loop_ms(body);
    const bool ok = sanity_finite(O);
    std::printf("%-34s %8.3f ms   %s\n", tag, ms, ok ? "ok" : "NOT FINITE");
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("SAM/ViTDet attention bench (ViT-B shapes), avg of %d iters\n", ITERS);
    // Global block: 64x64 grid -> L=4096.
    bench("global  64x64 D768 H12  FP32", false, 64, 64, 0, 768, 12, Dtype::FP32);
    bench("global  64x64 D768 H12  FP16", false, 64, 64, 0, 768, 12, Dtype::FP16);
    // Windowed block: 64x64 grid, window 14 -> 25 windows of 196 tokens.
    bench("window  64x64 w14 D768 H12 FP32", true, 64, 64, 14, 768, 12, Dtype::FP32);
    bench("window  64x64 w14 D768 H12 FP16", true, 64, 64, 14, 768, 12, Dtype::FP16);
    return 0;
}
