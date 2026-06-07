// Micro-benchmark: fused vs composite filtered_lrelu at StyleGAN3-R-ish shapes.
//
// NOT registered with ctest — invoke manually:
//   ./build-cuda/tests/Release/brotensor_bench_filtered_lrelu
//
// The fused path triggers when up_buf/act_buf are uncommitted; the composite
// (baseline) is forced by handing it pre-committed up_buf/act_buf. Each row
// sanity-checks the fused output against the composite so a fast-but-wrong
// kernel can't pass silently. Timing uses cudaEvent_t with warmup.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

namespace {

constexpr int WARMUP = 5;
constexpr int ITERS  = 50;

float time_loop_ms(int iters, const std::function<void()>& body) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < WARMUP; ++i) body();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) body();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

std::vector<float> rnd(size_t n, uint32_t seed, float s = 0.3f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-s, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

Tensor up_dtype(const std::vector<float>& v, int r, int c, Dtype dt) {
    Tensor g = Tensor::from_host_on(Device::CPU, v.data(), r, c).to(Device::CUDA);
    if (dt == Dtype::FP32) return g;
    Tensor out;
    brotensor::cast(g, out, dt);   // FP32 → FP16/BF16 on device
    return out;
}

std::vector<float> to_host_f32(const Tensor& g) {
    Tensor f32;
    brotensor::cast(g, f32, Dtype::FP32);
    brotensor::sync(Device::CUDA);
    return f32.to_host_vector();
}

void run(const char* tag, Dtype dt, int C, int H, int W) {
    const int N = 1;
    const int up = 2, down = 2;
    const int px0 = 2, px1 = 3, py0 = 2, py1 = 3;
    const int fuH = 6, fuW = 6, fdH = 6, fdW = 6;
    const float gain = std::sqrt(2.0f), slope = 0.2f, clamp = 0.8f;

    Tensor X  = up_dtype(rnd((size_t)N * C * H * W, 1), N, C * H * W, dt);
    Tensor Fu = up_dtype(rnd(fuH * fuW, 2), fuH, fuW, dt);
    Tensor Fd = up_dtype(rnd(fdH * fdW, 3), fdH, fdW, dt);
    Tensor B  = up_dtype(rnd(C, 4), C, 1, dt);

    // Fused: uncommitted caches.
    Tensor ub_f, ab_f, Y_f;
    auto fused = [&]() {
        brotensor::filtered_lrelu_forward(X, Fu, Fd, &B, N, C, H, W, up, down,
                                          px0, px1, py0, py1, gain, slope, clamp,
                                          ub_f, ab_f, Y_f);
    };
    // Composite: pre-commit up_buf/act_buf so the dispatcher takes the fallback.
    fused();                              // populate Y_f once for the diff
    brotensor::sync(Device::CUDA);
    Tensor ub_c = ub_f, ab_c = ab_f, Y_c; // dummy committed caches
    // Make ub_c/ab_c genuinely committed with the right (composite) shapes by
    // running the composite once via committed buffers.
    {
        Tensor u, a;
        // up_buf must be committed for the composite branch; allocate via a
        // first composite call using freshly committed (non-null) placeholders.
        u = Tensor::zeros_on(Device::CUDA, 1, 1, dt);
        a = Tensor::zeros_on(Device::CUDA, 1, 1, dt);
        brotensor::filtered_lrelu_forward(X, Fu, Fd, &B, N, C, H, W, up, down,
                                          px0, px1, py0, py1, gain, slope, clamp,
                                          u, a, Y_c);
        ub_c = u; ab_c = a;
    }
    auto composite = [&]() {
        Tensor u = ub_c, a = ab_c, y;
        brotensor::filtered_lrelu_forward(X, Fu, Fd, &B, N, C, H, W, up, down,
                                          px0, px1, py0, py1, gain, slope, clamp,
                                          u, a, y);
    };

    // Correctness: fused vs composite output.
    brotensor::sync(Device::CUDA);
    std::vector<float> yf = to_host_f32(Y_f);
    std::vector<float> yc = to_host_f32(Y_c);
    double md = 0; for (size_t i = 0; i < yf.size(); ++i) md = std::max(md, (double)std::fabs(yf[i] - yc[i]));

    const float t_fused = time_loop_ms(ITERS, fused);
    const float t_comp  = time_loop_ms(ITERS, composite);
    std::printf("  %-5s C=%-4d %dx%-4d  composite %7.3f ms   fused %7.3f ms   speedup %4.2fx   max|d|=%.2e\n",
                tag, C, H, W, t_comp, t_fused, t_comp / t_fused, md);
}

} // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA not available — skipping.\n");
        return 0;
    }
    std::printf("filtered_lrelu  up=down=2 fu=fd=6x6 (composite=baseline, fused=tiled):\n");
    struct Shape { int C, H, W; };
    for (Shape s : {Shape{512, 32, 32}, Shape{256, 64, 64}, Shape{128, 128, 128},
                    Shape{128, 256, 256}, Shape{64, 256, 256}}) {
        for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
            const char* tag = dt == Dtype::FP32 ? "fp32" : (dt == Dtype::FP16 ? "fp16" : "bf16");
            run(tag, dt, s.C, s.H, s.W);
        }
        std::printf("\n");
    }
    return 0;
}
