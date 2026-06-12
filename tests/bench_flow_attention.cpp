// Micro-benchmark: non-causal MHA over pre-projected packed QKV at the
// TripoSplat flow-DiT shapes — flash_attention_forward (per-head WMMA path,
// materialises an (Lq, Lk) score matrix per head) vs
// flash_attention_varlen_forward with batch=1, cu=[0, L] (fused online-softmax
// tile kernel), which is semantically identical for this workload.
//
// Flow-DiT call sites per forward (D=1024, nh=16, hd=64):
//   2x noise_refiner   L=8192
//   2x context_refiner L=4101
//   24x joint blocks   L=12294   (8192 latent + 4101 cond + 1 cam)
//
// Each row sanity-checks the two implementations against each other on a
// 32-element spot check so a fast-but-wrong kernel can't pass silently.
//
// NOT registered with ctest — invoke manually:
//   ./build/tests/Release/brotensor_bench_flow_attention

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

using brotensor::Dtype;
using brotensor::Tensor;

namespace {

constexpr int WARMUP = 2;
constexpr int ITERS  = 5;   // the slow path is ~600 ms/call at L=12294

void upload_rand_fp16(int rows, int cols, Tensor& g, std::mt19937& rng,
                      float scale = 0.3f) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(static_cast<size_t>(rows) * cols);
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        h[i] = brotensor::fp32_to_fp16_bits(d(rng));
    g = Tensor::from_host_fp16_on(brotensor::Device::CUDA, h.data(), rows, cols);
}

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
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

// Spot-check the two outputs agree. Attention output is a convex mix of V
// (|V| <= ~0.3 here); FP16 round-off across the two accumulation orders stays
// well inside 2e-2 absolute at these magnitudes.
bool outputs_agree(const Tensor& a, const Tensor& b, float tol = 2e-2f) {
    std::vector<uint16_t> ha(a.size()), hb(b.size());
    a.copy_to_host_fp16(ha.data());
    b.copy_to_host_fp16(hb.data());
    brotensor::sync_all();
    if (ha.size() != hb.size()) return false;
    const int n = static_cast<int>(ha.size());
    const int step = n > 4096 ? n / 4096 : 1;
    for (int i = 0; i < n; i += step) {
        const float fa = brotensor::fp16_bits_to_fp32(ha[i]);
        const float fb = brotensor::fp16_bits_to_fp32(hb[i]);
        if (!std::isfinite(fa) || !std::isfinite(fb)) return false;
        if (std::fabs(fa - fb) > tol) return false;
    }
    return true;
}

void bench_shape(int L, int nh, int hd) {
    const int D = nh * hd;
    std::mt19937 rng(42);
    Tensor Q, K, V, O_fa, O_vl;
    upload_rand_fp16(L, D, Q, rng);
    upload_rand_fp16(L, D, K, rng);
    upload_rand_fp16(L, D, V, rng);

    Tensor cu = Tensor::zeros_on(brotensor::Device::CPU, 2, 1, Dtype::INT32);
    static_cast<int32_t*>(cu.data)[0] = 0;
    static_cast<int32_t*>(cu.data)[1] = L;
    cu = cu.to(brotensor::Device::CUDA);
    const int32_t* cu_p = static_cast<const int32_t*>(cu.data);

    const float ms_fa = time_loop_ms(ITERS, [&] {
        brotensor::flash_attention_forward(Q, K, V, /*d_mask=*/nullptr, nh,
                                           /*causal=*/false, O_fa);
    });
    const float ms_vl = time_loop_ms(ITERS, [&] {
        brotensor::flash_attention_varlen_forward(Q, K, V, cu_p, cu_p,
                                                  /*batch=*/1, L, L, nh, hd,
                                                  /*causal=*/false, O_vl);
    });
    const bool ok = outputs_agree(O_fa, O_vl);

    // Useful-FLOP rate (2 matmuls, 2*L*L*D each fused or not): 4*L^2*D.
    const double flop = 4.0 * static_cast<double>(L) * L * D;
    std::printf("L=%6d nh=%d hd=%d   fa: %9.2f ms (%6.1f GFLOP/s)   "
                "varlen: %9.2f ms (%6.1f GFLOP/s)   speedup %6.1fx   %s\n",
                L, nh, hd,
                ms_fa, flop / (ms_fa * 1e6),
                ms_vl, flop / (ms_vl * 1e6),
                ms_fa / ms_vl, ok ? "outputs agree" : "!! MISMATCH !!");
}

}  // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("brotensor_bench_flow_attention  (warmup=%d, iters=%d)\n",
                WARMUP, ITERS);
    bench_shape(4101, 16, 64);    // context_refiner
    bench_shape(8192, 16, 64);    // noise_refiner
    bench_shape(12294, 16, 64);   // joint blocks
    return 0;
}
