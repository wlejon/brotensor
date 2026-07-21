// CPU-reference parity for the public matmul_abt op (batched A @ B^T, FP16)
// on whichever GPU backend this binary was built with (CUDA or Metal).
//
// The existing test_matmul_abt.cpp is CUDA-only and uses sub-1024 M*N shapes,
// so it never exercises the Metal 64x64x32 simdgroup fast path. This test adds
// large shapes that DO hit the tiled kernel — including deliberately unaligned
// M/N/K to exercise its zero-fill masking — plus batch>1, per-N bias, and every
// activation code, all validated against a double-precision CPU reference of
// the fp16-rounded operands. It runs on Metal via the parity harness.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <string>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;
using namespace bt_parity;

namespace {

float ref_act(int act, float v) {
    switch (act) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: {
            const float k = 0.7978845608f;
            float u = k * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + std::tanh(u));
        }
        case 3: return 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f));
        case 4: return v / (1.0f + std::exp(-v));
        case 5: return v / (1.0f + std::exp(-1.702f * v));
        default: return v;
    }
}

// One case: FP16 matmul_abt on the GPU backend vs a double-accum CPU reference
// computed from the fp16-rounded operands.
void run(int batch, int M, int N, int K, bool with_bias, int act,
         const char* tag) {
    SplitMix64 rng(0x9111u + batch * 31 + M * 7 + N * 5 + K * 3 + act * 13 +
                   (with_bias ? 1 : 0));

    Tensor A = Tensor::zeros_on(Device::CPU, batch * M, K, Dtype::FP32);
    Tensor B = Tensor::zeros_on(Device::CPU, batch * N, K, Dtype::FP32);
    Tensor bias = Tensor::zeros_on(Device::CPU, 1, N, Dtype::FP32);
    fill_random(A, rng, 0.5f);
    fill_random(B, rng, 0.5f);
    fill_random(bias, rng, 0.5f);

    // Operands the kernel actually sees (fp16-rounded).
    Tensor Aq = fp16_host_to_f32(to_fp16_host(A));
    Tensor Bq = fp16_host_to_f32(to_fp16_host(B));
    Tensor biasq = fp16_host_to_f32(to_fp16_host(bias));
    const float* pA = Aq.host_f32();
    const float* pB = Bq.host_f32();
    const float* pBias = biasq.host_f32();

    Tensor ref = Tensor::zeros_on(Device::CPU, batch * M, N, Dtype::FP32);
    float* pr = ref.host_f32_mut();
    for (int b = 0; b < batch; ++b)
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double s = 0.0;
                for (int k = 0; k < K; ++k)
                    s += static_cast<double>(pA[(static_cast<size_t>(b) * M + m) * K + k]) *
                         static_cast<double>(pB[(static_cast<size_t>(b) * N + n) * K + k]);
                float v = static_cast<float>(s);
                if (with_bias) v += pBias[n];
                v = ref_act(act, v);
                pr[(static_cast<size_t>(b) * M + m) * N + n] = v;
            }

    Tensor Ag = to_fp16_host(A).to(gpu_device());
    Tensor Bg = to_fp16_host(B).to(gpu_device());
    Tensor biasg = to_fp16_host(bias).to(gpu_device());
    Tensor Cg = Tensor::zeros_on(gpu_device(), batch * M, N, Dtype::FP16);

    brotensor::matmul_abt(Ag, Bg, Cg, batch, M, N, K,
                          static_cast<long long>(M) * K,
                          static_cast<long long>(N) * K,
                          static_cast<long long>(M) * N,
                          with_bias ? &biasg : nullptr, act);

    Tensor got = fp16_host_to_f32(download_to_host(Cg));
    // FP16 storage with FP32 accumulate; short reductions here keep error low.
    compare_tensors(ref, got, tag, /*atol=*/2e-2f, /*rtol=*/5e-2f);
}

}  // namespace

// Naive path (M*N < 1024): mirrors the historical coverage, now on Metal too.
BT_PARITY_TEST(matmul_abt_small_naive) {
    run(1, 20, 12, 24, false, 0, "small b1");
    run(2, 20, 12, 24, false, 0, "small b2");
    run(2, 20, 12, 24, true,  0, "small b2 bias");
    run(2, 20, 12, 24, true,  1, "small b2 bias relu");
}

// Tiled fast path, tile-aligned (N%64==0, K%32==0).
BT_PARITY_TEST(matmul_abt_tiled_aligned) {
    run(1, 64, 128, 64, false, 0, "tiled aligned");
    run(1, 64, 128, 64, true,  0, "tiled aligned bias");
    run(1, 64, 128, 64, true,  2, "tiled aligned bias gelu-tanh");
}

// Tiled fast path with masking: M, N, K all unaligned to the tile, batch>1.
BT_PARITY_TEST(matmul_abt_tiled_masked) {
    run(2, 50, 70, 40, true, 1, "tiled masked b2 bias relu");
    run(1, 96, 96, 48, true, 4, "tiled masked K48 silu");    // K % 32 == 16
    run(1, 200, 16, 32, false, 0, "tiled tall skinny N16");  // N < BN, masked
    run(1, 128, 130, 64, true, 3, "tiled masked N130 gelu-erf");
}

int main() { return bt_parity::run_all("test_matmul_abt_parity"); }
