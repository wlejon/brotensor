// WMMA INT8W batched linear parity vs FP16-dequant reference. Exercises the
// SDXL UNet projection shapes (Q/K/V/O at D=320/640/1280, GEGLU FF1/FF2)
// plus a small odd-K fallback case that misses the WMMA dispatcher and must
// land on the tiled int8w kernel.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#if defined(BROTENSOR_HAS_CUDA)
#include <cuda_runtime.h>
#else
#include <cstring>
static inline void cudaMemcpy(void* dst, const void* src, size_t n, int) {
    std::memcpy(dst, src, n);
}
#define cudaMemcpyHostToDevice 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while(0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void run_case(const char* label, int B, int M, int K, bool with_bias,
                     uint32_t seed, float tol = 1.5e-2f) {
    std::printf("  %s B=%d M=%d K=%d bias=%d\n", label, B, M, K, (int)with_bias);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dx(-0.5f, 0.5f);
    std::uniform_real_distribution<float> dw(-0.1f, 0.1f);
    std::uniform_real_distribution<float> db(-0.05f, 0.05f);

    std::vector<float> Xf(static_cast<size_t>(B) * K);
    std::vector<float> Wf(static_cast<size_t>(M) * K);
    std::vector<float> Bf(M);
    for (auto& v : Xf) v = dx(rng);
    for (auto& v : Wf) v = dw(rng);
    for (auto& v : Bf) v = db(rng);
    auto Xh = to_fp16(Xf), Wh = to_fp16(Wf), Bh = to_fp16(Bf);

    std::vector<int8_t> Wq(static_cast<size_t>(M) * K);
    std::vector<float>  scales(M);
    brotensor::quantize_int8_per_row_host(Wh.data(), M, K, Wq.data(), scales.data());

    // Reference: dequantise W → FP16, run FP16 batched linear.
    std::vector<uint16_t> Wdeq(static_cast<size_t>(M) * K);
    for (int r = 0; r < M; ++r) {
        const float s = scales[r];
        for (int c = 0; c < K; ++c) {
            Wdeq[r * K + c] = brotensor::fp32_to_fp16_bits(
                static_cast<float>(Wq[r * K + c]) * s);
        }
    }

    GpuTensor Xg, Wdeq_g, Bg, Y_ref_g;
    brotensor::upload_fp16(Xh.data(),   B, K, Xg);
    brotensor::upload_fp16(Wdeq.data(), M, K, Wdeq_g);
    if (with_bias) brotensor::upload_fp16(Bh.data(), M, 1, Bg);
    brotensor::linear_forward_batched_fp16_gpu(
        Wdeq_g, with_bias ? &Bg : nullptr, Xg, Y_ref_g);
    std::vector<uint16_t> ref(Y_ref_g.size());
    brotensor::download_fp16(Y_ref_g, ref.data());

    // INT8W path.
    GpuTensor W_int8_g(M, K, Dtype::INT8);
    GpuTensor S_g, Y_g;
    cudaMemcpy(W_int8_g.data, Wq.data(),
               static_cast<size_t>(M) * K * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    brotensor::upload(scales.data(), M, 1, S_g);
    brotensor::linear_forward_batched_int8w_fp16_gpu(
        W_int8_g, S_g, with_bias ? &Bg : nullptr, Xg, Y_g);
    CHECK(Y_g.dtype == Dtype::FP16 && Y_g.rows == B && Y_g.cols == M);
    std::vector<uint16_t> got(Y_g.size());
    brotensor::download_fp16(Y_g, got.data());
    brotensor::cuda_sync();

    float max_err = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float r = brotensor::fp16_bits_to_fp32(ref[i]);
        const float e = std::fabs(g - r);
        if (e > max_err) max_err = e;
        if (e > 1e-2f + 1e-2f * std::fabs(r)) ++bad;
    }
    std::printf("    max_err=%g bad=%d/%zu\n", max_err, bad, ref.size());
    CHECK(max_err < tol);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_int8_linear_wmma\n");

    // SDXL Q/K/V/O projections at three head-channel widths.
    run_case("QKVO D=320",  64, 320,  320,  false, 0x1001);
    run_case("QKVO D=640",  64, 640,  640,  false, 0x1002);
    run_case("QKVO D=1280", 16, 1280, 1280, false, 0x1003);

    // GEGLU FF1 (wide expand) — rectangular M = 2*D.
    run_case("GEGLU FF1 D=640",  16, 2 * 640,  640,  false, 0x2001);
    run_case("GEGLU FF1 D=1280", 16, 2 * 1280, 1280, false, 0x2002);

    // FF2 contraction — K = 2*D.
    run_case("FF2 D=1280", 16, 1280, 2 * 1280, false, 0x2101);

    // Bias coverage on at least one shape.
    run_case("QKVO D=640 +bias", 64, 640, 640, true, 0x3001);

    // Fallback paths: K%8 != 0 must bypass WMMA and still match the reference.
    run_case("fallback K=24", 16, 64,  24, false, 0x4001);
    run_case("fallback K=33", 16, 64,  33, false, 0x4002);

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
