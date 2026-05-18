// WMMA INT8W conv2d parity vs FP16-dequant reference. Exercises the
// SD-typical 3x3 s1, 1x1 s1, 3x3 s2 shapes plus a 5x5 case that must fall
// through the WMMA dispatcher to the naive int8w kernel.

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

static void run_case(const char* label,
                     int N, int C_in, int H, int W,
                     int C_out, int kH, int kW,
                     int stride, int pad,
                     uint32_t seed) {
    const int dil = 1, groups = 1;
    std::printf("  %s N=%d Ci=%d Co=%d HxW=%dx%d k=%dx%d s=%d p=%d\n",
                label, N, C_in, C_out, H, W, kH, kW, stride, pad);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dx(-0.5f, 0.5f);
    std::uniform_real_distribution<float> dw(-0.1f, 0.1f);

    std::vector<float> Xf(static_cast<size_t>(N) * C_in * H * W);
    std::vector<float> Wf(static_cast<size_t>(C_out) * C_in * kH * kW);
    for (auto& v : Xf) v = dx(rng);
    for (auto& v : Wf) v = dw(rng);
    auto Xh = to_fp16(Xf), Wh = to_fp16(Wf);

    const int win = C_in * kH * kW;
    std::vector<int8_t> Wq(static_cast<size_t>(C_out) * win);
    std::vector<float>  scales(C_out);
    brotensor::quantize_int8_per_row_host(Wh.data(), C_out, win, Wq.data(), scales.data());

    // Reference: dequantise to FP16 then run conv2d_forward_gpu (which itself
    // takes the FP16 WMMA fast path where applicable).
    std::vector<uint16_t> Wdeq(static_cast<size_t>(C_out) * win);
    for (int r = 0; r < C_out; ++r) {
        const float s = scales[r];
        for (int c = 0; c < win; ++c) {
            Wdeq[r * win + c] = brotensor::fp32_to_fp16_bits(
                static_cast<float>(Wq[r * win + c]) * s);
        }
    }

    GpuTensor Xg, Wdeq_g, Y_ref_g;
    brotensor::upload_fp16(Xh.data(), N, C_in * H * W, Xg);
    brotensor::upload_fp16(Wdeq.data(), C_out, win, Wdeq_g);
    brotensor::conv2d_forward_gpu(Xg, Wdeq_g, nullptr,
                                  N, C_in, H, W, C_out, kH, kW,
                                  stride, stride, pad, pad, dil, dil,
                                  groups, Y_ref_g);
    std::vector<uint16_t> ref(Y_ref_g.size());
    brotensor::download_fp16(Y_ref_g, ref.data());

    GpuTensor W_int8_g(C_out, win, Dtype::INT8);
    GpuTensor S_g, Y_g;
    cudaMemcpy(W_int8_g.data, Wq.data(),
               static_cast<size_t>(C_out) * win * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    brotensor::upload(scales.data(), C_out, 1, S_g);
    brotensor::conv2d_int8w_fp16_forward_gpu(Xg, W_int8_g, S_g, nullptr,
                                             N, C_in, H, W, C_out, kH, kW,
                                             stride, stride, pad, pad, dil, dil,
                                             groups, Y_g);
    CHECK(Y_g.size() == Y_ref_g.size() && Y_g.dtype == Dtype::FP16);
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
        if (e > 1.5e-2f + 1.5e-2f * std::fabs(r)) ++bad;
    }
    std::printf("    max_err=%g bad=%d/%zu\n", max_err, bad, ref.size());
    CHECK(max_err < 1.5e-2f);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_int8_conv_wmma\n");

    // 3x3 s1 p1 — three SD-typical channel scales (downsized for test speed).
    run_case("3x3 s1 p1 64ch",   1,  64, 32, 32,  64, 3, 3, 1, 1, 0x11);
    run_case("3x3 s1 p1 128ch",  1, 128, 16, 16, 128, 3, 3, 1, 1, 0x12);
    run_case("3x3 s1 p1 256ch",  1, 256,  8,  8, 256, 3, 3, 1, 1, 0x13);

    // 1x1 s1 p0.
    run_case("1x1 s1 p0 128ch",  1, 128, 16, 16, 128, 1, 1, 1, 0, 0x21);

    // 3x3 s2 p1 (downsampler).
    run_case("3x3 s2 p1",        1, 128, 32, 32, 256, 3, 3, 2, 1, 0x31);

    // 5x5 — should fall through WMMA dispatcher to naive int8w kernel.
    run_case("5x5 s1 p2 (naive)", 1, 32, 16, 16, 32, 5, 5, 1, 2, 0x41);

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
