// W8A16 parity: matmul_int8w_fp16 vs matmul(dequant_as_fp16, X), and
// conv2d_int8w_fp16_forward vs conv2d_forward(X, dequant_as_fp16).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#if defined(BROTENSOR_HAS_CUDA)
#include <cuda_runtime.h>
#else
#include <cstring>
// On Metal a device Tensor's .data points to a host-shared MTLBuffer, so a
// plain memcpy is equivalent to a H2D copy. Provide a tiny shim so the test
// body stays backend-agnostic.
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

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while(0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void test_matmul_int8w() {
    std::printf("  matmul_int8w_fp16_gpu 256x256\n");
    const int OUT = 256, IN = 256, B = 32;
    std::mt19937 rng(0x55);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Wf(OUT * IN), Xf(IN * B);
    for (auto& v : Wf) v = dist(rng);
    for (auto& v : Xf) v = dist(rng);
    auto Wh = to_fp16(Wf), Xh = to_fp16(Xf);

    // Host quantise.
    std::vector<int8_t> Wq(OUT * IN);
    std::vector<float>  scales(OUT);
    brotensor::quantize_int8_per_row_host(Wh.data(), OUT, IN, Wq.data(), scales.data());

    // Reference: dequantise to FP16 on host, then run plain matmul_gpu.
    std::vector<uint16_t> Wdeq(OUT * IN);
    for (int r = 0; r < OUT; ++r) {
        const float s = scales[r];
        for (int c = 0; c < IN; ++c) {
            const float v = static_cast<float>(Wq[r * IN + c]) * s;
            Wdeq[r * IN + c] = brotensor::fp32_to_fp16_bits(v);
        }
    }

    Tensor Y_ref_g;
    Tensor W_deq_g = Tensor::from_host_fp16_on(Device::CUDA, Wdeq.data(), OUT, IN);
    Tensor X_g     = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),   IN,  B);
    brotensor::matmul(W_deq_g, X_g, Y_ref_g);
    std::vector<uint16_t> ref(OUT * B);
    Y_ref_g.copy_to_host_fp16(ref.data());

    // W8A16 path.
    Tensor W_int8_g = Tensor::empty_on(Device::CUDA, OUT, IN, Dtype::INT8);
    Tensor Y_g;
    cudaMemcpy(W_int8_g.data, Wq.data(), OUT * IN * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    Tensor S_g = Tensor::from_host_on(Device::CUDA, scales.data(), OUT, 1);
    brotensor::matmul_int8w_fp16(W_int8_g, S_g, X_g, Y_g);
    CHECK(Y_g.dtype == Dtype::FP16 && Y_g.rows == OUT && Y_g.cols == B);
    std::vector<uint16_t> got(OUT * B);
    brotensor::sync_all();
    Y_g.copy_to_host_fp16(got.data());

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
    // The dequantised reference is *identical* up to FP16 rounding; the two
    // paths share the same K=256 reduction in FP32, so error should be tiny.
    CHECK(max_err < 1e-2f);
}

static void test_conv2d_int8w() {
    std::printf("  conv2d_int8w_fp16_forward_gpu small NCHW\n");
    const int N = 1, C_in = 8, H = 12, W = 12;
    const int C_out = 16, kH = 3, kW = 3;
    const int stride = 1, pad = 1, dil = 1, groups = 1;
    std::mt19937 rng(0x66);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Xf(N * C_in * H * W), Wf(C_out * C_in * kH * kW);
    for (auto& v : Xf) v = dist(rng);
    for (auto& v : Wf) v = dist(rng);
    auto Xh = to_fp16(Xf), Wh = to_fp16(Wf);

    const int win = C_in * kH * kW;
    std::vector<int8_t> Wq(C_out * win);
    std::vector<float>  scales(C_out);
    brotensor::quantize_int8_per_row_host(Wh.data(), C_out, win, Wq.data(), scales.data());

    // Reference: dequantise to FP16 then run conv2d_forward_gpu.
    std::vector<uint16_t> Wdeq(C_out * win);
    for (int r = 0; r < C_out; ++r) {
        const float s = scales[r];
        for (int c = 0; c < win; ++c) {
            Wdeq[r * win + c] = brotensor::fp32_to_fp16_bits(
                static_cast<float>(Wq[r * win + c]) * s);
        }
    }

    Tensor Y_ref_g;
    Tensor Xg     = Tensor::from_host_fp16_on(Device::CUDA, Xh.data(),   N, C_in * H * W);
    Tensor Wdeq_g = Tensor::from_host_fp16_on(Device::CUDA, Wdeq.data(), C_out, win);
    brotensor::conv2d_forward(Xg, Wdeq_g, nullptr,
                              N, C_in, H, W, C_out, kH, kW,
                              stride, stride, pad, pad, dil, dil,
                              groups, Y_ref_g);
    std::vector<uint16_t> ref(Y_ref_g.size());
    Y_ref_g.copy_to_host_fp16(ref.data());

    // W8A16 path.
    Tensor W_int8_g = Tensor::empty_on(Device::CUDA, C_out, win, Dtype::INT8);
    Tensor Y_g;
    cudaMemcpy(W_int8_g.data, Wq.data(), C_out * win * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    Tensor S_g = Tensor::from_host_on(Device::CUDA, scales.data(), C_out, 1);
    brotensor::conv2d_int8w_fp16_forward(Xg, W_int8_g, S_g, nullptr,
                                         N, C_in, H, W, C_out, kH, kW,
                                         stride, stride, pad, pad, dil, dil,
                                         groups, Y_g);
    CHECK(Y_g.size() == Y_ref_g.size() && Y_g.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Y_g.size());
    brotensor::sync_all();
    Y_g.copy_to_host_fp16(got.data());

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
    CHECK(max_err < 1e-2f);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_int8_quant\n");
    test_matmul_int8w();
    test_conv2d_int8w();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
