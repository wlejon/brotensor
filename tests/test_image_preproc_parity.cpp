// CPU↔GPU parity tests for image_normalize and image_u8_to_f32_nhwc_to_nchw.
//
// For image_u8_to_f32_nhwc_to_nchw, the `src` pointer is on the SAME device
// as the output: on CPU we pass a host buffer; on the GPU backend (CUDA) we
// upload the uint8 buffer to a device tensor first and pass its device data
// pointer.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Dtype;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-5f;

void run_image_normalize(int N, int C, int H, int W, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);
    Tensor mean = Tensor::mat(C, 1);
    Tensor std_ = Tensor::mat(C, 1);
    for (int c = 0; c < C; ++c) {
        mean.host_f32_mut()[c] = rng.next_unit() * 0.5f;
        // Positive non-zero std.
        std_.host_f32_mut()[c] = 0.2f + std::fabs(rng.next_unit());
    }

    Tensor cpu_Y;
    brotensor::image_normalize(X, mean, std_, N, C, H, W, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gm = mean.to(gpu_device());
    Tensor gs = std_.to(gpu_device());
    Tensor gY;
    brotensor::image_normalize(gX, gm, gs, N, C, H, W, gY);
    compare_tensors(cpu_Y, download_to_host(gY), "img_norm_Y", kAtol, kRtol);
}

void run_u8_to_f32(int N, int H, int W, int C,
                   float scale, float bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int total = N * H * W * C;
    std::vector<uint8_t> src(total);
    for (int i = 0; i < total; ++i) {
        src[i] = static_cast<uint8_t>(rng.next_u64() & 0xFF);
    }

    Tensor cpu_Y;
    brotensor::image_u8_to_f32_nhwc_to_nchw(src.data(), N, H, W, C,
                                            scale, bias, cpu_Y);

    // Upload the u8 buffer to GPU. Use INT8 storage (size_bytes == count) —
    // contents are uninterpreted bytes, we just need a device pointer.
    Tensor src_gpu = brotensor::Tensor::zeros_on(brotensor::Device::CPU,
                                                 total, 1, Dtype::INT8);
    std::memcpy(src_gpu.host_raw_mut(), src.data(), total);
    src_gpu = src_gpu.to(gpu_device());
    const uint8_t* d_src = static_cast<const uint8_t*>(src_gpu.data);

    // Pin gY to the GPU device so the dispatcher resolves to CUDA rather than
    // CPU (no tensor inputs in this op — Y is the only operand). Allocate as
    // empty so the op resizes it to the correct shape.
    Tensor gY = Tensor::zeros_on(gpu_device(), N, C * H * W, Dtype::FP32);
    brotensor::image_u8_to_f32_nhwc_to_nchw(d_src, N, H, W, C,
                                            scale, bias, gY);
    compare_tensors(cpu_Y, download_to_host(gY), "u8_nhwc_nchw", kAtol, kRtol);
}

} // namespace

BT_PARITY_TEST(img_norm_small)    { run_image_normalize(2, 3, 4, 5, 0xC500ull); }
BT_PARITY_TEST(img_norm_bigger)   { run_image_normalize(1, 4, 8, 11, 0xC501ull); }

BT_PARITY_TEST(u8_nhwc_nchw_id)   { run_u8_to_f32(1, 4, 5, 3, 1.0f,         0.0f, 0xC510ull); }
BT_PARITY_TEST(u8_nhwc_nchw_01)   { run_u8_to_f32(2, 3, 4, 3, 1.0f/255.0f,  0.0f, 0xC511ull); }
BT_PARITY_TEST(u8_nhwc_nchw_m11)  { run_u8_to_f32(2, 5, 7, 3, 2.0f/255.0f, -1.0f, 0xC512ull); }

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    return run_all("image_preproc cpu/gpu parity");
}
