// CPU↔GPU parity tests for adaptive_avg_pool2d + max_pool2d.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Dtype;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-4f;

// adaptive_avg_pool2d

void run_aap_fwd(int N, int C, int H, int W, int Ho, int Wo, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::adaptive_avg_pool2d_forward(X, N, C, H, W, Ho, Wo, cpu_Y);
    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::adaptive_avg_pool2d_forward(gX, N, C, H, W, Ho, Wo, gpu_Y);
    compare_tensors(cpu_Y, download_to_host(gpu_Y), "aap_fwd", kAtol, kRtol);
}

void run_aap_bwd(int N, int C, int H, int W, int Ho, int Wo, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(N, C * Ho * Wo);
    fill_random(dY, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::adaptive_avg_pool2d_backward(dY, N, C, H, W, Ho, Wo, cpu_dX);
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::adaptive_avg_pool2d_backward(gdY, N, C, H, W, Ho, Wo, gpu_dX);
    compare_tensors(cpu_dX, download_to_host(gpu_dX), "aap_bwd", kAtol, kRtol);
}

// max_pool2d

void run_mp_fwd(int N, int C, int H, int W,
                int kH, int kW, int sh, int sw, int ph, int pw,
                uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y, cpu_I;
    brotensor::max_pool2d_forward(X, N, C, H, W, kH, kW, sh, sw, ph, pw,
                                  cpu_Y, cpu_I);
    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y, gpu_I;
    brotensor::max_pool2d_forward(gX, N, C, H, W, kH, kW, sh, sw, ph, pw,
                                  gpu_Y, gpu_I);
    compare_tensors(cpu_Y, download_to_host(gpu_Y), "mp_fwd_Y", kAtol, kRtol);

    Tensor host_I = download_to_host(gpu_I);
    BT_CHECK(host_I.rows == cpu_I.rows && host_I.cols == cpu_I.cols);
    const int32_t* a = static_cast<const int32_t*>(cpu_I.data);
    const int32_t* b = static_cast<const int32_t*>(host_I.data);
    for (int i = 0; i < cpu_I.size(); ++i) BT_CHECK(a[i] == b[i]);
}

void run_mp_bwd(int N, int C, int H, int W,
                int kH, int kW, int sh, int sw, int ph, int pw,
                uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);

    // Forward to get Idx (CPU); use same Idx for both backends.
    Tensor cpu_Y, cpu_I;
    brotensor::max_pool2d_forward(X, N, C, H, W, kH, kW, sh, sw, ph, pw,
                                  cpu_Y, cpu_I);
    const int Ho = cpu_Y.cols / (C * (cpu_Y.cols / (C * (cpu_Y.cols / C))));
    (void)Ho;
    const int H_out = (H + 2 * ph - kH) / sh + 1;
    const int W_out = (W + 2 * pw - kW) / sw + 1;

    Tensor dY = Tensor::mat(N, C * H_out * W_out);
    fill_random(dY, rng, 1.0f);

    Tensor cpu_dX;
    brotensor::max_pool2d_backward(dY, cpu_I, N, C, H, W, H_out, W_out, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gI  = cpu_I.to(gpu_device());
    Tensor gpu_dX;
    brotensor::max_pool2d_backward(gdY, gI, N, C, H, W, H_out, W_out, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "mp_bwd", kAtol, kRtol);
}

} // namespace

// adaptive_avg_pool2d
BT_PARITY_TEST(aap_fwd_id)        { run_aap_fwd(2, 3, 8, 9,  8, 9, 0xC400ull); }
BT_PARITY_TEST(aap_fwd_global)    { run_aap_fwd(2, 4, 7, 11, 1, 1, 0xC401ull); }
BT_PARITY_TEST(aap_fwd_uneven)    { run_aap_fwd(2, 3, 9, 7,  4, 3, 0xC402ull); }
BT_PARITY_TEST(aap_bwd_uneven)    { run_aap_bwd(2, 3, 9, 7,  4, 3, 0xC403ull); }
BT_PARITY_TEST(aap_bwd_global)    { run_aap_bwd(2, 4, 7, 11, 1, 1, 0xC404ull); }

// max_pool2d
BT_PARITY_TEST(mp_fwd_2x2s2)      { run_mp_fwd(2, 3, 8, 8, 2, 2, 2, 2, 0, 0, 0xC410ull); }
BT_PARITY_TEST(mp_fwd_3x3s1p1)    { run_mp_fwd(2, 3, 6, 7, 3, 3, 1, 1, 1, 1, 0xC411ull); }
BT_PARITY_TEST(mp_fwd_pad)        { run_mp_fwd(1, 2, 5, 5, 3, 3, 2, 2, 1, 1, 0xC412ull); }
BT_PARITY_TEST(mp_bwd_2x2s2)      { run_mp_bwd(2, 3, 8, 8, 2, 2, 2, 2, 0, 0, 0xC413ull); }
BT_PARITY_TEST(mp_bwd_3x3s1p1)    { run_mp_bwd(2, 3, 6, 7, 3, 3, 1, 1, 1, 1, 0xC414ull); }
BT_PARITY_TEST(mp_bwd_overlap)    { run_mp_bwd(1, 2, 6, 6, 3, 3, 1, 1, 0, 0, 0xC415ull); }

int main() { return run_all("pool2d cpu/gpu parity"); }
