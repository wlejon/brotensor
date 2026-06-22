// CPU↔GPU parity for patch_unpack_forward (DiT unpatchify — pure gather,
// bit-exact FP32). Mirrors the reference scatter in a host loop and checks the
// CPU and GPU ops both match it, in both channel orderings and with channel drop.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

struct Shape { int hp, wp, P, C_total, C_keep; };

// Reference: Y[c, i*P+py, j*P+px] = tokens[i*wp+j, col].
std::vector<float> reference(const std::vector<float>& tok, const Shape& s,
                             bool channel_major) {
    const int PP = s.P * s.P;
    const int H = s.hp * s.P, W = s.wp * s.P;
    const int row_stride = PP * s.C_total;
    std::vector<float> Y(static_cast<size_t>(s.C_keep) * H * W, 0.0f);
    for (int c = 0; c < s.C_keep; ++c)
        for (int i = 0; i < s.hp; ++i)
            for (int py = 0; py < s.P; ++py)
                for (int j = 0; j < s.wp; ++j)
                    for (int px = 0; px < s.P; ++px) {
                        const int block = py * s.P + px;
                        const int col = channel_major ? c * PP + block
                                                      : block * s.C_total + c;
                        const int y = i * s.P + py, x = j * s.P + px;
                        Y[(static_cast<size_t>(c) * H + y) * W + x] =
                            tok[static_cast<size_t>(i * s.wp + j) * row_stride + col];
                    }
    return Y;
}

void run_shape(const Shape& s, uint64_t seed, bool channel_major) {
    SplitMix64 rng(seed);
    const int N = s.hp * s.wp;
    Tensor X = Tensor::mat(N, s.P * s.P * s.C_total);
    fill_random(X, rng);
    const std::vector<float> host = X.to_host_vector();

    const std::vector<float> ref = reference(host, s, channel_major);
    const int cols = s.C_keep * (s.hp * s.P) * (s.wp * s.P);
    Tensor ref_t = Tensor::mat(1, cols);
    for (int k = 0; k < cols; ++k) ref_t.host_f32_mut()[k] = ref[static_cast<size_t>(k)];

    Tensor cpu_Y;
    brotensor::patch_unpack_forward(X, s.hp, s.wp, s.P, s.C_total, s.C_keep,
                                    channel_major, cpu_Y);
    compare_tensors(ref_t, cpu_Y, "patch_unpack_cpu_vs_ref", 0.0f, 0.0f);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::patch_unpack_forward(gX, s.hp, s.wp, s.P, s.C_total, s.C_keep,
                                    channel_major, gpu_Y);
    compare_tensors(ref_t, download_to_host(gpu_Y),
                    "patch_unpack_gpu_vs_ref", 0.0f, 0.0f);
}

void run_shape(const Shape& s, uint64_t seed) {
    run_shape(s, seed, /*channel_major=*/false);
    run_shape(s, seed ^ 0xABCD, /*channel_major=*/true);
}

BT_PARITY_TEST(patch_unpack_small) {
    run_shape({2, 2, 2, 2, 2}, 0xD001);
}
BT_PARITY_TEST(patch_unpack_rect) {
    run_shape({3, 5, 2, 4, 4}, 0xD002);
}
BT_PARITY_TEST(patch_unpack_pixart_drop) {
    // PixArt-Sigma: P=2, C_total=8 (eps+variance), keep first 4 (eps). 1024px
    // gives a 64x64 token grid; use a small grid here, block-major.
    run_shape({8, 8, 2, 8, 4}, 0xD003, /*channel_major=*/false);
}
BT_PARITY_TEST(patch_unpack_pixart_1024_grid) {
    // Full PixArt 1024px token grid (64x64), keep-half. Larger gather.
    run_shape({64, 64, 2, 8, 4}, 0xD004, /*channel_major=*/false);
}

} // namespace

int main() {
    return run_all("test_patch_unpack_parity");
}
