// CPU↔GPU parity for spatial_merge_2x2_forward (pure gather, bit-exact FP32).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

struct Shape { int N, C, H, W; };

void run_shape(const Shape& s, uint64_t seed, bool channel_major) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(s.N, s.C * s.H * s.W);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::spatial_merge_2x2_forward(X, s.N, s.C, s.H, s.W, channel_major, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::spatial_merge_2x2_forward(gX, s.N, s.C, s.H, s.W, channel_major, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y),
                    "spatial_merge_2x2", 0.0f, 0.0f);
}

// Both channel orderings exercise the same gather, so run each shape twice.
void run_shape(const Shape& s, uint64_t seed) {
    run_shape(s, seed, /*channel_major=*/false);
    run_shape(s, seed ^ 0xABCD, /*channel_major=*/true);
}

BT_PARITY_TEST(spatial_merge_small) {
    run_shape({1, 2, 4, 4}, 0xC001);
}
BT_PARITY_TEST(spatial_merge_rect) {
    run_shape({2, 3, 4, 6}, 0xC002);
}
BT_PARITY_TEST(spatial_merge_more_chan) {
    run_shape({1, 5, 6, 6}, 0xC003);
}
BT_PARITY_TEST(spatial_merge_batched) {
    run_shape({3, 4, 8, 8}, 0xC004);
}
BT_PARITY_TEST(spatial_merge_qwenvl_like) {
    // 16x16 patch grid with 64 channels — small but VL-shaped.
    run_shape({1, 64, 16, 16}, 0xC005);
}
BT_PARITY_TEST(spatial_merge_flux2_like) {
    // 32 latent channels, channel-major pixel-unshuffle (Flux.2 VAE tail).
    run_shape({1, 32, 32, 32}, 0xC006, /*channel_major=*/true);
}

} // namespace

int main() {
    return run_all("test_spatial_merge_parity");
}
