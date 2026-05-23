// CPU↔GPU parity tests for brotensor::l2_norm_forward / l2_norm_backward.
//
// Per-head, last-dim L2 normalisation used by the gated delta-rule text path.
// Layout is (L, num_heads * head_dim). FP32-only on both sides.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

void run_fwd(int L, int num_heads, int head_dim, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::l2_norm_forward(X, head_dim, num_heads, eps, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::l2_norm_forward(gX, head_dim, num_heads, eps, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y),
                    "l2_norm_fwd", 1e-5f, 1e-4f);
}

void run_bwd(int L, int num_heads, int head_dim, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(L, num_heads * head_dim);
    Tensor dY = Tensor::mat(L, num_heads * head_dim);
    fill_random(X,  rng);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::l2_norm_backward(X, head_dim, num_heads, eps, dY, cpu_dX);

    Tensor gX  = X.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::l2_norm_backward(gX, head_dim, num_heads, eps, gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX),
                    "l2_norm_bwd", 1e-4f, 1e-3f);
}

} // namespace

BT_PARITY_TEST(l2_norm_fwd_1h_8d)    { run_fwd(1, 1, 8,   1e-6f, 0x7300ull); }
BT_PARITY_TEST(l2_norm_fwd_4h_64d)   { run_fwd(7, 4, 64,  1e-6f, 0x7301ull); }
BT_PARITY_TEST(l2_norm_fwd_2h_128d)  { run_fwd(3, 2, 128, 1e-6f, 0x7302ull); }
BT_PARITY_TEST(l2_norm_fwd_odd_dim)  { run_fwd(5, 3, 17,  1e-6f, 0x7303ull); }

BT_PARITY_TEST(l2_norm_bwd_1h_8d)    { run_bwd(1, 1, 8,   1e-6f, 0x7310ull); }
BT_PARITY_TEST(l2_norm_bwd_4h_64d)   { run_bwd(7, 4, 64,  1e-6f, 0x7311ull); }
BT_PARITY_TEST(l2_norm_bwd_2h_128d)  { run_bwd(3, 2, 128, 1e-6f, 0x7312ull); }
BT_PARITY_TEST(l2_norm_bwd_odd_dim)  { run_bwd(5, 3, 17,  1e-6f, 0x7313ull); }

int main() { return run_all("l2_norm cpu/gpu parity"); }
