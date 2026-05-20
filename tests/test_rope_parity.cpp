// CPU↔GPU parity tests for brotensor::rope_forward and rope_backward.
//
// CHUNK 2. X / Y layout is (L, num_heads * head_dim). Tests cover seq_offset
// != 0, multiple heads, and a non-default theta_base.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_fwd(int L, int num_heads, int head_dim, int seq_offset,
             float theta_base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::rope_forward(X, head_dim, num_heads, seq_offset, theta_base, cpu_Y);

    Tensor gX = X.to(Device::CUDA);
    Tensor gpu_Y;
    brotensor::rope_forward(gX, head_dim, num_heads, seq_offset, theta_base, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "rope_fwd", 1e-4f, 1e-3f);
}

void run_bwd(int L, int num_heads, int head_dim, int seq_offset,
             float theta_base, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(L, num_heads * head_dim);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::rope_backward(dY, head_dim, num_heads, seq_offset, theta_base, cpu_dX);

    Tensor gdY = dY.to(Device::CUDA);
    Tensor gpu_dX;
    brotensor::rope_backward(gdY, head_dim, num_heads, seq_offset, theta_base, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "rope_bwd", 1e-4f, 1e-3f);
}

} // namespace

// ─── forward ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(rope_fwd_1h_off0)    { run_fwd(8, 1, 16, 0, 10000.0f, 0x6000ull); }
BT_PARITY_TEST(rope_fwd_4h_off0)    { run_fwd(12, 4, 8, 0, 10000.0f, 0x6001ull); }
BT_PARITY_TEST(rope_fwd_offset)     { run_fwd(6, 2, 16, 37, 10000.0f, 0x6002ull); }
BT_PARITY_TEST(rope_fwd_small_dim)  { run_fwd(5, 3, 2, 11, 10000.0f, 0x6003ull); }
BT_PARITY_TEST(rope_fwd_theta500)   { run_fwd(7, 2, 8, 4, 500.0f, 0x6004ull); }

// ─── backward ──────────────────────────────────────────────────────────────
BT_PARITY_TEST(rope_bwd_1h_off0)    { run_bwd(8, 1, 16, 0, 10000.0f, 0x6010ull); }
BT_PARITY_TEST(rope_bwd_4h_off0)    { run_bwd(12, 4, 8, 0, 10000.0f, 0x6011ull); }
BT_PARITY_TEST(rope_bwd_offset)     { run_bwd(6, 2, 16, 37, 10000.0f, 0x6012ull); }
BT_PARITY_TEST(rope_bwd_theta500)   { run_bwd(7, 2, 8, 4, 500.0f, 0x6013ull); }

int main() { return run_all("rope cpu/gpu parity"); }
