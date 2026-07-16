// CPU↔GPU parity tests for brotensor::l2_norm_forward / l2_norm_backward.
//
// Per-head, last-dim L2 normalisation used by the gated delta-rule text path.
// Layout is (L, num_heads * head_dim). CPU is FP32-only; the GPU backends
// dispatch on X.dtype (FP32 / FP16 / BF16) with FP32 math.

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

void run_fwd_fp16(int L, int num_heads, int head_dim, float eps,
                  uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);
    Tensor X_q = fp16_host_to_f32(to_fp16_host(X));   // shared quantised input

    Tensor cpu_Y;
    brotensor::l2_norm_forward(X_q, head_dim, num_heads, eps, cpu_Y);

    Tensor gX = to_fp16_gpu(X_q);
    Tensor gpu_Y;
    brotensor::l2_norm_forward(gX, head_dim, num_heads, eps, gpu_Y);

    compare_tensors(cpu_Y, fp16_host_to_f32(download_to_host(gpu_Y)),
                    "l2_norm_fp16_fwd", 1e-2f, 1e-2f);
}

void run_bwd_fp16(int L, int num_heads, int head_dim, float eps,
                  uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(L, num_heads * head_dim);
    Tensor dY = Tensor::mat(L, num_heads * head_dim);
    fill_random(X,  rng);
    fill_random(dY, rng);
    Tensor X_q  = fp16_host_to_f32(to_fp16_host(X));
    Tensor dY_q = fp16_host_to_f32(to_fp16_host(dY));

    Tensor cpu_dX;
    brotensor::l2_norm_backward(X_q, head_dim, num_heads, eps, dY_q, cpu_dX);

    Tensor gX  = to_fp16_gpu(X_q);
    Tensor gdY = to_fp16_gpu(dY_q);
    Tensor gpu_dX;
    brotensor::l2_norm_backward(gX, head_dim, num_heads, eps, gdY, gpu_dX);

    compare_tensors(cpu_dX, fp16_host_to_f32(download_to_host(gpu_dX)),
                    "l2_norm_fp16_bwd", 1e-2f, 1e-2f);
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

// FP16 storage — the TripoSplat flow-sampler shape class.
BT_PARITY_TEST(l2_norm_fp16_fwd_4h_64d) { run_fwd_fp16(7, 4, 64, 1e-6f, 0x7320ull); }
BT_PARITY_TEST(l2_norm_fp16_fwd_odd)    { run_fwd_fp16(5, 3, 17, 1e-6f, 0x7321ull); }
BT_PARITY_TEST(l2_norm_fp16_bwd_4h_64d) { run_bwd_fp16(7, 4, 64, 1e-6f, 0x7322ull); }

int main() { return run_all("l2_norm cpu/gpu parity"); }
