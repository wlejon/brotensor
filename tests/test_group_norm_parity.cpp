// CPU↔GPU parity tests for the group_norm op family (CHUNK 3).
//
//   group_norm_forward   — Y  OVERWRITTEN.
//   group_norm_backward  — dX OVERWRITTEN; dGamma / dBeta ACCUMULATE (+=).
//                          The GPU atomicAdds per-channel grads into FP32
//                          scratch then folds into the caller's dGamma/dBeta.
//                          The backward test pre-fills dGamma/dBeta with a
//                          non-zero baseline to verify the += contract.
//
// NCHW activations; per-(sample, group) mean/var normalization then per-channel
// affine. Covers several num_groups including groups == C (instance norm) and
// groups == 1 (layer norm over CHW).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

struct GnCfg {
    int N, C, H, W, num_groups;
};

constexpr float kEps = 1e-5f;

// ─── forward ───────────────────────────────────────────────────────────────
void run_fwd(const GnCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int spatial = c.H * c.W;
    Tensor X = Tensor::mat(c.N, c.C * spatial);
    Tensor gamma = Tensor::vec(c.C);
    Tensor beta = Tensor::vec(c.C);
    fill_random(X, rng);
    // gamma centered near 1, beta small — realistic affine params.
    for (int i = 0; i < gamma.size(); ++i) gamma[i] = 0.5f + rng.next_unit() * 0.5f;
    for (int i = 0; i < beta.size(); ++i)  beta[i]  = rng.next_unit() * 0.2f;

    Tensor cpu_Y;
    brotensor::group_norm_forward(X, gamma, beta, c.N, c.C, c.H, c.W,
                                  c.num_groups, kEps, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gG = gamma.to(gpu_device());
    Tensor gB = beta.to(gpu_device());
    Tensor gpu_Y;
    brotensor::group_norm_forward(gX, gG, gB, c.N, c.C, c.H, c.W,
                                  c.num_groups, kEps, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "group_norm_fwd",
                    1e-4f, 1e-3f);
}

// ─── backward — dX overwrites; dGamma/dBeta accumulate. ─────────────────────
void run_bwd(const GnCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int spatial = c.H * c.W;
    Tensor X = Tensor::mat(c.N, c.C * spatial);
    Tensor gamma = Tensor::vec(c.C);
    Tensor dY = Tensor::mat(c.N, c.C * spatial);
    Tensor dG0 = Tensor::vec(c.C);
    Tensor dB0 = Tensor::vec(c.C);
    fill_random(X, rng);
    fill_random(dY, rng);
    for (int i = 0; i < gamma.size(); ++i) gamma[i] = 0.5f + rng.next_unit() * 0.5f;
    fill_random(dG0, rng, 0.5f);   // non-zero baseline to verify += contract
    fill_random(dB0, rng, 0.5f);

    Tensor cpu_dX;
    Tensor cpu_dG = dG0;           // deep copy
    Tensor cpu_dB = dB0;
    brotensor::group_norm_backward(X, gamma, dY, c.N, c.C, c.H, c.W,
                                   c.num_groups, kEps,
                                   cpu_dX, cpu_dG, cpu_dB);

    Tensor gX = X.to(gpu_device());
    Tensor gG = gamma.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    Tensor gpu_dG = dG0.to(gpu_device());   // same baseline on GPU
    Tensor gpu_dB = dB0.to(gpu_device());
    brotensor::group_norm_backward(gX, gG, gdY, c.N, c.C, c.H, c.W,
                                   c.num_groups, kEps,
                                   gpu_dX, gpu_dG, gpu_dB);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "group_norm_bwd_dX",
                    1e-4f, 1e-3f);
    // Per-channel grads accumulate across N*H*W — relax the tolerance.
    compare_tensors(cpu_dG, download_to_host(gpu_dG), "group_norm_bwd_dGamma",
                    1e-3f, 1e-3f);
    compare_tensors(cpu_dB, download_to_host(gpu_dB), "group_norm_bwd_dBeta",
                    1e-3f, 1e-3f);
}

// Config bank — { N, C, H, W, num_groups }.
const GnCfg kTiny     {1, 4,  2, 2, 2};
const GnCfg kStandard {2, 8,  4, 4, 4};
const GnCfg kGroups1  {1, 6,  3, 3, 1};   // layer norm over CHW
const GnCfg kGroupsC  {1, 4,  5, 5, 4};   // instance norm (groups == C)
const GnCfg kSDish    {2, 32, 6, 6, 8};
const GnCfg kRect     {1, 6,  3, 7, 3};

} // namespace

// ─── forward ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(group_norm_fwd_tiny)      { run_fwd(kTiny,     0x7000ull); }
BT_PARITY_TEST(group_norm_fwd_standard)  { run_fwd(kStandard, 0x7001ull); }
BT_PARITY_TEST(group_norm_fwd_groups1)   { run_fwd(kGroups1,  0x7002ull); }
BT_PARITY_TEST(group_norm_fwd_groupsC)   { run_fwd(kGroupsC,  0x7003ull); }
BT_PARITY_TEST(group_norm_fwd_sdish)     { run_fwd(kSDish,    0x7004ull); }
BT_PARITY_TEST(group_norm_fwd_rect)      { run_fwd(kRect,     0x7005ull); }

// ─── backward (dGamma/dBeta accumulate) ────────────────────────────────────
BT_PARITY_TEST(group_norm_bwd_tiny)      { run_bwd(kTiny,     0x7010ull); }
BT_PARITY_TEST(group_norm_bwd_standard)  { run_bwd(kStandard, 0x7011ull); }
BT_PARITY_TEST(group_norm_bwd_groups1)   { run_bwd(kGroups1,  0x7012ull); }
BT_PARITY_TEST(group_norm_bwd_groupsC)   { run_bwd(kGroupsC,  0x7013ull); }
BT_PARITY_TEST(group_norm_bwd_sdish)     { run_bwd(kSDish,    0x7014ull); }
BT_PARITY_TEST(group_norm_bwd_rect)      { run_bwd(kRect,     0x7015ull); }

// ─── BF16 forward + backward parity ─────────────────────────────────────────
// BF16 is GPU-only. Round FP32 inputs to BF16, run on CUDA, widen back and
// compare against the FP32 CPU reference with loose tolerances.

namespace {

void run_fwd_bf16(const GnCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int spatial = c.H * c.W;
    Tensor X_f32 = Tensor::mat(c.N, c.C * spatial);
    Tensor gamma_f32 = Tensor::vec(c.C);
    Tensor beta_f32 = Tensor::vec(c.C);
    fill_random(X_f32, rng);
    for (int i = 0; i < gamma_f32.size(); ++i) gamma_f32[i] = 0.5f + rng.next_unit() * 0.5f;
    for (int i = 0; i < beta_f32.size(); ++i)  beta_f32[i]  = rng.next_unit() * 0.2f;

    // CPU FP32 reference.
    Tensor cpu_Y;
    brotensor::group_norm_forward(X_f32, gamma_f32, beta_f32, c.N, c.C, c.H, c.W,
                                  c.num_groups, kEps, cpu_Y);

    // BF16 GPU path.
    Tensor gX     = to_bf16_cuda(X_f32);
    Tensor gGamma = to_bf16_cuda(gamma_f32);
    Tensor gBeta  = to_bf16_cuda(beta_f32);
    Tensor gpu_Y;
    brotensor::group_norm_forward(gX, gGamma, gBeta, c.N, c.C, c.H, c.W,
                                  c.num_groups, kEps, gpu_Y);
    brotensor::sync_all();

    Tensor Y_h = bf16_host_to_f32(download_to_host(gpu_Y));
    compare_tensors(cpu_Y, Y_h, "group_norm_bf16_fwd", 3e-2f, 3e-2f);
}

void run_bwd_bf16(const GnCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int spatial = c.H * c.W;
    Tensor X_f32 = Tensor::mat(c.N, c.C * spatial);
    Tensor gamma_f32 = Tensor::vec(c.C);
    Tensor dY_f32 = Tensor::mat(c.N, c.C * spatial);
    Tensor dG0_f32 = Tensor::vec(c.C);
    Tensor dB0_f32 = Tensor::vec(c.C);
    fill_random(X_f32, rng);
    fill_random(dY_f32, rng);
    for (int i = 0; i < gamma_f32.size(); ++i) gamma_f32[i] = 0.5f + rng.next_unit() * 0.5f;
    fill_random(dG0_f32, rng, 0.5f);
    fill_random(dB0_f32, rng, 0.5f);

    // CPU FP32 reference.
    Tensor cpu_dX;
    Tensor cpu_dG = dG0_f32;
    Tensor cpu_dB = dB0_f32;
    brotensor::group_norm_backward(X_f32, gamma_f32, dY_f32, c.N, c.C, c.H, c.W,
                                   c.num_groups, kEps, cpu_dX, cpu_dG, cpu_dB);

    // BF16 GPU path.
    Tensor gX     = to_bf16_cuda(X_f32);
    Tensor gG     = to_bf16_cuda(gamma_f32);
    Tensor gdY    = to_bf16_cuda(dY_f32);
    Tensor gpu_dX;
    Tensor gpu_dG = Tensor::zeros_on(Device::CUDA, c.C, 1, brotensor::Dtype::BF16);
    Tensor gpu_dB = Tensor::zeros_on(Device::CUDA, c.C, 1, brotensor::Dtype::BF16);
    // Pre-load the non-zero baseline into BF16 accumulator buffers.
    {
        Tensor dG0_bf16 = to_bf16_cuda(dG0_f32);
        Tensor dB0_bf16 = to_bf16_cuda(dB0_f32);
        gpu_dG = dG0_bf16;
        gpu_dB = dB0_bf16;
    }
    brotensor::group_norm_backward(gX, gG, gdY, c.N, c.C, c.H, c.W,
                                   c.num_groups, kEps, gpu_dX, gpu_dG, gpu_dB);
    brotensor::sync_all();

    Tensor dX_h = bf16_host_to_f32(download_to_host(gpu_dX));
    Tensor dG_h = bf16_host_to_f32(download_to_host(gpu_dG));
    Tensor dB_h = bf16_host_to_f32(download_to_host(gpu_dB));

    compare_tensors(cpu_dX, dX_h, "group_norm_bf16_bwd_dX",     3e-2f, 3e-2f);
    compare_tensors(cpu_dG, dG_h, "group_norm_bf16_bwd_dGamma", 6e-2f, 6e-2f);
    compare_tensors(cpu_dB, dB_h, "group_norm_bf16_bwd_dBeta",  6e-2f, 6e-2f);
}

} // namespace (BF16 helpers)

BT_PARITY_TEST(group_norm_bf16_fwd_tiny)     { run_fwd_bf16(kTiny,     0x7080ull); }
BT_PARITY_TEST(group_norm_bf16_fwd_standard) { run_fwd_bf16(kStandard, 0x7081ull); }
BT_PARITY_TEST(group_norm_bf16_fwd_sdish)    { run_fwd_bf16(kSDish,    0x7082ull); }

BT_PARITY_TEST(group_norm_bf16_bwd_tiny)     { run_bwd_bf16(kTiny,     0x7090ull); }
BT_PARITY_TEST(group_norm_bf16_bwd_standard) { run_bwd_bf16(kStandard, 0x7091ull); }
BT_PARITY_TEST(group_norm_bf16_bwd_sdish)    { run_bwd_bf16(kSDish,    0x7092ull); }

int main() { return run_all("group_norm cpu/gpu parity"); }
