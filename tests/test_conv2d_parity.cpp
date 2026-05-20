// CPU↔GPU parity tests for the conv2d op family (CHUNK 3).
//
//   conv2d_forward          — Y  OVERWRITTEN.
//   conv2d_backward_input   — dX OVERWRITTEN.
//   conv2d_backward_weight  — dWt ACCUMULATES (+=); GPU folds an FP32 scratch
//                             into the caller's dWt. The accumulation tests
//                             pre-fill dWt with a non-zero baseline.
//   conv2d_backward_bias    — dB  ACCUMULATES (+=); likewise.
//
// NCHW activations; weight layout OIHW (C_out, C_in/groups, kH, kW). Groups
// convention: output channel oc belongs to group oc/(C_out/groups).
//
// Configs cover 1x1, 3x3 stride-1 pad-1, strided, dilated, and grouped conv
// (incl. depthwise where groups == C_in), with and without bias.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

struct ConvCfg {
    int N, C_in, H, W, C_out, kH, kW;
    int stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups;
};

inline int out_dim(int in, int pad, int dil, int k, int stride) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

// ─── forward ───────────────────────────────────────────────────────────────
void run_fwd(const ConvCfg& c, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    Tensor X = Tensor::mat(c.N, c.C_in * c.H * c.W);
    Tensor Wt = Tensor::mat(c.C_out, Cg_in * c.kH * c.kW);
    fill_random(X, rng, 0.5f);
    fill_random(Wt, rng, 0.5f);
    Tensor B;
    Tensor* Bp = nullptr;
    if (has_bias) {
        B = Tensor::vec(c.C_out);
        fill_random(B, rng, 0.5f);
        Bp = &B;
    }

    Tensor cpu_Y;
    brotensor::conv2d_forward(X, Wt, Bp, c.N, c.C_in, c.H, c.W, c.C_out,
                              c.kH, c.kW, c.stride_h, c.stride_w,
                              c.pad_h, c.pad_w, c.dil_h, c.dil_w, c.groups,
                              cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gW = Wt.to(gpu_device());
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) { gB = B.to(gpu_device()); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::conv2d_forward(gX, gW, gBp, c.N, c.C_in, c.H, c.W, c.C_out,
                              c.kH, c.kW, c.stride_h, c.stride_w,
                              c.pad_h, c.pad_w, c.dil_h, c.dil_w, c.groups,
                              gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "conv2d_fwd", 1e-4f, 1e-3f);
}

// ─── backward input ────────────────────────────────────────────────────────
void run_bwd_input(const ConvCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    const int H_out = out_dim(c.H, c.pad_h, c.dil_h, c.kH, c.stride_h);
    const int W_out = out_dim(c.W, c.pad_w, c.dil_w, c.kW, c.stride_w);
    Tensor Wt = Tensor::mat(c.C_out, Cg_in * c.kH * c.kW);
    Tensor dY = Tensor::mat(c.N, c.C_out * H_out * W_out);
    fill_random(Wt, rng, 0.5f);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::conv2d_backward_input(Wt, dY, c.N, c.C_in, c.H, c.W, c.C_out,
                                     c.kH, c.kW, c.stride_h, c.stride_w,
                                     c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                     c.groups, cpu_dX);

    Tensor gW = Wt.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::conv2d_backward_input(gW, gdY, c.N, c.C_in, c.H, c.W, c.C_out,
                                     c.kH, c.kW, c.stride_h, c.stride_w,
                                     c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                     c.groups, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "conv2d_bwd_dX",
                    1e-4f, 1e-3f);
}

// ─── backward weight — dWt accumulates; baseline pre-filled. ────────────────
void run_bwd_weight(const ConvCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    const int H_out = out_dim(c.H, c.pad_h, c.dil_h, c.kH, c.stride_h);
    const int W_out = out_dim(c.W, c.pad_w, c.dil_w, c.kW, c.stride_w);
    Tensor X = Tensor::mat(c.N, c.C_in * c.H * c.W);
    Tensor dY = Tensor::mat(c.N, c.C_out * H_out * W_out);
    Tensor dW0 = Tensor::mat(c.C_out, Cg_in * c.kH * c.kW);
    fill_random(X, rng, 0.5f);
    fill_random(dY, rng, 0.5f);
    fill_random(dW0, rng, 0.5f);   // non-zero baseline to verify += contract

    Tensor cpu_dW = dW0;           // deep copy
    brotensor::conv2d_backward_weight(X, dY, c.N, c.C_in, c.H, c.W, c.C_out,
                                      c.kH, c.kW, c.stride_h, c.stride_w,
                                      c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                      c.groups, cpu_dW);

    Tensor gX = X.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dW = dW0.to(gpu_device());   // same baseline on GPU
    brotensor::conv2d_backward_weight(gX, gdY, c.N, c.C_in, c.H, c.W, c.C_out,
                                      c.kH, c.kW, c.stride_h, c.stride_w,
                                      c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                      c.groups, gpu_dW);

    compare_tensors(cpu_dW, download_to_host(gpu_dW), "conv2d_bwd_dW",
                    1e-4f, 1e-3f);
}

// ─── backward bias — dB accumulates; baseline pre-filled. ───────────────────
void run_bwd_bias(int N, int C_out, int H_out, int W_out, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(N, C_out * H_out * W_out);
    Tensor dB0 = Tensor::vec(C_out);
    fill_random(dY, rng, 0.5f);
    fill_random(dB0, rng, 0.5f);   // non-zero baseline to verify += contract

    Tensor cpu_dB = dB0;           // deep copy
    brotensor::conv2d_backward_bias(dY, N, C_out, H_out, W_out, cpu_dB);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dB = dB0.to(gpu_device());   // same baseline on GPU
    brotensor::conv2d_backward_bias(gdY, N, C_out, H_out, W_out, gpu_dB);

    compare_tensors(cpu_dB, download_to_host(gpu_dB), "conv2d_bwd_dB",
                    1e-4f, 1e-3f);
}

// ─── BF16 forward (BF16-on-CUDA vs FP32 CPU reference) ──────────────────────
void run_fwd_bf16(const ConvCfg& c, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    Tensor X = Tensor::mat(c.N, c.C_in * c.H * c.W);
    Tensor Wt = Tensor::mat(c.C_out, Cg_in * c.kH * c.kW);
    fill_random(X, rng, 0.5f);
    fill_random(Wt, rng, 0.5f);
    Tensor B;
    Tensor* Bp = nullptr;
    if (has_bias) {
        B = Tensor::vec(c.C_out);
        fill_random(B, rng, 0.5f);
        Bp = &B;
    }

    Tensor cpu_Y;
    brotensor::conv2d_forward(X, Wt, Bp, c.N, c.C_in, c.H, c.W, c.C_out,
                              c.kH, c.kW, c.stride_h, c.stride_w,
                              c.pad_h, c.pad_w, c.dil_h, c.dil_w, c.groups,
                              cpu_Y);

    Tensor gX = to_bf16_cuda(X);
    Tensor gW = to_bf16_cuda(Wt);
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) { gB = to_bf16_cuda(B); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::conv2d_forward(gX, gW, gBp, c.N, c.C_in, c.H, c.W, c.C_out,
                              c.kH, c.kW, c.stride_h, c.stride_w,
                              c.pad_h, c.pad_w, c.dil_h, c.dil_w, c.groups,
                              gpu_Y);

    Tensor gpu_host = bf16_host_to_f32(download_to_host(gpu_Y));
    compare_tensors(cpu_Y, gpu_host, "conv2d_fwd_bf16", 5e-2f, 5e-2f);
}

// ─── BF16 backward input ────────────────────────────────────────────────────
void run_bwd_input_bf16(const ConvCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    const int H_out = out_dim(c.H, c.pad_h, c.dil_h, c.kH, c.stride_h);
    const int W_out = out_dim(c.W, c.pad_w, c.dil_w, c.kW, c.stride_w);
    Tensor Wt = Tensor::mat(c.C_out, Cg_in * c.kH * c.kW);
    Tensor dY = Tensor::mat(c.N, c.C_out * H_out * W_out);
    fill_random(Wt, rng, 0.5f);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::conv2d_backward_input(Wt, dY, c.N, c.C_in, c.H, c.W, c.C_out,
                                     c.kH, c.kW, c.stride_h, c.stride_w,
                                     c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                     c.groups, cpu_dX);

    Tensor gW = to_bf16_cuda(Wt);
    Tensor gdY = to_bf16_cuda(dY);
    Tensor gpu_dX;
    brotensor::conv2d_backward_input(gW, gdY, c.N, c.C_in, c.H, c.W, c.C_out,
                                     c.kH, c.kW, c.stride_h, c.stride_w,
                                     c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                     c.groups, gpu_dX);

    Tensor gpu_host = bf16_host_to_f32(download_to_host(gpu_dX));
    compare_tensors(cpu_dX, gpu_host, "conv2d_bwd_dX_bf16", 8e-2f, 8e-2f);
}

// ─── BF16 backward weight (accumulate) ──────────────────────────────────────
void run_bwd_weight_bf16(const ConvCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    const int H_out = out_dim(c.H, c.pad_h, c.dil_h, c.kH, c.stride_h);
    const int W_out = out_dim(c.W, c.pad_w, c.dil_w, c.kW, c.stride_w);
    Tensor X = Tensor::mat(c.N, c.C_in * c.H * c.W);
    Tensor dY = Tensor::mat(c.N, c.C_out * H_out * W_out);
    Tensor dW0 = Tensor::mat(c.C_out, Cg_in * c.kH * c.kW);
    fill_random(X, rng, 0.5f);
    fill_random(dY, rng, 0.5f);
    fill_random(dW0, rng, 0.5f);

    Tensor cpu_dW = dW0;
    brotensor::conv2d_backward_weight(X, dY, c.N, c.C_in, c.H, c.W, c.C_out,
                                      c.kH, c.kW, c.stride_h, c.stride_w,
                                      c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                      c.groups, cpu_dW);

    Tensor gX = to_bf16_cuda(X);
    Tensor gdY = to_bf16_cuda(dY);
    Tensor gpu_dW = to_bf16_cuda(dW0);
    brotensor::conv2d_backward_weight(gX, gdY, c.N, c.C_in, c.H, c.W, c.C_out,
                                      c.kH, c.kW, c.stride_h, c.stride_w,
                                      c.pad_h, c.pad_w, c.dil_h, c.dil_w,
                                      c.groups, gpu_dW);

    Tensor gpu_host = bf16_host_to_f32(download_to_host(gpu_dW));
    compare_tensors(cpu_dW, gpu_host, "conv2d_bwd_dW_bf16", 8e-2f, 8e-2f);
}

// ─── BF16 backward bias (accumulate) ────────────────────────────────────────
void run_bwd_bias_bf16(int N, int C_out, int H_out, int W_out, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(N, C_out * H_out * W_out);
    Tensor dB0 = Tensor::vec(C_out);
    fill_random(dY, rng, 0.5f);
    fill_random(dB0, rng, 0.5f);

    Tensor cpu_dB = dB0;
    brotensor::conv2d_backward_bias(dY, N, C_out, H_out, W_out, cpu_dB);

    Tensor gdY = to_bf16_cuda(dY);
    Tensor gpu_dB = to_bf16_cuda(dB0);
    brotensor::conv2d_backward_bias(gdY, N, C_out, H_out, W_out, gpu_dB);

    Tensor gpu_host = bf16_host_to_f32(download_to_host(gpu_dB));
    compare_tensors(cpu_dB, gpu_host, "conv2d_bwd_dB_bf16", 8e-2f, 8e-2f);
}

// Standard config bank — { N,Cin,H,W, Cout,kH,kW, sh,sw,ph,pw,dh,dw,groups }.
const ConvCfg k1x1     {2, 6, 7, 7,   8, 1, 1,  1,1, 0,0, 1,1, 1};
const ConvCfg k3x3_s1p1{2, 4, 8, 8,   6, 3, 3,  1,1, 1,1, 1,1, 1};
const ConvCfg k3x3_s2  {1, 3, 9, 9,   5, 3, 3,  2,2, 1,1, 1,1, 1};
const ConvCfg kDilated {1, 3, 11, 11, 4, 3, 3,  1,1, 2,2, 2,2, 1};
const ConvCfg kGrouped {2, 8, 6, 6,   8, 3, 3,  1,1, 1,1, 1,1, 4};
const ConvCfg kDepthw  {1, 6, 7, 7,   6, 3, 3,  1,1, 1,1, 1,1, 6};
const ConvCfg kAsym    {1, 4, 7, 9,   5, 3, 2,  2,1, 1,0, 1,1, 1};

} // namespace

// ─── forward ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(conv2d_fwd_1x1_bias)      { run_fwd(k1x1,      true,  0x6000ull); }
BT_PARITY_TEST(conv2d_fwd_1x1_nobias)    { run_fwd(k1x1,      false, 0x6001ull); }
BT_PARITY_TEST(conv2d_fwd_3x3_s1p1)      { run_fwd(k3x3_s1p1, true,  0x6002ull); }
BT_PARITY_TEST(conv2d_fwd_3x3_s2)        { run_fwd(k3x3_s2,   true,  0x6003ull); }
BT_PARITY_TEST(conv2d_fwd_dilated)       { run_fwd(kDilated,  false, 0x6004ull); }
BT_PARITY_TEST(conv2d_fwd_grouped)       { run_fwd(kGrouped,  true,  0x6005ull); }
BT_PARITY_TEST(conv2d_fwd_depthwise)     { run_fwd(kDepthw,   true,  0x6006ull); }
BT_PARITY_TEST(conv2d_fwd_asym)          { run_fwd(kAsym,     false, 0x6007ull); }

// ─── backward input ────────────────────────────────────────────────────────
BT_PARITY_TEST(conv2d_bwd_input_1x1)     { run_bwd_input(k1x1,      0x6010ull); }
BT_PARITY_TEST(conv2d_bwd_input_3x3)     { run_bwd_input(k3x3_s1p1, 0x6011ull); }
BT_PARITY_TEST(conv2d_bwd_input_s2)      { run_bwd_input(k3x3_s2,   0x6012ull); }
BT_PARITY_TEST(conv2d_bwd_input_dilated) { run_bwd_input(kDilated,  0x6013ull); }
BT_PARITY_TEST(conv2d_bwd_input_grouped) { run_bwd_input(kGrouped,  0x6014ull); }
BT_PARITY_TEST(conv2d_bwd_input_depthw)  { run_bwd_input(kDepthw,   0x6015ull); }
BT_PARITY_TEST(conv2d_bwd_input_asym)    { run_bwd_input(kAsym,     0x6016ull); }

// ─── backward weight (accumulate) ──────────────────────────────────────────
BT_PARITY_TEST(conv2d_bwd_weight_1x1)     { run_bwd_weight(k1x1,      0x6020ull); }
BT_PARITY_TEST(conv2d_bwd_weight_3x3)     { run_bwd_weight(k3x3_s1p1, 0x6021ull); }
BT_PARITY_TEST(conv2d_bwd_weight_s2)      { run_bwd_weight(k3x3_s2,   0x6022ull); }
BT_PARITY_TEST(conv2d_bwd_weight_dilated) { run_bwd_weight(kDilated,  0x6023ull); }
BT_PARITY_TEST(conv2d_bwd_weight_grouped) { run_bwd_weight(kGrouped,  0x6024ull); }
BT_PARITY_TEST(conv2d_bwd_weight_depthw)  { run_bwd_weight(kDepthw,   0x6025ull); }

// ─── backward bias (accumulate) ────────────────────────────────────────────
BT_PARITY_TEST(conv2d_bwd_bias_small)  { run_bwd_bias(2, 8, 4, 4, 0x6030ull); }
BT_PARITY_TEST(conv2d_bwd_bias_single) { run_bwd_bias(1, 5, 9, 9, 0x6031ull); }
BT_PARITY_TEST(conv2d_bwd_bias_wide)   { run_bwd_bias(3, 16, 5, 7, 0x6032ull); }

// ─── BF16 forward (BF16-on-CUDA vs FP32 CPU reference) ──────────────────────
BT_PARITY_TEST(conv2d_fwd_bf16_1x1_bias)   { run_fwd_bf16(k1x1,      true,  0x6100ull); }
BT_PARITY_TEST(conv2d_fwd_bf16_3x3_s1p1)   { run_fwd_bf16(k3x3_s1p1, true,  0x6101ull); }
BT_PARITY_TEST(conv2d_fwd_bf16_3x3_s2)     { run_fwd_bf16(k3x3_s2,   true,  0x6102ull); }
BT_PARITY_TEST(conv2d_fwd_bf16_dilated)    { run_fwd_bf16(kDilated,  false, 0x6103ull); }
BT_PARITY_TEST(conv2d_fwd_bf16_grouped)    { run_fwd_bf16(kGrouped,  true,  0x6104ull); }
BT_PARITY_TEST(conv2d_fwd_bf16_depthwise)  { run_fwd_bf16(kDepthw,   true,  0x6105ull); }
BT_PARITY_TEST(conv2d_fwd_bf16_asym)       { run_fwd_bf16(kAsym,     false, 0x6106ull); }

// ─── BF16 backward input ────────────────────────────────────────────────────
BT_PARITY_TEST(conv2d_bwd_input_bf16_1x1)     { run_bwd_input_bf16(k1x1,      0x6110ull); }
BT_PARITY_TEST(conv2d_bwd_input_bf16_3x3)     { run_bwd_input_bf16(k3x3_s1p1, 0x6111ull); }
BT_PARITY_TEST(conv2d_bwd_input_bf16_s2)      { run_bwd_input_bf16(k3x3_s2,   0x6112ull); }
BT_PARITY_TEST(conv2d_bwd_input_bf16_grouped) { run_bwd_input_bf16(kGrouped,  0x6113ull); }

// ─── BF16 backward weight (accumulate) ──────────────────────────────────────
BT_PARITY_TEST(conv2d_bwd_weight_bf16_1x1) { run_bwd_weight_bf16(k1x1,      0x6120ull); }
BT_PARITY_TEST(conv2d_bwd_weight_bf16_3x3) { run_bwd_weight_bf16(k3x3_s1p1, 0x6121ull); }
BT_PARITY_TEST(conv2d_bwd_weight_bf16_grp) { run_bwd_weight_bf16(kGrouped,  0x6122ull); }

// ─── BF16 backward bias (accumulate) ────────────────────────────────────────
BT_PARITY_TEST(conv2d_bwd_bias_bf16_small) { run_bwd_bias_bf16(2, 8, 4, 4, 0x6130ull); }
BT_PARITY_TEST(conv2d_bwd_bias_bf16_wide)  { run_bwd_bias_bf16(3, 16, 5, 7, 0x6131ull); }

int main() { return run_all("conv2d cpu/gpu parity"); }
