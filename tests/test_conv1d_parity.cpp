// CPU↔GPU parity tests for the brosoundml 1D-convolution family (CHUNK 3).
//
//   conv_transpose1d_forward          — Y  OVERWRITTEN.
//   conv_transpose1d_backward_input   — dX OVERWRITTEN.
//   conv_transpose1d_backward_weight  — dWt ACCUMULATES (+=); the accumulation
//                                       tests pre-fill dWt with a non-zero
//                                       baseline to verify the contract.
//   conv_transpose1d_backward_bias    — dB  ACCUMULATES (+=); likewise.
//   causal_conv1d_update              — Y OVERWRITTEN; `state` rolled in place,
//                                       so the test compares both Y and state.
//   pad1d_forward / pad1d_backward    — zero / reflect / replicate modes.
//
// NCL activations; conv_transpose1d weights are OIL (C_in, (C_out/groups)*kL);
// causal_conv1d_update weights are depthwise (C, kL). FP32-only on every
// backend — there is no BF16 path for these audio ops.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-3f;

struct ConvTCfg {
    int N, C_in, L, C_out, kL;
    int stride, padding, output_padding, dilation, groups;
};

// torch ConvTranspose1d output-length formula.
int convt1d_out_len(const ConvTCfg& c) {
    return (c.L - 1) * c.stride - 2 * c.padding
           + c.dilation * (c.kL - 1) + c.output_padding + 1;
}

// ─── conv_transpose1d_forward ──────────────────────────────────────────────
void run_convt_fwd(const ConvTCfg& c, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_out = c.C_out / c.groups;
    Tensor X  = Tensor::mat(c.N, c.C_in * c.L);
    Tensor Wt = Tensor::mat(c.C_in, Cg_out * c.kL);
    fill_random(X, rng, 0.5f);
    fill_random(Wt, rng, 0.5f);
    Tensor B;
    Tensor* Bp = nullptr;
    if (has_bias) { B = Tensor::vec(c.C_out); fill_random(B, rng, 0.5f); Bp = &B; }

    Tensor cpu_Y;
    brotensor::conv_transpose1d_forward(X, Wt, Bp, c.N, c.C_in, c.L, c.C_out,
                                        c.kL, c.stride, c.padding,
                                        c.output_padding, c.dilation, c.groups,
                                        cpu_Y);

    Tensor gX = X.to(gpu_device()), gW = Wt.to(gpu_device());
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) { gB = B.to(gpu_device()); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::conv_transpose1d_forward(gX, gW, gBp, c.N, c.C_in, c.L, c.C_out,
                                        c.kL, c.stride, c.padding,
                                        c.output_padding, c.dilation, c.groups,
                                        gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "convt1d_fwd",
                    kAtol, kRtol);
}

// ─── conv_transpose1d_backward_input ───────────────────────────────────────
void run_convt_bwd_input(const ConvTCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_out = c.C_out / c.groups;
    const int L_out = convt1d_out_len(c);
    Tensor Wt = Tensor::mat(c.C_in, Cg_out * c.kL);
    Tensor dY = Tensor::mat(c.N, c.C_out * L_out);
    fill_random(Wt, rng, 0.5f);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::conv_transpose1d_backward_input(Wt, dY, c.N, c.C_in, c.L,
                                               c.C_out, c.kL, c.stride,
                                               c.padding, c.output_padding,
                                               c.dilation, c.groups, cpu_dX);

    Tensor gW = Wt.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::conv_transpose1d_backward_input(gW, gdY, c.N, c.C_in, c.L,
                                               c.C_out, c.kL, c.stride,
                                               c.padding, c.output_padding,
                                               c.dilation, c.groups, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "convt1d_bwd_dX",
                    kAtol, kRtol);
}

// ─── conv_transpose1d_backward_weight — dWt accumulates ────────────────────
void run_convt_bwd_weight(const ConvTCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_out = c.C_out / c.groups;
    const int L_out = convt1d_out_len(c);
    Tensor X  = Tensor::mat(c.N, c.C_in * c.L);
    Tensor dY = Tensor::mat(c.N, c.C_out * L_out);
    Tensor dW0 = Tensor::mat(c.C_in, Cg_out * c.kL);
    fill_random(X, rng, 0.5f);
    fill_random(dY, rng, 0.5f);
    fill_random(dW0, rng, 0.5f);   // non-zero baseline to verify += contract

    Tensor cpu_dW = dW0;           // deep copy
    brotensor::conv_transpose1d_backward_weight(X, dY, c.N, c.C_in, c.L,
                                                c.C_out, c.kL, c.stride,
                                                c.padding, c.output_padding,
                                                c.dilation, c.groups, cpu_dW);

    Tensor gX = X.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gpu_dW = dW0.to(gpu_device());   // same baseline on GPU
    brotensor::conv_transpose1d_backward_weight(gX, gdY, c.N, c.C_in, c.L,
                                                c.C_out, c.kL, c.stride,
                                                c.padding, c.output_padding,
                                                c.dilation, c.groups, gpu_dW);

    compare_tensors(cpu_dW, download_to_host(gpu_dW), "convt1d_bwd_dW",
                    kAtol, kRtol);
}

// ─── conv_transpose1d_backward_bias — dB accumulates ───────────────────────
void run_convt_bwd_bias(const ConvTCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int L_out = convt1d_out_len(c);
    Tensor dY = Tensor::mat(c.N, c.C_out * L_out);
    Tensor dB0 = Tensor::vec(c.C_out);
    fill_random(dY, rng, 0.5f);
    fill_random(dB0, rng, 0.5f);   // non-zero baseline to verify += contract

    Tensor cpu_dB = dB0;           // deep copy
    brotensor::conv_transpose1d_backward_bias(dY, c.N, c.C_out, L_out, cpu_dB);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dB = dB0.to(gpu_device());   // same baseline on GPU
    brotensor::conv_transpose1d_backward_bias(gdY, c.N, c.C_out, L_out, gpu_dB);

    compare_tensors(cpu_dB, download_to_host(gpu_dB), "convt1d_bwd_dB",
                    kAtol, kRtol);
}

// ─── causal_conv1d_update — compares both Y and the rolled state ───────────
void run_causal(int N, int C, int L_step, int kL, int dilation,
                bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int hist = (kL - 1) * dilation;
    Tensor X  = Tensor::mat(N, C * L_step);
    Tensor Wt = Tensor::mat(C, kL);
    Tensor S0 = Tensor::mat(N, C * hist);
    fill_random(X, rng, 0.5f);
    fill_random(Wt, rng, 0.5f);
    fill_random(S0, rng, 0.5f);
    Tensor B;
    Tensor* Bp = nullptr;
    if (has_bias) { B = Tensor::vec(C); fill_random(B, rng, 0.5f); Bp = &B; }

    Tensor cpu_state = S0;         // deep copy — rolled in place
    Tensor cpu_Y;
    brotensor::causal_conv1d_update(X, Wt, Bp, N, C, L_step, kL, dilation,
                                    cpu_state, cpu_Y);

    Tensor gX = X.to(gpu_device()), gW = Wt.to(gpu_device());
    Tensor gpu_state = S0.to(gpu_device());
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) { gB = B.to(gpu_device()); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::causal_conv1d_update(gX, gW, gBp, N, C, L_step, kL, dilation,
                                    gpu_state, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "causal_conv1d_Y",
                    kAtol, kRtol);
    compare_tensors(cpu_state, download_to_host(gpu_state),
                    "causal_conv1d_state", kAtol, kRtol);
}

// ─── pad1d_forward ─────────────────────────────────────────────────────────
void run_pad_fwd(int N, int C, int L, int pl, int pr, int mode,
                 uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * L);
    fill_random(X, rng, 0.5f);

    Tensor cpu_Y;
    brotensor::pad1d_forward(X, N, C, L, pl, pr, mode, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::pad1d_forward(gX, N, C, L, pl, pr, mode, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "pad1d_fwd", kAtol, kRtol);
}

// ─── pad1d_backward ────────────────────────────────────────────────────────
void run_pad_bwd(int N, int C, int L, int pl, int pr, int mode,
                 uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(N, C * (L + pl + pr));
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::pad1d_backward(dY, N, C, L, pl, pr, mode, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::pad1d_backward(gdY, N, C, L, pl, pr, mode, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "pad1d_bwd",
                    kAtol, kRtol);
}

// Standard config bank — { N,Cin,L, Cout,kL, stride,pad,out_pad,dil,groups }.
const ConvTCfg kPlain   {2, 3, 6,  4, 3,  1, 0, 0, 1, 1};
const ConvTCfg kStride2 {1, 3, 5,  5, 4,  2, 1, 1, 1, 1};
const ConvTCfg kPadded  {2, 4, 7,  6, 3,  1, 1, 0, 1, 1};
const ConvTCfg kDilated {1, 3, 8,  4, 3,  1, 0, 0, 2, 1};
const ConvTCfg kGrouped {2, 6, 6,  8, 3,  2, 1, 0, 1, 2};

} // namespace

// ─── conv_transpose1d_forward ──────────────────────────────────────────────
BT_PARITY_TEST(convt1d_fwd_plain_bias)   { run_convt_fwd(kPlain,   true,  0x7000ull); }
BT_PARITY_TEST(convt1d_fwd_plain_nobias) { run_convt_fwd(kPlain,   false, 0x7001ull); }
BT_PARITY_TEST(convt1d_fwd_stride2)      { run_convt_fwd(kStride2, true,  0x7002ull); }
BT_PARITY_TEST(convt1d_fwd_padded)       { run_convt_fwd(kPadded,  false, 0x7003ull); }
BT_PARITY_TEST(convt1d_fwd_dilated)      { run_convt_fwd(kDilated, true,  0x7004ull); }
BT_PARITY_TEST(convt1d_fwd_grouped)      { run_convt_fwd(kGrouped, true,  0x7005ull); }

// ─── conv_transpose1d_backward_input ───────────────────────────────────────
BT_PARITY_TEST(convt1d_bwd_input_plain)   { run_convt_bwd_input(kPlain,   0x7010ull); }
BT_PARITY_TEST(convt1d_bwd_input_stride2) { run_convt_bwd_input(kStride2, 0x7011ull); }
BT_PARITY_TEST(convt1d_bwd_input_padded)  { run_convt_bwd_input(kPadded,  0x7012ull); }
BT_PARITY_TEST(convt1d_bwd_input_dilated) { run_convt_bwd_input(kDilated, 0x7013ull); }
BT_PARITY_TEST(convt1d_bwd_input_grouped) { run_convt_bwd_input(kGrouped, 0x7014ull); }

// ─── conv_transpose1d_backward_weight (accumulate) ─────────────────────────
BT_PARITY_TEST(convt1d_bwd_weight_plain)   { run_convt_bwd_weight(kPlain,   0x7020ull); }
BT_PARITY_TEST(convt1d_bwd_weight_stride2) { run_convt_bwd_weight(kStride2, 0x7021ull); }
BT_PARITY_TEST(convt1d_bwd_weight_dilated) { run_convt_bwd_weight(kDilated, 0x7022ull); }
BT_PARITY_TEST(convt1d_bwd_weight_grouped) { run_convt_bwd_weight(kGrouped, 0x7023ull); }

// ─── conv_transpose1d_backward_bias (accumulate) ───────────────────────────
BT_PARITY_TEST(convt1d_bwd_bias_plain)   { run_convt_bwd_bias(kPlain,   0x7030ull); }
BT_PARITY_TEST(convt1d_bwd_bias_stride2) { run_convt_bwd_bias(kStride2, 0x7031ull); }
BT_PARITY_TEST(convt1d_bwd_bias_grouped) { run_convt_bwd_bias(kGrouped, 0x7032ull); }

// ─── causal_conv1d_update ──────────────────────────────────────────────────
BT_PARITY_TEST(causal_conv1d_k3_bias)   { run_causal(2, 4, 5,  3, 1, true,  0x7040ull); }
BT_PARITY_TEST(causal_conv1d_k3_nobias) { run_causal(2, 4, 5,  3, 1, false, 0x7041ull); }
BT_PARITY_TEST(causal_conv1d_k4_dil2)   { run_causal(1, 3, 6,  4, 2, true,  0x7042ull); }
BT_PARITY_TEST(causal_conv1d_step1)     { run_causal(3, 8, 1,  5, 1, true,  0x7043ull); }

// ─── pad1d_forward ─────────────────────────────────────────────────────────
BT_PARITY_TEST(pad1d_fwd_zero)      { run_pad_fwd(2, 3, 7, 2, 3, 0, 0x7050ull); }
BT_PARITY_TEST(pad1d_fwd_reflect)   { run_pad_fwd(2, 3, 7, 3, 2, 1, 0x7051ull); }
BT_PARITY_TEST(pad1d_fwd_replicate) { run_pad_fwd(1, 4, 5, 4, 4, 2, 0x7052ull); }

// ─── pad1d_backward ────────────────────────────────────────────────────────
BT_PARITY_TEST(pad1d_bwd_zero)      { run_pad_bwd(2, 3, 7, 2, 3, 0, 0x7060ull); }
BT_PARITY_TEST(pad1d_bwd_reflect)   { run_pad_bwd(2, 3, 7, 3, 2, 1, 0x7061ull); }
BT_PARITY_TEST(pad1d_bwd_replicate) { run_pad_bwd(1, 4, 5, 4, 4, 2, 0x7062ull); }

int main() { return run_all("conv1d cpu/gpu parity"); }
