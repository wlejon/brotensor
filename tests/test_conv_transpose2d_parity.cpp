// CPU↔GPU parity tests for conv_transpose2d_* (forward + 3 backwards).
//
// dWt / dB ACCUMULATE — tests pre-fill them with a non-zero baseline to
// verify the += contract on Metal.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

struct CT2 {
    int N, C_in, H, W, C_out, kH, kW;
    int stride_h, stride_w, pad_h, pad_w;
    int output_padding_h, output_padding_w;
    int dil_h, dil_w, groups;
};

inline int out_dim(int L, int stride, int padding, int output_padding,
                   int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

void run_fwd(const CT2& c, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_out = c.C_out / c.groups;
    Tensor X = Tensor::mat(c.N, c.C_in * c.H * c.W);
    Tensor Wt = Tensor::mat(c.C_in, Cg_out * c.kH * c.kW);
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
    brotensor::conv_transpose2d_forward(X, Wt, Bp, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gW = Wt.to(gpu_device());
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) { gB = B.to(gpu_device()); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::conv_transpose2d_forward(gX, gW, gBp, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "ct2d_fwd",
                    1e-4f, 1e-3f);
}

void run_bwd_input(const CT2& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_out = c.C_out / c.groups;
    const int H_out = out_dim(c.H, c.stride_h, c.pad_h, c.output_padding_h,
                              c.dil_h, c.kH);
    const int W_out = out_dim(c.W, c.stride_w, c.pad_w, c.output_padding_w,
                              c.dil_w, c.kW);
    Tensor Wt = Tensor::mat(c.C_in, Cg_out * c.kH * c.kW);
    Tensor dY = Tensor::mat(c.N, c.C_out * H_out * W_out);
    fill_random(Wt, rng, 0.5f);
    fill_random(dY, rng, 0.5f);

    Tensor cpu_dX;
    brotensor::conv_transpose2d_backward_input(Wt, dY, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, cpu_dX);

    Tensor gW = Wt.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::conv_transpose2d_backward_input(gW, gdY, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "ct2d_bwd_dX",
                    1e-4f, 1e-3f);
}

void run_bwd_weight(const CT2& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_out = c.C_out / c.groups;
    const int H_out = out_dim(c.H, c.stride_h, c.pad_h, c.output_padding_h,
                              c.dil_h, c.kH);
    const int W_out = out_dim(c.W, c.stride_w, c.pad_w, c.output_padding_w,
                              c.dil_w, c.kW);
    Tensor X = Tensor::mat(c.N, c.C_in * c.H * c.W);
    Tensor dY = Tensor::mat(c.N, c.C_out * H_out * W_out);
    Tensor dW0 = Tensor::mat(c.C_in, Cg_out * c.kH * c.kW);
    fill_random(X, rng, 0.5f);
    fill_random(dY, rng, 0.5f);
    fill_random(dW0, rng, 0.5f);

    Tensor cpu_dW = dW0;  // deep copy
    brotensor::conv_transpose2d_backward_weight(X, dY, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, cpu_dW);

    Tensor gX = X.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dW = dW0.to(gpu_device());
    brotensor::conv_transpose2d_backward_weight(gX, gdY, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, gpu_dW);

    compare_tensors(cpu_dW, download_to_host(gpu_dW), "ct2d_bwd_dW",
                    1e-4f, 1e-3f);
}

void run_bwd_bias(const CT2& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int H_out = out_dim(c.H, c.stride_h, c.pad_h, c.output_padding_h,
                              c.dil_h, c.kH);
    const int W_out = out_dim(c.W, c.stride_w, c.pad_w, c.output_padding_w,
                              c.dil_w, c.kW);
    Tensor dY = Tensor::mat(c.N, c.C_out * H_out * W_out);
    Tensor dB0 = Tensor::vec(c.C_out);
    fill_random(dY, rng, 0.5f);
    fill_random(dB0, rng, 0.5f);

    Tensor cpu_dB = dB0;
    brotensor::conv_transpose2d_backward_bias(dY, c.N, c.C_out, H_out, W_out,
                                              cpu_dB);

    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dB = dB0.to(gpu_device());
    brotensor::conv_transpose2d_backward_bias(gdY, c.N, c.C_out, H_out, W_out,
                                              gpu_dB);

    compare_tensors(cpu_dB, download_to_host(gpu_dB), "ct2d_bwd_dB",
                    1e-4f, 1e-3f);
}

// ── FP16 / BF16 forward (low-precision-on-GPU vs FP32 CPU reference) ───────
// CUDA-only paths: the CPU backend stays FP32, so the reference runs FP32 and
// the GPU runs the same values rounded to 16-bit storage.
void run_fwd_16(const CT2& c, bool has_bias, bool bf16, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_out = c.C_out / c.groups;
    Tensor X = Tensor::mat(c.N, c.C_in * c.H * c.W);
    Tensor Wt = Tensor::mat(c.C_in, Cg_out * c.kH * c.kW);
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
    brotensor::conv_transpose2d_forward(X, Wt, Bp, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, cpu_Y);

    Tensor gX = bf16 ? to_bf16_gpu(X) : to_fp16_gpu(X);
    Tensor gW = bf16 ? to_bf16_gpu(Wt) : to_fp16_gpu(Wt);
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) {
        gB = bf16 ? to_bf16_gpu(B) : to_fp16_gpu(B);
        gBp = &gB;
    }
    Tensor gpu_Y;
    brotensor::conv_transpose2d_forward(gX, gW, gBp, c.N, c.C_in, c.H, c.W,
        c.C_out, c.kH, c.kW, c.stride_h, c.stride_w,
        c.pad_h, c.pad_w, c.output_padding_h, c.output_padding_w,
        c.dil_h, c.dil_w, c.groups, gpu_Y);

    Tensor host = download_to_host(gpu_Y);
    Tensor wide = bf16 ? bf16_host_to_f32(host) : fp16_host_to_f32(host);
    compare_tensors(cpu_Y, wide, bf16 ? "ct2d_fwd_bf16" : "ct2d_fwd_fp16",
                    bf16 ? 5e-2f : 2e-2f, bf16 ? 5e-2f : 2e-2f);
}

// brovisionml shapes (scaled spatially so the FP32 CPU reference stays fast):
// lineart upsamplers 3x3 s2 p1 op1 (256->128 and 128->64) and the SAM/DPT
// 2x2 s2 upscaler.
const CT2 kLineartUp1 {1, 256, 12, 14, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1};
const CT2 kLineartUp2 {1, 128, 17, 15,  64, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1};
const CT2 kUp2x2      {1, 256, 11, 13,  64, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1};

} // namespace

// ── FP16 / BF16 forward ─────────────────────────────────────────────────────
BT_PARITY_TEST(ct2d_fwd_fp16_lineart_up1) { run_fwd_16(kLineartUp1, true,  false, 0xC640ull); }
BT_PARITY_TEST(ct2d_fwd_fp16_lineart_up2) { run_fwd_16(kLineartUp2, true,  false, 0xC641ull); }
BT_PARITY_TEST(ct2d_fwd_fp16_up2x2)       { run_fwd_16(kUp2x2,      false, false, 0xC642ull); }
BT_PARITY_TEST(ct2d_fwd_bf16_lineart_up1) { run_fwd_16(kLineartUp1, true,  true,  0xC643ull); }
BT_PARITY_TEST(ct2d_fwd_bf16_up2x2)       { run_fwd_16(kUp2x2,      true,  true,  0xC644ull); }

// ── forward ────────────────────────────────────────────────────────────────
// {N, C_in, H, W, C_out, kH, kW, sh, sw, ph, pw, oph, opw, dh, dw, groups}
BT_PARITY_TEST(ct2d_fwd_2x_stride2) {
    run_fwd({2, 4, 5, 6, 3, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, 1}, true, 0xC600ull);
}
BT_PARITY_TEST(ct2d_fwd_3x3s1) {
    run_fwd({2, 3, 6, 7, 4, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1, 1}, false, 0xC601ull);
}
BT_PARITY_TEST(ct2d_fwd_outpad) {
    run_fwd({1, 2, 4, 5, 2, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1}, true, 0xC602ull);
}
BT_PARITY_TEST(ct2d_fwd_dil) {
    run_fwd({1, 2, 6, 6, 2, 3, 3, 1, 1, 1, 1, 0, 0, 2, 2, 1}, false, 0xC603ull);
}
BT_PARITY_TEST(ct2d_fwd_groups) {
    run_fwd({1, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2}, true, 0xC604ull);
}

// ── backward_input ────────────────────────────────────────────────────────
BT_PARITY_TEST(ct2d_bwd_input_2x) {
    run_bwd_input({2, 4, 5, 6, 3, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, 1}, 0xC610ull);
}
BT_PARITY_TEST(ct2d_bwd_input_outpad) {
    run_bwd_input({1, 2, 4, 5, 2, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1}, 0xC611ull);
}
BT_PARITY_TEST(ct2d_bwd_input_groups) {
    run_bwd_input({1, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2}, 0xC612ull);
}

// ── backward_weight (dWt += ) ─────────────────────────────────────────────
BT_PARITY_TEST(ct2d_bwd_weight_2x) {
    run_bwd_weight({2, 4, 5, 6, 3, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, 1}, 0xC620ull);
}
BT_PARITY_TEST(ct2d_bwd_weight_3x3s1) {
    run_bwd_weight({2, 3, 6, 7, 4, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1, 1}, 0xC621ull);
}
BT_PARITY_TEST(ct2d_bwd_weight_groups) {
    run_bwd_weight({1, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2}, 0xC622ull);
}

// ── backward_bias (dB += ) ────────────────────────────────────────────────
BT_PARITY_TEST(ct2d_bwd_bias_2x) {
    run_bwd_bias({2, 4, 5, 6, 3, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1, 1}, 0xC630ull);
}

int main() { return run_all("conv_transpose2d cpu/gpu parity"); }
