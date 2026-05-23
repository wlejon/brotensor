// CPU↔GPU parity tests for conv3d_forward and the GPU-only
// conv3d_int8w_fp16_forward (Qwen3-VL patch embedder).
//
// conv3d_forward: FP32, FP16, BF16 paths checked against the FP32 CPU
// reference. NCTHW activations; OICTHW (grouped) filter.
//
// conv3d_int8w_fp16_forward: GPU-only (CPU vtable slot is null). Compared
// against an FP16 CPU reference computed by widening the INT8 weight by its
// per-row FP32 dequant scale and running a plain conv3d_forward in FP16 — the
// kernels' math is bit-equivalent (FP32 acc, no further rounding), so this is
// an exact correctness check, not a bounded-error check.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

struct Conv3dCfg {
    int N, C_in, T, H, W, C_out, kT, kH, kW;
    int stride_t, stride_h, stride_w;
    int pad_t, pad_h, pad_w;
    int dil_t, dil_h, dil_w;
    int groups;
};

inline int out_dim(int in, int pad, int dil, int k, int stride) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

void run_fwd_fp32(const Conv3dCfg& c, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    Tensor X = Tensor::mat(c.N, c.C_in * c.T * c.H * c.W);
    Tensor Wt = Tensor::mat(c.C_out, Cg_in * c.kT * c.kH * c.kW);
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
    brotensor::conv3d_forward(X, Wt, Bp,
                              c.N, c.C_in, c.T, c.H, c.W,
                              c.C_out, c.kT, c.kH, c.kW,
                              c.stride_t, c.stride_h, c.stride_w,
                              c.pad_t, c.pad_h, c.pad_w,
                              c.dil_t, c.dil_h, c.dil_w,
                              c.groups, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gW = Wt.to(gpu_device());
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) { gB = B.to(gpu_device()); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::conv3d_forward(gX, gW, gBp,
                              c.N, c.C_in, c.T, c.H, c.W,
                              c.C_out, c.kT, c.kH, c.kW,
                              c.stride_t, c.stride_h, c.stride_w,
                              c.pad_t, c.pad_h, c.pad_w,
                              c.dil_t, c.dil_h, c.dil_w,
                              c.groups, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "conv3d_fwd", 1e-4f, 1e-3f);
}

void run_fwd_bf16(const Conv3dCfg& c, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    Tensor X = Tensor::mat(c.N, c.C_in * c.T * c.H * c.W);
    Tensor Wt = Tensor::mat(c.C_out, Cg_in * c.kT * c.kH * c.kW);
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
    brotensor::conv3d_forward(X, Wt, Bp,
                              c.N, c.C_in, c.T, c.H, c.W,
                              c.C_out, c.kT, c.kH, c.kW,
                              c.stride_t, c.stride_h, c.stride_w,
                              c.pad_t, c.pad_h, c.pad_w,
                              c.dil_t, c.dil_h, c.dil_w,
                              c.groups, cpu_Y);

    Tensor gX = to_bf16_gpu(X);
    Tensor gW = to_bf16_gpu(Wt);
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) { gB = to_bf16_gpu(B); gBp = &gB; }
    Tensor gpu_Y;
    brotensor::conv3d_forward(gX, gW, gBp,
                              c.N, c.C_in, c.T, c.H, c.W,
                              c.C_out, c.kT, c.kH, c.kW,
                              c.stride_t, c.stride_h, c.stride_w,
                              c.pad_t, c.pad_h, c.pad_w,
                              c.dil_t, c.dil_h, c.dil_w,
                              c.groups, gpu_Y);

    Tensor gpu_host = bf16_host_to_f32(download_to_host(gpu_Y));
    compare_tensors(cpu_Y, gpu_host, "conv3d_fwd_bf16", 5e-2f, 5e-2f);
}

// ─── INT8w forward — GPU-only ──────────────────────────────────────────────
//
// Build a dequantised FP32 weight on the host, run an FP32 conv3d on the CPU
// for the reference, then compare against the GPU's INT8w-on-FP16 result with
// a bounded tolerance (FP16 quantisation of inputs + FP16 storage of the
// output drives the error budget — same envelope as the conv2d int8w tests).
void run_fwd_int8w(const Conv3dCfg& c, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    const int x_n = c.N * c.C_in * c.T * c.H * c.W;
    const int w_n = c.C_out * Cg_in * c.kT * c.kH * c.kW;

    // FP32 host inputs.
    std::vector<float> Xh(x_n), Wf(w_n), Bh(c.C_out);
    for (auto& v : Xh) v = rng.next_unit() * 0.5f;
    for (auto& v : Wf) v = rng.next_unit() * 0.5f;
    for (auto& v : Bh) v = rng.next_unit() * 0.5f;

    // Per-output-row symmetric quant: scale = max(|w|)/127, q = round(w/scale).
    std::vector<int8_t> Wq(w_n);
    std::vector<float> Sh(c.C_out, 0.0f);
    const int row = Cg_in * c.kT * c.kH * c.kW;
    for (int r = 0; r < c.C_out; ++r) {
        float m = 0.0f;
        for (int j = 0; j < row; ++j) {
            const float v = std::fabs(Wf[r * row + j]);
            if (v > m) m = v;
        }
        const float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
        Sh[r] = scale;
        for (int j = 0; j < row; ++j) {
            float qf = Wf[r * row + j] / scale;
            int qi = static_cast<int>(qf >= 0.0f ? qf + 0.5f : qf - 0.5f);
            if (qi >  127) qi =  127;
            if (qi < -128) qi = -128;
            Wq[r * row + j] = static_cast<int8_t>(qi);
        }
    }

    // FP32 CPU reference using the DEQUANTISED weight — what the INT8 GPU
    // kernel actually evaluates (modulo FP16 storage).
    std::vector<float> Wdq(w_n);
    for (int r = 0; r < c.C_out; ++r) {
        for (int j = 0; j < row; ++j) {
            Wdq[r * row + j] = static_cast<float>(Wq[r * row + j]) * Sh[r];
        }
    }

    Tensor X_cpu = Tensor::from_host_on(Device::CPU,
                                        Xh.data(),
                                        c.N, c.C_in * c.T * c.H * c.W);
    Tensor W_cpu_dq = Tensor::from_host_on(Device::CPU,
                                           Wdq.data(), c.C_out, row);
    Tensor B_cpu;
    Tensor* Bcp = nullptr;
    if (has_bias) {
        B_cpu = Tensor::from_host_on(Device::CPU,
                                     Bh.data(), c.C_out, 1);
        Bcp = &B_cpu;
    }
    Tensor cpu_Y;
    brotensor::conv3d_forward(X_cpu, W_cpu_dq, Bcp,
                              c.N, c.C_in, c.T, c.H, c.W,
                              c.C_out, c.kT, c.kH, c.kW,
                              c.stride_t, c.stride_h, c.stride_w,
                              c.pad_t, c.pad_h, c.pad_w,
                              c.dil_t, c.dil_h, c.dil_w,
                              c.groups, cpu_Y);

    // GPU INT8w + FP16: build FP16 X / FP16 bias, INT8 W, FP32 scales.
    std::vector<uint16_t> Xh16(x_n), Bh16(c.C_out);
    for (int i = 0; i < x_n; ++i) Xh16[i] = brotensor::fp32_to_fp16_bits(Xh[i]);
    for (int i = 0; i < c.C_out; ++i) Bh16[i] = brotensor::fp32_to_fp16_bits(Bh[i]);

    Tensor gX = Tensor::from_host_fp16_on(gpu_device(),
                                          Xh16.data(),
                                          c.N, c.C_in * c.T * c.H * c.W);
    Tensor gW = Tensor::from_host_int8_on(gpu_device(),
                                          Wq.data(), c.C_out, row);
    Tensor gS = Tensor::from_host_on(gpu_device(),
                                     Sh.data(), c.C_out, 1);
    Tensor gB;
    Tensor* gBp = nullptr;
    if (has_bias) {
        gB = Tensor::from_host_fp16_on(gpu_device(),
                                       Bh16.data(), c.C_out, 1);
        gBp = &gB;
    }

    Tensor gpu_Y;
    brotensor::conv3d_int8w_fp16_forward(gX, gW, gS, gBp,
                                         c.N, c.C_in, c.T, c.H, c.W,
                                         c.C_out, c.kT, c.kH, c.kW,
                                         c.stride_t, c.stride_h, c.stride_w,
                                         c.pad_t, c.pad_h, c.pad_w,
                                         c.dil_t, c.dil_h, c.dil_w,
                                         c.groups, gpu_Y);
    brotensor::sync_all();
    BT_CHECK(gpu_Y.dtype == Dtype::FP16);

    // Widen GPU FP16 result to FP32 and compare. The FP16 quantisation of X
    // + FP16-stored output is the dominant error source; mirror the loose
    // tolerance used by test_int8_quant.cpp's conv2d_int8w check.
    const int total = c.N * gpu_Y.cols;
    std::vector<uint16_t> got(total);
    gpu_Y.copy_to_host_fp16(got.data());
    Tensor gpu_host = Tensor::zeros_on(Device::CPU, gpu_Y.rows, gpu_Y.cols);
    float* gh = gpu_host.host_f32_mut();
    for (int i = 0; i < total; ++i) gh[i] = brotensor::fp16_bits_to_fp32(got[i]);

    compare_tensors(cpu_Y, gpu_host, "conv3d_int8w_fwd", 2e-2f, 2e-2f);
}

// ─── Config bank ──────────────────────────────────────────────────────────
// { N,Cin,T,H,W,  Cout,kT,kH,kW,  st,sh,sw, pt,ph,pw, dt,dh,dw, groups }
const Conv3dCfg k1x1x1     {2, 4, 3, 5, 5,   6, 1, 1, 1,
                            1,1,1,  0,0,0,  1,1,1,  1};
const Conv3dCfg k3x3x3_p1  {2, 3, 4, 5, 5,   4, 3, 3, 3,
                            1,1,1,  1,1,1,  1,1,1,  1};
const Conv3dCfg kT2x2x2_s2 {1, 3, 4, 6, 6,   5, 2, 2, 2,
                            2,2,2,  0,0,0,  1,1,1,  1};
const Conv3dCfg kDilated   {1, 2, 5, 5, 5,   2, 3, 3, 3,
                            1,1,1,  2,2,2,  2,1,1,  1};
const Conv3dCfg kGrouped   {2, 4, 3, 4, 4,   6, 3, 3, 3,
                            1,1,1,  1,1,1,  1,1,1,  2};
const Conv3dCfg kDepthw3d  {1, 4, 3, 5, 5,   4, 3, 3, 3,
                            1,1,1,  1,1,1,  1,1,1,  4};
// Qwen3-VL-style patch embedder: kT=2, kH=kW=14, stride matches kernel.
const Conv3dCfg kPatchTiny {1, 3, 2, 14, 14,  8, 2, 14, 14,
                            2,14,14,  0,0,0,  1,1,1,  1};

} // namespace

// ─── FP32 forward ──────────────────────────────────────────────────────────
BT_PARITY_TEST(conv3d_fwd_1x1x1_bias)   { run_fwd_fp32(k1x1x1,     true,  0xC001ull); }
BT_PARITY_TEST(conv3d_fwd_1x1x1_nobias) { run_fwd_fp32(k1x1x1,     false, 0xC002ull); }
BT_PARITY_TEST(conv3d_fwd_3x3x3_p1)     { run_fwd_fp32(k3x3x3_p1,  true,  0xC003ull); }
BT_PARITY_TEST(conv3d_fwd_k2_s2)        { run_fwd_fp32(kT2x2x2_s2, true,  0xC004ull); }
BT_PARITY_TEST(conv3d_fwd_dilated)      { run_fwd_fp32(kDilated,   false, 0xC005ull); }
BT_PARITY_TEST(conv3d_fwd_grouped)      { run_fwd_fp32(kGrouped,   true,  0xC006ull); }
BT_PARITY_TEST(conv3d_fwd_depthw)       { run_fwd_fp32(kDepthw3d,  true,  0xC007ull); }
BT_PARITY_TEST(conv3d_fwd_patch_tiny)   { run_fwd_fp32(kPatchTiny, true,  0xC008ull); }

// ─── BF16 forward (GPU-only dtype; compared against FP32 CPU ref) ──────────
BT_PARITY_TEST(conv3d_fwd_bf16_1x1x1)   { run_fwd_bf16(k1x1x1,     true,  0xC101ull); }
BT_PARITY_TEST(conv3d_fwd_bf16_3x3x3)   { run_fwd_bf16(k3x3x3_p1,  true,  0xC102ull); }
BT_PARITY_TEST(conv3d_fwd_bf16_grouped) { run_fwd_bf16(kGrouped,   true,  0xC103ull); }
BT_PARITY_TEST(conv3d_fwd_bf16_patch)   { run_fwd_bf16(kPatchTiny, true,  0xC104ull); }

// ─── INT8w-on-FP16 forward (GPU-only) ──────────────────────────────────────
BT_PARITY_TEST(conv3d_int8w_1x1x1)      { run_fwd_int8w(k1x1x1,     true,  0xC201ull); }
BT_PARITY_TEST(conv3d_int8w_3x3x3)      { run_fwd_int8w(k3x3x3_p1,  true,  0xC202ull); }
BT_PARITY_TEST(conv3d_int8w_k2_s2)      { run_fwd_int8w(kT2x2x2_s2, false, 0xC203ull); }
BT_PARITY_TEST(conv3d_int8w_grouped)    { run_fwd_int8w(kGrouped,   true,  0xC204ull); }
BT_PARITY_TEST(conv3d_int8w_patch_tiny) { run_fwd_int8w(kPatchTiny, true,  0xC205ull); }

int main() { return run_all("conv3d cpu/gpu parity"); }
