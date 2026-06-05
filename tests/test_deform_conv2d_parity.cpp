// CPU↔GPU parity + torchvision golden for deform_conv2d_forward (modulated
// deformable conv2d v2, forward/inference only).
//
// Two checks:
//   * Parity: random inputs, CPU vs GPU backend, tight FP32 tolerance.
//   * Golden: torchvision.ops.deform_conv2d reference, loaded from an
//     out-of-repo .bin (D:/projects/_splat_assets/deform_conv_golden.bin,
//     gen_deform_conv_golden.py). Skips cleanly when the file is absent — the
//     parity check above always runs.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

struct DC {
    int N, C_in, H, W, C_out, kH, kW;
    int stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
    int groups, deform_groups;
};

inline int out_dim(int L, int pad, int dil, int k, int s) {
    return (L + 2 * pad - dil * (k - 1) - 1) / s + 1;
}

// Build an FP32 host tensor (rows, cols) filled with rng noise * scale.
Tensor rnd(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    fill_random(t, rng, scale);
    return t;
}

void run_parity(const DC& c, bool has_mask, bool has_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Cg_in = c.C_in / c.groups;
    const int H_out = out_dim(c.H, c.pad_h, c.dil_h, c.kH, c.stride_h);
    const int W_out = out_dim(c.W, c.pad_w, c.dil_w, c.kW, c.stride_w);
    const int ksz = c.kH * c.kW;

    Tensor X      = rnd(c.N, c.C_in * c.H * c.W, rng, 1.0f);
    Tensor offset = rnd(c.N, c.deform_groups * 2 * ksz * H_out * W_out, rng, 1.5f);
    Tensor Wt     = rnd(c.C_out, Cg_in * ksz, rng, 0.5f);
    Tensor mask, bias;
    Tensor* maskp = nullptr;
    Tensor* biasp = nullptr;
    if (has_mask) {
        mask = rnd(c.N, c.deform_groups * ksz * H_out * W_out, rng, 0.5f);
        // shift into a positive-ish modulator range
        for (int i = 0; i < mask.size(); ++i) mask.ptr()[i] += 1.0f;
        maskp = &mask;
    }
    if (has_bias) { bias = rnd(c.C_out, 1, rng, 0.5f); biasp = &bias; }

    Tensor cpu_Y;
    brotensor::deform_conv2d_forward(X, offset, maskp, Wt, biasp,
        c.N, c.C_in, c.H, c.W, c.C_out, c.kH, c.kW,
        c.stride_h, c.stride_w, c.pad_h, c.pad_w, c.dil_h, c.dil_w,
        c.groups, c.deform_groups, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gOff = offset.to(gpu_device());
    Tensor gW = Wt.to(gpu_device());
    Tensor gMask, gBias;
    Tensor* gMaskp = nullptr;
    Tensor* gBiasp = nullptr;
    if (has_mask) { gMask = mask.to(gpu_device()); gMaskp = &gMask; }
    if (has_bias) { gBias = bias.to(gpu_device()); gBiasp = &gBias; }
    Tensor gpu_Y;
    brotensor::deform_conv2d_forward(gX, gOff, gMaskp, gW, gBiasp,
        c.N, c.C_in, c.H, c.W, c.C_out, c.kH, c.kW,
        c.stride_h, c.stride_w, c.pad_h, c.pad_w, c.dil_h, c.dil_w,
        c.groups, c.deform_groups, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "deform_conv2d_fwd",
                    1e-4f, 1e-3f);
}

// ── parity cases ───────────────────────────────────────────────────────────
BT_PARITY_TEST(deform_conv2d_3x3_mask_bias) {
    run_parity({1, 8, 10, 12, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, true, true, 0xD100ull);
}
BT_PARITY_TEST(deform_conv2d_3x3_nomask) {
    run_parity({1, 8, 10, 12, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, false, true, 0xD101ull);
}
BT_PARITY_TEST(deform_conv2d_1x1) {
    run_parity({2, 6, 9, 9, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}, true, false, 0xD102ull);
}
BT_PARITY_TEST(deform_conv2d_7x7) {
    run_parity({1, 16, 11, 13, 8, 7, 7, 1, 1, 3, 3, 1, 1, 1, 1}, true, true, 0xD103ull);
}
BT_PARITY_TEST(deform_conv2d_grouped) {
    run_parity({1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2}, true, true, 0xD104ull);
}
BT_PARITY_TEST(deform_conv2d_stride2) {
    run_parity({1, 8, 12, 12, 6, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1}, true, true, 0xD105ull);
}

// ── torchvision golden (out-of-repo, gated) ─────────────────────────────────

std::vector<float> read_f32(FILE* f, size_t n) {
    std::vector<float> v(n);
    size_t got = std::fread(v.data(), sizeof(float), n, f);
    if (got != n) { std::printf("    golden: short read\n"); throw 0; }
    return v;
}

Tensor mat_from(const std::vector<float>& src, int rows, int cols) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < rows * cols; ++i) t.ptr()[i] = src[i];
    return t;
}

BT_PARITY_TEST(deform_conv2d_torchvision_golden) {
    const char* path = "D:/projects/_splat_assets/deform_conv_golden.bin";
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("    [skip] golden file absent (%s)\n", path);
        return;
    }
    int32_t num_cases = 0;
    if (std::fread(&num_cases, sizeof(int32_t), 1, f) != 1) { std::fclose(f); throw 0; }
    for (int ci = 0; ci < num_cases; ++ci) {
        int32_t h[16];
        if (std::fread(h, sizeof(int32_t), 16, f) != 16) { std::fclose(f); throw 0; }
        const int N = h[0], Cin = h[1], H = h[2], W = h[3], Cout = h[4];
        const int kH = h[5], kW = h[6], sh = h[7], sw = h[8];
        const int ph = h[9], pw = h[10], dh = h[11], dw = h[12];
        const int g = h[13], dg = h[14], um = h[15];
        const int Hout = out_dim(H, ph, dh, kH, sh);
        const int Wout = out_dim(W, pw, dw, kW, sw);
        const int ksz = kH * kW;

        Tensor X   = mat_from(read_f32(f, (size_t)N * Cin * H * W), N, Cin * H * W);
        Tensor off = mat_from(read_f32(f, (size_t)N * 2 * dg * ksz * Hout * Wout),
                              N, 2 * dg * ksz * Hout * Wout);
        Tensor mask, bias;
        Tensor* maskp = nullptr;
        if (um) {
            mask = mat_from(read_f32(f, (size_t)N * dg * ksz * Hout * Wout),
                            N, dg * ksz * Hout * Wout);
            maskp = &mask;
        }
        Tensor Wt = mat_from(read_f32(f, (size_t)Cout * (Cin / g) * ksz),
                             Cout, (Cin / g) * ksz);
        bias = mat_from(read_f32(f, (size_t)Cout), Cout, 1);
        std::vector<float> Yexp = read_f32(f, (size_t)N * Cout * Hout * Wout);

        Tensor cpu_Y;
        brotensor::deform_conv2d_forward(X, off, maskp, Wt, &bias,
            N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, g, dg, cpu_Y);

        Tensor exp = mat_from(Yexp, N, Cout * Hout * Wout);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "golden[%d] k%d g%d dg%d m%d", ci, kH, g, dg, um);
        compare_tensors(cpu_Y, exp, tag, 2e-4f, 1e-3f);

        // also check the GPU backend matches the reference
        Tensor gX = X.to(gpu_device()), gOff = off.to(gpu_device());
        Tensor gW = Wt.to(gpu_device()), gB = bias.to(gpu_device()), gMask;
        Tensor* gMaskp = nullptr;
        if (um) { gMask = mask.to(gpu_device()); gMaskp = &gMask; }
        Tensor gpu_Y;
        brotensor::deform_conv2d_forward(gX, gOff, gMaskp, gW, &gB,
            N, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, g, dg, gpu_Y);
        compare_tensors(exp, download_to_host(gpu_Y), tag, 5e-3f, 5e-3f);
    }
    std::fclose(f);
    std::printf("    golden: %d cases matched torchvision\n", num_cases);
}

} // namespace

int main() { return run_all("deform_conv2d cpu/gpu parity + golden"); }
