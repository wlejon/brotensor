// CPU<->GPU parity for the diffusion ResBlock (CHUNK 6).
//
//   resblock_forward  — GroupNorm→SiLU→conv→[t_emb shift]→GroupNorm→SiLU→conv
//                       + skip (optional 1x1 Wskip conv).
//   resblock_backward — full backward of the above.
//
// DTYPE NOTES
//   The CUDA resblock runs FP16 internally (GroupNorm/SiLU/conv2d FP16 with
//   FP32 accumulators). The CPU backend is FP32-only. We quantise all inputs
//   through FP16 so both backends start identical, feed FP16 to the GPU and
//   FP32 to the CPU, and compare with the loose FP16-scale tolerance the
//   test_resblock.cpp smoke test uses (atol 1e-2, rtol 1e-2). A ResBlock is a
//   long composite chain (two GroupNorms + two 3x3 convs), so FP16 rounding
//   compounds — a tight FP32 tolerance would not survive the cross-backend
//   FP16/FP32 comparison.
//
// ACCUMULATION
//   resblock_backward: dX OVERWRITTEN (GN1 backward, then skip path added);
//   dGamma1/dBeta1/dGamma2/dBeta2, dW1/dW2/dWskip, db1/db2/dbskip and
//   dt_emb_shift all ACCUMULATE (+=). The grad buffers are pre-filled with a
//   non-zero baseline (identical on both backends) to verify accumulation.
//
// COVERAGE: C_in==C_out (identity skip) and C_in!=C_out (1x1 Wskip conv),
// with and without the t_emb shift.

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

inline float q16(float v) {
    return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

// CPU FP32 tensor with FP16-quantised random values.
Tensor make_q16_cpu(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = q16(rng.next_unit() * scale);
    return t;
}

Tensor to_fp16_cuda(const Tensor& cpu) {
    const int n = cpu.size();
    std::vector<uint16_t> h(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h[i] = brotensor::fp32_to_fp16_bits(cpu[i]);
    return Tensor::from_host_fp16_on(gpu_device(), h.data(), cpu.rows, cpu.cols);
}

Tensor fp16_cuda_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    std::vector<uint16_t> h = g.to_host_vector_fp16();
    Tensor out = Tensor::mat(g.rows, g.cols);
    for (int i = 0; i < out.size(); ++i)
        out.ptr()[i] = brotensor::fp16_bits_to_fp32(h[i]);
    return out;
}

// ─── BF16 variants ─────────────────────────────────────────────────────────

inline float qbf16(float v) {
    return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v));
}

// CPU FP32 tensor with BF16-quantised random values.
Tensor make_qbf16_cpu(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = qbf16(rng.next_unit() * scale);
    return t;
}

// ─── forward parity ───────────────────────────────────────────────────────

void run_forward(int N, int C_in, int C_out, int H, int W, int num_groups,
                 uint64_t seed, bool with_temb, bool need_skip) {
    SplitMix64 rng(seed);
    const int spatial = H * W;
    const float eps = 1e-5f;

    Tensor X  = make_q16_cpu(N, C_in * spatial, rng, 0.3f);
    Tensor g1 = make_q16_cpu(C_in, 1, rng, 0.3f);
    Tensor b1 = make_q16_cpu(C_in, 1, rng, 0.3f);
    Tensor W1 = make_q16_cpu(C_out, C_in * 9, rng, 0.3f);
    Tensor bc1 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor g2 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor b2 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor W2 = make_q16_cpu(C_out, C_out * 9, rng, 0.3f);
    Tensor bc2 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor Wsk, bsk, temb;
    if (need_skip) {
        Wsk = make_q16_cpu(C_out, C_in, rng, 0.3f);
        bsk = make_q16_cpu(C_out, 1, rng, 0.3f);
    }
    if (with_temb) temb = make_q16_cpu(N, C_out, rng, 0.3f);

    const Tensor* temb_c = with_temb ? &temb : nullptr;
    const Tensor* Wsk_c  = need_skip ? &Wsk : nullptr;
    const Tensor* bsk_c  = need_skip ? &bsk : nullptr;

    // CPU path.
    Tensor Y_c;
    brotensor::resblock_forward(X, g1, b1, W1, &bc1, temb_c,
                                g2, b2, W2, &bc2, Wsk_c, bsk_c,
                                N, C_in, C_out, H, W, num_groups, eps, Y_c);

    // GPU path.
    Tensor gX  = to_fp16_cuda(X);
    Tensor gg1 = to_fp16_cuda(g1), gb1 = to_fp16_cuda(b1);
    Tensor gW1 = to_fp16_cuda(W1), gbc1 = to_fp16_cuda(bc1);
    Tensor gg2 = to_fp16_cuda(g2), gb2 = to_fp16_cuda(b2);
    Tensor gW2 = to_fp16_cuda(W2), gbc2 = to_fp16_cuda(bc2);
    Tensor gWsk, gbsk, gtemb;
    if (need_skip) { gWsk = to_fp16_cuda(Wsk); gbsk = to_fp16_cuda(bsk); }
    if (with_temb) gtemb = to_fp16_cuda(temb);
    const Tensor* gtemb_p = with_temb ? &gtemb : nullptr;
    const Tensor* gWsk_p  = need_skip ? &gWsk : nullptr;
    const Tensor* gbsk_p  = need_skip ? &gbsk : nullptr;

    Tensor gY;
    brotensor::resblock_forward(gX, gg1, gb1, gW1, &gbc1, gtemb_p,
                                gg2, gb2, gW2, &gbc2, gWsk_p, gbsk_p,
                                N, C_in, C_out, H, W, num_groups, eps, gY);

    compare_tensors(Y_c, fp16_cuda_to_cpu(gY), "resblock.Y", 1e-2f, 1e-2f);
}

// ─── backward parity ──────────────────────────────────────────────────────

void run_backward(int N, int C_in, int C_out, int H, int W, int num_groups,
                  uint64_t seed, bool with_temb, bool need_skip) {
    SplitMix64 rng(seed);
    const int spatial = H * W;
    const float eps = 1e-5f;

    Tensor X  = make_q16_cpu(N, C_in * spatial, rng, 0.3f);
    Tensor g1 = make_q16_cpu(C_in, 1, rng, 0.3f);
    Tensor b1 = make_q16_cpu(C_in, 1, rng, 0.3f);
    Tensor W1 = make_q16_cpu(C_out, C_in * 9, rng, 0.3f);
    Tensor bc1 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor g2 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor b2 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor W2 = make_q16_cpu(C_out, C_out * 9, rng, 0.3f);
    Tensor bc2 = make_q16_cpu(C_out, 1, rng, 0.3f);
    Tensor dY = make_q16_cpu(N, C_out * spatial, rng, 0.3f);
    Tensor Wsk, bsk, temb;
    if (need_skip) {
        Wsk = make_q16_cpu(C_out, C_in, rng, 0.3f);
        bsk = make_q16_cpu(C_out, 1, rng, 0.3f);
    }
    if (with_temb) temb = make_q16_cpu(N, C_out, rng, 0.3f);

    const Tensor* temb_c = with_temb ? &temb : nullptr;
    const Tensor* Wsk_c  = need_skip ? &Wsk : nullptr;
    const Tensor* bsk_c  = need_skip ? &bsk : nullptr;

    // Non-zero gradient baselines (accumulation contract).
    Tensor dG1_i = make_q16_cpu(C_in, 1, rng, 0.1f);
    Tensor dB1_i = make_q16_cpu(C_in, 1, rng, 0.1f);
    Tensor dW1_i = make_q16_cpu(C_out, C_in * 9, rng, 0.1f);
    Tensor db1_i = make_q16_cpu(C_out, 1, rng, 0.1f);
    Tensor dG2_i = make_q16_cpu(C_out, 1, rng, 0.1f);
    Tensor dB2_i = make_q16_cpu(C_out, 1, rng, 0.1f);
    Tensor dW2_i = make_q16_cpu(C_out, C_out * 9, rng, 0.1f);
    Tensor db2_i = make_q16_cpu(C_out, 1, rng, 0.1f);
    Tensor dt_i, dWsk_i, dbsk_i;
    if (with_temb) dt_i = make_q16_cpu(N, C_out, rng, 0.1f);
    if (need_skip) {
        dWsk_i = make_q16_cpu(C_out, C_in, rng, 0.1f);
        dbsk_i = make_q16_cpu(C_out, 1, rng, 0.1f);
    }

    // CPU path.
    Tensor dX_c = Tensor::mat(N, C_in * spatial);
    Tensor dG1_c = dG1_i, dB1_c = dB1_i, dW1_c = dW1_i, db1_c = db1_i;
    Tensor dG2_c = dG2_i, dB2_c = dB2_i, dW2_c = dW2_i, db2_c = db2_i;
    Tensor dt_c, dWsk_c, dbsk_c;
    Tensor *dt_cp = nullptr, *dWsk_cp = nullptr, *dbsk_cp = nullptr;
    if (with_temb) { dt_c = dt_i; dt_cp = &dt_c; }
    if (need_skip) {
        dWsk_c = dWsk_i; dbsk_c = dbsk_i;
        dWsk_cp = &dWsk_c; dbsk_cp = &dbsk_c;
    }
    brotensor::resblock_backward(
        X, g1, b1, W1, &bc1, temb_c, g2, b2, W2, &bc2, Wsk_c, bsk_c,
        N, C_in, C_out, H, W, num_groups, eps, dY,
        dX_c, dG1_c, dB1_c, dW1_c, &db1_c, dt_cp,
        dG2_c, dB2_c, dW2_c, &db2_c, dWsk_cp, dbsk_cp);

    // GPU path.
    Tensor gX  = to_fp16_cuda(X);
    Tensor gg1 = to_fp16_cuda(g1), gb1 = to_fp16_cuda(b1);
    Tensor gW1 = to_fp16_cuda(W1), gbc1 = to_fp16_cuda(bc1);
    Tensor gg2 = to_fp16_cuda(g2), gb2 = to_fp16_cuda(b2);
    Tensor gW2 = to_fp16_cuda(W2), gbc2 = to_fp16_cuda(bc2);
    Tensor gdY = to_fp16_cuda(dY);
    Tensor gWsk, gbsk, gtemb;
    if (need_skip) { gWsk = to_fp16_cuda(Wsk); gbsk = to_fp16_cuda(bsk); }
    if (with_temb) gtemb = to_fp16_cuda(temb);
    const Tensor* gtemb_p = with_temb ? &gtemb : nullptr;
    const Tensor* gWsk_p  = need_skip ? &gWsk : nullptr;
    const Tensor* gbsk_p  = need_skip ? &gbsk : nullptr;

    Tensor gdX  = Tensor::empty_on(gpu_device(), N, C_in * spatial, Dtype::FP16);
    Tensor gdG1 = to_fp16_cuda(dG1_i), gdB1 = to_fp16_cuda(dB1_i);
    Tensor gdW1 = to_fp16_cuda(dW1_i), gdb1 = to_fp16_cuda(db1_i);
    Tensor gdG2 = to_fp16_cuda(dG2_i), gdB2 = to_fp16_cuda(dB2_i);
    Tensor gdW2 = to_fp16_cuda(dW2_i), gdb2 = to_fp16_cuda(db2_i);
    Tensor gdt, gdWsk, gdbsk;
    Tensor *gdt_p = nullptr, *gdWsk_p = nullptr, *gdbsk_p = nullptr;
    if (with_temb) { gdt = to_fp16_cuda(dt_i); gdt_p = &gdt; }
    if (need_skip) {
        gdWsk = to_fp16_cuda(dWsk_i); gdbsk = to_fp16_cuda(dbsk_i);
        gdWsk_p = &gdWsk; gdbsk_p = &gdbsk;
    }
    brotensor::resblock_backward(
        gX, gg1, gb1, gW1, &gbc1, gtemb_p, gg2, gb2, gW2, &gbc2,
        gWsk_p, gbsk_p,
        N, C_in, C_out, H, W, num_groups, eps, gdY,
        gdX, gdG1, gdB1, gdW1, &gdb1, gdt_p,
        gdG2, gdB2, gdW2, &gdb2, gdWsk_p, gdbsk_p);

    const float atol = 1e-2f, rtol = 1e-2f;
    compare_tensors(dX_c,  fp16_cuda_to_cpu(gdX),  "resblock.dX",  atol, rtol);
    compare_tensors(dG1_c, fp16_cuda_to_cpu(gdG1), "resblock.dGamma1", atol, rtol);
    compare_tensors(dB1_c, fp16_cuda_to_cpu(gdB1), "resblock.dBeta1",  atol, rtol);
    compare_tensors(dW1_c, fp16_cuda_to_cpu(gdW1), "resblock.dW1", atol, rtol);
    compare_tensors(db1_c, fp16_cuda_to_cpu(gdb1), "resblock.db1", atol, rtol);
    compare_tensors(dG2_c, fp16_cuda_to_cpu(gdG2), "resblock.dGamma2", atol, rtol);
    compare_tensors(dB2_c, fp16_cuda_to_cpu(gdB2), "resblock.dBeta2",  atol, rtol);
    compare_tensors(dW2_c, fp16_cuda_to_cpu(gdW2), "resblock.dW2", atol, rtol);
    compare_tensors(db2_c, fp16_cuda_to_cpu(gdb2), "resblock.db2", atol, rtol);
    if (with_temb)
        compare_tensors(dt_c, fp16_cuda_to_cpu(gdt),
                        "resblock.dt_emb_shift", atol, rtol);
    if (need_skip) {
        compare_tensors(dWsk_c, fp16_cuda_to_cpu(gdWsk),
                        "resblock.dWskip", atol, rtol);
        compare_tensors(dbsk_c, fp16_cuda_to_cpu(gdbsk),
                        "resblock.dbskip", atol, rtol);
    }
}

// ─── BF16 forward parity (BF16-on-CUDA vs FP32 CPU reference) ──────────────

void run_forward_bf16(int N, int C_in, int C_out, int H, int W, int num_groups,
                      uint64_t seed, bool with_temb, bool need_skip) {
    SplitMix64 rng(seed);
    const int spatial = H * W;
    const float eps = 1e-5f;

    Tensor X  = make_qbf16_cpu(N, C_in * spatial, rng, 0.3f);
    Tensor g1 = make_qbf16_cpu(C_in, 1, rng, 0.3f);
    Tensor b1 = make_qbf16_cpu(C_in, 1, rng, 0.3f);
    Tensor W1 = make_qbf16_cpu(C_out, C_in * 9, rng, 0.3f);
    Tensor bc1 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor g2 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor b2 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor W2 = make_qbf16_cpu(C_out, C_out * 9, rng, 0.3f);
    Tensor bc2 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor Wsk, bsk, temb;
    if (need_skip) {
        Wsk = make_qbf16_cpu(C_out, C_in, rng, 0.3f);
        bsk = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    }
    if (with_temb) temb = make_qbf16_cpu(N, C_out, rng, 0.3f);

    const Tensor* temb_c = with_temb ? &temb : nullptr;
    const Tensor* Wsk_c  = need_skip ? &Wsk : nullptr;
    const Tensor* bsk_c  = need_skip ? &bsk : nullptr;

    // CPU FP32 reference.
    Tensor Y_c;
    brotensor::resblock_forward(X, g1, b1, W1, &bc1, temb_c,
                                g2, b2, W2, &bc2, Wsk_c, bsk_c,
                                N, C_in, C_out, H, W, num_groups, eps, Y_c);

    // GPU BF16 path.
    Tensor gX  = to_bf16_cuda(X);
    Tensor gg1 = to_bf16_cuda(g1), gb1 = to_bf16_cuda(b1);
    Tensor gW1 = to_bf16_cuda(W1), gbc1 = to_bf16_cuda(bc1);
    Tensor gg2 = to_bf16_cuda(g2), gb2 = to_bf16_cuda(b2);
    Tensor gW2 = to_bf16_cuda(W2), gbc2 = to_bf16_cuda(bc2);
    Tensor gWsk, gbsk, gtemb;
    if (need_skip) { gWsk = to_bf16_cuda(Wsk); gbsk = to_bf16_cuda(bsk); }
    if (with_temb) gtemb = to_bf16_cuda(temb);
    const Tensor* gtemb_p = with_temb ? &gtemb : nullptr;
    const Tensor* gWsk_p  = need_skip ? &gWsk : nullptr;
    const Tensor* gbsk_p  = need_skip ? &gbsk : nullptr;

    Tensor gY;
    brotensor::resblock_forward(gX, gg1, gb1, gW1, &gbc1, gtemb_p,
                                gg2, gb2, gW2, &gbc2, gWsk_p, gbsk_p,
                                N, C_in, C_out, H, W, num_groups, eps, gY);

    Tensor gY_host = bf16_host_to_f32(download_to_host(gY));
    compare_tensors(Y_c, gY_host, "resblock.Y.bf16", 1.2e-1f, 1.2e-1f);
}

// ─── BF16 backward parity ──────────────────────────────────────────────────

void run_backward_bf16(int N, int C_in, int C_out, int H, int W, int num_groups,
                       uint64_t seed, bool with_temb, bool need_skip) {
    SplitMix64 rng(seed);
    const int spatial = H * W;
    const float eps = 1e-5f;

    Tensor X  = make_qbf16_cpu(N, C_in * spatial, rng, 0.3f);
    Tensor g1 = make_qbf16_cpu(C_in, 1, rng, 0.3f);
    Tensor b1 = make_qbf16_cpu(C_in, 1, rng, 0.3f);
    Tensor W1 = make_qbf16_cpu(C_out, C_in * 9, rng, 0.3f);
    Tensor bc1 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor g2 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor b2 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor W2 = make_qbf16_cpu(C_out, C_out * 9, rng, 0.3f);
    Tensor bc2 = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    Tensor dY = make_qbf16_cpu(N, C_out * spatial, rng, 0.3f);
    Tensor Wsk, bsk, temb;
    if (need_skip) {
        Wsk = make_qbf16_cpu(C_out, C_in, rng, 0.3f);
        bsk = make_qbf16_cpu(C_out, 1, rng, 0.3f);
    }
    if (with_temb) temb = make_qbf16_cpu(N, C_out, rng, 0.3f);

    const Tensor* temb_c = with_temb ? &temb : nullptr;
    const Tensor* Wsk_c  = need_skip ? &Wsk : nullptr;
    const Tensor* bsk_c  = need_skip ? &bsk : nullptr;

    Tensor dG1_i = make_qbf16_cpu(C_in, 1, rng, 0.1f);
    Tensor dB1_i = make_qbf16_cpu(C_in, 1, rng, 0.1f);
    Tensor dW1_i = make_qbf16_cpu(C_out, C_in * 9, rng, 0.1f);
    Tensor db1_i = make_qbf16_cpu(C_out, 1, rng, 0.1f);
    Tensor dG2_i = make_qbf16_cpu(C_out, 1, rng, 0.1f);
    Tensor dB2_i = make_qbf16_cpu(C_out, 1, rng, 0.1f);
    Tensor dW2_i = make_qbf16_cpu(C_out, C_out * 9, rng, 0.1f);
    Tensor db2_i = make_qbf16_cpu(C_out, 1, rng, 0.1f);
    Tensor dt_i, dWsk_i, dbsk_i;
    if (with_temb) dt_i = make_qbf16_cpu(N, C_out, rng, 0.1f);
    if (need_skip) {
        dWsk_i = make_qbf16_cpu(C_out, C_in, rng, 0.1f);
        dbsk_i = make_qbf16_cpu(C_out, 1, rng, 0.1f);
    }

    // CPU FP32 reference.
    Tensor dX_c = Tensor::mat(N, C_in * spatial);
    Tensor dG1_c = dG1_i, dB1_c = dB1_i, dW1_c = dW1_i, db1_c = db1_i;
    Tensor dG2_c = dG2_i, dB2_c = dB2_i, dW2_c = dW2_i, db2_c = db2_i;
    Tensor dt_c, dWsk_c, dbsk_c;
    Tensor *dt_cp = nullptr, *dWsk_cp = nullptr, *dbsk_cp = nullptr;
    if (with_temb) { dt_c = dt_i; dt_cp = &dt_c; }
    if (need_skip) {
        dWsk_c = dWsk_i; dbsk_c = dbsk_i;
        dWsk_cp = &dWsk_c; dbsk_cp = &dbsk_c;
    }
    brotensor::resblock_backward(
        X, g1, b1, W1, &bc1, temb_c, g2, b2, W2, &bc2, Wsk_c, bsk_c,
        N, C_in, C_out, H, W, num_groups, eps, dY,
        dX_c, dG1_c, dB1_c, dW1_c, &db1_c, dt_cp,
        dG2_c, dB2_c, dW2_c, &db2_c, dWsk_cp, dbsk_cp);

    // GPU BF16 path.
    Tensor gX  = to_bf16_cuda(X);
    Tensor gg1 = to_bf16_cuda(g1), gb1 = to_bf16_cuda(b1);
    Tensor gW1 = to_bf16_cuda(W1), gbc1 = to_bf16_cuda(bc1);
    Tensor gg2 = to_bf16_cuda(g2), gb2 = to_bf16_cuda(b2);
    Tensor gW2 = to_bf16_cuda(W2), gbc2 = to_bf16_cuda(bc2);
    Tensor gdY = to_bf16_cuda(dY);
    Tensor gWsk, gbsk, gtemb;
    if (need_skip) { gWsk = to_bf16_cuda(Wsk); gbsk = to_bf16_cuda(bsk); }
    if (with_temb) gtemb = to_bf16_cuda(temb);
    const Tensor* gtemb_p = with_temb ? &gtemb : nullptr;
    const Tensor* gWsk_p  = need_skip ? &gWsk : nullptr;
    const Tensor* gbsk_p  = need_skip ? &gbsk : nullptr;

    Tensor gdX  = Tensor::empty_on(Device::CUDA, N, C_in * spatial, Dtype::BF16);
    Tensor gdG1 = to_bf16_cuda(dG1_i), gdB1 = to_bf16_cuda(dB1_i);
    Tensor gdW1 = to_bf16_cuda(dW1_i), gdb1 = to_bf16_cuda(db1_i);
    Tensor gdG2 = to_bf16_cuda(dG2_i), gdB2 = to_bf16_cuda(dB2_i);
    Tensor gdW2 = to_bf16_cuda(dW2_i), gdb2 = to_bf16_cuda(db2_i);
    Tensor gdt, gdWsk, gdbsk;
    Tensor *gdt_p = nullptr, *gdWsk_p = nullptr, *gdbsk_p = nullptr;
    if (with_temb) { gdt = to_bf16_cuda(dt_i); gdt_p = &gdt; }
    if (need_skip) {
        gdWsk = to_bf16_cuda(dWsk_i); gdbsk = to_bf16_cuda(dbsk_i);
        gdWsk_p = &gdWsk; gdbsk_p = &gdbsk;
    }
    brotensor::resblock_backward(
        gX, gg1, gb1, gW1, &gbc1, gtemb_p, gg2, gb2, gW2, &gbc2,
        gWsk_p, gbsk_p,
        N, C_in, C_out, H, W, num_groups, eps, gdY,
        gdX, gdG1, gdB1, gdW1, &gdb1, gdt_p,
        gdG2, gdB2, gdW2, &gdb2, gdWsk_p, gdbsk_p);

    const float atol = 1.5e-1f, rtol = 1.5e-1f;
    compare_tensors(dX_c,  bf16_host_to_f32(download_to_host(gdX)),
                    "resblock.dX.bf16",  atol, rtol);
    compare_tensors(dG1_c, bf16_host_to_f32(download_to_host(gdG1)),
                    "resblock.dGamma1.bf16", atol, rtol);
    compare_tensors(dB1_c, bf16_host_to_f32(download_to_host(gdB1)),
                    "resblock.dBeta1.bf16",  atol, rtol);
    compare_tensors(dW1_c, bf16_host_to_f32(download_to_host(gdW1)),
                    "resblock.dW1.bf16", atol, rtol);
    compare_tensors(db1_c, bf16_host_to_f32(download_to_host(gdb1)),
                    "resblock.db1.bf16", atol, rtol);
    compare_tensors(dG2_c, bf16_host_to_f32(download_to_host(gdG2)),
                    "resblock.dGamma2.bf16", atol, rtol);
    compare_tensors(dB2_c, bf16_host_to_f32(download_to_host(gdB2)),
                    "resblock.dBeta2.bf16",  atol, rtol);
    compare_tensors(dW2_c, bf16_host_to_f32(download_to_host(gdW2)),
                    "resblock.dW2.bf16", atol, rtol);
    compare_tensors(db2_c, bf16_host_to_f32(download_to_host(gdb2)),
                    "resblock.db2.bf16", atol, rtol);
    if (with_temb)
        compare_tensors(dt_c, bf16_host_to_f32(download_to_host(gdt)),
                        "resblock.dt_emb_shift.bf16", atol, rtol);
    if (need_skip) {
        compare_tensors(dWsk_c, bf16_host_to_f32(download_to_host(gdWsk)),
                        "resblock.dWskip.bf16", atol, rtol);
        compare_tensors(dbsk_c, bf16_host_to_f32(download_to_host(gdbsk)),
                        "resblock.dbskip.bf16", atol, rtol);
    }
}

} // namespace

// ─── forward: C_in==C_out (identity skip) and C_in!=C_out (1x1 Wskip) ─────
BT_PARITY_TEST(resblock_fwd_same_8x8_g4) {
    run_forward(1, 32, 32, 8, 8, 4, 0x800ull, false, false);
}
BT_PARITY_TEST(resblock_fwd_same_8x8_g4_temb) {
    run_forward(2, 32, 32, 8, 8, 4, 0x801ull, true, false);
}
BT_PARITY_TEST(resblock_fwd_up_8x8_g4_temb) {
    run_forward(1, 32, 64, 8, 8, 4, 0x802ull, true, true);
}
BT_PARITY_TEST(resblock_fwd_up_4x4_g2) {
    run_forward(2, 16, 32, 4, 4, 2, 0x803ull, false, true);
}

// ─── backward: same configs ───────────────────────────────────────────────
BT_PARITY_TEST(resblock_bwd_same_8x8_g4) {
    run_backward(1, 8, 8, 8, 8, 4, 0x810ull, false, false);
}
BT_PARITY_TEST(resblock_bwd_same_8x8_g4_temb) {
    run_backward(1, 8, 8, 8, 8, 4, 0x811ull, true, false);
}
BT_PARITY_TEST(resblock_bwd_up_4x4_g4_temb) {
    run_backward(1, 8, 16, 4, 4, 4, 0x812ull, true, true);
}
BT_PARITY_TEST(resblock_bwd_up_4x4_g2) {
    run_backward(2, 8, 16, 4, 4, 2, 0x813ull, false, true);
}

// ─── BF16 forward (BF16-on-CUDA vs FP32 CPU reference) ──────────────────────
BT_PARITY_TEST(resblock_fwd_bf16_same_8x8_g4) {
    run_forward_bf16(1, 32, 32, 8, 8, 4, 0x820ull, false, false);
}
BT_PARITY_TEST(resblock_fwd_bf16_same_8x8_g4_temb) {
    run_forward_bf16(2, 32, 32, 8, 8, 4, 0x821ull, true, false);
}
BT_PARITY_TEST(resblock_fwd_bf16_up_8x8_g4_temb) {
    run_forward_bf16(1, 32, 64, 8, 8, 4, 0x822ull, true, true);
}
BT_PARITY_TEST(resblock_fwd_bf16_up_4x4_g2) {
    run_forward_bf16(2, 16, 32, 4, 4, 2, 0x823ull, false, true);
}

// ─── BF16 backward ──────────────────────────────────────────────────────────
BT_PARITY_TEST(resblock_bwd_bf16_same_8x8_g4) {
    run_backward_bf16(1, 8, 8, 8, 8, 4, 0x830ull, false, false);
}
BT_PARITY_TEST(resblock_bwd_bf16_same_8x8_g4_temb) {
    run_backward_bf16(1, 8, 8, 8, 8, 4, 0x831ull, true, false);
}
BT_PARITY_TEST(resblock_bwd_bf16_up_4x4_g4_temb) {
    run_backward_bf16(1, 8, 16, 4, 4, 4, 0x832ull, true, true);
}
BT_PARITY_TEST(resblock_bwd_bf16_up_4x4_g2) {
    run_backward_bf16(2, 8, 16, 4, 4, 2, 0x833ull, false, true);
}

int main() { return run_all("resblock cpu/gpu parity"); }
