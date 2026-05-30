// CPU<->GPU parity for self_attention_decomposed_rel_pos_forward (SAM/ViTDet).
//
// FP32 CPU vs FP32 GPU at tight tolerance is the primary kernel check; BF16 on
// the GPU backend vs the FP32 CPU reference (loose tolerance) exercises the
// half-precision path. Run with and without the optional qkv/output projection
// biases, across window-sized, non-square, and global-block-sized grids.

#include "parity_helpers.h"

#include <brotensor/ops.h>

#include <cmath>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_fp32(int gh, int gw, int D, int H, bool with_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int L = gh * gw, dh = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    Tensor X  = Tensor::mat(L, D);
    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    Tensor rh = Tensor::mat(2 * gh - 1, dh), rw = Tensor::mat(2 * gw - 1, dh);
    fill_random(X, rng, 0.5f);
    fill_random(Wq, rng, 0.3f); fill_random(Wk, rng, 0.3f);
    fill_random(Wv, rng, 0.3f); fill_random(Wo, rng, 0.3f);
    fill_random(rh, rng, 0.4f); fill_random(rw, rng, 0.4f);

    Tensor bq = Tensor::vec(D), bk = Tensor::vec(D),
           bv = Tensor::vec(D), bo = Tensor::vec(D);
    if (with_bias) {
        fill_random(bq, rng, 0.2f); fill_random(bk, rng, 0.2f);
        fill_random(bv, rng, 0.2f); fill_random(bo, rng, 0.2f);
    }
    const Tensor* pbq = with_bias ? &bq : nullptr;
    const Tensor* pbk = with_bias ? &bk : nullptr;
    const Tensor* pbv = with_bias ? &bv : nullptr;
    const Tensor* pbo = with_bias ? &bo : nullptr;

    Tensor O_c;
    brotensor::self_attention_decomposed_rel_pos_forward(
        X, Wq, pbq, Wk, pbk, Wv, pbv, Wo, pbo, rh, rw, H, gh, gw, scale, O_c);

    Tensor gX = X.to(gpu_device());
    Tensor gWq = Wq.to(gpu_device()), gWk = Wk.to(gpu_device()),
           gWv = Wv.to(gpu_device()), gWo = Wo.to(gpu_device());
    Tensor grh = rh.to(gpu_device()), grw = rw.to(gpu_device());
    Tensor gbq = bq.to(gpu_device()), gbk = bk.to(gpu_device()),
           gbv = bv.to(gpu_device()), gbo = bo.to(gpu_device());
    const Tensor* gpbq = with_bias ? &gbq : nullptr;
    const Tensor* gpbk = with_bias ? &gbk : nullptr;
    const Tensor* gpbv = with_bias ? &gbv : nullptr;
    const Tensor* gpbo = with_bias ? &gbo : nullptr;

    Tensor gO;
    brotensor::self_attention_decomposed_rel_pos_forward(
        gX, gWq, gpbq, gWk, gpbk, gWv, gpbv, gWo, gpbo, grh, grw,
        H, gh, gw, scale, gO);

    compare_tensors(O_c, download_to_host(gO), "rdp.O", 1e-4f, 1e-3f);
}

void run_bf16(int gh, int gw, int D, int H, bool with_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int L = gh * gw, dh = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    Tensor X  = Tensor::mat(L, D);
    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    Tensor rh = Tensor::mat(2 * gh - 1, dh), rw = Tensor::mat(2 * gw - 1, dh);
    fill_random(X, rng, 0.5f);
    fill_random(Wq, rng, 0.3f); fill_random(Wk, rng, 0.3f);
    fill_random(Wv, rng, 0.3f); fill_random(Wo, rng, 0.3f);
    fill_random(rh, rng, 0.4f); fill_random(rw, rng, 0.4f);

    Tensor bq = Tensor::vec(D), bk = Tensor::vec(D),
           bv = Tensor::vec(D), bo = Tensor::vec(D);
    if (with_bias) {
        fill_random(bq, rng, 0.2f); fill_random(bk, rng, 0.2f);
        fill_random(bv, rng, 0.2f); fill_random(bo, rng, 0.2f);
    }
    const Tensor* pbq = with_bias ? &bq : nullptr;
    const Tensor* pbk = with_bias ? &bk : nullptr;
    const Tensor* pbv = with_bias ? &bv : nullptr;
    const Tensor* pbo = with_bias ? &bo : nullptr;

    Tensor O_c;
    brotensor::self_attention_decomposed_rel_pos_forward(
        X, Wq, pbq, Wk, pbk, Wv, pbv, Wo, pbo, rh, rw, H, gh, gw, scale, O_c);

    Tensor gX = to_bf16_gpu(X);
    Tensor gWq = to_bf16_gpu(Wq), gWk = to_bf16_gpu(Wk),
           gWv = to_bf16_gpu(Wv), gWo = to_bf16_gpu(Wo);
    Tensor grh = to_bf16_gpu(rh), grw = to_bf16_gpu(rw);
    Tensor gbq = to_bf16_gpu(bq), gbk = to_bf16_gpu(bk),
           gbv = to_bf16_gpu(bv), gbo = to_bf16_gpu(bo);
    const Tensor* gpbq = with_bias ? &gbq : nullptr;
    const Tensor* gpbk = with_bias ? &gbk : nullptr;
    const Tensor* gpbv = with_bias ? &gbv : nullptr;
    const Tensor* gpbo = with_bias ? &gbo : nullptr;

    Tensor gO;
    brotensor::self_attention_decomposed_rel_pos_forward(
        gX, gWq, gpbq, gWk, gpbk, gWv, gpbv, gWo, gpbo, grh, grw,
        H, gh, gw, scale, gO);

    compare_tensors(O_c, bf16_host_to_f32(download_to_host(gO)),
                    "rdp.O.bf16", 8e-2f, 8e-2f);
}

// Windowed variant: rel-pos sized for the window, padding handled internally.
void run_windowed_fp32(int gh, int gw, int window, int D, int H,
                       bool with_bias, uint64_t seed) {
    SplitMix64 rng(seed);
    const int L = gh * gw, dh = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    Tensor X  = Tensor::mat(L, D);
    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    Tensor rh = Tensor::mat(2 * window - 1, dh), rw = Tensor::mat(2 * window - 1, dh);
    fill_random(X, rng, 0.5f);
    fill_random(Wq, rng, 0.3f); fill_random(Wk, rng, 0.3f);
    fill_random(Wv, rng, 0.3f); fill_random(Wo, rng, 0.3f);
    fill_random(rh, rng, 0.4f); fill_random(rw, rng, 0.4f);

    Tensor bq = Tensor::vec(D), bk = Tensor::vec(D),
           bv = Tensor::vec(D), bo = Tensor::vec(D);
    if (with_bias) {
        fill_random(bq, rng, 0.2f); fill_random(bk, rng, 0.2f);
        fill_random(bv, rng, 0.2f); fill_random(bo, rng, 0.2f);
    }
    const Tensor* pbq = with_bias ? &bq : nullptr;
    const Tensor* pbk = with_bias ? &bk : nullptr;
    const Tensor* pbv = with_bias ? &bv : nullptr;
    const Tensor* pbo = with_bias ? &bo : nullptr;

    Tensor O_c;
    brotensor::self_attention_decomposed_rel_pos_windowed_forward(
        X, Wq, pbq, Wk, pbk, Wv, pbv, Wo, pbo, rh, rw,
        H, gh, gw, window, scale, O_c);

    Tensor gX = X.to(gpu_device());
    Tensor gWq = Wq.to(gpu_device()), gWk = Wk.to(gpu_device()),
           gWv = Wv.to(gpu_device()), gWo = Wo.to(gpu_device());
    Tensor grh = rh.to(gpu_device()), grw = rw.to(gpu_device());
    Tensor gbq = bq.to(gpu_device()), gbk = bk.to(gpu_device()),
           gbv = bv.to(gpu_device()), gbo = bo.to(gpu_device());
    const Tensor* gpbq = with_bias ? &gbq : nullptr;
    const Tensor* gpbk = with_bias ? &gbk : nullptr;
    const Tensor* gpbv = with_bias ? &gbv : nullptr;
    const Tensor* gpbo = with_bias ? &gbo : nullptr;

    Tensor gO;
    brotensor::self_attention_decomposed_rel_pos_windowed_forward(
        gX, gWq, gpbq, gWk, gpbk, gWv, gpbv, gWo, gpbo, grh, grw,
        H, gh, gw, window, scale, gO);

    compare_tensors(O_c, download_to_host(gO), "rdp.win.O", 1e-4f, 1e-3f);
}

void run_windowed_bf16(int gh, int gw, int window, int D, int H, uint64_t seed) {
    SplitMix64 rng(seed);
    const int L = gh * gw, dh = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    Tensor X  = Tensor::mat(L, D);
    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    Tensor rh = Tensor::mat(2 * window - 1, dh), rw = Tensor::mat(2 * window - 1, dh);
    fill_random(X, rng, 0.5f);
    fill_random(Wq, rng, 0.3f); fill_random(Wk, rng, 0.3f);
    fill_random(Wv, rng, 0.3f); fill_random(Wo, rng, 0.3f);
    fill_random(rh, rng, 0.4f); fill_random(rw, rng, 0.4f);

    Tensor bq = Tensor::vec(D), bk = Tensor::vec(D),
           bv = Tensor::vec(D), bo = Tensor::vec(D);
    fill_random(bq, rng, 0.2f); fill_random(bk, rng, 0.2f);
    fill_random(bv, rng, 0.2f); fill_random(bo, rng, 0.2f);

    Tensor O_c;
    brotensor::self_attention_decomposed_rel_pos_windowed_forward(
        X, Wq, &bq, Wk, &bk, Wv, &bv, Wo, &bo, rh, rw,
        H, gh, gw, window, scale, O_c);

    Tensor gX = to_bf16_gpu(X);
    Tensor gWq = to_bf16_gpu(Wq), gWk = to_bf16_gpu(Wk),
           gWv = to_bf16_gpu(Wv), gWo = to_bf16_gpu(Wo);
    Tensor grh = to_bf16_gpu(rh), grw = to_bf16_gpu(rw);
    Tensor gbq = to_bf16_gpu(bq), gbk = to_bf16_gpu(bk),
           gbv = to_bf16_gpu(bv), gbo = to_bf16_gpu(bo);

    Tensor gO;
    brotensor::self_attention_decomposed_rel_pos_windowed_forward(
        gX, gWq, &gbq, gWk, &gbk, gWv, &gbv, gWo, &gbo, grh, grw,
        H, gh, gw, window, scale, gO);

    compare_tensors(O_c, bf16_host_to_f32(download_to_host(gO)),
                    "rdp.win.O.bf16", 8e-2f, 8e-2f);
}

}  // namespace

BT_PARITY_TEST(rdp_fp32_g4x4_D32_h4_nobias) { run_fp32(4, 4, 32, 4, false, 0x7001ull); }
BT_PARITY_TEST(rdp_fp32_g4x4_D32_h4_bias)   { run_fp32(4, 4, 32, 4, true,  0x7002ull); }
BT_PARITY_TEST(rdp_fp32_g2x3_D16_h2_bias)   { run_fp32(2, 3, 16, 2, true,  0x7003ull); }
BT_PARITY_TEST(rdp_fp32_g8x8_D64_h8_bias)   { run_fp32(8, 8, 64, 8, true,  0x7004ull); }
BT_PARITY_TEST(rdp_fp32_g5x7_D24_h3_bias)   { run_fp32(5, 7, 24, 3, true,  0x7005ull); }

BT_PARITY_TEST(rdp_bf16_g4x4_D32_h4_bias) { run_bf16(4, 4, 32, 4, true, 0x7101ull); }
BT_PARITY_TEST(rdp_bf16_g8x8_D64_h8_bias) { run_bf16(8, 8, 64, 8, true, 0x7102ull); }

// Windowed: window==grid (no pad), exact multiple, and padded grids.
BT_PARITY_TEST(rdp_win_fp32_g4x4_w4_nobias) { run_windowed_fp32(4, 4, 4, 32, 4, false, 0x7201ull); }
BT_PARITY_TEST(rdp_win_fp32_g8x8_w4_bias)   { run_windowed_fp32(8, 8, 4, 32, 4, true,  0x7202ull); }
BT_PARITY_TEST(rdp_win_fp32_g10x10_w4_bias) { run_windowed_fp32(10, 10, 4, 32, 4, true, 0x7203ull); }
BT_PARITY_TEST(rdp_win_fp32_g7x5_w3_bias)   { run_windowed_fp32(7, 5, 3, 24, 3, true,  0x7204ull); }

BT_PARITY_TEST(rdp_win_bf16_g10x10_w4_bias) { run_windowed_bf16(10, 10, 4, 32, 4, 0x7301ull); }

int main() { return run_all("self_attention_decomposed_rel_pos cpu/gpu parity"); }
