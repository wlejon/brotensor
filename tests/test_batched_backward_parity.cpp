// CPU↔GPU parity tests for the batched (training) backward kernels and the
// batched per-sample loss kernels:
//   linear_backward_batched / relu_backward_batched / tanh_backward_batched
//   mse_vec_per_sample / softmax_xent_fused_batched
//
// Each test builds host inputs, runs the CPU backend op, builds GPU copies of
// the same inputs, runs the GPU backend op, and compares the results.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

// ─── linear_backward_batched ───────────────────────────────────────────────
//
// dX overwritten; dW / dB accumulate. Pre-fill dW / dB with an identical
// non-zero baseline on both backends to verify the accumulation contract.

void run_linear_backward_batched(int B, int in_dim, int out_dim,
                                 uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = Tensor::mat(out_dim, in_dim);
    Tensor X_BD = Tensor::mat(B, in_dim);
    Tensor dY_BD = Tensor::mat(B, out_dim);
    fill_random(W, rng);
    fill_random(X_BD, rng);
    fill_random(dY_BD, rng);

    // Non-zero accumulation baseline, identical on both backends.
    Tensor dW_init = Tensor::mat(out_dim, in_dim);
    Tensor dB_init = Tensor::vec(out_dim);
    fill_random(dW_init, rng, 0.25f);
    fill_random(dB_init, rng, 0.25f);

    // CPU path.
    Tensor dX_cpu = Tensor::mat(B, in_dim);
    Tensor dW_cpu = dW_init;
    Tensor dB_cpu = dB_init;
    brotensor::linear_backward_batched(W, X_BD, dY_BD, dX_cpu, dW_cpu, dB_cpu);

    // GPU path with the same starting accumulators.
    Tensor gW = W.to(gpu_device());
    Tensor gX = X_BD.to(gpu_device());
    Tensor gdY = dY_BD.to(gpu_device());
    Tensor gdX = Tensor::zeros_on(gpu_device(), B, in_dim);
    Tensor gdW = dW_init.to(gpu_device());
    Tensor gdB = dB_init.to(gpu_device());
    brotensor::linear_backward_batched(gW, gX, gdY, gdX, gdW, gdB);

    compare_tensors(dX_cpu, download_to_host(gdX), "linear_backward_batched.dX");
    compare_tensors(dW_cpu, download_to_host(gdW), "linear_backward_batched.dW");
    compare_tensors(dB_cpu, download_to_host(gdB), "linear_backward_batched.dB");
}

// ─── relu_backward_batched ─────────────────────────────────────────────────
//
// dX = dY * (X > 0); reads the forward input X.

void run_relu_backward_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_BD = Tensor::mat(B, D);
    Tensor dY_BD = Tensor::mat(B, D);
    fill_random(X_BD, rng);
    fill_random(dY_BD, rng);

    Tensor dX_cpu = Tensor::mat(B, D);
    brotensor::relu_backward_batched(X_BD, dY_BD, dX_cpu);

    Tensor gX = X_BD.to(gpu_device());
    Tensor gdY = dY_BD.to(gpu_device());
    Tensor gdX = Tensor::zeros_on(gpu_device(), B, D);
    brotensor::relu_backward_batched(gX, gdY, gdX);

    compare_tensors(dX_cpu, download_to_host(gdX), "relu_backward_batched");
}

// ─── tanh_backward_batched ─────────────────────────────────────────────────
//
// dX = dY * (1 - Y*Y); reads the forward OUTPUT Y. Compute Y first via
// tanh_forward_batched.

void run_tanh_backward_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_BD = Tensor::mat(B, D);
    Tensor dY_BD = Tensor::mat(B, D);
    fill_random(X_BD, rng);
    fill_random(dY_BD, rng);

    // Forward output Y on CPU.
    Tensor Y_cpu = Tensor::mat(B, D);
    brotensor::tanh_forward_batched(X_BD, Y_cpu);
    Tensor dX_cpu = Tensor::mat(B, D);
    brotensor::tanh_backward_batched(Y_cpu, dY_BD, dX_cpu);

    // Forward output Y on GPU.
    Tensor gX = X_BD.to(gpu_device());
    Tensor gY = Tensor::zeros_on(gpu_device(), B, D);
    brotensor::tanh_forward_batched(gX, gY);
    Tensor gdY = dY_BD.to(gpu_device());
    Tensor gdX = Tensor::zeros_on(gpu_device(), B, D);
    brotensor::tanh_backward_batched(gY, gdY, gdX);

    compare_tensors(Y_cpu, download_to_host(gY), "tanh_forward_batched");
    compare_tensors(dX_cpu, download_to_host(gdX), "tanh_backward_batched");
}

// ─── mse_vec_per_sample ────────────────────────────────────────────────────
//
// pred / target / dPred / loss_per_sample all (B, 1).
//   dPred           = pred - target
//   loss_per_sample = 0.5 * (pred - target)^2

void run_mse_vec_per_sample(int B, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor pred = Tensor::vec(B), target = Tensor::vec(B);
    fill_random(pred, rng);
    fill_random(target, rng);

    Tensor dPred_cpu = Tensor::vec(B), loss_cpu = Tensor::vec(B);
    brotensor::mse_vec_per_sample(pred, target, dPred_cpu, loss_cpu);

    Tensor gpred = pred.to(gpu_device());
    Tensor gtarget = target.to(gpu_device());
    Tensor gdPred = Tensor::zeros_on(gpu_device(), B, 1);
    Tensor gloss = Tensor::zeros_on(gpu_device(), B, 1);
    brotensor::mse_vec_per_sample(gpred, gtarget, gdPred, gloss);

    compare_tensors(dPred_cpu, download_to_host(gdPred),
                    "mse_vec_per_sample.dPred");
    compare_tensors(loss_cpu, download_to_host(gloss),
                    "mse_vec_per_sample.loss");
}

// ─── softmax_xent_fused_batched ────────────────────────────────────────────
//
// logits/target/probs/dLogits all (B, L) where L = head_offsets.back().
// d_head_offsets: length n_heads+1, cumulative. For the CPU call the offsets
// and mask must be host-resident; for the GPU call they must be GPU-resident.

void run_softmax_xent_fused_batched(int B, int n_heads,
                                    const std::vector<int>& head_offsets,
                                    bool with_mask, uint64_t seed) {
    BT_CHECK(static_cast<int>(head_offsets.size()) == n_heads + 1);
    const int L = head_offsets.back();
    SplitMix64 rng(seed);

    Tensor logits = Tensor::mat(B, L);
    fill_random(logits, rng);

    // Optional mask: keep at least one valid entry per head-slice.
    std::vector<float> mask_vec;
    const bool use_mask = with_mask;
    if (use_mask) {
        mask_vec.assign(static_cast<size_t>(B) * L, 1.0f);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_heads; ++h) {
                const int s = head_offsets[h], e = head_offsets[h + 1];
                // Mask out roughly the back half of each slice but always
                // keep the first entry valid.
                for (int i = s; i < e; ++i) {
                    const int local = i - s;
                    const int width = e - s;
                    if (local > 0 && local >= (width + 1) / 2)
                        mask_vec[static_cast<size_t>(b) * L + i] = 0.0f;
                }
            }
        }
    }

    // Build per-head-slice valid probability distributions in target_BL.
    Tensor target = Tensor::mat(B, L);
    target.zero();
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            const int s = head_offsets[h], e = head_offsets[h + 1];
            float tsum = 0.0f;
            for (int i = s; i < e; ++i) {
                if (use_mask &&
                    mask_vec[static_cast<size_t>(b) * L + i] == 0.0f)
                    continue;
                const float v = rng.next_f01() + 0.05f;
                target[static_cast<size_t>(b) * L + i] = v;
                tsum += v;
            }
            if (tsum > 0.0f)
                for (int i = s; i < e; ++i)
                    target[static_cast<size_t>(b) * L + i] /= tsum;
        }
    }

    // CPU path — host-resident offsets and mask.
    Tensor probs_cpu = Tensor::mat(B, L);
    Tensor dLogits_cpu = Tensor::mat(B, L);
    Tensor loss_cpu = Tensor::vec(B);
    brotensor::softmax_xent_fused_batched(
        logits, target,
        use_mask ? mask_vec.data() : nullptr,
        head_offsets.data(), n_heads,
        probs_cpu, dLogits_cpu, loss_cpu);

    // GPU path — device-resident offsets and mask.
    Tensor glogits = logits.to(gpu_device());
    Tensor gtarget = target.to(gpu_device());
    Tensor gprobs = Tensor::zeros_on(gpu_device(), B, L);
    Tensor gdLogits = Tensor::zeros_on(gpu_device(), B, L);
    Tensor gloss = Tensor::zeros_on(gpu_device(), B, 1);

    Tensor d_off = upload_offsets(head_offsets);
    Tensor d_mask_buf = upload_mask(use_mask ? &mask_vec : nullptr);
    const float* d_mask =
        use_mask ? static_cast<const float*>(d_mask_buf.data) : nullptr;

    brotensor::softmax_xent_fused_batched(
        glogits, gtarget, d_mask,
        static_cast<const int*>(d_off.data), n_heads,
        gprobs, gdLogits, gloss);

    compare_tensors(probs_cpu, download_to_host(gprobs),
                    "softmax_xent_fused_batched.probs");
    compare_tensors(dLogits_cpu, download_to_host(gdLogits),
                    "softmax_xent_fused_batched.dLogits");
    compare_tensors(loss_cpu, download_to_host(gloss),
                    "softmax_xent_fused_batched.loss");
}

// ─── bce_with_logits_fused_batched ─────────────────────────────────────────
//
// logits/target/probs/dLogits all (B, L); optional (B, L) mask. pos_weight
// scales the positive-class loss / gradient term.

void run_bce_with_logits_fused_batched(int B, int L, float pos_weight,
                                       bool with_mask, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor logits = Tensor::mat(B, L);
    fill_random(logits, rng);

    // Targets: mix of soft and hard binary in [0,1].
    Tensor target = Tensor::mat(B, L);
    for (int i = 0; i < B * L; ++i) {
        const float r = rng.next_f01();
        target.host_f32_mut()[i] = (r < 0.33f) ? 0.0f
                                    : (r < 0.66f) ? 1.0f
                                                  : r;
    }

    // Optional mask: keep at least one valid entry per row.
    std::vector<float> mask_vec;
    if (with_mask) {
        mask_vec.assign(static_cast<size_t>(B) * L, 1.0f);
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < L; ++i) {
                if (i > 0 && (i % 2) == 1)
                    mask_vec[static_cast<size_t>(b) * L + i] = 0.0f;
            }
        }
    }

    // CPU path.
    Tensor probs_cpu = Tensor::mat(B, L);
    Tensor dLogits_cpu = Tensor::mat(B, L);
    Tensor loss_cpu = Tensor::vec(B);
    brotensor::bce_with_logits_fused_batched(
        logits, target,
        with_mask ? mask_vec.data() : nullptr,
        pos_weight, probs_cpu, dLogits_cpu, loss_cpu);

    // GPU path.
    Tensor glogits = logits.to(gpu_device());
    Tensor gtarget = target.to(gpu_device());
    Tensor gprobs = Tensor::zeros_on(gpu_device(), B, L);
    Tensor gdLogits = Tensor::zeros_on(gpu_device(), B, L);
    Tensor gloss = Tensor::zeros_on(gpu_device(), B, 1);

    Tensor d_mask_buf = upload_mask(with_mask ? &mask_vec : nullptr);
    const float* d_mask =
        with_mask ? static_cast<const float*>(d_mask_buf.data) : nullptr;

    brotensor::bce_with_logits_fused_batched(
        glogits, gtarget, d_mask, pos_weight,
        gprobs, gdLogits, gloss);

    compare_tensors(probs_cpu, download_to_host(gprobs),
                    "bce_with_logits_fused_batched.probs");
    compare_tensors(dLogits_cpu, download_to_host(gdLogits),
                    "bce_with_logits_fused_batched.dLogits");
    compare_tensors(loss_cpu, download_to_host(gloss),
                    "bce_with_logits_fused_batched.loss");
}

} // namespace

// ─── linear_backward_batched ───────────────────────────────────────────────
BT_PARITY_TEST(linear_backward_batched_B1)  {
    run_linear_backward_batched(1, 32, 16, 0x100ull);
}
BT_PARITY_TEST(linear_backward_batched_B8)  {
    run_linear_backward_batched(8, 64, 32, 0x101ull);
}
BT_PARITY_TEST(linear_backward_batched_skinny) {
    run_linear_backward_batched(8, 1, 7, 0x102ull);
}

// ─── relu_backward_batched ─────────────────────────────────────────────────
BT_PARITY_TEST(relu_backward_batched_B1)  { run_relu_backward_batched(1, 32, 0x110ull); }
BT_PARITY_TEST(relu_backward_batched_B8)  { run_relu_backward_batched(8, 64, 0x111ull); }
BT_PARITY_TEST(relu_backward_batched_B64) { run_relu_backward_batched(64, 32, 0x112ull); }

// ─── tanh_backward_batched ─────────────────────────────────────────────────
BT_PARITY_TEST(tanh_backward_batched_B1)  { run_tanh_backward_batched(1, 32, 0x120ull); }
BT_PARITY_TEST(tanh_backward_batched_B8)  { run_tanh_backward_batched(8, 64, 0x121ull); }
BT_PARITY_TEST(tanh_backward_batched_B64) { run_tanh_backward_batched(64, 32, 0x122ull); }

// ─── mse_vec_per_sample ────────────────────────────────────────────────────
BT_PARITY_TEST(mse_vec_per_sample_B1)   { run_mse_vec_per_sample(1, 0x130ull); }
BT_PARITY_TEST(mse_vec_per_sample_B8)   { run_mse_vec_per_sample(8, 0x131ull); }
BT_PARITY_TEST(mse_vec_per_sample_B64)  { run_mse_vec_per_sample(64, 0x132ull); }

// ─── softmax_xent_fused_batched ────────────────────────────────────────────
BT_PARITY_TEST(sxfb_1head_B1_nomask) {
    run_softmax_xent_fused_batched(1, 1, {0, 32}, false, 0x140ull);
}
BT_PARITY_TEST(sxfb_1head_B8_nomask) {
    run_softmax_xent_fused_batched(8, 1, {0, 64}, false, 0x141ull);
}
BT_PARITY_TEST(sxfb_1head_B8_mask) {
    run_softmax_xent_fused_batched(8, 1, {0, 64}, true, 0x142ull);
}
BT_PARITY_TEST(sxfb_3heads_B8_nomask) {
    run_softmax_xent_fused_batched(8, 3, {0, 16, 48, 64}, false, 0x143ull);
}
BT_PARITY_TEST(sxfb_3heads_B8_mask) {
    run_softmax_xent_fused_batched(8, 3, {0, 16, 48, 64}, true, 0x144ull);
}
BT_PARITY_TEST(sxfb_4heads_B1_mask) {
    run_softmax_xent_fused_batched(1, 4, {0, 8, 16, 24, 32}, true, 0x145ull);
}

// ─── bce_with_logits_fused_batched ─────────────────────────────────────────
BT_PARITY_TEST(bce_B1_L1_nomask) {
    run_bce_with_logits_fused_batched(1, 1, 1.0f, false, 0x150ull);
}
BT_PARITY_TEST(bce_B8_L1_nomask) {
    run_bce_with_logits_fused_batched(8, 1, 1.0f, false, 0x151ull);
}
BT_PARITY_TEST(bce_B8_L4_mask) {
    run_bce_with_logits_fused_batched(8, 4, 1.0f, true, 0x152ull);
}
BT_PARITY_TEST(bce_B4_L2_posweight) {
    run_bce_with_logits_fused_batched(4, 2, 3.0f, false, 0x153ull);
}

int main() { return run_all("batched-backward cpu/gpu parity"); }
