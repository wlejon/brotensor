// Parity tests for the batched (inference-only) GPU kernels.
//
// For each batched op we compare its B-row output against B independent
// calls to the corresponding single-sample kernel.

#include "parity_helpers.h"

#include <brotensor/ops.h>

#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

// ─── linear_forward_batched ────────────────────────────────────────────────

static void run_linear_batched(int B, int in_dim, int out_dim, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = Tensor::mat(out_dim, in_dim), b = Tensor::vec(out_dim);
    fill_random(W, rng);
    fill_random(b, rng);

    // Build (B, in_dim) input matrix.
    Tensor X_BD = Tensor::mat(B, in_dim);
    fill_random(X_BD, rng);

    // Reference: B sequential single-sample GPU calls into a (B, out_dim) buffer.
    Tensor gW = W.to(gpu_device()), gb = b.to(gpu_device()),
           gX_BD = X_BD.to(gpu_device());
    Tensor Y_ref = Tensor::mat(B, out_dim);
    for (int i = 0; i < B; ++i) {
        Tensor xi = Tensor::vec(in_dim);
        for (int j = 0; j < in_dim; ++j)
            xi[j] = X_BD[static_cast<size_t>(i) * in_dim + j];
        Tensor gxi = xi.to(gpu_device());
        Tensor gyi = Tensor::zeros_on(gpu_device(), out_dim, 1);
        brotensor::linear_forward(gW, gb, gxi, gyi);
        Tensor yi = download_to_host(gyi);
        for (int j = 0; j < out_dim; ++j)
            Y_ref[static_cast<size_t>(i) * out_dim + j] = yi[j];
    }

    // Batched call.
    Tensor gY_BD;
    brotensor::linear_forward_batched(gW, gb, gX_BD, gY_BD);
    Tensor Y_batched = download_to_host(gY_BD);
    BT_CHECK(Y_batched.rows == B);
    BT_CHECK(Y_batched.cols == out_dim);
    compare_tensors(Y_ref, Y_batched, "linear_forward_batched");
}

BT_PARITY_TEST(linear_batched_B1)   { run_linear_batched(1,  16, 32, 0xC1ull); }
BT_PARITY_TEST(linear_batched_B4)   { run_linear_batched(4,  64, 32, 0xC2ull); }
BT_PARITY_TEST(linear_batched_B64)  { run_linear_batched(64, 128, 96, 0xC3ull); }
BT_PARITY_TEST(linear_batched_skinny) { run_linear_batched(8, 1, 7, 0xC4ull); }

// ─── relu_forward_batched ──────────────────────────────────────────────────

static void run_relu_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_BD = Tensor::mat(B, D);
    fill_random(X_BD, rng);

    // Reference: per-row single-sample kernel.
    Tensor Y_ref = Tensor::mat(B, D);
    for (int i = 0; i < B; ++i) {
        Tensor xi = Tensor::vec(D);
        for (int j = 0; j < D; ++j)
            xi[j] = X_BD[static_cast<size_t>(i) * D + j];
        Tensor gxi = xi.to(gpu_device());
        Tensor gyi = Tensor::zeros_on(gpu_device(), D, 1);
        brotensor::relu_forward(gxi, gyi);
        Tensor yi = download_to_host(gyi);
        for (int j = 0; j < D; ++j)
            Y_ref[static_cast<size_t>(i) * D + j] = yi[j];
    }

    Tensor gX = X_BD.to(gpu_device());
    Tensor gY;
    brotensor::relu_forward_batched(gX, gY);
    Tensor Y_batched = download_to_host(gY);
    compare_tensors(Y_ref, Y_batched, "relu_forward_batched");
}

BT_PARITY_TEST(relu_batched_B1)  { run_relu_batched(1, 64, 0xD1ull); }
BT_PARITY_TEST(relu_batched_B4)  { run_relu_batched(4, 64, 0xD2ull); }
BT_PARITY_TEST(relu_batched_B64) { run_relu_batched(64, 32, 0xD3ull); }

// ─── tanh_forward_batched ──────────────────────────────────────────────────

static void run_tanh_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_BD = Tensor::mat(B, D);
    fill_random(X_BD, rng);

    Tensor Y_ref = Tensor::mat(B, D);
    for (int i = 0; i < B; ++i) {
        Tensor xi = Tensor::vec(D);
        for (int j = 0; j < D; ++j)
            xi[j] = X_BD[static_cast<size_t>(i) * D + j];
        Tensor gxi = xi.to(gpu_device());
        Tensor gyi = Tensor::zeros_on(gpu_device(), D, 1);
        brotensor::tanh_forward(gxi, gyi);
        Tensor yi = download_to_host(gyi);
        for (int j = 0; j < D; ++j)
            Y_ref[static_cast<size_t>(i) * D + j] = yi[j];
    }

    Tensor gX = X_BD.to(gpu_device());
    Tensor gY;
    brotensor::tanh_forward_batched(gX, gY);
    Tensor Y_batched = download_to_host(gY);
    compare_tensors(Y_ref, Y_batched, "tanh_forward_batched");
}

BT_PARITY_TEST(tanh_batched_B1)  { run_tanh_batched(1, 8, 0xE1ull); }
BT_PARITY_TEST(tanh_batched_B4)  { run_tanh_batched(4, 16, 0xE2ull); }
BT_PARITY_TEST(tanh_batched_B64) { run_tanh_batched(64, 1, 0xE3ull); }

// ─── add_inplace_batched ───────────────────────────────────────────────────

static void run_add_batched(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Y_init = Tensor::mat(B, D), X = Tensor::mat(B, D);
    fill_random(Y_init, rng);
    fill_random(X, rng);

    // Reference: per-row single-sample kernel.
    Tensor Y_ref = Y_init;
    for (int i = 0; i < B; ++i) {
        Tensor yi = Tensor::vec(D), xi = Tensor::vec(D);
        for (int j = 0; j < D; ++j) {
            yi[j] = Y_ref[static_cast<size_t>(i) * D + j];
            xi[j] = X[static_cast<size_t>(i) * D + j];
        }
        Tensor gyi = yi.to(gpu_device()), gxi = xi.to(gpu_device());
        brotensor::add_inplace(gyi, gxi);
        Tensor out = download_to_host(gyi);
        for (int j = 0; j < D; ++j)
            Y_ref[static_cast<size_t>(i) * D + j] = out[j];
    }

    Tensor gY = Y_init.to(gpu_device()), gX = X.to(gpu_device());
    brotensor::add_inplace_batched(gY, gX);
    Tensor Y_batched = download_to_host(gY);
    compare_tensors(Y_ref, Y_batched, "add_inplace_batched");
}

BT_PARITY_TEST(add_batched_B1)  { run_add_batched(1, 8, 0xF1ull); }
BT_PARITY_TEST(add_batched_B4)  { run_add_batched(4, 16, 0xF2ull); }
BT_PARITY_TEST(add_batched_B64) { run_add_batched(64, 32, 0xF3ull); }

int main() { return run_all("batched ops cpu/gpu parity"); }
