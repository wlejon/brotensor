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
using brotensor::Dtype;

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

// ─── linear_forward_batched, 16-bit weights ────────────────────────────────
//
// W stored FP16/BF16 with FP32 activations and accumulation. Both backends
// widen the identical 16-bit weight values to FP32 before the dot product, so
// CPU vs GPU differ only by reduction order — the standard tolerance holds.

static void run_linear_batched_w16(int B, int in_dim, int out_dim,
                                   brotensor::Dtype wdt, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = Tensor::mat(out_dim, in_dim), b = Tensor::vec(out_dim);
    fill_random(W, rng);
    fill_random(b, rng);
    Tensor X_BD = Tensor::mat(B, in_dim);
    fill_random(X_BD, rng);

    // Round W to the 16-bit storage dtype on the host.
    Tensor W16 = Tensor::zeros_on(brotensor::Device::CPU, out_dim, in_dim, wdt);
    {
        const float* s = W.host_f32();
        uint16_t* d = wdt == Dtype::BF16 ? W16.host_bf16_mut() : W16.host_fp16_mut();
        for (int i = 0; i < W.size(); ++i)
            d[i] = wdt == Dtype::BF16 ? brotensor::fp32_to_bf16_bits(s[i])
                                      : brotensor::fp32_to_fp16_bits(s[i]);
    }

    // CPU reference: the CPU backend's own 16-bit-weight widening path.
    Tensor Y_ref;
    brotensor::linear_forward_batched(W16, b, X_BD, Y_ref);

    // GPU op over the same 16-bit weights.
    Tensor gW16 = W16.to(gpu_device()), gb = b.to(gpu_device()),
           gX_BD = X_BD.to(gpu_device());
    Tensor gY_BD;
    brotensor::linear_forward_batched(gW16, gb, gX_BD, gY_BD);
    Tensor Y_batched = download_to_host(gY_BD);
    BT_CHECK(Y_batched.rows == B);
    BT_CHECK(Y_batched.cols == out_dim);
    compare_tensors(Y_ref, Y_batched, "linear_forward_batched_w16",
                    1e-4f, 1e-4f);
}

// B<=32 even in_dim -> GEMV; in_dim%4!=0 exercises the T2 tail; odd in_dim
// and B=64 take the tiled wide-batch kernel.
BT_PARITY_TEST(linear_batched_bf16w_B1)   { run_linear_batched_w16(1, 512, 96, Dtype::BF16, 0xB161ull); }
BT_PARITY_TEST(linear_batched_bf16w_B4_tail) { run_linear_batched_w16(4, 30, 17, Dtype::BF16, 0xB162ull); }
BT_PARITY_TEST(linear_batched_bf16w_odd)  { run_linear_batched_w16(2, 33, 17, Dtype::BF16, 0xB163ull); }
BT_PARITY_TEST(linear_batched_bf16w_B64)  { run_linear_batched_w16(64, 128, 96, Dtype::BF16, 0xB164ull); }
BT_PARITY_TEST(linear_batched_fp16w_B1)   { run_linear_batched_w16(1, 512, 96, Dtype::FP16, 0xB165ull); }
BT_PARITY_TEST(linear_batched_fp16w_B64)  { run_linear_batched_w16(64, 128, 96, Dtype::FP16, 0xB166ull); }

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

// ─── linear_backward_batched BF16 ─────────────────────────────────────────

static void run_linear_backward_batched_bf16(int B, int in_dim, int out_dim,
                                             uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W_f32  = Tensor::mat(out_dim, in_dim);
    Tensor X_f32  = Tensor::mat(B, in_dim);
    Tensor dY_f32 = Tensor::mat(B, out_dim);
    fill_random(W_f32, rng);
    fill_random(X_f32, rng);
    fill_random(dY_f32, rng);

    // CPU FP32 reference for dX, dW, dB.
    Tensor dX_ref = Tensor::mat(B, in_dim);
    Tensor dW_ref = Tensor::mat(out_dim, in_dim);
    Tensor dB_ref = Tensor::vec(out_dim);
    dX_ref.zero(); dW_ref.zero(); dB_ref.zero();
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < in_dim; ++j) {
            float a = 0.0f;
            for (int i = 0; i < out_dim; ++i)
                a += W_f32[static_cast<size_t>(i)*in_dim+j] *
                     dY_f32[static_cast<size_t>(b)*out_dim+i];
            dX_ref[static_cast<size_t>(b)*in_dim+j] = a;
        }
    for (int i = 0; i < out_dim; ++i)
        for (int j = 0; j < in_dim; ++j) {
            float a = 0.0f;
            for (int b = 0; b < B; ++b)
                a += dY_f32[static_cast<size_t>(b)*out_dim+i] *
                     X_f32 [static_cast<size_t>(b)*in_dim +j];
            dW_ref[static_cast<size_t>(i)*in_dim+j] = a;
        }
    for (int i = 0; i < out_dim; ++i) {
        float a = 0.0f;
        for (int b = 0; b < B; ++b) a += dY_f32[static_cast<size_t>(b)*out_dim+i];
        dB_ref[i] = a;
    }

    // BF16 GPU run.
    Tensor gW  = to_bf16_gpu(W_f32);
    Tensor gX  = to_bf16_gpu(X_f32);
    Tensor gdY = to_bf16_gpu(dY_f32);
    Tensor gdX;
    Tensor gdW  = Tensor::zeros_on(gpu_device(), out_dim, in_dim, Dtype::BF16);
    Tensor gdB  = Tensor::zeros_on(gpu_device(), out_dim, 1,      Dtype::BF16);

    brotensor::linear_backward_batched(gW, gX, gdY, gdX, gdW, gdB);
    BT_CHECK(gdX.dtype == Dtype::BF16);
    BT_CHECK(gdW.dtype == Dtype::BF16);
    BT_CHECK(gdB.dtype == Dtype::BF16);

    Tensor dX_gpu = bf16_host_to_f32(download_to_host(gdX));
    Tensor dW_gpu = bf16_host_to_f32(download_to_host(gdW));
    Tensor dB_gpu = bf16_host_to_f32(download_to_host(gdB));

    compare_tensors(dX_ref, dX_gpu, "linear_bwd_batched_bf16.dX", 3e-2f, 3e-2f);
    compare_tensors(dW_ref, dW_gpu, "linear_bwd_batched_bf16.dW", 3e-2f, 3e-2f);
    compare_tensors(dB_ref, dB_gpu, "linear_bwd_batched_bf16.dB", 3e-2f, 3e-2f);
}

BT_PARITY_TEST(linear_bwd_batched_bf16_B3_7_5)  { run_linear_backward_batched_bf16(3,  7,  5,  0xBF01ull); }
BT_PARITY_TEST(linear_bwd_batched_bf16_B8_32_16){ run_linear_backward_batched_bf16(8,  32, 16, 0xBF02ull); }
BT_PARITY_TEST(linear_bwd_batched_bf16_B16_64_32){ run_linear_backward_batched_bf16(16, 64, 32, 0xBF03ull); }

// ─── linear_forward_batched_fp16 / _act, BF16 storage ──────────────────────
//
// Exercises the WMMA BF16 forward GEMM (fp16_internal::launch_matmul_ABT, BF16
// overload) and its float-staged store epilogue, including the fused
// bias + activation path. Y = X @ W^T (+ bias) (+ act).

static float ref_linear_act(int act, float v) {
    switch (act) {
        case 1:  return v > 0.0f ? v : 0.0f;                                  // relu
        case 2: { const float k = 0.7978845608f;                             // gelu(tanh)
                  const float u = k * (v + 0.044715f * v * v * v);
                  return 0.5f * v * (1.0f + std::tanh(u)); }
        case 3:  return 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f)); // gelu(exact)
        case 4:  return v / (1.0f + std::exp(-v));                            // silu
        case 5:  return v / (1.0f + std::exp(-1.702f * v));                   // quick_gelu
        default: return v;
    }
}

static void run_linear_forward_batched_bf16(int B, int in_dim, int out_dim,
                                            bool with_bias, int act,
                                            uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W_f32 = Tensor::mat(out_dim, in_dim);
    Tensor X_f32 = Tensor::mat(B, in_dim);
    Tensor b_f32 = Tensor::vec(out_dim);
    fill_random(W_f32, rng);
    fill_random(X_f32, rng);
    fill_random(b_f32, rng);

    // CPU FP32 reference.
    Tensor Y_ref = Tensor::mat(B, out_dim);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < out_dim; ++i) {
            float a = with_bias ? b_f32[i] : 0.0f;
            for (int k = 0; k < in_dim; ++k)
                a += X_f32[static_cast<size_t>(b)*in_dim+k] *
                     W_f32[static_cast<size_t>(i)*in_dim+k];
            Y_ref[static_cast<size_t>(b)*out_dim+i] = ref_linear_act(act, a);
        }

    // BF16 GPU run.
    Tensor gW = to_bf16_gpu(W_f32);
    Tensor gX = to_bf16_gpu(X_f32);
    Tensor gb = to_bf16_gpu(b_f32);
    Tensor gY;
    const Tensor* biasp = with_bias ? &gb : nullptr;
    if (act == 0)
        brotensor::linear_forward_batched_fp16(gW, biasp, gX, gY);
    else
        brotensor::linear_forward_batched_fp16_act(gW, biasp, gX, act, gY);
    BT_CHECK(gY.dtype == Dtype::BF16);
    BT_CHECK(gY.rows == B);
    BT_CHECK(gY.cols == out_dim);

    Tensor Y_gpu = bf16_host_to_f32(download_to_host(gY));
    const char* label = with_bias ? "linear_fwd_batched_bf16"
                                  : "linear_fwd_batched_bf16_nobias";
    compare_tensors(Y_ref, Y_gpu, label, 3e-2f, 3e-2f);
}

// Plain forward (bias + no-bias), both the WMMA path (B,N,K multiples of 8) and
// the naive fallback (skinny / unaligned).
BT_PARITY_TEST(linear_fwd_batched_bf16_B8_32_16)  { run_linear_forward_batched_bf16(8,  32, 16, true,  0, 0xBF11ull); }
BT_PARITY_TEST(linear_fwd_batched_bf16_B64_128_96){ run_linear_forward_batched_bf16(64, 128, 96, true,  0, 0xBF12ull); }
BT_PARITY_TEST(linear_fwd_batched_bf16_nobias)    { run_linear_forward_batched_bf16(16, 64, 32, false, 0, 0xBF13ull); }
BT_PARITY_TEST(linear_fwd_batched_bf16_skinny)    { run_linear_forward_batched_bf16(8,  7,  5,  true,  0, 0xBF14ull); }
// Fused activation epilogue over the float-staged BF16 store.
BT_PARITY_TEST(linear_fwd_batched_bf16_silu)      { run_linear_forward_batched_bf16(32, 64, 64, true,  4, 0xBF15ull); }
BT_PARITY_TEST(linear_fwd_batched_bf16_gelu)      { run_linear_forward_batched_bf16(32, 64, 64, true,  2, 0xBF16ull); }

// ─── linear_forward_batched large-B (grid.y > 65535) ──────────────────────
//
// Regression for the Kokoro CUDA error 9: linear_forward_batched mapped the
// batch dim onto gridDim.y, which CUDA caps at 65535. Anything that flattens
// long sequences into B (TTS vocoder frame batches, etc.) hit invalid-launch.
// The launcher now chunks B; this test exercises the boundary by running a
// large-B input as one call and comparing against two half-B calls of the
// same op — equivalence implies chunking is correct, without needing a
// per-row scalar reference.
static void run_linear_batched_large_B(int B, int in_dim, int out_dim,
                                       uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = Tensor::mat(out_dim, in_dim), b = Tensor::vec(out_dim);
    fill_random(W, rng);
    fill_random(b, rng);
    Tensor X_BD = Tensor::mat(B, in_dim);
    fill_random(X_BD, rng);

    Tensor gW = W.to(gpu_device()), gb = b.to(gpu_device()),
           gX_BD = X_BD.to(gpu_device());

    // One big batched call — must not raise CUDA error 9.
    Tensor gY_full;
    brotensor::linear_forward_batched(gW, gb, gX_BD, gY_full);
    Tensor Y_full = download_to_host(gY_full);
    BT_CHECK(Y_full.rows == B);
    BT_CHECK(Y_full.cols == out_dim);

    // Reference: split X into two halves, run batched on each, concatenate.
    const int B0 = B / 2;
    const int B1 = B - B0;
    Tensor X0 = Tensor::mat(B0, in_dim);
    Tensor X1 = Tensor::mat(B1, in_dim);
    std::memcpy(X0.data, X_BD.data,
                static_cast<size_t>(B0) * in_dim * sizeof(float));
    std::memcpy(X1.data,
                static_cast<float*>(X_BD.data)
                    + static_cast<size_t>(B0) * in_dim,
                static_cast<size_t>(B1) * in_dim * sizeof(float));
    Tensor gX0 = X0.to(gpu_device()), gX1 = X1.to(gpu_device());
    Tensor gY0, gY1;
    brotensor::linear_forward_batched(gW, gb, gX0, gY0);
    brotensor::linear_forward_batched(gW, gb, gX1, gY1);
    Tensor Y0 = download_to_host(gY0), Y1 = download_to_host(gY1);

    Tensor Y_ref = Tensor::mat(B, out_dim);
    std::memcpy(Y_ref.data, Y0.data,
                static_cast<size_t>(B0) * out_dim * sizeof(float));
    std::memcpy(static_cast<float*>(Y_ref.data)
                    + static_cast<size_t>(B0) * out_dim,
                Y1.data,
                static_cast<size_t>(B1) * out_dim * sizeof(float));

    compare_tensors(Y_ref, Y_full, "linear_forward_batched_largeB");
}

BT_PARITY_TEST(linear_batched_B_just_over_65535) {
    run_linear_batched_large_B(65536, 8, 4, 0xC101ull);
}
BT_PARITY_TEST(linear_batched_B_multi_chunk) {
    run_linear_batched_large_B(130000, 8, 4, 0xC102ull);
}

int main() { return run_all("batched ops cpu/gpu parity"); }
