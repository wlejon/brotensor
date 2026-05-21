// CPU↔GPU parity tests for brotensor::matmul and brotensor::matmul_backward.
//
// CHUNK 2. Row-major: C(M,N) = A(M,K) @ B(K,N).
//   - matmul_backward dA / dB ACCUMULATE (+=); the GPU atomic-adds partials
//     into the caller-provided buffers. The accumulation test below exercises
//     this by pre-filling dA / dB with a non-zero baseline.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_matmul(int M, int K, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = Tensor::mat(M, K);
    Tensor B = Tensor::mat(K, N);
    fill_random(A, rng);
    fill_random(B, rng);

    Tensor cpu_C;
    brotensor::matmul(A, B, cpu_C);

    Tensor gA = A.to(gpu_device());
    Tensor gB = B.to(gpu_device());
    Tensor gpu_C;
    brotensor::matmul(gA, gB, gpu_C);

    // Relax tolerance: K-length sums accumulate float rounding.
    compare_tensors(cpu_C, download_to_host(gpu_C), "matmul", 1e-4f, 1e-3f);
}

// Backward with zero-initialised dA / dB (gradient from scratch).
void run_matmul_bwd(int M, int K, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A  = Tensor::mat(M, K);
    Tensor B  = Tensor::mat(K, N);
    Tensor dC = Tensor::mat(M, N);
    fill_random(A, rng);
    fill_random(B, rng);
    fill_random(dC, rng);

    Tensor cpu_dA = Tensor::mat(M, K);  // zero-initialised
    Tensor cpu_dB = Tensor::mat(K, N);
    brotensor::matmul_backward(A, B, dC, cpu_dA, cpu_dB);

    Tensor gA  = A.to(gpu_device());
    Tensor gB  = B.to(gpu_device());
    Tensor gdC = dC.to(gpu_device());
    Tensor gpu_dA = Tensor::zeros_on(gpu_device(), M, K);
    Tensor gpu_dB = Tensor::zeros_on(gpu_device(), K, N);
    brotensor::matmul_backward(gA, gB, gdC, gpu_dA, gpu_dB);

    compare_tensors(cpu_dA, download_to_host(gpu_dA), "matmul_bwd_dA", 1e-4f, 1e-3f);
    compare_tensors(cpu_dB, download_to_host(gpu_dB), "matmul_bwd_dB", 1e-4f, 1e-3f);
}

// Backward where dA / dB carry a pre-existing baseline — verifies the +=
// accumulation contract holds identically on CPU and GPU.
void run_matmul_bwd_accum(int M, int K, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A  = Tensor::mat(M, K);
    Tensor B  = Tensor::mat(K, N);
    Tensor dC = Tensor::mat(M, N);
    Tensor dA0 = Tensor::mat(M, K);
    Tensor dB0 = Tensor::mat(K, N);
    fill_random(A, rng);
    fill_random(B, rng);
    fill_random(dC, rng);
    fill_random(dA0, rng);
    fill_random(dB0, rng);

    Tensor cpu_dA = dA0;               // deep copy of baseline (CPU)
    Tensor cpu_dB = dB0;
    brotensor::matmul_backward(A, B, dC, cpu_dA, cpu_dB);

    Tensor gA  = A.to(gpu_device());
    Tensor gB  = B.to(gpu_device());
    Tensor gdC = dC.to(gpu_device());
    Tensor gpu_dA = dA0.to(gpu_device());  // same baseline on GPU
    Tensor gpu_dB = dB0.to(gpu_device());
    brotensor::matmul_backward(gA, gB, gdC, gpu_dA, gpu_dB);

    compare_tensors(cpu_dA, download_to_host(gpu_dA), "matmul_bwd_accum_dA", 1e-4f, 1e-3f);
    compare_tensors(cpu_dB, download_to_host(gpu_dB), "matmul_bwd_accum_dB", 1e-4f, 1e-3f);
}

// ─── BF16 parity (GPU-only): BF16-on-CUDA vs FP32 CPU reference ─────────────

void run_matmul_bf16(int M, int K, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = Tensor::mat(M, K);
    Tensor B = Tensor::mat(K, N);
    fill_random(A, rng);
    fill_random(B, rng);

    Tensor cpu_C;
    brotensor::matmul(A, B, cpu_C);

    Tensor gA = to_bf16_gpu(A);
    Tensor gB = to_bf16_gpu(B);
    Tensor gpu_C;
    brotensor::matmul(gA, gB, gpu_C);

    // BF16: 8 mantissa bits over a K-length reduction — loose tolerance.
    compare_tensors(cpu_C, bf16_host_to_f32(download_to_host(gpu_C)),
                    "matmul_bf16", 5e-2f, 5e-2f);
}

void run_matmul_bwd_bf16(int M, int K, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A  = Tensor::mat(M, K);
    Tensor B  = Tensor::mat(K, N);
    Tensor dC = Tensor::mat(M, N);
    fill_random(A, rng);
    fill_random(B, rng);
    fill_random(dC, rng);

    Tensor cpu_dA = Tensor::mat(M, K);  // zero-initialised
    Tensor cpu_dB = Tensor::mat(K, N);
    brotensor::matmul_backward(A, B, dC, cpu_dA, cpu_dB);

    Tensor gA  = to_bf16_gpu(A);
    Tensor gB  = to_bf16_gpu(B);
    Tensor gdC = to_bf16_gpu(dC);
    Tensor gpu_dA = Tensor::zeros_on(gpu_device(), M, K, brotensor::Dtype::BF16);
    Tensor gpu_dB = Tensor::zeros_on(gpu_device(), K, N, brotensor::Dtype::BF16);
    brotensor::matmul_backward(gA, gB, gdC, gpu_dA, gpu_dB);

    // Backward over a reduction — keep K modest and tolerance loose.
    compare_tensors(cpu_dA, bf16_host_to_f32(download_to_host(gpu_dA)),
                    "matmul_bwd_bf16_dA", 1e-1f, 1e-1f);
    compare_tensors(cpu_dB, bf16_host_to_f32(download_to_host(gpu_dB)),
                    "matmul_bwd_bf16_dB", 1e-1f, 1e-1f);
}

} // namespace

// ─── forward ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(matmul_1x1x1)     { run_matmul(1, 1, 1, 0x5000ull); }
BT_PARITY_TEST(matmul_square)    { run_matmul(16, 16, 16, 0x5001ull); }
BT_PARITY_TEST(matmul_nonsquare) { run_matmul(7, 19, 11, 0x5002ull); }
BT_PARITY_TEST(matmul_wide)      { run_matmul(3, 40, 5, 0x5003ull); }
BT_PARITY_TEST(matmul_rowvec)    { run_matmul(1, 13, 9, 0x5004ull); }
BT_PARITY_TEST(matmul_colvec)    { run_matmul(9, 13, 1, 0x5005ull); }

// ─── backward ──────────────────────────────────────────────────────────────
BT_PARITY_TEST(matmul_bwd_1x1x1)     { run_matmul_bwd(1, 1, 1, 0x5010ull); }
BT_PARITY_TEST(matmul_bwd_square)    { run_matmul_bwd(16, 16, 16, 0x5011ull); }
BT_PARITY_TEST(matmul_bwd_nonsquare) { run_matmul_bwd(7, 19, 11, 0x5012ull); }
BT_PARITY_TEST(matmul_bwd_accum)     { run_matmul_bwd_accum(9, 13, 5, 0x5013ull); }

// ─── BF16 parity ─────────────────────────────────────────────────────────────
BT_PARITY_TEST(matmul_bf16_square)    { run_matmul_bf16(16, 16, 16, 0x5020ull); }
BT_PARITY_TEST(matmul_bf16_nonsquare) { run_matmul_bf16(7, 19, 11, 0x5021ull); }
BT_PARITY_TEST(matmul_bf16_rowvec)    { run_matmul_bf16(1, 13, 9, 0x5022ull); }

BT_PARITY_TEST(matmul_bwd_bf16_square)    { run_matmul_bwd_bf16(16, 16, 16, 0x5030ull); }
BT_PARITY_TEST(matmul_bwd_bf16_nonsquare) { run_matmul_bwd_bf16(7, 19, 11, 0x5031ull); }

int main() { return run_all("matmul/matmul_backward cpu/gpu parity"); }
