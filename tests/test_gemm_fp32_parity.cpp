// FP32 GEMM CPU<->GPU parity, through every public op that reaches the GPU's
// FP32 GEMM, over shapes that hit each of its paths: whole 64x64x16 tiles,
// ragged M / N / K edges, K not a multiple of 4 (rows not 16-byte aligned, so
// the scalar loads), an operand at an odd element offset, and M <= 8 (the
// skinny kernel). Each operand layout is covered:
//   NN  matmul, linear_backward_batched dX, conv2d 1x1 forward
//   NT  linear_forward_batched(_ex), matmul_backward dA
//   TN  matmul_backward dB, linear_backward_batched dW, conv2d 1x1 dX
// along with the bias / activation / accumulate / GeGLU epilogues, a batched
// launch (the 1x1 conv over N images) and BF16 operands (matmul).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

Tensor rnd(int r, int c, SplitMix64& rng) {
    Tensor t = Tensor::mat(r, c);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = rng.next_unit() * 2.0f - 1.0f;
    return t;
}

Tensor gpu(const Tensor& t) { return t.to(gpu_device()); }

void cmp(const Tensor& cpu, const Tensor& g, const char* tag) {
    compare_tensors(cpu, download_to_host(g), tag, 1e-4f, 1e-4f);
}

// {M, N, K}: tiled (whole + ragged tiles, unaligned K), then skinny M.
constexpr int kShapes[][3] = {
    {64, 64, 16}, {130, 77, 301}, {257, 129, 1030}, {65, 200, 67},
    {1, 257, 300}, {3, 64, 129}, {8, 70, 33}, {5, 31, 4097},
};

void run_matmul(int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = rnd(M, K, rng), B = rnd(K, N, rng), C;
    brotensor::matmul(A, B, C);
    Tensor gC;
    brotensor::matmul(gpu(A), gpu(B), gC);
    cmp(C, gC, "matmul");
}

void run_matmul_bwd(int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = rnd(M, K, rng), B = rnd(K, N, rng), dC = rnd(M, N, rng);
    Tensor dA = rnd(M, K, rng), dB = rnd(K, N, rng);   // accumulated into
    Tensor gdA = gpu(dA), gdB = gpu(dB);
    brotensor::matmul_backward(A, B, dC, dA, dB);
    brotensor::matmul_backward(gpu(A), gpu(B), gpu(dC), gdA, gdB);
    cmp(dA, gdA, "matmul_bwd_dA");
    cmp(dB, gdB, "matmul_bwd_dB");
}

void run_linear(int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = rnd(N, K, rng), b = rnd(N, 1, rng), X = rnd(M, K, rng), Y, gY;
    brotensor::linear_forward_batched(W, b, X, Y);
    brotensor::linear_forward_batched(gpu(W), gpu(b), gpu(X), gY);
    cmp(Y, gY, "linear_fw");

    // _ex: bias + gelu-tanh, accumulate, and GeGLU (even N).
    Tensor gW = gpu(W), gb = gpu(b), gX = gpu(X);
    brotensor::linear_forward_batched_ex(W, &b, X, 2, brotensor::kLinearEpiStore, nullptr, Y);
    brotensor::linear_forward_batched_ex(gW, &gb, gX, 2, brotensor::kLinearEpiStore, nullptr, gY);
    cmp(Y, gY, "linear_ex_gelu");
    Tensor Yacc = rnd(M, N, rng), gYacc = gpu(Yacc);
    brotensor::linear_forward_batched_ex(W, &b, X, 0, brotensor::kLinearEpiAccumulate, nullptr, Yacc);
    brotensor::linear_forward_batched_ex(gW, &gb, gX, 0, brotensor::kLinearEpiAccumulate, nullptr, gYacc);
    cmp(Yacc, gYacc, "linear_ex_accumulate");
    if (N % 2 == 0) {
        brotensor::linear_forward_batched_ex(W, &b, X, 0, brotensor::kLinearEpiGeglu, nullptr, Y);
        brotensor::linear_forward_batched_ex(gW, &gb, gX, 0, brotensor::kLinearEpiGeglu, nullptr, gY);
        cmp(Y, gY, "linear_ex_geglu");
    }
}

void run_linear_bwd(int M, int N, int K, uint64_t seed) {
    // B = M rows, out = N, in = K.
    SplitMix64 rng(seed);
    Tensor W = rnd(N, K, rng), X = rnd(M, K, rng), dY = rnd(M, N, rng);
    Tensor dX, dW = rnd(N, K, rng), dB = rnd(N, 1, rng);
    Tensor gdX, gdW = gpu(dW), gdB = gpu(dB);
    brotensor::linear_backward_batched(W, X, dY, dX, dW, dB);
    brotensor::linear_backward_batched(gpu(W), gpu(X), gpu(dY), gdX, gdW, gdB);
    cmp(dX, gdX, "linear_bwd_dX");
    cmp(dW, gdW, "linear_bwd_dW");
    cmp(dB, gdB, "linear_bwd_dB");
}

// B read from an odd element offset inside a larger buffer: rows are no longer
// 16-byte aligned even though N is a multiple of 4.
void run_matmul_offset(uint64_t seed) {
    const int M = 70, N = 68, K = 36;
    SplitMix64 rng(seed);
    Tensor A = rnd(M, K, rng), Bbig = rnd(1, K * N + 1, rng), C;
    Tensor B = Tensor::mat(K, N);
    for (int i = 0; i < K * N; ++i) B.ptr()[i] = Bbig.ptr()[i + 1];
    brotensor::matmul(A, B, C);
    Tensor gBbig = gpu(Bbig);
    Tensor gB = Tensor::view(gpu_device(), static_cast<float*>(gBbig.data) + 1, K, N);
    Tensor gC;
    brotensor::matmul(gpu(A), gB, gC);
    cmp(C, gC, "matmul_offset");
}

void run_conv1x1(int N, int C_in, int C_out, int H, int W, uint64_t seed) {
    SplitMix64 rng(seed);
    const int HW = H * W;
    Tensor X = rnd(N, C_in * HW, rng), Wt = rnd(C_out, C_in, rng), dY = rnd(N, C_out * HW, rng);
    Tensor Y, gY, dX, gdX;
    brotensor::conv2d_forward(X, Wt, nullptr, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, Y);
    brotensor::conv2d_forward(gpu(X), gpu(Wt), nullptr, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, gY);
    cmp(Y, gY, "conv1x1_fw");
    brotensor::conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX);
    brotensor::conv2d_backward_input(gpu(Wt), gpu(dY), N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, gdX);
    cmp(dX, gdX, "conv1x1_dX");
}

void run_matmul_bf16(int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = bf16_host_to_f32(to_bf16_host(rnd(M, K, rng)));
    Tensor B = bf16_host_to_f32(to_bf16_host(rnd(K, N, rng)));
    Tensor C;
    brotensor::matmul(A, B, C);
    Tensor gC;
    brotensor::matmul(to_bf16_gpu(A), to_bf16_gpu(B), gC);
    // One BF16 rounding of the FP32-accumulated result.
    compare_tensors(C, bf16_host_to_f32(download_to_host(gC)), "matmul_bf16", 1e-3f, 8e-3f);
}

}  // namespace

BT_PARITY_TEST(gemm_fp32_matmul) {
    uint64_t s = 0x6000ull;
    for (const auto& sh : kShapes) run_matmul(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp32_matmul_bwd) {
    uint64_t s = 0x6100ull;
    for (const auto& sh : kShapes) run_matmul_bwd(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp32_linear) {
    uint64_t s = 0x6200ull;
    for (const auto& sh : kShapes) run_linear(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp32_linear_bwd) {
    uint64_t s = 0x6300ull;
    for (const auto& sh : kShapes) run_linear_bwd(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp32_matmul_offset) { run_matmul_offset(0x6400ull); }
BT_PARITY_TEST(gemm_fp32_conv1x1_batched) { run_conv1x1(3, 19, 70, 13, 11, 0x6500ull); }
BT_PARITY_TEST(gemm_fp32_conv1x1_skinny) { run_conv1x1(2, 5, 3, 16, 20, 0x6501ull); }
BT_PARITY_TEST(gemm_fp32_matmul_bf16) {
    run_matmul_bf16(130, 77, 301, 0x6600ull);
    run_matmul_bf16(4, 64, 129, 0x6601ull);
}

int main() { return run_all("fp32 gemm cpu/gpu parity"); }
