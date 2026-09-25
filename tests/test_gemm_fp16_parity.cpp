// FP16 GEMM CPU<->GPU parity, through every public op that reaches the GPU's
// FP16 GEMM, over shapes that hit each of its paths: whole and ragged 64x64
// tiles at both K steps (BK 32 on a large grid, BK 16 on a small one), K not
// a multiple of 4 (the scalar loads), an operand at an odd element offset,
// M == 1 (the decode GEMV) and 1 < M <= 8 (the skinny kernel's staged rows).
// Each operand layout is covered:
//   NN  matmul, linear_backward_batched dX, conv2d 1x1 forward
//   NT  linear_forward_batched(_ex / _fp16 / _fp16_act), matmul_abt,
//       matmul_backward dA
//   TN  matmul_backward dB, linear_backward_batched dW, conv2d 1x1 dX
// along with the bias / activation / accumulate / GeGLU epilogues, a batched
// matmul_abt and the FP16-weight / FP32-activation linear.
//
// The CPU reference runs in FP32 on the FP16-rounded inputs; the GPU
// accumulates in FP32 and rounds the result to FP16 once, so the tolerance is
// one FP16 rounding of the output plus accumulation-order noise.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>

using namespace bt_parity;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

// FP32 host tensor in [-1, 1] whose values are exactly representable in FP16.
Tensor rnd16(int r, int c, SplitMix64& rng) {
    Tensor t = Tensor::mat(r, c);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = rng.next_unit();
    return fp16_host_to_f32(to_fp16_host(t));
}

Tensor g16(const Tensor& t) { return to_fp16_gpu(t); }

void cmp16(const Tensor& cpu, const Tensor& g, const char* tag) {
    compare_tensors(cpu, fp16_host_to_f32(download_to_host(g)), tag, 2e-3f, 2e-3f);
}

// {M, N, K}: tiled (BK 32 grid, BK 16 grids with whole / ragged tiles and an
// unaligned K), then M == 1 and the staged skinny rows.
constexpr int kShapes[][3] = {
    {1024, 800, 100}, {64, 64, 32}, {130, 77, 301}, {257, 129, 1030}, {65, 200, 67},
    {1, 257, 300}, {1, 64, 4099}, {3, 64, 129}, {8, 70, 33}, {5, 31, 1101},
};

void run_matmul(int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = rnd16(M, K, rng), B = rnd16(K, N, rng), C, gC;
    brotensor::matmul(A, B, C);
    brotensor::matmul(g16(A), g16(B), gC);
    cmp16(C, gC, "matmul");
}

void run_matmul_bwd(int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = rnd16(M, K, rng), B = rnd16(K, N, rng), dC = rnd16(M, N, rng);
    Tensor dA = rnd16(M, K, rng), dB = rnd16(K, N, rng);   // accumulated into
    Tensor gdA = g16(dA), gdB = g16(dB);
    brotensor::matmul_backward(A, B, dC, dA, dB);
    brotensor::matmul_backward(g16(A), g16(B), g16(dC), gdA, gdB);
    cmp16(dA, gdA, "matmul_bwd_dA");
    cmp16(dB, gdB, "matmul_bwd_dB");
}

void run_linear(int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor W = rnd16(N, K, rng), b = rnd16(N, 1, rng), X = rnd16(M, K, rng), Y, gY;
    Tensor gW = g16(W), gb = g16(b), gX = g16(X);

    brotensor::linear_forward_batched(W, b, X, Y);
    brotensor::linear_forward_batched_fp16(gW, &gb, gX, gY);
    cmp16(Y, gY, "linear_fp16");

    brotensor::linear_forward_batched_ex(W, &b, X, 3, brotensor::kLinearEpiStore, nullptr, Y);
    brotensor::linear_forward_batched_fp16_act(gW, &gb, gX, 3, gY);
    cmp16(Y, gY, "linear_fp16_act_gelu_erf");

    brotensor::linear_forward_batched_ex(W, &b, X, 4, brotensor::kLinearEpiStore, nullptr, Y);
    brotensor::linear_forward_batched_ex(gW, &gb, gX, 4, brotensor::kLinearEpiStore, nullptr, gY);
    cmp16(Y, gY, "linear_ex_silu");

    Tensor Yacc = rnd16(M, N, rng), gYacc = g16(Yacc);
    brotensor::linear_forward_batched_ex(W, &b, X, 0, brotensor::kLinearEpiAccumulate, nullptr, Yacc);
    brotensor::linear_forward_batched_ex(gW, &gb, gX, 0, brotensor::kLinearEpiAccumulate, nullptr, gYacc);
    cmp16(Yacc, gYacc, "linear_ex_accumulate");
    if (N % 2 == 0) {
        brotensor::linear_forward_batched_ex(W, &b, X, 0, brotensor::kLinearEpiGeglu, nullptr, Y);
        brotensor::linear_forward_batched_ex(gW, &gb, gX, 0, brotensor::kLinearEpiGeglu, nullptr, gY);
        cmp16(Y, gY, "linear_ex_geglu");
    }

    // FP16 weight under FP32 activations: FP32 in, FP32 out, FP32 math.
    Tensor gY32;
    brotensor::linear_forward_batched(W, b, X, Y);
    brotensor::linear_forward_batched(gW, b.to(gpu_device()), X.to(gpu_device()), gY32);
    compare_tensors(Y, download_to_host(gY32), "linear_w16_a32", 1e-4f, 1e-4f);
}

// Batched matmul_abt with bias + relu: C[b] = relu(A[b] @ B[b]^T + bias).
void run_matmul_abt(int batch, int M, int N, int K, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor A = rnd16(batch * M, K, rng), B = rnd16(batch * N, K, rng), b = rnd16(N, 1, rng);
    Tensor ref = Tensor::mat(batch * M, N);
    for (int z = 0; z < batch; ++z)
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += A.ptr()[(z * M + m) * K + k] * B.ptr()[(z * N + n) * K + k];
                acc += b.ptr()[n];
                ref.ptr()[(z * M + m) * N + n] = acc > 0.0f ? acc : 0.0f;
            }
    Tensor gC = Tensor::zeros_on(gpu_device(), batch * M, N, Dtype::FP16), gb = g16(b);
    brotensor::matmul_abt(g16(A), g16(B), gC, batch, M, N, K, static_cast<long long>(M) * K,
                          static_cast<long long>(N) * K, static_cast<long long>(M) * N, &gb, 1);
    cmp16(ref, gC, "matmul_abt");
}

void run_linear_bwd(int M, int N, int K, uint64_t seed) {
    // B = M rows, out = N, in = K.
    SplitMix64 rng(seed);
    Tensor W = rnd16(N, K, rng), X = rnd16(M, K, rng), dY = rnd16(M, N, rng);
    Tensor dX, dW = rnd16(N, K, rng), dB = rnd16(N, 1, rng);
    Tensor gdX, gdW = g16(dW), gdB = g16(dB);
    brotensor::linear_backward_batched(W, X, dY, dX, dW, dB);
    brotensor::linear_backward_batched(g16(W), g16(X), g16(dY), gdX, gdW, gdB);
    cmp16(dX, gdX, "linear_bwd_dX");
    cmp16(dW, gdW, "linear_bwd_dW");
    cmp16(dB, gdB, "linear_bwd_dB");
}

// B read from an odd element offset inside a larger buffer: rows are no longer
// 8-byte aligned even though N is a multiple of 4.
void run_matmul_offset(uint64_t seed) {
    const int M = 70, N = 68, K = 36;
    SplitMix64 rng(seed);
    Tensor A = rnd16(M, K, rng), Bbig = rnd16(1, K * N + 1, rng), C;
    Tensor B = Tensor::mat(K, N);
    for (int i = 0; i < K * N; ++i) B.ptr()[i] = Bbig.ptr()[i + 1];
    brotensor::matmul(A, B, C);
    Tensor gBbig = g16(Bbig);
    Tensor gB = Tensor::view(gpu_device(), static_cast<uint16_t*>(gBbig.data) + 1, K, N, Dtype::FP16);
    Tensor gC;
    brotensor::matmul(g16(A), gB, gC);
    cmp16(C, gC, "matmul_offset");
}

void run_conv1x1(int N, int C_in, int C_out, int H, int W, uint64_t seed) {
    SplitMix64 rng(seed);
    const int HW = H * W;
    Tensor X = rnd16(N, C_in * HW, rng), Wt = rnd16(C_out, C_in, rng), dY = rnd16(N, C_out * HW, rng);
    Tensor Y, gY, dX, gdX;
    brotensor::conv2d_forward(X, Wt, nullptr, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, Y);
    brotensor::conv2d_forward(g16(X), g16(Wt), nullptr, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, gY);
    cmp16(Y, gY, "conv1x1_fw");
    brotensor::conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX);
    brotensor::conv2d_backward_input(g16(Wt), g16(dY), N, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, gdX);
    cmp16(dX, gdX, "conv1x1_dX");
}

}  // namespace

BT_PARITY_TEST(gemm_fp16_matmul) {
    uint64_t s = 0x7000ull;
    for (const auto& sh : kShapes) run_matmul(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp16_matmul_bwd) {
    uint64_t s = 0x7100ull;
    for (const auto& sh : kShapes) run_matmul_bwd(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp16_linear) {
    uint64_t s = 0x7200ull;
    for (const auto& sh : kShapes) run_linear(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp16_linear_bwd) {
    uint64_t s = 0x7300ull;
    for (const auto& sh : kShapes) run_linear_bwd(sh[0], sh[1], sh[2], s++);
}
BT_PARITY_TEST(gemm_fp16_matmul_abt) {
    run_matmul_abt(3, 70, 67, 45, 0x7400ull);
    run_matmul_abt(2, 1, 130, 300, 0x7401ull);
    run_matmul_abt(4, 6, 9, 17, 0x7402ull);   // M*N below the tiled floor: naive
}
BT_PARITY_TEST(gemm_fp16_matmul_offset) { run_matmul_offset(0x7500ull); }
BT_PARITY_TEST(gemm_fp16_conv1x1_batched) { run_conv1x1(3, 19, 70, 13, 11, 0x7600ull); }
BT_PARITY_TEST(gemm_fp16_conv1x1_skinny) { run_conv1x1(2, 5, 3, 16, 20, 0x7601ull); }

int main() { return run_all("fp16 gemm cpu/gpu parity"); }
