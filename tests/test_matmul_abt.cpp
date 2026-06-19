// Parity for the public matmul_abt op (batched A @ B^T, FP16/BF16) against a
// CPU reference triple loop. Covers batch=1 and batch=2, FP16 and BF16, with
// and without fused bias + relu.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// Quantise a vector to/from the storage dtype so the reference sees the same
// rounded operands the kernel does.
static uint16_t to_bits(float v, bool fp16) {
    return fp16 ? brotensor::fp32_to_fp16_bits(v) : brotensor::fp32_to_bf16_bits(v);
}
static float from_bits(uint16_t b, bool fp16) {
    return fp16 ? brotensor::fp16_bits_to_fp32(b) : brotensor::bf16_bits_to_fp32(b);
}

// C[b][m,n] = sum_k A[b][m,k]*B[b][n,k] (+ bias[n], then relu if act==1).
static void ref_abt(const std::vector<float>& A, const std::vector<float>& B,
                    const std::vector<float>* bias, int act,
                    std::vector<float>& C,
                    int batch, int M, int N, int K) {
    C.assign(static_cast<size_t>(batch) * M * N, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) {
                    s += static_cast<double>(A[(static_cast<size_t>(b) * M + m) * K + k]) *
                         static_cast<double>(B[(static_cast<size_t>(b) * N + n) * K + k]);
                }
                float v = static_cast<float>(s);
                if (bias) v += (*bias)[n];
                if (act == 1) v = v > 0.0f ? v : 0.0f;
                C[(static_cast<size_t>(b) * M + m) * N + n] = v;
            }
}

static void run_case(bool fp16, int batch, bool with_bias, int act) {
    const char* dt = fp16 ? "fp16" : "bf16";
    std::printf("  matmul_abt %s batch=%d bias=%d act=%d\n",
                dt, batch, (int)with_bias, act);
    const int M = 20, K = 24, N = 12;
    std::mt19937 rng(0xAB70u + batch * 131 + (fp16 ? 7 : 3) + act);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> A(static_cast<size_t>(batch) * M * K);
    std::vector<float> B(static_cast<size_t>(batch) * N * K);
    std::vector<float> bias(N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    // Quantise operands so the reference matches the rounded inputs.
    auto Aq = A, Bq = B, biasq = bias;
    for (auto& v : Aq)    v = from_bits(to_bits(v, fp16), fp16);
    for (auto& v : Bq)    v = from_bits(to_bits(v, fp16), fp16);
    for (auto& v : biasq) v = from_bits(to_bits(v, fp16), fp16);

    std::vector<float> ref;
    ref_abt(Aq, Bq, with_bias ? &biasq : nullptr, act, ref, batch, M, N, K);

    std::vector<uint16_t> Ah(A.size()), Bh(B.size()), biash(N);
    for (size_t i = 0; i < A.size(); ++i) Ah[i] = to_bits(A[i], fp16);
    for (size_t i = 0; i < B.size(); ++i) Bh[i] = to_bits(B[i], fp16);
    for (int i = 0; i < N; ++i)           biash[i] = to_bits(bias[i], fp16);

    auto make = [&](const uint16_t* p, int rows, int cols) {
        return fp16 ? Tensor::from_host_fp16_on(Device::CUDA, p, rows, cols)
                    : Tensor::from_host_bf16_on(Device::CUDA, p, rows, cols);
    };
    Tensor Ag = make(Ah.data(), batch * M, K);
    Tensor Bg = make(Bh.data(), batch * N, K);
    Tensor biasg = make(biash.data(), 1, N);
    // C is NOT auto-resized — pre-size/dtype it (zero-filled).
    std::vector<uint16_t> Cz(static_cast<size_t>(batch) * M * N, to_bits(0.0f, fp16));
    Tensor Cg = make(Cz.data(), batch * M, N);

    brotensor::matmul_abt(Ag, Bg, Cg, batch, M, N, K,
                          (long long)M * K, (long long)N * K, (long long)M * N,
                          with_bias ? &biasg : nullptr, act);

    std::vector<uint16_t> got(Cg.size());
    brotensor::sync_all();
    if (fp16) Cg.copy_to_host_fp16(got.data());
    else      Cg.copy_to_host_bf16(got.data());

    float max_err = 0.0f;
    int bad = 0;
    // BF16 has ~8 mantissa bits — looser tolerance than FP16.
    const float atol = fp16 ? 2e-2f : 2e-1f;
    const float rtol = fp16 ? 5e-2f : 1e-1f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = from_bits(got[i], fp16);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(ref[i])) ++bad;
    }
    std::printf("    %s max_err=%g bad=%d/%zu\n", dt, max_err, bad, ref.size());
    CHECK(bad == 0);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_matmul_abt\n");
    for (bool fp16 : {true, false}) {
        run_case(fp16, /*batch=*/1, /*with_bias=*/false, /*act=*/0);
        run_case(fp16, /*batch=*/2, /*with_bias=*/false, /*act=*/0);
        run_case(fp16, /*batch=*/2, /*with_bias=*/true,  /*act=*/0);
        run_case(fp16, /*batch=*/2, /*with_bias=*/true,  /*act=*/1);
    }
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
