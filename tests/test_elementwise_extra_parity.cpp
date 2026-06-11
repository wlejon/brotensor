// CPU↔GPU parity tests for brotensor::clamp and brotensor::mul_inplace.
//
// CHUNK 1. test_elementwise_parity.cpp already covers the activation ops;
// this file covers the two newly-CPU-ported elementwise ops.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_clamp(int r, int c, float lo, float hi, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor base = Tensor::mat(r, c);
    fill_random(base, rng, 3.0f);  // wide range so clamping actually bites

    Tensor cpu = base;             // deep copy (CPU)
    brotensor::clamp(cpu, lo, hi);

    Tensor gpu = base.to(gpu_device());
    brotensor::clamp(gpu, lo, hi);

    Tensor gpu_h = download_to_host(gpu);
    compare_tensors(cpu, gpu_h, "clamp");
}

void run_clamp_bf16(int r, int c, float lo, float hi, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor base = Tensor::mat(r, c);
    fill_random(base, rng, 3.0f);

    Tensor cpu = base;
    brotensor::clamp(cpu, lo, hi);  // FP32 CPU reference

    Tensor gpu = to_bf16_gpu(base);
    brotensor::clamp(gpu, lo, hi);

    compare_tensors(cpu, bf16_host_to_f32(download_to_host(gpu)), "clamp_bf16", 2e-2f, 2e-2f);
}

void run_mul(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor a = Tensor::mat(r, c), b = Tensor::mat(r, c);
    fill_random(a, rng);
    fill_random(b, rng);

    Tensor cpu = a;                // deep copy (CPU)
    brotensor::mul_inplace(cpu, b);

    Tensor ga = a.to(gpu_device());
    Tensor gb = b.to(gpu_device());
    brotensor::mul_inplace(ga, gb);

    Tensor gpu_h = download_to_host(ga);
    compare_tensors(cpu, gpu_h, "mul_inplace");
}

void run_mul_bf16(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor a = Tensor::mat(r, c), b = Tensor::mat(r, c);
    fill_random(a, rng);
    fill_random(b, rng);

    Tensor cpu = a;
    brotensor::mul_inplace(cpu, b);  // FP32 CPU reference

    Tensor ga = to_bf16_gpu(a);
    Tensor gb = to_bf16_gpu(b);
    brotensor::mul_inplace(ga, gb);

    compare_tensors(cpu, bf16_host_to_f32(download_to_host(ga)), "mul_inplace_bf16", 2e-2f, 2e-2f);
}

} // namespace

BT_PARITY_TEST(clamp_1x1)        { run_clamp(1, 1, -0.5f, 0.5f, 0x700ull); }
BT_PARITY_TEST(clamp_8x32)       { run_clamp(8, 32, -1.0f, 1.0f, 0x701ull); }
BT_PARITY_TEST(clamp_asym)       { run_clamp(16, 16, -0.25f, 0.75f, 0x702ull); }
BT_PARITY_TEST(clamp_relu_like)  { run_clamp(7, 13, 0.0f, 3.4e38f, 0x703ull); }

BT_PARITY_TEST(mul_1x1)          { run_mul(1, 1, 0x710ull); }
BT_PARITY_TEST(mul_8x32)         { run_mul(8, 32, 0x711ull); }
BT_PARITY_TEST(mul_vec)          { run_mul(64, 1, 0x712ull); }

// ─── BF16 parity tests ─────────────────────────────────────────────────────
BT_PARITY_TEST(clamp_bf16_8x32)   { run_clamp_bf16(8, 32, -1.0f, 1.0f, 0x720ull); }
BT_PARITY_TEST(clamp_bf16_16x16)  { run_clamp_bf16(16, 16, -0.25f, 0.75f, 0x721ull); }

BT_PARITY_TEST(mul_bf16_8x32)     { run_mul_bf16(8, 32, 0x730ull); }
BT_PARITY_TEST(mul_bf16_64x1)     { run_mul_bf16(64, 1, 0x731ull); }

// ─── threshold_u8 ───────────────────────────────────────────────────────────

namespace {

void check_u8_equal(const Tensor& a, const Tensor& b, const char* tag) {
    BT_CHECK(a.rows == b.rows && a.cols == b.cols);
    BT_CHECK(a.dtype == brotensor::Dtype::INT8 &&
             b.dtype == brotensor::Dtype::INT8);
    const int8_t* ap = static_cast<const int8_t*>(a.data);
    const int8_t* bp = static_cast<const int8_t*>(b.data);
    for (int i = 0; i < a.size(); ++i) {
        if (ap[i] != bp[i]) {
            std::printf("    [%s] byte mismatch at i=%d: %d vs %d\n",
                        tag, i, int(ap[i]), int(bp[i]));
            throw 0;
        }
    }
}

// Random matrix: CPU and CUDA byte masks must agree exactly.
void run_threshold_u8(int r, int c, float t, bool fp16, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(r, c);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    Tensor gX;
    if (fp16) {
        Tensor Xq = fp16_host_to_f32(to_fp16_host(X));
        brotensor::threshold_u8(Xq, t, cpu_Y);
        gX = to_fp16_gpu(X);
    } else {
        brotensor::threshold_u8(X, t, cpu_Y);
        gX = X.to(gpu_device());
    }
    BT_CHECK(cpu_Y.rows == r && cpu_Y.cols == c);

    Tensor gpu_Y;
    brotensor::threshold_u8(gX, t, gpu_Y);
    check_u8_equal(cpu_Y, download_to_host(gpu_Y), "threshold_u8");
}

// Handwritten values including exact ties at the threshold: strict > means a
// tie maps to 0.
void run_threshold_u8_handwritten() {
    const float t = 0.5f;
    const float vals[6]    = {0.49f, 0.5f, 0.51f, -0.5f, 2.0f, 0.0f};
    const int8_t expect[6] = {0,     0,    1,      0,     1,    0};

    Tensor X = Tensor::mat(2, 3);
    for (int i = 0; i < 6; ++i) X.host_f32_mut()[i] = vals[i];

    Tensor cpu_Y;
    brotensor::threshold_u8(X, t, cpu_Y);
    const int8_t* cp = static_cast<const int8_t*>(cpu_Y.data);
    for (int i = 0; i < 6; ++i) BT_CHECK(cp[i] == expect[i]);

    // GPU — FP32 and FP16 (0.5 is FP16-exact, so the tie stays a tie).
    for (int pass = 0; pass < 2; ++pass) {
        Tensor gX = (pass == 0) ? X.to(gpu_device()) : to_fp16_gpu(X);
        Tensor gpu_Y;
        brotensor::threshold_u8(gX, t, gpu_Y);
        Tensor host = download_to_host(gpu_Y);
        const int8_t* gp = static_cast<const int8_t*>(host.data);
        for (int i = 0; i < 6; ++i) BT_CHECK(gp[i] == expect[i]);
    }
}

} // namespace (threshold_u8)

BT_PARITY_TEST(thr_u8_small)       { run_threshold_u8(4, 9, 0.1f, false, 0x740ull); }
BT_PARITY_TEST(thr_u8_wide)        { run_threshold_u8(3, 1027, -0.3f, false, 0x741ull); }
BT_PARITY_TEST(thr_u8_fp16)        { run_threshold_u8(5, 257, 0.2f, true, 0x742ull); }
BT_PARITY_TEST(thr_u8_handwritten) { run_threshold_u8_handwritten(); }

int main() { return run_all("clamp/mul_inplace cpu/gpu parity"); }
