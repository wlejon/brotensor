// CPU↔GPU parity tests for elementwise activations and adds.

#include "parity_helpers.h"

#include <brotensor/ops.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void test_relu(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::vec(n), dY = Tensor::vec(n);
    fill_random(x, rng);
    fill_random(dY, rng);

    Tensor y_cpu = Tensor::vec(n), dX_cpu = Tensor::vec(n);
    brotensor::relu_forward(x, y_cpu);
    brotensor::relu_backward(x, dY, dX_cpu);

    Tensor gx = x.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gy = Tensor::zeros_on(gpu_device(), n, 1);
    Tensor gdX = Tensor::zeros_on(gpu_device(), n, 1);
    brotensor::relu_forward(gx, gy);
    brotensor::relu_backward(gx, gdY, gdX);

    compare_tensors(y_cpu, download_to_host(gy), "relu_forward");
    compare_tensors(dX_cpu, download_to_host(gdX), "relu_backward");
}

void test_tanh(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::vec(n), dY = Tensor::vec(n);
    fill_random(x, rng);
    fill_random(dY, rng);

    Tensor y_cpu = Tensor::vec(n), dX_cpu = Tensor::vec(n);
    brotensor::tanh_forward(x, y_cpu);
    brotensor::tanh_backward(y_cpu, dY, dX_cpu);

    Tensor gx = x.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gy = Tensor::zeros_on(gpu_device(), n, 1);
    Tensor gdX = Tensor::zeros_on(gpu_device(), n, 1);
    brotensor::tanh_forward(gx, gy);
    Tensor y_gpu = download_to_host(gy);
    brotensor::tanh_backward(gy, gdY, gdX);

    compare_tensors(y_cpu, y_gpu, "tanh_forward");
    compare_tensors(dX_cpu, download_to_host(gdX), "tanh_backward");
}

void test_sigmoid(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::vec(n), dY = Tensor::vec(n);
    fill_random(x, rng);
    fill_random(dY, rng);

    Tensor y_cpu = Tensor::vec(n), dX_cpu = Tensor::vec(n);
    brotensor::sigmoid_forward(x, y_cpu);
    brotensor::sigmoid_backward(y_cpu, dY, dX_cpu);

    Tensor gx = x.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gy = Tensor::zeros_on(gpu_device(), n, 1);
    Tensor gdX = Tensor::zeros_on(gpu_device(), n, 1);
    brotensor::sigmoid_forward(gx, gy);
    Tensor y_gpu = download_to_host(gy);
    brotensor::sigmoid_backward(gy, gdY, gdX);

    compare_tensors(y_cpu, y_gpu, "sigmoid_forward");
    compare_tensors(dX_cpu, download_to_host(gdX), "sigmoid_backward");
}

void test_add_inplace(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::vec(n), x = Tensor::vec(n);
    fill_random(y, rng);
    fill_random(x, rng);

    Tensor y_cpu = y;
    brotensor::add_inplace(y_cpu, x);

    Tensor gy = y.to(gpu_device()), gx = x.to(gpu_device());
    brotensor::add_inplace(gy, gx);

    compare_tensors(y_cpu, download_to_host(gy), "add_inplace");
}

void test_add_inplace_bf16(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::vec(n), x = Tensor::vec(n);
    fill_random(y, rng);
    fill_random(x, rng);

    Tensor y_ref = y;
    brotensor::add_inplace(y_ref, x);

    Tensor gy = to_bf16_gpu(y), gx = to_bf16_gpu(x);
    brotensor::add_inplace(gy, gx);

    compare_tensors(y_ref, bf16_host_to_f32(download_to_host(gy)), "add_inplace_bf16", 2e-2f, 2e-2f);
}

void test_add_scalar_inplace(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::vec(n);
    fill_random(y, rng);
    const float s = 0.375f;

    Tensor y_cpu = y;
    brotensor::add_scalar_inplace(y_cpu, s);

    Tensor gy = y.to(gpu_device());
    brotensor::add_scalar_inplace(gy, s);

    compare_tensors(y_cpu, download_to_host(gy), "add_scalar_inplace");
}

void test_add_scalar_inplace_bf16(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::vec(n);
    fill_random(y, rng);
    const float s = 0.375f;

    Tensor y_ref = y;
    brotensor::add_scalar_inplace(y_ref, s);

    Tensor gy = to_bf16_gpu(y);
    brotensor::add_scalar_inplace(gy, s);

    compare_tensors(y_ref, bf16_host_to_f32(download_to_host(gy)), "add_scalar_inplace_bf16", 2e-2f, 2e-2f);
}

} // namespace

BT_PARITY_TEST(relu_n1)    { test_relu(1, 0x10ull); }
BT_PARITY_TEST(relu_n7)    { test_relu(7, 0x11ull); }
BT_PARITY_TEST(relu_n256)  { test_relu(256, 0x12ull); }
BT_PARITY_TEST(relu_n1024) { test_relu(1024, 0x13ull); }

BT_PARITY_TEST(tanh_n1)    { test_tanh(1, 0x20ull); }
BT_PARITY_TEST(tanh_n7)    { test_tanh(7, 0x21ull); }
BT_PARITY_TEST(tanh_n256)  { test_tanh(256, 0x22ull); }
BT_PARITY_TEST(tanh_n1024) { test_tanh(1024, 0x23ull); }

BT_PARITY_TEST(sigmoid_n1)    { test_sigmoid(1, 0x30ull); }
BT_PARITY_TEST(sigmoid_n7)    { test_sigmoid(7, 0x31ull); }
BT_PARITY_TEST(sigmoid_n256)  { test_sigmoid(256, 0x32ull); }
BT_PARITY_TEST(sigmoid_n1024) { test_sigmoid(1024, 0x33ull); }

BT_PARITY_TEST(add_inplace_n1)    { test_add_inplace(1, 0x40ull); }
BT_PARITY_TEST(add_inplace_n7)    { test_add_inplace(7, 0x41ull); }
BT_PARITY_TEST(add_inplace_n256)  { test_add_inplace(256, 0x42ull); }
BT_PARITY_TEST(add_inplace_n1024) { test_add_inplace(1024, 0x43ull); }

BT_PARITY_TEST(add_scalar_n1)    { test_add_scalar_inplace(1, 0x50ull); }
BT_PARITY_TEST(add_scalar_n7)    { test_add_scalar_inplace(7, 0x51ull); }
BT_PARITY_TEST(add_scalar_n256)  { test_add_scalar_inplace(256, 0x52ull); }
BT_PARITY_TEST(add_scalar_n1024) { test_add_scalar_inplace(1024, 0x53ull); }

// ─── BF16 parity tests ─────────────────────────────────────────────────────
BT_PARITY_TEST(add_inplace_bf16_n256)  { test_add_inplace_bf16(256, 0x60ull); }
BT_PARITY_TEST(add_inplace_bf16_n1024) { test_add_inplace_bf16(1024, 0x61ull); }

BT_PARITY_TEST(add_scalar_bf16_n256)  { test_add_scalar_inplace_bf16(256, 0x62ull); }
BT_PARITY_TEST(add_scalar_bf16_n1024) { test_add_scalar_inplace_bf16(1024, 0x63ull); }

// FP16 forward parity for the relu/tanh/sigmoid trio — the qwen35 attention
// gate runs sigmoid on FP16 activations. Quantise inputs through FP16, run
// CPU in FP32 and GPU in FP16 storage, compare loosely.
namespace {
void test_unary_fp16(void (*op)(const Tensor&, Tensor&), const char* tag,
                     int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::vec(n);
    fill_random(x, rng);
    Tensor x_q = fp16_host_to_f32(to_fp16_host(x));

    Tensor y_cpu = Tensor::vec(n);
    op(x_q, y_cpu);

    Tensor gx = to_fp16_gpu(x_q);
    Tensor gy;
    op(gx, gy);

    compare_tensors(y_cpu, fp16_host_to_f32(download_to_host(gy)),
                    tag, 1e-3f, 1e-3f);
}
} // namespace

BT_PARITY_TEST(relu_fp16_n256)    { test_unary_fp16(brotensor::relu_forward,    "relu_fp16",    256, 0x70ull); }
BT_PARITY_TEST(tanh_fp16_n256)    { test_unary_fp16(brotensor::tanh_forward,    "tanh_fp16",    256, 0x71ull); }
BT_PARITY_TEST(sigmoid_fp16_n256) { test_unary_fp16(brotensor::sigmoid_forward, "sigmoid_fp16", 256, 0x72ull); }

// ─── Misaligned 16-bit in-place ops ────────────────────────────────────────
//
// The 16-bit in-place kernels read eight elements at a time as an int4, which
// needs a 16-byte-aligned base. Every tensor the rest of this suite builds is
// cudaMalloc-backed and so is 256-byte aligned, which means the kernels' scalar
// head — the elements before the first boundary — is unreachable from any other
// test here. (Verified: poisoning that loop left all 167 tests green.) But
// Tensor::view() hands the ops a caller-supplied pointer, so misalignment is
// reachable in real use.
//
// These build a view starting at element `off` of a larger buffer, which puts
// the base at 2*off mod 16, and check every residue class. For the two-operand
// ops the offsets are chosen to cover both the congruent case (same residue,
// vector path with a head) and the incongruent one (different residues, no
// vector path at all).
namespace {

using brotensor::Dtype;

Tensor q16(const Tensor& f32, bool bf16) {
    return bf16 ? bf16_host_to_f32(to_bf16_host(f32)) : fp16_host_to_f32(to_fp16_host(f32));
}
Tensor up16(const Tensor& f32, bool bf16) {
    return bf16 ? to_bf16_gpu(f32) : to_fp16_gpu(f32);
}
Tensor down16(const Tensor& g, bool bf16) {
    return bf16 ? bf16_host_to_f32(download_to_host(g)) : fp16_host_to_f32(download_to_host(g));
}
// A view of `n` elements starting `off` elements into `g`. The view aliases
// g's storage, so g must outlive it — callers keep both in scope.
Tensor sub_view(const Tensor& g, int off, int n, bool bf16) {
    return Tensor::view(gpu_device(), static_cast<uint16_t*>(g.data) + off, n, 1,
                        bf16 ? Dtype::BF16 : Dtype::FP16);
}
// Everything outside [off, off+n) must be byte-identical to what was uploaded.
void check_untouched(const Tensor& before, const Tensor& after, int off, int n,
                     int total, const char* tag) {
    for (int i = 0; i < total; ++i) {
        if (i >= off && i < off + n) continue;
        if (before[i] != after[i]) {
            std::fprintf(stderr, "%s: wrote outside the view at %d\n", tag, i);
            throw std::runtime_error(tag);
        }
    }
}

void test_scale_inplace_misaligned(int off, int n, bool bf16, uint64_t seed) {
    SplitMix64 rng(seed);
    const int total = off + n + 3;          // trailing slack guards the tail too
    Tensor h = Tensor::vec(total);
    fill_random(h, rng);
    Tensor hq = q16(h, bf16);
    const float s = 0.375f;                 // exact in FP16 and BF16

    // Rounded back through the storage type: the kernel widens to FP32,
    // multiplies and rounds once, so the reference has to round once too.
    Tensor ref = Tensor::vec(n);
    for (int i = 0; i < n; ++i) ref.ptr()[i] = hq[off + i] * s;
    ref = q16(ref, bf16);

    Tensor g = up16(hq, bf16);
    Tensor v = sub_view(g, off, n, bf16);
    brotensor::scale_inplace(v, s);
    brotensor::sync_all();

    Tensor back = down16(g, bf16);
    Tensor got = Tensor::vec(n);
    for (int i = 0; i < n; ++i) got.ptr()[i] = back[off + i];
    compare_tensors(ref, got, "scale_inplace_misaligned", 1e-3f, 1e-3f);
    check_untouched(hq, back, off, n, total, "scale_inplace_misaligned");
}

// y_off and x_off differing by an odd multiple of 8 elements makes the two
// bases incongruent mod 16, which disables the vector path entirely.
void test_mul_inplace_misaligned(int y_off, int x_off, int n, bool bf16,
                                 uint64_t seed) {
    SplitMix64 rng(seed);
    const int total = (y_off > x_off ? y_off : x_off) + n + 3;
    Tensor hy = Tensor::vec(total), hx = Tensor::vec(total);
    fill_random(hy, rng);
    fill_random(hx, rng);
    Tensor hyq = q16(hy, bf16), hxq = q16(hx, bf16);

    Tensor ref = Tensor::vec(n);
    for (int i = 0; i < n; ++i) ref.ptr()[i] = hyq[y_off + i] * hxq[x_off + i];
    ref = q16(ref, bf16);

    Tensor gy = up16(hyq, bf16), gx = up16(hxq, bf16);
    Tensor vy = sub_view(gy, y_off, n, bf16);
    Tensor vx = sub_view(gx, x_off, n, bf16);
    brotensor::mul_inplace(vy, vx);
    brotensor::sync_all();

    Tensor back = down16(gy, bf16);
    Tensor got = Tensor::vec(n);
    for (int i = 0; i < n; ++i) got.ptr()[i] = back[y_off + i];
    compare_tensors(ref, got, "mul_inplace_misaligned", 1e-3f, 1e-3f);
    check_untouched(hyq, back, y_off, n, total, "mul_inplace_misaligned");
}

}  // namespace

// Every base residue mod 16 (offset 0..7 elements), at a length that is not a
// multiple of 8 so the trailing scalar loop runs as well.
BT_PARITY_TEST(scale_inplace_fp16_off0)  { test_scale_inplace_misaligned(0, 1021, false, 0x80ull); }
BT_PARITY_TEST(scale_inplace_fp16_off1)  { test_scale_inplace_misaligned(1, 1021, false, 0x81ull); }
BT_PARITY_TEST(scale_inplace_fp16_off2)  { test_scale_inplace_misaligned(2, 1021, false, 0x82ull); }
BT_PARITY_TEST(scale_inplace_fp16_off3)  { test_scale_inplace_misaligned(3, 1021, false, 0x83ull); }
BT_PARITY_TEST(scale_inplace_fp16_off4)  { test_scale_inplace_misaligned(4, 1021, false, 0x84ull); }
BT_PARITY_TEST(scale_inplace_fp16_off5)  { test_scale_inplace_misaligned(5, 1021, false, 0x85ull); }
BT_PARITY_TEST(scale_inplace_fp16_off6)  { test_scale_inplace_misaligned(6, 1021, false, 0x86ull); }
BT_PARITY_TEST(scale_inplace_fp16_off7)  { test_scale_inplace_misaligned(7, 1021, false, 0x87ull); }
BT_PARITY_TEST(scale_inplace_bf16_off3)  { test_scale_inplace_misaligned(3, 1021, true,  0x88ull); }
BT_PARITY_TEST(scale_inplace_bf16_off5)  { test_scale_inplace_misaligned(5, 1021, true,  0x89ull); }
// Shorter than one vector: head clamps to n and the vector loop does nothing.
BT_PARITY_TEST(scale_inplace_fp16_tiny)  { test_scale_inplace_misaligned(3, 5, false, 0x8Aull); }

// Congruent bases (vector path with a head) and incongruent ones (no vector
// path at all).
BT_PARITY_TEST(mul_inplace_fp16_cong)    { test_mul_inplace_misaligned(3, 11, 1021, false, 0x90ull); }
BT_PARITY_TEST(mul_inplace_fp16_incong)  { test_mul_inplace_misaligned(3,  6, 1021, false, 0x91ull); }
BT_PARITY_TEST(mul_inplace_bf16_cong)    { test_mul_inplace_misaligned(5, 13, 1021, true,  0x92ull); }
BT_PARITY_TEST(mul_inplace_bf16_incong)  { test_mul_inplace_misaligned(5,  2, 1021, true,  0x93ull); }

int main() { return run_all("elementwise cpu/gpu parity"); }
