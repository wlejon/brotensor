// CPU↔GPU parity tests for elementwise activations and adds.

#include "parity_helpers.h"

#include <brotensor/ops.h>

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

    Tensor gy = to_bf16_cuda(y), gx = to_bf16_cuda(x);
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

    Tensor gy = to_bf16_cuda(y);
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

int main() { return run_all("elementwise cpu/gpu parity"); }
