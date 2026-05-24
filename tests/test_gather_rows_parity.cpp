// CPU↔GPU parity tests for gather_rows / scatter_rows_add.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Dtype;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-4f;

Tensor make_idx_host(const std::vector<int32_t>& v) {
    Tensor h = Tensor::zeros_on(brotensor::Device::CPU,
                                static_cast<int>(v.size()), 1,
                                Dtype::INT32);
    auto* p = static_cast<int32_t*>(h.host_raw_mut());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return h;
}

void run_gather(int R, int C, const std::vector<int32_t>& idx,
                uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(R, C);
    fill_random(X, rng, 1.0f);
    Tensor I_cpu = make_idx_host(idx);

    Tensor cpu_Y;
    brotensor::gather_rows(X, I_cpu, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gI = I_cpu.to(gpu_device());
    Tensor gpu_Y;
    brotensor::gather_rows(gX, gI, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "gather", kAtol, kRtol);
}

void run_scatter(int R, int C, const std::vector<int32_t>& idx,
                 uint64_t seed) {
    SplitMix64 rng(seed);
    const int M = static_cast<int>(idx.size());
    Tensor dY = Tensor::mat(M, C);
    fill_random(dY, rng, 1.0f);
    Tensor I_cpu = make_idx_host(idx);

    Tensor cpu_dX;
    brotensor::scatter_rows_add(dY, I_cpu, R, cpu_dX);

    Tensor gdY = dY.to(gpu_device());
    Tensor gI  = I_cpu.to(gpu_device());
    Tensor gpu_dX;
    brotensor::scatter_rows_add(gdY, gI, R, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "scatter",
                    kAtol, kRtol);
}

} // namespace

BT_PARITY_TEST(gather_basic) {
    run_gather(16, 8, {0, 5, 2, 15, 9, 3}, 0xC500ull);
}
BT_PARITY_TEST(gather_dup) {
    run_gather(8, 12, {3, 3, 3, 0, 7, 0, 1, 1, 1, 7}, 0xC501ull);
}
BT_PARITY_TEST(gather_wide) {
    std::vector<int32_t> v;
    for (int i = 0; i < 64; ++i) v.push_back(i % 32);
    run_gather(32, 256, v, 0xC502ull);
}

BT_PARITY_TEST(scatter_basic) {
    run_scatter(16, 8, {0, 5, 2, 15, 9, 3}, 0xC510ull);
}
BT_PARITY_TEST(scatter_dup) {
    run_scatter(8, 12, {3, 3, 3, 0, 7, 0, 1, 1, 1, 7}, 0xC511ull);
}
BT_PARITY_TEST(scatter_wide) {
    std::vector<int32_t> v;
    for (int i = 0; i < 64; ++i) v.push_back(i % 32);
    run_scatter(32, 256, v, 0xC512ull);
}

int main() { return run_all("gather_rows cpu/gpu parity"); }
