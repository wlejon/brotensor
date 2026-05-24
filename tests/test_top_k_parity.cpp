// CPU↔GPU parity tests for top_k_rows.
//
// X (R, C) FP32 -> Vals (R, k) FP32 + Idx (R, k) INT32. Deterministic; ties
// broken by smaller column index — same rule on both backends.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <cstring>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Dtype;

namespace {

void run(int R, int C, int k, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(R, C);
    fill_random(X, rng, 1.0f);

    Tensor cpu_V, cpu_I;
    brotensor::top_k_rows(X, k, cpu_V, cpu_I);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_V, gpu_I;
    brotensor::top_k_rows(gX, k, gpu_V, gpu_I);

    compare_tensors(cpu_V, download_to_host(gpu_V), "top_k_vals",
                    1e-5f, 1e-4f);

    // Indices: compare as int32 directly.
    Tensor host_I = download_to_host(gpu_I);
    BT_CHECK(host_I.rows == cpu_I.rows && host_I.cols == cpu_I.cols);
    const int32_t* a = static_cast<const int32_t*>(cpu_I.data);
    const int32_t* b = static_cast<const int32_t*>(host_I.data);
    for (int i = 0; i < cpu_I.size(); ++i) {
        BT_CHECK(a[i] == b[i]);
    }
}

} // namespace

BT_PARITY_TEST(top_k_basic)     { run(8, 64,  5, 0xC300ull); }
BT_PARITY_TEST(top_k_small_k)   { run(4, 16,  1, 0xC301ull); }
BT_PARITY_TEST(top_k_full)      { run(3, 12, 12, 0xC302ull); }
BT_PARITY_TEST(top_k_wide)      { run(2, 1024, 16, 0xC303ull); }
BT_PARITY_TEST(top_k_many_rows) { run(64, 32, 4, 0xC304ull); }

int main() { return run_all("top_k_rows cpu/gpu parity"); }
