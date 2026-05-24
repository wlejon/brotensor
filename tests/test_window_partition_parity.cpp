// CPU↔GPU parity tests for window_partition_forward / window_reverse_forward.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-4f;

void run_partition(int N, int C, int H, int W, int window, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::window_partition_forward(X, N, C, H, W, window, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::window_partition_forward(gX, N, C, H, W, window, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "wpart_fwd",
                    kAtol, kRtol);
}

void run_reverse(int N, int C, int H, int W, int window, uint64_t seed) {
    SplitMix64 rng(seed);
    const int nw_h = H / window;
    const int nw_w = W / window;
    Tensor X = Tensor::mat(N * nw_h * nw_w, C * window * window);
    fill_random(X, rng, 1.0f);

    Tensor cpu_Y;
    brotensor::window_reverse_forward(X, N, C, H, W, window, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::window_reverse_forward(gX, N, C, H, W, window, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "wrev_fwd",
                    kAtol, kRtol);
}

} // namespace

BT_PARITY_TEST(wpart_basic)    { run_partition(2, 3, 8, 8, 4, 0xC700ull); }
BT_PARITY_TEST(wpart_uneven)   { run_partition(1, 5, 12, 8, 4, 0xC701ull); }
BT_PARITY_TEST(wpart_single)   { run_partition(2, 4, 6, 6, 6, 0xC702ull); }
BT_PARITY_TEST(wpart_w7)       { run_partition(2, 3, 14, 21, 7, 0xC703ull); }

BT_PARITY_TEST(wrev_basic)     { run_reverse(2, 3, 8, 8, 4, 0xC710ull); }
BT_PARITY_TEST(wrev_uneven)    { run_reverse(1, 5, 12, 8, 4, 0xC711ull); }
BT_PARITY_TEST(wrev_w7)        { run_reverse(2, 3, 14, 21, 7, 0xC712ull); }

int main() { return run_all("window_partition cpu/gpu parity"); }
