// brotensor::idle(): a non-blocking "has the queued work finished". Queued
// work that takes a while reads as not idle at first, polling alone (no sync)
// reaches idle, and after a sync the device is idle. CPU is always idle.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

int main() {
    bt::init();
    CHECK(bt::idle(Device::CPU));
    const Device dev = bt_parity::gpu_device();
    if (dev == Device::CPU) {
        std::printf("no GPU device - CPU only\n");
        return g_failures ? 1 : 0;
    }
    std::printf("test_idle on %s\n", bt::to_string(dev).c_str());

    const int n = 2048;
    std::vector<float> h(static_cast<std::size_t>(n) * n, 0.001f), hb(n, 0.0f);
    Tensor W = Tensor::from_host_on(dev, h.data(), n, n);
    Tensor B = Tensor::from_host_on(dev, hb.data(), n, 1);
    Tensor x = Tensor::from_host_on(dev, h.data(), n, n);
    bt::sync(dev);
    CHECK(bt::idle(dev));

    // Seconds of matmuls on any GPU: queued at once, so not finished yet.
    auto queue = [&](int k) {
        Tensor y;
        for (int i = 0; i < k; ++i) bt::linear_forward_batched(W, B, x, y);
        return y;
    };
    Tensor y = queue(200);
    CHECK(!bt::idle(dev));

    // Polling alone finishes it (a backend that batches commands must
    // submit them for idle() to ever become true).
    const auto t0 = std::chrono::steady_clock::now();
    int polls = 0;
    while (!bt::idle(dev)) {
        ++polls;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  idle after %d polls, %.3f s\n", polls, s);
    CHECK(polls > 0);
    CHECK(s < 60);
    CHECK(bt::idle(dev));

    y = queue(20);
    bt::sync(dev);
    CHECK(bt::idle(dev));

    std::printf(g_failures ? "FAILED (%d)\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
