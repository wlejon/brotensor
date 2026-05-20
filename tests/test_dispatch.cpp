// Coverage for the runtime dispatch layer: init() idempotence, device
// availability, the default-device policy (set_default_device + DeviceScope),
// device-tagged tensor construction, a CPU op routed through the dispatcher,
// and mixed-device misuse detection.
//
// In the "always built" group — every CUDA-specific assertion is guarded by
// is_available(Device::CUDA) so this builds and passes on a CPU-only build.
//
// Convention matches the rest of tests/: plain executable, CHECK macro,
// exits non-zero on failure, prints progress.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool contains(const std::vector<Device>& v, Device d) {
    return std::find(v.begin(), v.end(), d) != v.end();
}

// 1. init() is idempotent.
static void test_init_idempotent() {
    std::printf("test_init_idempotent\n");
    bool threw = false;
    try {
        brotensor::init();
        brotensor::init();  // second call must be a harmless no-op
    } catch (const std::exception& e) {
        threw = true;
        std::printf("  unexpected throw: %s\n", e.what());
    }
    CHECK(!threw);
}

// 2. CPU is always available and listed.
static void test_cpu_always_available() {
    std::printf("test_cpu_always_available\n");
    CHECK(brotensor::is_available(Device::CPU));

    std::vector<Device> devs = brotensor::available_devices();
    CHECK(!devs.empty());
    CHECK(contains(devs, Device::CPU));

    // available_devices() and is_available() must agree.
    for (Device d : devs) CHECK(brotensor::is_available(d));
}

// 3. default_device() returns a registered device.
static void test_default_device_registered() {
    std::printf("test_default_device_registered\n");
    const Device d = brotensor::default_device();
    CHECK(brotensor::is_available(d));
    CHECK(contains(brotensor::available_devices(), d));
}

// 4. set_default_device(CPU) makes the next zeros() land on CPU.
static void test_set_default_device() {
    std::printf("test_set_default_device\n");
    const Device saved = brotensor::default_device();

    brotensor::set_default_device(Device::CPU);
    CHECK(brotensor::default_device() == Device::CPU);

    Tensor t = Tensor::zeros(3, 4);
    CHECK(t.device == Device::CPU);
    CHECK(t.rows == 3 && t.cols == 4);

    // Restore the prior policy.
    brotensor::set_default_device(saved);
    CHECK(brotensor::default_device() == saved);
}

// 5. DeviceScope overrides the default for its lifetime and restores after.
static void test_device_scope() {
    std::printf("test_device_scope\n");
    // Establish a known default that is NOT CPU if possible, so we can prove
    // the scope both overrides and restores. Falls back to CPU otherwise.
    const Device outer = brotensor::default_device();

    {
        brotensor::DeviceScope scope(Device::CPU);
        CHECK(brotensor::default_device() == Device::CPU);
        Tensor inside = Tensor::zeros(2, 2);
        CHECK(inside.device == Device::CPU);
    }

    // Previous default restored on scope exit.
    CHECK(brotensor::default_device() == outer);

    // Nested scopes restore correctly.
    {
        brotensor::DeviceScope s1(Device::CPU);
        CHECK(brotensor::default_device() == Device::CPU);
        {
            brotensor::DeviceScope s2(Device::CPU);
            CHECK(brotensor::default_device() == Device::CPU);
        }
        CHECK(brotensor::default_device() == Device::CPU);
    }
    CHECK(brotensor::default_device() == outer);
}

// 6. A CPU op runs through the dispatcher and produces correct values.
static void test_cpu_op_through_dispatch() {
    std::printf("test_cpu_op_through_dispatch\n");
    Tensor x = Tensor::zeros_on(Device::CPU, 5, 1);
    Tensor y = Tensor::zeros_on(Device::CPU, 5, 1);
    float* xp = x.host_f32_mut();
    xp[0] = -3.0f; xp[1] = -0.5f; xp[2] = 0.0f; xp[3] = 0.25f; xp[4] = 7.0f;

    brotensor::relu_forward(x, y);

    CHECK(y.device == Device::CPU);
    CHECK(y.host_f32()[0] == 0.0f);
    CHECK(y.host_f32()[1] == 0.0f);
    CHECK(y.host_f32()[2] == 0.0f);
    CHECK(y.host_f32()[3] == 0.25f);
    CHECK(y.host_f32()[4] == 7.0f);
}

// 7. Mixed-device operands passed to one op must throw. Only meaningful when
//    CUDA is available (we need a second, distinct device to mix).
static void test_mixed_device_misuse() {
    std::printf("test_mixed_device_misuse\n");
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("  CUDA not available - skipping mixed-device sub-case\n");
        return;
    }

    Tensor x_cpu  = Tensor::zeros_on(Device::CPU,  4, 1);
    Tensor y_cuda = Tensor::zeros_on(Device::CUDA, 4, 1);

    bool threw = false;
    try {
        // relu_forward dispatches on its operands' device; a CPU input with a
        // CUDA output is an operand-consistency violation.
        brotensor::relu_forward(x_cpu, y_cuda);
    } catch (const std::runtime_error&) {
        threw = true;
    } catch (const std::exception&) {
        // Any std::exception subtype is acceptable evidence the op rejected
        // the mismatch rather than corrupting memory.
        threw = true;
    }
    CHECK(threw);
}

// 8. If CUDA is available, set_default_device(CUDA) lands a tensor on CUDA.
static void test_cuda_default_device() {
    std::printf("test_cuda_default_device\n");
    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("  CUDA not available - skipping CUDA-default sub-case\n");
        return;
    }
    const Device saved = brotensor::default_device();

    brotensor::set_default_device(Device::CUDA);
    CHECK(brotensor::default_device() == Device::CUDA);
    Tensor t = Tensor::zeros(3, 3);
    CHECK(t.device == Device::CUDA);

    // DeviceScope variant too.
    {
        brotensor::set_default_device(Device::CPU);
        brotensor::DeviceScope scope(Device::CUDA);
        Tensor inside = Tensor::zeros(2, 2);
        CHECK(inside.device == Device::CUDA);
    }
    CHECK(brotensor::default_device() == Device::CPU);

    brotensor::set_default_device(saved);
}

int main() {
    brotensor::init();
    std::printf("test_dispatch\n");

    test_init_idempotent();
    test_cpu_always_available();
    test_default_device_registered();
    test_set_default_device();
    test_device_scope();
    test_cpu_op_through_dispatch();
    test_mixed_device_misuse();
    test_cuda_default_device();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll dispatch checks passed.\n");
    return 0;
}
