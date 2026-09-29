// Test suite for Phase 1 AMD GPU (ROCm/HIP) backend integration in brotensor.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

static bool contains(const std::vector<Device>& v, Device d) {
    return std::find(v.begin(), v.end(), d) != v.end();
}

static void test_hip_probe_and_registration() {
    std::printf("test_hip_probe_and_registration\n");
    brotensor::init();

    CHECK(brotensor::is_available(Device::HIP));

    int count = brotensor::hip_device_count();
    std::printf("  HIP device count: %d\n", count);
    CHECK(count >= 1);

    std::vector<Device> devs = brotensor::available_devices();
    CHECK(contains(devs, Device::HIP));
    CHECK(contains(devs, Device::hip(0)));

    Device def = brotensor::default_device();
    std::printf("  Default device: %s\n", brotensor::device_name(def));
    CHECK(def == Device::HIP);

    CHECK(std::strcmp(brotensor::device_name(Device::HIP), "hip") == 0);
    CHECK(brotensor::to_string(Device::HIP) == "hip");

    std::string prod = brotensor::device_product_name(Device::HIP);
    std::printf("  HIP device 0 product name: %s\n", prod.c_str());
    CHECK(!prod.empty());
}

static void test_hip_alloc_and_lifecycle() {
    std::printf("test_hip_alloc_and_lifecycle\n");
    Tensor a = Tensor::zeros_on(Device::HIP, 4, 8);
    CHECK(a.rows == 4);
    CHECK(a.cols == 8);
    CHECK(a.size() == 32);
    CHECK(a.data != nullptr);
    CHECK(a.device == Device::HIP);
    CHECK(a.device.is_hip());
    CHECK(a.device.is_gpu());
    CHECK(!a.device.is_cpu());

    a.zero();

    a.resize(2, 3);
    CHECK(a.rows == 2);
    CHECK(a.cols == 3);
    CHECK(a.size() == 6);
    CHECK(a.device == Device::HIP);

    // Move ctor
    Tensor b = std::move(a);
    CHECK(b.rows == 2 && b.cols == 3);
    CHECK(b.device == Device::HIP);
    CHECK(a.data == nullptr);

    // Move assignment
    Tensor c;
    c = std::move(b);
    CHECK(c.rows == 2 && c.cols == 3);
    CHECK(c.device == Device::HIP);
    CHECK(b.data == nullptr);
}

static void test_hip_host_device_transfer() {
    std::printf("test_hip_host_device_transfer\n");
    std::vector<float> host_in = {1.5f, -2.25f, 3.125f, 0.0f, -42.0f, 100.5f};
    Tensor d = Tensor::from_host_on(Device::HIP, host_in.data(), 2, 3);
    CHECK(d.rows == 2);
    CHECK(d.cols == 3);
    CHECK(d.device == Device::HIP);

    std::vector<float> host_out = d.to_host_vector();
    CHECK(host_out.size() == host_in.size());
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(host_out[i] - host_in[i]) < 1e-6f);
    }

    // Clone on HIP
    Tensor d_cloned = d.clone();
    CHECK(d_cloned.rows == 2);
    CHECK(d_cloned.cols == 3);
    CHECK(d_cloned.device == Device::HIP);
    CHECK(d_cloned.data != d.data);

    std::vector<float> clone_out = d_cloned.to_host_vector();
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(clone_out[i] - host_in[i]) < 1e-6f);
    }

    // Inter-device transfer to CPU and back to HIP
    Tensor cpu_t = d.to(Device::CPU);
    CHECK(cpu_t.device == Device::CPU);
    CHECK(cpu_t.rows == 2);
    CHECK(cpu_t.cols == 3);
    const float* cp = cpu_t.host_f32();
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(cp[i] - host_in[i]) < 1e-6f);
    }

    Tensor back_hip = cpu_t.to(Device::HIP);
    CHECK(back_hip.device == Device::HIP);
    std::vector<float> back_out = back_hip.to_host_vector();
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        CHECK(std::fabs(back_out[i] - host_in[i]) < 1e-6f);
    }
}

static void test_hip_memory_and_sync() {
    std::printf("test_hip_memory_and_sync\n");
    std::size_t free_b = 0, total_b = 0;
    bool ok = brotensor::device_mem_info(Device::HIP, free_b, total_b);
    CHECK(ok);
    std::printf("  HIP mem info: free=%.2f MB, total=%.2f MB\n",
                static_cast<double>(free_b) / (1024.0 * 1024.0),
                static_cast<double>(total_b) / (1024.0 * 1024.0));
    CHECK(total_b > 0);

    brotensor::sync(Device::HIP);
    brotensor::sync_all();

    (void)brotensor::device_mem_trim(Device::HIP, 0);
}

static void test_hip_device_scope() {
    std::printf("test_hip_device_scope\n");
    const Device outer = brotensor::default_device();

    {
        brotensor::DeviceScope s_cpu(Device::CPU);
        CHECK(brotensor::default_device() == Device::CPU);
        Tensor t_cpu = Tensor::zeros(2, 2);
        CHECK(t_cpu.device == Device::CPU);

        {
            brotensor::DeviceScope s_hip(Device::HIP);
            CHECK(brotensor::default_device() == Device::HIP);
            Tensor t_hip = Tensor::zeros(2, 2);
            CHECK(t_hip.device == Device::HIP);
        }
        CHECK(brotensor::default_device() == Device::CPU);
    }
    CHECK(brotensor::default_device() == outer);
}

int main() {
    std::printf("test_hip running...\n");
    test_hip_probe_and_registration();
    test_hip_alloc_and_lifecycle();
    test_hip_host_device_transfer();
    test_hip_memory_and_sync();
    test_hip_device_scope();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll HIP Phase 1 checks passed successfully.\n");
    return 0;
}
