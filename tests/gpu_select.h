#pragma once

// The GPU a backend-neutral test runs on, one rule for every suite:
// BROTENSOR_TEST_GPU=vulkan|cuda|metal picks that backend when it is
// registered, otherwise the first registered of CUDA, Metal, Vulkan.
// Device::CPU means "no GPU backend" (the suite skips). ctest registers the
// generic GPU suites a second time as <name>_vulkan with
// BROTENSOR_TEST_GPU=vulkan and BROTENSOR_DEFAULT_DEVICE=vulkan, so the same
// test (and every Device::CUDA it names, which aliases to Vulkan then) runs
// on Vulkan; in a Vulkan-only build Vulkan is the only choice.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cstdlib>
#include <initializer_list>
#include <string>

namespace bt_test {

inline brotensor::Device gpu() {
    using brotensor::Device;
    static const Device d = [] {
        if (const char* e = std::getenv("BROTENSOR_TEST_GPU")) {
            const std::string want(e);
            const Device pick = want == "vulkan" ? Device::VULKAN
                              : want == "cuda"   ? Device::CUDA
                              : want == "metal"  ? Device::Metal
                                                 : Device::CPU;
            if (!pick.is_cpu() && brotensor::is_available(pick)) return pick;
        }
        for (Device c : {Device::CUDA, Device::Metal, Device::VULKAN}) {
            if (brotensor::is_available(c)) return c;
        }
        return Device::CPU;
    }();
    return d;
}

inline bool has_gpu() { return !gpu().is_cpu(); }

inline const char* gpu_name() {
    switch (gpu().type) {
        case brotensor::DeviceType::CUDA: return "CUDA";
        case brotensor::DeviceType::Metal: return "Metal";
        case brotensor::DeviceType::VULKAN: return "Vulkan";
        default: return "CPU";
    }
}

}  // namespace bt_test
