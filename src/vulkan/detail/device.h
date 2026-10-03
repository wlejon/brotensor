#pragma once

// The Vulkan backend's process and device state.
//
//   Instance   process-wide: the loader, the VkInstance, and the list of
//              eligible physical devices (Device::vulkan(i) is entry i).
//   DeviceCtx  one per eligible physical device, created on first use: the
//              VkDevice, its graphics queue, and the Stream / Allocator /
//              Pipelines that every op on that device goes through.
//
// Eligible = Vulkan >= 1.2 with buffer device address, timeline semaphores,
// 64-bit integers, 8/16-bit storage and FP16 arithmetic in shaders, subgroup
// arithmetic in compute, and a queue family with graphics + compute. CPU
// implementations (llvmpipe, lavapipe) are skipped unless
// BROTENSOR_VK_ALLOW_CPU=1. BROTENSOR_DISABLE_VULKAN=1 skips the backend.
//
// Both objects are created once and intentionally never destroyed: tensors
// can outlive main() (statics, leaked caches) and a free at exit must still
// find its device. The driver reclaims everything when the process ends.

#include "allocator.h"
#include "pipelines.h"
#include "stream.h"
#include "vk_fns.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace brotensor { struct Tensor; }

namespace brotensor::detail::vulkan {

struct PhysInfo {
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    std::uint32_t queue_family = 0;
    bool graphics_queue = true;          // family has graphics + compute
    std::uint32_t subgroup_size = 0;
    std::uint32_t min_subgroup = 0, max_subgroup = 0;
    bool subgroup_size_control = false;  // pipelines may require a size
    bool memory_budget = false;          // VK_EXT_memory_budget
    bool cooperative_matrix = false;     // VK_KHR_cooperative_matrix (enabled when present)
    bool memory_model = false;           // vulkanMemoryModel (+ device scope), enabled when present
    bool shader_float64 = false;         // shaderFloat64, enabled when present (custom kernels)
    bool coopmat_f16 = false;            // 16x16x16 FP16 -> FP32 fragments at subgroup 32: the
                                         // GEMM kernels' tensor-core path (BROTENSOR_VK_NO_COOPMAT=1 off)
    std::string name;
    std::string driver;
};

class Instance {
public:
    // Loads the loader and enumerates devices. Never throws: when there is no
    // loader or no eligible device, count() is 0 and why() says why.
    static Instance& get();

    int count() const { return static_cast<int>(phys_.size()); }
    const PhysInfo& phys(int i) const { return phys_.at(static_cast<std::size_t>(i)); }
    const std::string& why() const { return why_; }
    VkInstance handle() const { return inst_; }
    const LoaderFns& fns() const { return fns_; }

private:
    Instance();
    LoaderFns fns_{};
    VkInstance inst_ = VK_NULL_HANDLE;
    std::uint32_t api_ = 0;
    std::vector<PhysInfo> phys_;
    std::string why_;
};

class DeviceCtx {
public:
    DeviceCtx(int index, const PhysInfo& info);
    ~DeviceCtx();
    DeviceCtx(const DeviceCtx&) = delete;
    DeviceCtx& operator=(const DeviceCtx&) = delete;

    int index() const { return index_; }
    const PhysInfo& info() const { return info_; }
    VkDevice device() const { return dev_; }
    VkQueue queue() const { return queue_; }
    const DeviceFns& fn() const { return fn_; }

    Stream&    stream()    { return *stream_; }
    Allocator& allocator() { return *alloc_; }
    Pipelines& pipelines() { return *pipes_; }

private:
    int index_;
    PhysInfo info_;
    DeviceFns fn_{};
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::unique_ptr<Stream> stream_;
    std::unique_ptr<Allocator> alloc_;
    std::unique_ptr<Pipelines> pipes_;
};

// The context for Device::vulkan(index), created on first use. Throws if the
// index is out of range or the device cannot be created.
DeviceCtx& device(int index);

// The context if it has been created, else null. For calls that have nothing
// to do on an untouched device (sync, trim) and must not create one.
DeviceCtx* device_if_created(int index);

// Shorthand used by the op files: the context a tensor lives on, and its
// device address.
DeviceCtx& device_of(const ::brotensor::Tensor& t);
inline std::uint64_t addr(const void* p) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p));
}

}  // namespace brotensor::detail::vulkan
